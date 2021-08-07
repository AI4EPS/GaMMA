import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from gmma import mixture

to_seconds = lambda t: t.timestamp(tz="UTC")
from_seconds = lambda t: pd.Timestamp.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
# to_seconds = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
# from_seconds = lambda t: [datetime.utcfromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for x in t]


def convert_picks_csv(picks, stations, config):
    t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    a = picks["amp"].apply(lambda x: np.log10(x * 1e2)).to_numpy()
    data = np.stack([t, a]).T
    meta = stations.merge(picks["id"], how="right", on="id")
    locs = meta[config["dims"]].to_numpy()
    phase_type = picks["type"].apply(lambda x: x.lower()).to_numpy()
    phase_weight = picks["prob"].to_numpy()[:, np.newaxis]
    return data, locs, phase_type, phase_weight


def association(data, locs, phase_type, phase_weight, num_sta, pick_idx, event_idx0, config, stations, pbar=None):

    db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(
        np.hstack([data[:, 0:1], locs[:, :2] / 6.0])
    )
    labels = db.labels_
    unique_labels = set(labels)
    events = []
    preds = []
    probs = []

    assignment = []
    for k in unique_labels:
        if k == -1:

            continue

        class_mask = labels == k
        data_ = data[class_mask]
        locs_ = locs[class_mask]
        phase_type_ = phase_type[class_mask]
        phase_weight_ = phase_weight[class_mask]
        pick_idx_ = pick_idx[class_mask]

        if len(pick_idx_) < config["min_picks_per_eq"]:
            continue

        if pbar is not None:
            pbar.set_description(f"Process {len(data_)} picks")

        time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)

        # initialization with 5 horizontal points and N time points
        num_event_loc_init = 5
        num_event_init = min(
            max(int(len(data_) / min(num_sta, 20) * config["oversample_factor"]), 1),
            max(len(data_) // num_event_loc_init, 1),
        )
        x0, xn = config["x(km)"]
        y0, yn = config["y(km)"]
        x1, y1 = np.mean(config["x(km)"]), np.mean(config["y(km)"])
        event_loc_init = [
            ((x0 + x1) / 2, (y0 + y1) / 2),
            ((x0 + x1) / 2, (yn + y1) / 2),
            ((xn + x1) / 2, (y0 + y1) / 2),
            ((xn + x1) / 2, (yn + y1) / 2),
            (x1, y1),
        ]
        num_event_time_init = max(num_event_init // num_event_loc_init, 1)
        centers_init = np.vstack(
            [
                np.vstack(
                    [
                        np.ones(num_event_time_init) * x,
                        np.ones(num_event_time_init) * y,
                        np.zeros(num_event_time_init),
                        np.linspace(
                            data_[:, 0].min() - 0.1 * time_range,
                            data_[:, 0].max() + 0.1 * time_range,
                            num_event_time_init,
                        ),
                    ]
                ).T
                for x, y in event_loc_init
            ]
        )

        # initialization with 1 horizontal center points and N time points
        if len(data_) < len(centers_init):
            num_event_init = min(max(int(len(data_) / min(num_sta, 20) * config["oversample_factor"]), 1), len(data_))
            centers_init = np.vstack(
                [
                    np.ones(num_event_init) * np.mean(stations["x(km)"]),
                    np.ones(num_event_init) * np.mean(stations["y(km)"]),
                    np.zeros(num_event_init),
                    np.linspace(
                        data_[:, 0].min() - 0.1 * time_range, data_[:, 0].max() + 0.1 * time_range, num_event_init
                    ),
                ]
            ).T

        mean_precision_prior = 0.1 / time_range
        if not config["use_amplitude"]:
            covariance_prior = np.array([[1]]) * 5
            data_ = data_[:, 0:1]
        else:
            covariance_prior = np.array([[1, 0], [0, 1]]) * 5

        gmm = mixture.BayesianGaussianMixture(
            n_components=len(centers_init),
            weight_concentration_prior=1 / len(centers_init),
            mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior,
            init_params="centers",
            centers_init=centers_init.copy(),
            station_locs=locs_,
            phase_type=phase_type_,
            phase_weight=phase_weight_,
            loss_type="l1",
            bounds=config["bfgs_bounds"],
            max_covar=20 ** 2,
        ).fit(data_)

        pred = gmm.predict(data_)
        prob_matrix = gmm.predict_proba(data_)
        prob_eq = prob_matrix.mean(axis=0)
        #  prob = prob_matrix[range(len(data_)), pred]
        #  score = gmm.score(data_)
        #  score_sample = gmm.score_samples(data_)
        prob = np.exp(gmm.score_samples(data_))

        ## filtering
        raw_idx = np.arange(len(centers_init))
        idx = np.array(
            [True if len(data_[pred == i, 0]) >= config["min_picks_per_eq"] else False for i in range(len(prob_eq))]
        )  # & (prob_eq > 1/num_event) #& (sigma_eq[:, 0,0] < 40)
        raw_idx = raw_idx[idx]
        time = gmm.centers_[idx, len(config["dims"])]
        loc = gmm.centers_[idx, : len(config["dims"])]
        if config["use_amplitude"]:
            mag = gmm.centers_[idx, len(config["dims"]) + 1]
        sigma_eq = gmm.covariances_[idx, ...]
        
        event_id = {} ## map from raw cluster id to filtered event id
        for i in range(len(time)):
            tmp = {"time(s)": time[i], "magnitude": mag[i], "sigma": sigma_eq[i].tolist()}
            for j, k in enumerate(config["dims"]):
                tmp[k] = loc[i][j]
            events.append(tmp)
            event_id[raw_idx[i]] = i

        for i in range(len(pick_idx_)):
            ## pred[i] is the raw cluster id; then event_id[pred[i]] maps to the new index of the selected events
            if pred[i] in event_id:
                assignment.append((pick_idx_[i], event_id[pred[i]] + event_idx0, prob[i]))

        event_idx0 += len(time)

    return events, assignment
