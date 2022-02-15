import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ._gaussian_mixture import GaussianMixture, calc_time, calc_amp
from ._bayesian_mixture import BayesianGaussianMixture


to_seconds = lambda t: t.timestamp(tz="UTC")
from_seconds = lambda t: pd.Timestamp.utcfromtimestamp(t).strftime(
    "%Y-%m-%dT%H:%M:%S.%f"
)[:-3]
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
    nan_idx = meta.isnull().any(axis=1)
    return (
        data[~nan_idx],
        locs[~nan_idx],
        phase_type[~nan_idx],
        phase_weight[~nan_idx],
        picks.index.to_numpy()[~nan_idx],
    )


def association(
    data,
    locs,
    phase_type,
    phase_weight,
    num_sta,
    pick_idx,
    event_idx0,
    config,
    method="BGMM",
    pbar=None,
):

    db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(
        np.hstack([data[:, 0:1], locs[:, :2] / 6.0])
    )
    # db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(data[:, 0:1])
    labels = db.labels_
    unique_labels = set(labels)
    events = []
    assignment = []  ## from picks to events
    event_idx = event_idx0

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

        ## initialization with 5 horizontal points and N time points
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

        ## initialization with 1 horizontal center points and N time points
        if len(data_) < len(centers_init):
            num_event_init = min(
                max(
                    int(len(data_) / min(num_sta, 20) * config["oversample_factor"]), 1
                ),
                len(data_),
            )
            centers_init = np.vstack(
                [
                    np.ones(num_event_init) * np.mean(config["x(km)"]),
                    np.ones(num_event_init) * np.mean(config["y(km)"]),
                    np.zeros(num_event_init),
                    np.linspace(
                        data_[:, 0].min() - 0.1 * time_range,
                        data_[:, 0].max() + 0.1 * time_range,
                        num_event_init,
                    ),
                ]
            ).T

        ## run clustering
        vel = config["vel"] if "vel" in config else {"p":6.0, "s":6.0/1.75}
        mean_precision_prior = 0.01 / time_range
        if not config["use_amplitude"]:
            covariance_prior = np.array([[1.0]]) * 5
            data_ = data_[:, 0:1]
        else:
            covariance_prior = np.array([[1.0, 0.0], [0.0, 0.5]]) * 5
        
        if method == "BGMM":
            gmm = BayesianGaussianMixture(
                n_components=len(centers_init),
                weight_concentration_prior=1 / len(centers_init),
                mean_precision_prior=mean_precision_prior,
                covariance_prior=covariance_prior,
                init_params="centers",
                centers_init=centers_init.copy(),
                station_locs=locs_,
                phase_type=phase_type_,
                phase_weight=phase_weight_,
                vel=vel,
                loss_type="l1",
                bounds=config["bfgs_bounds"],
                # max_covar=20 ** 2,
                # dummy_comp=True,
                # dummy_prob=0.1,
                # dummy_quantile=0.1,
            ).fit(data_)
        elif method == "GMM":
            gmm = GaussianMixture(
                n_components=len(centers_init),
                init_params="centers",
                centers_init=centers_init.copy(),
                station_locs=locs_,
                phase_type=phase_type_,
                phase_weight=phase_weight_,
                vel=vel,
                loss_type="l1",
                bounds=config["bfgs_bounds"],
                # max_covar=20 ** 2,
                dummy_comp=True,
                dummy_prob=1/(1 * np.sqrt(2*np.pi)) * np.exp(-1/2),
                dummy_quantile=0.1,
            ).fit(data_)
        else:
            raise(f"Unknown method {method}; Should be 'BGMM' or 'GMM'")

        ## run prediction
        pred = gmm.predict(data_)
        prob = np.exp(gmm.score_samples(data_))
        prob_matrix = gmm.predict_proba(data_)
        prob_eq = prob_matrix.sum(axis=0)
        #  prob = prob_matrix[range(len(data_)), pred]
        #  score = gmm.score(data_)
        #  score_sample = gmm.score_samples(data_)

        ## filtering
        for i in range(len(centers_init)):
            tmp_data = data_[pred == i]
            tmp_locs = locs_[pred == i]
            tmp_phase_type = phase_type_[pred == i]
            if len(tmp_data) < config["min_picks_per_eq"]:
                continue

            ## filter by time
            t_ = calc_time(gmm.centers_[i:i+1, :len(config["dims"])+1], tmp_locs, tmp_phase_type, vel=vel)
            diff_t = t_ - tmp_data[:,0:1]
            idx_t = (diff_t**2 < config["max_sigma11"]).squeeze()
            if len(tmp_data[idx_t]) < config["min_picks_per_eq"]:
                continue
            if config["use_amplitude"]:
                gmm.covariances_[i, 0, 0] = np.mean((diff_t[idx_t])**2)
            else:
                gmm.covariances_[i, 0] = np.mean((diff_t[idx_t])**2)

            ## filter by amplitude
            if config["use_amplitude"]:
                a_ = calc_amp(gmm.centers_[i:i+1, len(config["dims"])+1:len(config["dims"])+2], 
                                    gmm.centers_[i:i+1, :len(config["dims"])+1], 
                                    tmp_locs)
                diff_a = a_ - tmp_data[:,1:2]
                idx_a = (diff_a**2 < config["max_sigma22"]).squeeze()
                if len(tmp_data[idx_t & idx_a]) < config["min_picks_per_eq"]:
                    continue
                gmm.covariances_[i, 1, 1] = np.mean((diff_a[idx_a])**2)

            event = {
                "time(s)": gmm.centers_[i, len(config["dims"])],
                "magnitude": gmm.centers_[i, len(config["dims"]) + 1] if config["use_amplitude"] else 999,
                # "covariance": gmm.covariances_[i, ...],
                "sigma_time": gmm.covariances_[i, 0, 0] if config["use_amplitude"] else gmm.covariances_[i, 0],
                "sigma_amp": gmm.covariances_[i, 1, 1] if config["use_amplitude"] else 0,
                "sigma_cov": gmm.covariances_[i, 0, 1] if config["use_amplitude"] else 0,
                "prob_gamma": prob_eq[i],
                "event_idx": event_idx,
            }
            for j, k in enumerate(config["dims"]):  ## add location
                event[k] = gmm.centers_[i, j]
            events.append(event)
            for pi, pr in zip(pick_idx_[pred==i], prob):
                assignment.append((pi, event_idx, pr))
            event_idx += 1

    return events, assignment