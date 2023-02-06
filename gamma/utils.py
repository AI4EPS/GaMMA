from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture
from .seismic_ops import calc_amp, calc_time

to_seconds = lambda t: t.timestamp(tz="UTC")
from_seconds = lambda t: pd.Timestamp.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
# to_seconds = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
# from_seconds = lambda t: [datetime.utcfromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for x in t]


def convert_picks_csv(picks, stations, config):
    # t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    if type(picks["timestamp"].iloc[0]) is str:
        picks.loc[:, "timestamp"] = picks["timestamp"].apply(lambda x: datetime.fromisoformat(x))
    t = (
        picks["timestamp"]
        .apply(lambda x: x.tz_convert("UTC").timestamp() if x.tzinfo is not None else x.tz_localize("UTC").timestamp())
        .to_numpy()
    )
    # t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    timestamp0 = np.min(t)
    t = t - timestamp0
    if config["use_amplitude"]:
        a = picks["amp"].apply(lambda x: np.log10(x * 1e2)).to_numpy()  ##cm/s
        data = np.stack([t, a]).T
    else:
        data = t[:, np.newaxis]
    meta = stations.merge(picks["id"], how="right", on="id")
    locs = meta[config["dims"]].to_numpy()
    phase_type = picks["type"].apply(lambda x: x.lower()).to_numpy()
    phase_weight = picks["prob"].to_numpy()[:, np.newaxis]
    pick_station_id = picks.apply(lambda x: x.id + "_" + x.type, axis=1).to_numpy()
    nan_idx = meta.isnull().any(axis=1)
    return (
        data[~nan_idx],
        locs[~nan_idx],
        phase_type[~nan_idx],
        phase_weight[~nan_idx],
        picks.index.to_numpy()[~nan_idx],
        pick_station_id[~nan_idx],
        timestamp0,
    )


def association(picks, stations, config, event_idx0=0, method="BGMM", **kwargs):

    data, locs, phase_type, phase_weight, pick_idx, pick_station_id, timestamp0 = convert_picks_csv(
        picks, stations, config
    )

    vel = config["vel"] if "vel" in config else {"p": 6.0, "s": 6.0 / 1.73}
    if "covariance_prior" in config:
        covariance_prior_pre = config["covariance_prior"]
    else:
        covariance_prior_pre = [5.0, 2.0]

    if ("use_dbscan" in config) and config["use_dbscan"]:
        db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(
            np.hstack([data[:, 0:1], locs[:, :2] / vel["p"]])
        )
        # db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(data[:, 0:1])
        labels = db.labels_
        unique_labels = set(labels)
    else:
        labels = np.zeros(len(data))
        unique_labels = [0]

    events = []
    assignment = []  ## from picks to events
    event_idx = event_idx0

    pbar = tqdm(total=len(data), desc="Association")
    for k in unique_labels:

        if k == -1:
            data_ = data[labels == k]
            pbar.set_description(f"Skip {len(data_)} picks")
            pbar.update(len(data_))
            continue

        class_mask = labels == k
        data_ = data[class_mask]
        locs_ = locs[class_mask]
        phase_type_ = phase_type[class_mask]
        phase_weight_ = phase_weight[class_mask]
        pick_idx_ = pick_idx[class_mask]
        pick_station_id_ = pick_station_id[class_mask]

        if len(pick_idx_) < max(3, config["min_picks_per_eq"]):
            pbar.set_description(f"Skip {len(data_)} picks")
            pbar.update(len(data_))
            continue

        if pbar is not None:
            pbar.set_description(f"Process {len(data_)} picks")
            pbar.update(len(data_))

        time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)

        ## initialization with 5 horizontal points and N//5 time points
        centers_init = init_centers(config, data_, locs_, time_range)

        ## run clustering
        mean_precision_prior = 0.01 / time_range
        if not config["use_amplitude"]:
            covariance_prior = np.array([[covariance_prior_pre[0]]])
            data_ = data_[:, 0:1]
        else:
            covariance_prior = np.array([[covariance_prior_pre[0], 0.0], [0.0, covariance_prior_pre[1]]])

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
                n_components=len(centers_init) + 1,
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
                dummy_prob=1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2),
                dummy_quantile=0.1,
            ).fit(data_)
        else:
            raise (f"Unknown method {method}; Should be 'BGMM' or 'GMM'")

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
            tmp_pick_station_id = pick_station_id_[pred == i]
            tmp_phase_type = phase_type_[pred == i]
            if (len(tmp_data) == 0) or (len(tmp_data) < config["min_picks_per_eq"]):
                continue

            ## filter by time
            t_ = calc_time(gmm.centers_[i : i + 1, : len(config["dims"]) + 1], tmp_locs, tmp_phase_type, vel=vel)
            diff_t = np.abs(t_ - tmp_data[:, 0:1])
            idx_t = (diff_t < config["max_sigma11"]).squeeze(axis=1)
            idx_filter = idx_t
            if len(tmp_data[idx_filter]) < config["min_picks_per_eq"]:
                continue

            ## filter multiple picks at the same station
            unique_sta_id = {}
            for j, k in enumerate(tmp_pick_station_id):
                if (k not in unique_sta_id) or (diff_t[j] < unique_sta_id[k][1]):
                    unique_sta_id[k] = (j, diff_t[j])
            idx_s = np.zeros(len(idx_t)).astype(bool)  ## based on station
            for k in unique_sta_id:
                idx_s[unique_sta_id[k][0]] = True
            idx_filter = idx_filter & idx_s
            if len(tmp_data[idx_filter]) < config["min_picks_per_eq"]:
                continue
            gmm.covariances_[i, 0, 0] = np.mean((diff_t[idx_t]) ** 2)

            ## filter by amplitude
            if config["use_amplitude"]:
                a_ = calc_amp(
                    gmm.centers_[i : i + 1, len(config["dims"]) + 1 : len(config["dims"]) + 2],
                    gmm.centers_[i : i + 1, : len(config["dims"]) + 1],
                    tmp_locs,
                )
                diff_a = np.abs(a_ - tmp_data[:, 1:2])
                idx_a = (diff_a < config["max_sigma22"]).squeeze()
                idx_filter = idx_filter & idx_a
                if len(tmp_data[idx_filter]) < config["min_picks_per_eq"]:
                    continue

                if "max_sigma12" in config:
                    idx_cov = np.abs(gmm.covariances_[i, 0, 1]) < config["max_sigma12"]
                    idx_filter = idx_filter & idx_cov
                    if len(tmp_data[idx_filter]) < config["min_picks_per_eq"]:
                        continue

                gmm.covariances_[i, 1, 1] = np.mean((diff_a[idx_a]) ** 2)

            if "min_p_picks_per_eq" in config:
                if len(tmp_data[idx_filter & (tmp_phase_type == "p")]) < config["min_p_picks_per_eq"]:
                    continue
            if "min_s_picks_per_eq" in config:
                if len(tmp_data[idx_filter & (tmp_phase_type == "s")]) < config["min_s_picks_per_eq"]:
                    continue

            event = {
                # "time": from_seconds(gmm.centers_[i, len(config["dims"])]),
                "time": datetime.utcfromtimestamp(gmm.centers_[i, len(config["dims"])] + timestamp0).isoformat(
                    timespec="milliseconds"
                ),
                # "time(s)": gmm.centers_[i, len(config["dims"])],
                "magnitude": gmm.centers_[i, len(config["dims"]) + 1] if config["use_amplitude"] else 999,
                "sigma_time": np.sqrt(gmm.covariances_[i, 0, 0]),
                "sigma_amp": np.sqrt(gmm.covariances_[i, 1, 1]) if config["use_amplitude"] else 0,
                "cov_time_amp": gmm.covariances_[i, 0, 1] if config["use_amplitude"] else 0,
                "gamma_score": prob_eq[i],
                "event_index": event_idx,
            }
            for j, k in enumerate(config["dims"]):  ## add location
                event[k] = gmm.centers_[i, j]
            events.append(event)
            for pi, pr in zip(pick_idx_[pred == i][idx_filter], prob):
                assignment.append((pi, event_idx, pr))
            event_idx += 1

    return events, assignment

def init_centers(config, data_, locs_, time_range):

    if "initial_points" in config:
        initial_points = config["initial_points"]
        if not isinstance(initial_points, list):
            initial_points = [initial_points, initial_points, initial_points]
    else:
        initial_points = [1,1,1]
    
    if ((np.prod(initial_points) + 1) * config["oversample_factor"]) > len(data_):
        initial_points = [1,1,1]

    x_init = np.linspace(config["x(km)"][0], config["x(km)"][1], initial_points[0]+2)[1:-1]
    y_init = np.linspace(config["y(km)"][0], config["y(km)"][1], initial_points[1]+2)[1:-1]
    z_init = np.linspace(config["z(km)"][0], config["z(km)"][1], initial_points[2])
    x_init = np.broadcast_to(x_init[:, np.newaxis, np.newaxis], initial_points).reshape(-1)
    y_init = np.broadcast_to(y_init[np.newaxis, :, np.newaxis], initial_points).reshape(-1)
    z_init = np.broadcast_to(z_init[np.newaxis, np.newaxis, :], initial_points).reshape(-1)

    ## I found it helpful to add a point at the center of the area
    if (initial_points[0] == 2) and (initial_points[1] == 2):
        x_init = np.append(x_init, np.mean(config["x(km)"]))
        y_init = np.append(y_init, np.mean(config["y(km)"]))
        z_init = np.append(z_init, 0)
    num_xyz_init = len(x_init)

    num_sta = len(np.unique(locs_, axis=0))
    num_t_init = max(np.round(len(data_) / num_sta / num_xyz_init * config["oversample_factor"]), 1)
    num_t_init = min(int(num_t_init), len(data_)//num_xyz_init)
    t_init = np.sort(data_[:, 0])[::(len(data_[:, 0]) // num_t_init)][:num_t_init]
    # t_init = np.linspace(
    #         data_[:, 0].min() - 0.1 * time_range,
    #         data_[:, 0].max() + 0.1 * time_range,
    #         num_t_init)
    
    x_init = np.broadcast_to(x_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
    y_init = np.broadcast_to(y_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
    z_init = np.broadcast_to(z_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
    t_init = np.broadcast_to(t_init[np.newaxis, :], (num_xyz_init, num_t_init)).reshape(-1)

    if config["dims"] == ["x(km)", "y(km)", "z(km)"]:
        centers_init = np.vstack(
                [x_init, y_init, z_init, t_init]
            ).T
    elif config["dims"] == ["x(km)", "y(km)"]:
        centers_init = np.vstack(
                [x_init, y_init, t_init]
            ).T
    elif config["dims"] == ["x(km)"]:
        centers_init = np.vstack(
                [x_init, t_init]
            ).T
    else:
        raise (ValueError("Unsupported dims"))
    
    if config["use_amplitude"]:
        centers_init = np.hstack([centers_init, 1.0*np.ones((len(centers_init), 1))]) # init magnitude to 1.0

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(x_init, y_init)
    # plt.savefig("initial_points_xy.png")
    # plt.figure()
    # plt.scatter(x_init, z_init)
    # plt.savefig("initial_points_xz.png")
    # plt.figure()
    # plt.scatter(y_init, z_init)
    # plt.savefig("initial_points_yz.png")

    # plt.figure()
    # plt.scatter(t_init, x_init)
    # plt.savefig("initial_points_tx.png")
    # plt.figure()
    # plt.scatter(t_init, y_init)
    # plt.savefig("initial_points_ty.png")
    # plt.figure()
    # plt.scatter(t_init, z_init)
    # plt.savefig("initial_points_tz.png")
    
    return centers_init