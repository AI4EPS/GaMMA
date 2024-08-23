import multiprocessing as mp
import platform
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import DBSCAN

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture
from .seismic_ops import calc_amp, calc_time, initialize_eikonal

to_seconds = lambda t: t.timestamp(tz="UTC")
from_seconds = lambda t: pd.Timestamp.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
# to_seconds = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
# from_seconds = lambda t: [datetime.utcfromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for x in t]


# def estimate_station_spacing(stations):
def estimate_eps(stations, vp, sigma=3.0):
    X = stations[["x(km)", "y(km)", "z(km)"]].values
    D = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=-1))
    Tcsr = minimum_spanning_tree(D).toarray()

    # Tcsr = Tcsr[Tcsr > 0]
    # # mean = np.median(Tcsr)
    # mean = np.mean(Tcsr)
    # std = np.std(Tcsr)
    # eps = (mean + sigma * std) / vp

    eps = np.max(Tcsr) / vp * 1.5

    return eps


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
    meta = stations.merge(picks["id"], how="right", on="id", validate="one_to_many")
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


def hierarchical_dbscan_clustering(data, phase_loc, phase_type, phase_weight, vel, eps=15, min_samples=3):

    def dbscan2(t, xy, w, ph, vel, eps, min_samples, ratio=1.1):

        data = np.hstack([t * ratio, xy / vel["p"]])  # time, x, y
        db_ = DBSCAN(eps=eps, min_samples=min_samples).fit(data, sample_weight=np.squeeze(w, axis=-1))

        return db_.labels_

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(
        np.hstack([data[:, 0:1], phase_loc[:, :2] / vel["p"]]),  # time, x, y
        sample_weight=np.squeeze(phase_weight, axis=-1),
    )

    labels = db.labels_
    unique_labels = np.unique(labels)
    current_label = labels.max() + 1
    ratio = 1

    for _ in range(20):

        ratio *= 1.5
        unique_labels = np.unique(labels)
        current_label = unique_labels.max() + 1
        keep_split = False

        for label in unique_labels:

            if label == -1:
                continue

            idx = labels == label
            t = data[idx, 0:1]
            if (t.max() - t.min()) < 800:  # s
                continue

            xy = phase_loc[idx, :2]
            w = phase_weight[idx]
            ph = phase_type[idx]

            labels_ = dbscan2(t, xy, w, ph, vel, eps=eps, min_samples=min_samples, ratio=ratio)
            labels_ = np.where(labels_ == -1, -1, labels_ + current_label)
            labels[idx] = labels_

            current_label += labels_.max() + 1
            keep_split = True

        if not keep_split:
            break

    return labels


def association(picks, stations, config, event_idx0=0, method="BGMM", **kwargs):
    data, locs, phase_type, phase_weight, pick_idx, pick_station_id, timestamp0 = convert_picks_csv(
        picks, stations, config
    )

    if len(data) < config["min_picks_per_eq"]:
        return [], []

    vel = config["vel"] if "vel" in config else {"p": 6.0, "s": 6.0 / 1.73}
    if ("eikonal" not in config) or (config["eikonal"] is None):
        config["eikonal"] = None
    else:
        config["eikonal"] = initialize_eikonal(config["eikonal"])

    if ("use_dbscan" in config) and config["use_dbscan"]:
        # db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(
        #     np.hstack([data[:, 0:1], locs[:, :2] / np.average(vel["p"])]),
        #     sample_weight=np.squeeze(phase_weight),
        # )
        # labels = db.labels_
        labels = hierarchical_dbscan_clustering(
            data,
            locs,
            phase_type,
            phase_weight,
            vel,
            eps=config["dbscan_eps"],
            min_samples=config["dbscan_min_samples"],
        )
        unique_labels = set(labels)
        unique_labels = unique_labels.difference([-1])
    else:
        labels = np.zeros(len(data))
        unique_labels = [0]

    if "ncpu" not in config:
        config["ncpu"] = max(1, min(len(unique_labels) // 4, min(32, mp.cpu_count() - 1)))
    else:
        config["ncpu"] = min(mp.cpu_count(), config["ncpu"])

    if config["ncpu"] == 1:
        print(f"Associating {len(data)} picks with {config['ncpu']} CPUs")
        event_idx = 0
        events, assignment = [], []
        for unique_label in list(unique_labels):
            events_, assignment_ = associate(
                unique_label,
                labels,
                data,
                locs,
                phase_type,
                phase_weight,
                pick_idx,
                pick_station_id,
                config,
                timestamp0,
                vel,
                method,
                event_idx,
            )
            event_idx += len(events_)
            events.extend(events_)
            assignment.extend(assignment_)
    else:
        manager = mp.Manager()
        lock = manager.Lock()
        # event_idx0 - 1 as event_idx is increased before use
        event_idx = manager.Value("i", event_idx0 - 1)

        print(f"Associating {len(unique_labels)} clusters with {config['ncpu']} CPUs")

        # the following sort and shuffle is to make sure jobs are distributed evenly
        counter = Counter(labels)
        unique_labels = sorted(unique_labels, key=lambda x: counter[x], reverse=True)
        np.random.shuffle(unique_labels)

        # the default chunk_size is len(unique_labels)//(config["ncpu"]*4), which makes some jobs very heavy
        chunk_size = max(len(unique_labels) // (config["ncpu"] * 20), 1)

        # Check for OS to start a child process in multiprocessing
        # https://superfastpython.com/multiprocessing-context-in-python/
        if platform.system().lower() in ["darwin", "windows"]:
            context = "spawn"
        else:
            context = "fork"

        with mp.get_context(context).Pool(config["ncpu"]) as p:
            results = p.starmap(
                associate,
                [
                    [
                        k,
                        labels,
                        data,
                        locs,
                        phase_type,
                        phase_weight,
                        pick_idx,
                        pick_station_id,
                        config,
                        timestamp0,
                        vel,
                        method,
                        event_idx,
                        lock,
                    ]
                    for k in unique_labels
                ],
                chunksize=chunk_size,
            )
            # resuts is a list of tuples, each tuple contains two lists events and assignment
            # here we flatten the list of tuples into two lists
            events, assignment = [], []
            for each_events, each_assignment in results:
                events.extend(each_events)
                assignment.extend(each_assignment)

    return events, assignment  # , event_idx.value


def associate(
    k,
    labels,
    data,
    locs,
    phase_type,
    phase_weight,
    pick_idx,
    pick_station_id,
    config,
    timestamp0,
    vel,
    method,
    event_idx,
    lock=None,
):
    print(".", end="")

    data_ = data[labels == k]
    locs_ = locs[labels == k]
    phase_type_ = phase_type[labels == k]
    phase_weight_ = phase_weight[labels == k]
    pick_idx_ = pick_idx[labels == k]
    pick_station_id_ = pick_station_id[labels == k]

    max_num_event = max(Counter(pick_station_id_).values())

    if len(pick_idx_) < max(3, config["min_picks_per_eq"]):
        return [], []

    time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)
    if config["use_amplitude"]:
        amp_range = max(data_[:, 1].max() - data_[:, 1].min(), 1)

    ## initialization with [1,1,1] horizontal points and N time points
    centers_init = init_centers(config, data_, locs_, time_range, max_num_event)

    ## run clustering
    # mean_precision_prior = 0.01 / time_range
    if "covariance_prior" in config:
        covariance_prior_pre = config["covariance_prior"]
    else:
        # covariance_prior_pre = [5.0, 2.0]
        ## TODO: design a smark way to set covariance_prior
        weight = np.squeeze(phase_weight_)
        x_mean = np.average(locs_[:, 0], weights=weight)
        y_mean = np.average(locs_[:, 1], weights=weight)
        x_std = np.sqrt(np.average((locs_[:, 0] - x_mean) ** 2, weights=weight))
        y_std = np.sqrt(np.average((locs_[:, 1] - y_mean) ** 2, weights=weight))
        # x_std = np.std(locs_[:, 0])
        # y_std = np.std(locs_[:, 1])
        # t_std = np.std(data_[:, 0])
        ## option 1
        # scaler = max(np.sqrt(x_std**2 + y_std**2) / 6.0 , 0.1)
        ## option 2
        # scaler = max(np.sqrt(x_std**2 + y_std**2) / 6.0 / t_std, 0.1) * 10
        ## option 3
        # d, v = 50, 6.0
        # scaler = max((np.exp(np.sqrt(x_std**2 + y_std**2)/d) - 1)/(np.exp(1) - 1) * d / v / t_std, 0.2) * 10
        ## option 4
        rstd = np.sqrt(x_std**2 + y_std**2)
        # scaler = max(10.0, (rstd / 6.0) * (rstd / 60.0))  # 6.0 km/s, 60 km
        scaler = max(1.0, (rstd / 6.0) * (rstd / 30.0))  # 6.0 km/s, 30 km
        if config["use_amplitude"]:
            # covariance_prior_pre = [time_range * 10.0, amp_range * 10.0]
            covariance_prior_pre = [scaler, scaler]
        else:
            # covariance_prior_pre = [time_range * 10.0]
            covariance_prior_pre = [scaler]
    # print(f"covariance_prior_pre: {covariance_prior_pre}")
    if config["use_amplitude"]:
        covariance_prior = np.array([[covariance_prior_pre[0], 0.0], [0.0, covariance_prior_pre[1]]])
    else:
        covariance_prior = np.array([[covariance_prior_pre[0]]])
        data_ = data_[:, 0:1]

    if method == "BGMM":
        gmm = BayesianGaussianMixture(
            n_components=len(centers_init),
            weight_concentration_prior=1.0 / len(centers_init),
            # mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior,
            init_params="centers",
            centers_init=centers_init.copy(),
            station_locs=locs_,
            phase_type=phase_type_,
            phase_weight=phase_weight_,
            vel=vel,
            eikonal=config["eikonal"],
            bounds=config["bfgs_bounds"],
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
            eikonal=config["eikonal"],
            bounds=config["bfgs_bounds"],
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
    events = []
    assignment = []

    for i in range(len(centers_init)):
        tmp_data = data_[pred == i]
        tmp_locs = locs_[pred == i]
        tmp_pick_station_id = pick_station_id_[pred == i]
        tmp_phase_type = phase_type_[pred == i]
        if (len(tmp_data) == 0) or (len(tmp_data) < config["min_picks_per_eq"]):
            continue
        # idx_filter = np.ones(len(tmp_data)).astype(bool)

        ## filter by time
        t_ = calc_time(
            gmm.centers_[i : i + 1, : len(config["dims"]) + 1],
            tmp_locs,
            tmp_phase_type,
            vel=vel,
            eikonal=config["eikonal"],
        )
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

        if lock is not None:
            with lock:
                if not isinstance(event_idx, int):
                    event_idx.value += 1
                    event_idx_value = event_idx.value
                else:
                    event_idx += 1
                    event_idx_value = event_idx
        else:
            if not isinstance(event_idx, int):
                event_idx.value += 1
                event_idx_value = event_idx.value
            else:
                event_idx += 1
                event_idx_value = event_idx

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
            "num_picks": len(tmp_data[idx_filter]),
            "num_p_picks": len(tmp_data[idx_filter & (tmp_phase_type == "p")]),
            "num_s_picks": len(tmp_data[idx_filter & (tmp_phase_type == "s")]),
            "event_index": event_idx_value,
        }
        for j, k in enumerate(config["dims"]):  ## add location
            event[k] = gmm.centers_[i, j]
        events.append(event)

        for pi, pr in zip(pick_idx_[pred == i][idx_filter], prob):
            assignment.append((pi, event_idx_value, pr))

        if (event_idx_value + 1) % 100 == 0:
            print(f"\nAssociated {event_idx_value + 1} events")
    return events, assignment


# def init_centers(config, data_, locs_, time_range, max_num_event=1):
#     """
#     max_num_event: maximum number of events at one station
#     """

#     if "initial_points" in config:
#         initial_points = config["initial_points"]
#         if not isinstance(initial_points, list):
#             initial_points = [initial_points, initial_points, initial_points]
#     else:
#         initial_points = [1, 1, 1]

#     if (np.prod(initial_points) * max_num_event * config["oversample_factor"]) > len(data_):
#         initial_points = [1, 1, 1]

#     x_init = np.linspace(config["x(km)"][0], config["x(km)"][1], initial_points[0] + 2)[1:-1]
#     y_init = np.linspace(config["y(km)"][0], config["y(km)"][1], initial_points[1] + 2)[1:-1]
#     z_init = np.linspace(config["z(km)"][0], config["z(km)"][1], initial_points[2] + 2)[1:-1]
#     # z_init = np.linspace(config["z(km)"][0], config["z(km)"][1], initial_points[2]) + 1.0

#     ## manually set initial points
#     if "x_init" in config:
#         x_init = np.array(config["x_init"])
#     if "y_init" in config:
#         y_init = np.array(config["y_init"])
#     if "z_init" in config:
#         z_init = np.array(config["z_init"])

#     x_init = np.broadcast_to(x_init[:, np.newaxis, np.newaxis], initial_points).reshape(-1)
#     y_init = np.broadcast_to(y_init[np.newaxis, :, np.newaxis], initial_points).reshape(-1)
#     z_init = np.broadcast_to(z_init[np.newaxis, np.newaxis, :], initial_points).reshape(-1)

#     ## I found it helpful to add a point at the center of the area
#     if (initial_points[0] == 2) and (initial_points[1] == 2):
#         x_init = np.append(x_init, np.mean(config["x(km)"]))
#         y_init = np.append(y_init, np.mean(config["y(km)"]))
#         z_init = np.append(z_init, 0)

#     num_xyz_init = len(x_init)

#     # num_sta = len(np.unique(locs_, axis=0))
#     # num_t_init = max(np.round(len(data_) / num_sta / num_xyz_init * config["oversample_factor"]), 1)
#     # num_t_init = min(int(num_t_init), max(len(data_) // num_xyz_init, 1))
#     num_t_init = min(max(int(max_num_event * config["oversample_factor"]), 1), max(len(data_) // num_xyz_init, 1))
#     t_init = np.sort(data_[:, 0])[:: max(len(data_) // num_t_init, 1)][:num_t_init]
#     # t_init = np.linspace(
#     #         data_[:, 0].min() - 0.1 * time_range,
#     #         data_[:, 0].max() + 0.1 * time_range,
#     #         num_t_init)

#     x_init = np.broadcast_to(x_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
#     y_init = np.broadcast_to(y_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
#     z_init = np.broadcast_to(z_init[:, np.newaxis], (num_xyz_init, num_t_init)).reshape(-1)
#     t_init = np.broadcast_to(t_init[np.newaxis, :], (num_xyz_init, num_t_init)).reshape(-1)

#     if config["dims"] == ["x(km)", "y(km)", "z(km)"]:
#         centers_init = np.vstack([x_init, y_init, z_init, t_init]).T
#     elif config["dims"] == ["x(km)", "y(km)"]:
#         centers_init = np.vstack([x_init, y_init, t_init]).T
#     elif config["dims"] == ["x(km)"]:
#         centers_init = np.vstack([x_init, t_init]).T
#     else:
#         raise (ValueError("Unsupported dims"))

#     if config["use_amplitude"]:
#         centers_init = np.hstack([centers_init, 1.0 * np.ones((len(centers_init), 1))])  # init magnitude to 1.0

#     return centers_init


def init_centers(config, data_, locs_, time_range, max_num_event=1):
    """
    max_num_event: maximum number of events at one station
    """

    if "initial_points" in config:
        initial_points = config["initial_points"]
        if not isinstance(initial_points, list):
            initial_points = [initial_points, initial_points, initial_points]
    else:
        initial_points = [1, 1, 1]

    num_t_init = min(max(int(max_num_event * config["oversample_factor"]), 1), len(data_))

    index = np.argsort(data_[:, 0])[:: max(len(data_) // num_t_init, 1)][:num_t_init]
    t_init = data_[index, 0]
    x_init = locs_[:, 0][index]  # + np.random.uniform(low=-1, high=1, size=num_t_init) * np.std(locs_[:, 0])
    y_init = locs_[:, 1][index]  # + np.random.uniform(low=-1, high=1, size=num_t_init) * np.std(locs_[:, 1])
    # x_init, y_init = np.mean(locs_[:, 0]), np.mean(locs_[:, 1])
    # x_init = np.broadcast_to(x_init, (num_t_init)).reshape(-1)
    # y_init = np.broadcast_to(y_init, (num_t_init)).reshape(-1)
    z_init = np.linspace(config["z(km)"][0], config["z(km)"][1], initial_points[2] + 2)[1:-1]
    z_init = np.broadcast_to(z_init, (num_t_init)).reshape(-1)

    if config["dims"] == ["x(km)", "y(km)", "z(km)"]:
        centers_init = np.vstack([x_init, y_init, z_init, t_init]).T
    elif config["dims"] == ["x(km)", "y(km)"]:
        centers_init = np.vstack([x_init, y_init, t_init]).T
    elif config["dims"] == ["x(km)"]:
        centers_init = np.vstack([x_init, t_init]).T
    else:
        raise (ValueError("Unsupported dims"))

    if config["use_amplitude"]:
        centers_init = np.hstack([centers_init, 1.0 * np.ones((len(centers_init), 1))])  # init magnitude to 1.0

    return centers_init
