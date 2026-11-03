import multiprocessing as mp
import os
import platform
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture
from .seismic_ops import calc_amp, calc_loc_batch, calc_mag_batch, calc_time, initialize_eikonal


def random_seed():
    np.random.seed(42)
    random.seed(42)


# def estimate_station_spacing(stations):
def estimate_eps(stations, vp, sigma=2.0):
    X = stations[["x(km)", "y(km)", "z(km)"]].values
    D = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=-1))

    D[D == 0] = np.inf
    dist = np.sort(D, axis=1)[:, 1]
    # dist = np.max(dist)

    std = np.std(dist)
    dist = np.mean(dist) + sigma * std

    eps = dist / vp * 1.5

    # Tcsr = minimum_spanning_tree(D).toarray()

    # Tcsr = Tcsr[Tcsr > 0]
    # # mean = np.median(Tcsr)
    # mean = np.mean(Tcsr)
    # std = np.std(Tcsr)
    # eps = (mean + sigma * std) / vp

    # eps = np.max(Tcsr) / vp * 1.5

    return eps


def convert_picks_csv(picks, stations, config):
    # t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    if type(picks["timestamp"].iloc[0]) is str:
        # replace the column: writing datetimes into a string column in place fails on pandas >= 3
        picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.fromisoformat(x))
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


def hierarchical_dbscan_clustering(
    data,
    phase_loc,
    phase_type,
    phase_weight,
    vel,
    eps=15,
    min_samples=3,
    min_cluster_size=500,
    max_time_space_ratio=10,
):
    def dbscan2(t, xy, w, ph, vel, eps, min_samples, ratio=1.1):
        data = np.hstack([t, xy / vel["p"]])  # time, x, y
        db_ = DBSCAN(eps=eps * ratio, min_samples=min_samples, n_jobs=-1).fit(
            data, sample_weight=np.squeeze(w, axis=-1)
        )
        return db_.labels_

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(
        np.hstack([data[:, 0:1], phase_loc[:, :2] / vel["p"]]),  # time, x, y
        sample_weight=np.squeeze(phase_weight, axis=-1),
    )

    labels = db.labels_
    unique_labels = np.unique(labels)
    current_label = labels.max() + 1
    ratio = 1

    for _ in range(50):
        ratio /= 1.2
        unique_labels = np.unique(labels)
        current_label = unique_labels.max() + 1
        keep_split = False

        for label in tqdm(unique_labels, desc=f"Clustering (eps={eps * ratio:.2f})"):
            if label == -1:
                continue

            idx = labels == label
            if np.sum(idx) < min_cluster_size:
                continue

            t = data[idx, 0:1]
            xy = phase_loc[idx, :2]
            w = phase_weight[idx]
            ph = phase_type[idx]

            dxy = np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))
            dt = t.max() - t.min()
            if dt < dxy / vel["p"] * max_time_space_ratio:  # s
                continue
            # else:
            #     print(f"Splitting {len(t)} picks using eps={eps * ratio:.2f}")

            labels_ = dbscan2(t, xy, w, ph, vel, eps=eps, min_samples=min_samples, ratio=ratio)
            labels_ = np.where(labels_ == -1, -1, labels_ + current_label)
            labels[idx] = labels_

            current_label = labels_.max() + 1
            keep_split = True

        if not keep_split:
            break

    return labels


def association(picks, stations, config, event_idx0=0, method="BGMM", **kwargs):
    retired = {"max_sigma11": "max_residual_time", "max_sigma22": "max_residual_amplitude", "max_sigma12": None}
    for old, new in retired.items():  # fail loudly: a silently ignored limit would change the catalogue unseen
        if old in config:
            raise ValueError(f"config['{old}'] is retired; " + (f"use config['{new}']" if new else "remove it"))

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
            min_cluster_size=config["dbscan_min_cluster_size"] if "dbscan_min_cluster_size" in config else 500,
            max_time_space_ratio=(
                config["dbscan_max_time_space_ratio"] if "dbscan_max_time_space_ratio" in config else 10
            ),
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
        # event_idx0 - 1 as event_idx is increased before use, as in the parallel path
        event_idx = event_idx0 - 1
        events, assignment = [], []
        for unique_label in tqdm(sorted(unique_labels), desc=f"Associating {len(data)} picks (1 CPU)", unit="cluster"):
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
        # the following sort and shuffle is to make sure jobs are distributed evenly
        counter = Counter(labels)
        unique_labels = sorted(unique_labels, key=lambda x: counter[x], reverse=True)
        np.random.RandomState(42).shuffle(unique_labels)  # its own generator: the same order every run

        # the default chunk_size is len(unique_labels)//(config["ncpu"]*4), which makes some jobs very heavy
        chunk_size = max(len(unique_labels) // (config["ncpu"] * 20), 1)

        # Check for OS to start a child process in multiprocessing
        # https://superfastpython.com/multiprocessing-context-in-python/
        if platform.system().lower() in ["darwin", "windows"]:
            context = "spawn"
        else:
            context = "fork"

        # each task carries only its cluster's picks; config, timestamp0, vel and method reach each worker once through
        # the initializer. The eikonal tables (MBs) go through a memory-mapped file instead: in the initializer's
        # arguments they would block the parent until each worker had booted, one worker after another.
        with tempfile.TemporaryDirectory() as tmp:
            shared_config, tables = config, None
            if config["eikonal"] is not None:
                tables = os.path.join(tmp, "eikonal_tt.npy")
                np.save(tables, config["eikonal"]["tt"])
                eikonal = {k: v for k, v in config["eikonal"].items() if k not in ("tt", "up", "us")}
                shared_config = {**config, "eikonal": eikonal}
            shared = (shared_config, timestamp0, vel, method, tables)
            with mp.get_context(context).Pool(config["ncpu"], initializer=_init_worker, initargs=(shared,)) as p:
                args = []
                for k in unique_labels:
                    idx = labels == k
                    args.append(
                        (k, data[idx], locs[idx], phase_type[idx], phase_weight[idx], pick_idx[idx], pick_station_id[idx])
                    )
                # imap (in order, like starmap) so a progress bar can follow the clusters as they finish
                progress = tqdm(
                    p.imap(_associate_cluster, args, chunksize=chunk_size),
                    total=len(args),
                    desc=f"Associating {len(data)} picks ({config['ncpu']} CPUs)",
                    unit="cluster",
                )
                results = dict(zip(unique_labels, progress))

        # number events in cluster order from event_idx0, as the serial path does, not in the order clusters finish
        events, assignment, event_idx = [], [], event_idx0
        for k in sorted(results):
            each_events, each_assignment = results[k]
            for event in each_events:
                event["event_index"] += event_idx
            events.extend(each_events)
            assignment.extend((pi, ei + event_idx, pr) for pi, ei, pr in each_assignment)
            event_idx += len(each_events)

    print(f"Associated {len(events)} events with {len(assignment)} picks")
    return events, assignment  # , event_idx.value


_SHARED = None  # (config, timestamp0, vel, method), set once per worker


def _init_worker(shared):
    global _SHARED
    config, timestamp0, vel, method, tables = shared
    if tables is not None:  # the eikonal tables, shared through the page cache rather than copied
        config["eikonal"]["tt"] = np.load(tables, mmap_mode="r")
    _SHARED = (config, timestamp0, vel, method)


def _associate_cluster(task):
    """associate() on one cluster's picks; its events are numbered from 0 and renumbered by the caller"""
    k, data, locs, phase_type, phase_weight, pick_idx, pick_station_id = task
    config, timestamp0, vel, method = _SHARED
    labels = np.full(len(data), k)
    return associate(k, labels, data, locs, phase_type, phase_weight, pick_idx, pick_station_id, config, timestamp0, vel, method, -1)


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
):
    data_ = data[labels == k]
    locs_ = locs[labels == k]
    phase_type_ = phase_type[labels == k]
    phase_weight_ = phase_weight[labels == k]
    pick_idx_ = pick_idx[labels == k]
    pick_station_id_ = pick_station_id[labels == k]

    max_num_event = max(Counter(pick_station_id_).values()) * config["oversample_factor"]
    max_num_event = min(max_num_event, len(data_) // 3)

    if len(pick_idx_) < max(3, config["min_picks_per_eq"]):
        return [], []

    # time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)
    # if config["use_amplitude"]:
    #     amp_range = max(data_[:, 1].max() - data_[:, 1].min(), 1)

    # ## initialization with [1,1,1] horizontal points and N time points
    centers_init = init_centers(config, data_, locs_, phase_type_, phase_weight_, max_num_event, seed=42 + int(k))

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
        # scaler = max(1.0, (rstd / 6.0) * (rstd / 30.0))  # 6.0 km/s, 30 km
        a, b = 0.5, 15
        scaler = a + (b ** (rstd / 30) - 1) / (b - 1) * 4.0  # (0, a), (30, 4 + a)
        # print(f"rstd: {rstd}, scaler: {scaler}, max_num_event: {max_num_event}")
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

    random_state = 42
    if method == "BGMM":
        gmm = BayesianGaussianMixture(
            n_components=max_num_event,
            # weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / max_num_event,
            # mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior,
            # init_params="random",
            # init_params="k-means++",
            # init_params="kmeans",
            init_params="centers",
            centers_init=centers_init.copy(),
            station_locs=locs_,
            phase_type=phase_type_,
            phase_weight=phase_weight_,
            vel=vel,
            eikonal=config["eikonal"],
            bounds=config["bfgs_bounds"],
            random_state=random_state,
        ).fit(data_)
    elif method == "GMM":
        gmm = GaussianMixture(
            n_components=max_num_event,
            # init_params="random",
            # init_params="k-means++",
            # init_params="kmeans",
            init_params="centers",
            centers_init=centers_init.copy(),
            station_locs=locs_,
            phase_type=phase_type_,
            phase_weight=phase_weight_,
            vel=vel,
            eikonal=config["eikonal"],
            bounds=config["bfgs_bounds"],
            # dummy_comp=True,
            # dummy_prob=1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2),
            # dummy_quantile=0.1,
            random_state=random_state,
        ).fit(data_)
    else:
        raise ValueError(f"Unknown method {method}; Should be 'BGMM' or 'GMM'")

    ## run prediction
    pred = gmm.predict(data_)
    prob = np.exp(gmm.score_samples(data_))
    prob_matrix = gmm.predict_proba(data_)
    #  prob = prob_matrix[range(len(data_)), pred]
    #  score = gmm.score(data_)
    #  score_sample = gmm.score_samples(data_)

    ## post-filter: refit each event on its picks, drop the picks off the fit, keep the events with enough picks left
    D = len(config["dims"])
    events, assignment = [], []
    for i in range(max_num_event):
        idx = np.flatnonzero(pred == i)
        if len(idx) < config["min_picks_per_eq"]:
            continue
        keep, center, res = refit_event(
            gmm.centers_[i], data_[idx], locs_[idx], phase_type_[idx], phase_weight_[idx], config, vel
        )
        idx, res, ph = idx[keep], res[keep], phase_type_[idx[keep]]
        if (
            len(idx) < config["min_picks_per_eq"]
            or np.sum(ph == "p") < config.get("min_p_picks_per_eq", 0)
            or np.sum(ph == "s") < config.get("min_s_picks_per_eq", 0)
            or len(np.unique(locs_[idx], axis=0)) < config.get("min_stations", 0)
        ):
            continue

        event_idx += 1
        event = {
            "time": datetime.fromtimestamp(center[D] + timestamp0, timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="milliseconds"),
            "magnitude": center[D + 1] if config["use_amplitude"] else 999,
            "sigma_time": np.sqrt(np.mean(res[:, 0] ** 2)),
            "sigma_amp": np.sqrt(np.mean(res[:, 1] ** 2)) if config["use_amplitude"] else 0,
            "cov_time_amp": np.mean(res[:, 0] * res[:, 1]) if config["use_amplitude"] else 0,
            "gamma_score": prob_matrix[idx, i].sum(),
            "num_picks": len(idx),
            "num_p_picks": int(np.sum(ph == "p")),
            "num_s_picks": int(np.sum(ph == "s")),
            "event_index": event_idx,
            **dict(zip(config["dims"], center[:D])),
        }
        events.append(event)
        assignment.extend(zip(pick_idx_[idx], [event_idx] * len(idx), prob[idx]))  # each pick's own score

    return events, assignment


def refit_event(center, data, locs, phase_type, phase_weight, config, vel, max_iter=5):
    """ADLoc's RANSAC without the random sampling, since the mixture's centre is already a robust start: locate the
    event on its kept picks, keep the picks within max_residual_time (s) and max_residual_amplitude (log10) of that
    fit, and repeat until the kept picks stop changing. The 1.0 defaults are ~3 sigma of real PhaseNet residuals.

    center (D + 1[+ 1],) = (location, time[, magnitude]); data (n, 1 or 2) = (time[, log amplitude]).
    Returns the kept-pick mask (n,), the refitted center, and every pick's residuals (n, 1 or 2)."""
    D = locs.shape[1]
    keep = np.ones(len(data), bool)
    for _ in range(max_iter):
        w = phase_weight * keep[:, None]  # (n, 1)
        loc, _ = calc_loc_batch(
            data[:, :1], phase_type, locs, w, center[None, : D + 1], vel, config["eikonal"], config["bfgs_bounds"]
        )
        res = calc_time(loc, locs, phase_type, vel, config["eikonal"]) - data[:, :1]
        ok = np.abs(res[:, 0]) < config.get("max_residual_time", 1.0)
        if config["use_amplitude"]:
            mag = calc_mag_batch(data[:, 1:2], loc[:, :D], locs, w)
            res = np.hstack([res, calc_amp(mag, loc, locs) - data[:, 1:2]])
            ok &= np.abs(res[:, 1]) < config.get("max_residual_amplitude", 1.0)
            loc = np.hstack([loc, mag[:, None]])
        center = loc[0]
        if (ok == keep).all() or ok.sum() < 4:  # converged, or too few picks left to locate
            break
        keep = ok
    return ok, center, res


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


def init_centers(config, data_, locs_, type_, weight_, max_num_event=1, seed=42):
    """
    max_num_event: maximum number of events at one station
    seed: of this cluster's random start; differs between clusters so their starts are not offset alike
    """

    # t_init = np.random.choice(data_[:, 0], max_num_event, replace=False)
    # x_init = np.random.uniform(config["x(km)"][0], config["x(km)"][1], max_num_event)
    # y_init = np.random.uniform(config["y(km)"][0], config["y(km)"][1], max_num_event)
    # z_init = np.ones(max_num_event) * (config["z(km)"][0] + config["z(km)"][1]) / 2.0

    # index = np.argsort(data_[:, 0])[:: max(len(data_) // max_num_event, 1)][:max_num_event]
    # use p first, then s
    index = np.argsort(data_[:, 0])
    p_index = index[type_ == "p"]
    s_index = index[type_ == "s"]
    if len(p_index) >= max_num_event:
        index = p_index[:: max(len(p_index) // max_num_event, 1)][:max_num_event]
    else:
        num = max_num_event - len(p_index)
        index = np.concatenate([p_index, s_index[:: max(len(s_index) // num, 1)][:num]])

    weight_ = weight_.squeeze()
    mu_x = np.average(locs_[:, 0], weights=weight_)
    sigma_x = np.sqrt(np.average((locs_[:, 0] - mu_x) ** 2, weights=weight_))
    mu_y = np.average(locs_[:, 1], weights=weight_)
    sigma_y = np.sqrt(np.average((locs_[:, 1] - mu_y) ** 2, weights=weight_))
    t_init = (
        data_[index, 0]
        # + np.random.uniform(low=-1, high=0, size=max_num_event)
        # * (np.std(locs_[:, 0]) + np.std(locs_[:, 1]))
        # / 2.0
        # / 6.0
    )
    # a generator per cluster, so a cluster's start does not depend on the clusters run before it (CPU count, order)
    rng = np.random.RandomState(seed)
    x_init = locs_[index, 0] + rng.uniform(low=-1, high=1, size=max_num_event) * sigma_x
    y_init = locs_[index, 1] + rng.uniform(low=-1, high=1, size=max_num_event) * sigma_y
    z_init = (
        np.ones(max_num_event) * (config["z(km)"][0] + config["z(km)"][1]) / 2.0
        # + np.random.uniform(low=-1, high=1, size=max_num_event) * (config["z(km)"][1] - config["z(km)"][0]) / 6.0
    )

    if config["dims"] == ["x(km)", "y(km)", "z(km)"]:
        centers_init = np.vstack([x_init, y_init, z_init, t_init]).T
    elif config["dims"] == ["x(km)", "y(km)"]:
        centers_init = np.vstack([x_init, y_init, t_init]).T
    elif config["dims"] == ["x(km)"]:
        centers_init = np.vstack([x_init, t_init]).T
    else:
        raise ValueError("Unsupported dims")

    if config["use_amplitude"]:
        centers_init = np.hstack([centers_init, 1.0 * np.ones((len(centers_init), 1))])  # init magnitude to 1.0

    return centers_init
