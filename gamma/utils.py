import multiprocessing as mp
import os
import platform
import random
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import sort_graph_by_row_values
from tqdm import tqdm

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture
from .seismic_ops import calc_amp, calc_loc_batch, calc_mag_batch, calc_time, initialize_eikonal, traveltime_grad


def random_seed():
    np.random.seed(42)
    random.seed(42)


# def estimate_station_spacing(stations):
def estimate_eps(stations, vp, sigma=2.0):
    X = stations[["x_km", "y_km", "z_km"] if "x_km" in stations else ["x(km)", "y(km)", "z(km)"]].values
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
    if type(picks["phase_time"].iloc[0]) is str:
        # replace the column: writing datetimes into a string column in place fails on pandas >= 3
        picks["phase_time"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))
    t = (
        picks["phase_time"]
        .apply(lambda x: x.tz_convert("UTC").timestamp() if x.tzinfo is not None else x.tz_localize("UTC").timestamp())
        .to_numpy()
    )
    # t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    timestamp0 = np.min(t)
    t = t - timestamp0
    if config["use_amplitude"]:
        a = picks["phase_amplitude"].apply(lambda x: np.log10(x * 1e2)).to_numpy()  ##cm/s
        data = np.stack([t, a]).T
    else:
        data = t[:, np.newaxis]
    meta = stations.merge(picks["station_id"], how="right", on="station_id", validate="one_to_many")
    locs = meta[config["dims"]].to_numpy()
    phase_type = picks["phase_type"].apply(lambda x: x.lower()).to_numpy()
    phase_weight = picks["phase_score"].to_numpy()[:, np.newaxis]
    pick_station_id = picks.apply(lambda x: x.station_id + "_" + x.phase_type, axis=1).to_numpy()
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


_GEOCENTRIC = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


def geocentric(lonlat):
    """(n, 2) longitude, latitude in degrees -> (n, 3) Earth-centred x, y, z in km on the WGS84 ellipsoid. Straight-line
    distances there equal surface distances to 0.001 km within 100 km, anywhere on Earth."""
    return np.column_stack(_GEOCENTRIC.transform(lonlat[:, 0], lonlat[:, 1], np.zeros(len(lonlat)))) / 1e3


def local_projection(locs):
    """Azimuthal equidistant projection (WGS84, as in ADLoc) centred on the stations of locs = (longitude, latitude, z).
    Returns locs with longitude, latitude replaced by x, y in km, and the projection to map events back."""
    c = geocentric(np.unique(locs[:, :2], axis=0)).mean(0)
    lon0, lat0 = np.degrees(np.arctan2(c[1], c[0])), np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1])))
    proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=km +ellps=WGS84")
    x, y = proj(locs[:, 0], locs[:, 1])
    return np.column_stack([x, y, locs[:, 2:]]), proj


def link_picks(data, xy, phase_type, vel, max_km=100.0, tol=0.5):
    """Pairs of picks that one event could produce, as a sparse distance graph for DBSCAN(metric="precomputed").
    Stations closer than max_km link when the time difference fits the distance between them: same phase
    |dt| <= dx / v + tol; P at i then S at j within -dx / vp - tol <= tS - tP <= max_km (1/vs - 1/vp) + dx / vs + tol.
    Two picks at one station link only within tol, so two events there never link.
    xy (n, 2 or 3): station positions in km, projected x, y or Earth-centred x, y, z."""
    vp, vs = vel["p"], vel["s"]
    t, s = data[:, 0], phase_type == "s"
    sp_max = max_km * (1 / vs - 1 / vp)
    reach = np.hypot((sp_max + max_km / vs + tol) * vs, max_km)  # every compatible pair is within this in (t vs, x, y)
    i, j = cKDTree(np.column_stack([t * vs, xy])).query_pairs(reach, output_type="ndarray").T
    dx, dt = np.linalg.norm(xy[i] - xy[j], axis=1), t[j] - t[i]
    ps_ok = lambda d: (d >= -dx / vp - tol) & (d <= sp_max + dx / vs + tol)  # S minus P
    ok = (dx <= max_km) & (
        (~s[i] & ~s[j] & (np.abs(dt) <= dx / vp + tol))
        | (s[i] & s[j] & (np.abs(dt) <= dx / vs + tol))
        | (~s[i] & s[j] & ps_ok(dt))
        | (s[i] & ~s[j] & ps_ok(-dt))
    )
    i, j = i[ok], j[ok]
    n = np.arange(len(t))  # the zero diagonal stored up front: DBSCAN writes it, which would unsort the rows
    d = np.r_[np.zeros(len(t)), np.full(2 * len(i), tol / 2)]
    graph = coo_matrix((d, (np.r_[n, i, j], np.r_[n, j, i])), shape=(len(t), len(t))).tocsr()
    return sort_graph_by_row_values(graph, warn_when_not_sorted=False)


def focused_origin_time(data, locs, phase_type, vel, bw=1.0, grid_km=25.0, depth=5.0):
    """Each pick's origin time if its event sat at R, u = t - tt(R, station, phase), for the grid point R (over the
    picks' stations) that makes u sharpest: the most picks with another within +-bw. Picks of one event share u when
    R is near it, so R focuses on the activity. R is searched coarse to fine: a 4 * grid_km grid, then grid_km around
    its sharpest point (statewide, the full fine grid took most of the run time)."""
    def sharpest(xs, ys):
        best, R_best, u_best = -1, None, None
        for x in xs:
            for y in ys:
                R = np.r_[x, y, depth][: locs.shape[1]]
                u = data[:, 0] - traveltime_grad(np.broadcast_to(R, locs.shape), locs, phase_type, vel)[0]
                us = np.sort(u)
                sharp = np.sum(np.searchsorted(us, us + bw) - np.arange(len(us)))
                if sharp > best:
                    best, R_best, u_best = sharp, R, u
        return R_best, u_best

    lo, hi = locs[:, :2].min(0), locs[:, :2].max(0)
    c = 4 * grid_km
    R, _ = sharpest(np.arange(lo[0], hi[0] + c, c), np.arange(lo[1], hi[1] + c, c))
    return sharpest(np.arange(R[0] - c, R[0] + c + 1e-6, grid_km), np.arange(R[1] - c, R[1] + c + 1e-6, grid_km))[1]


def segment_picks(data, locs, phase_type, vel, max_picks=200, max_km=100.0, tol=0.5, min_samples=3, bw=1.0, geo=False):
    """Cut the picks into small segments for the mixture, each holding whole events. Returns a list of index arrays.

    Repeat on every piece until all are done:
    1. Space: split the piece into groups of linked picks (DBSCAN on the links of link_picks: picks one event could
       produce), which separates regions without breaking events; picks left as noise are dropped.
    2. Time: a group above max_picks is cut at the quietest moment of its focused origin time u (the fewest picks
       within +-bw; ties: the largest gap), and both halves go back to step 1.
    geo: locs are (longitude, latitude, z); picks link on the globe and each piece is projected around its stations."""
    graph = link_picks(data, geocentric(locs[:, :2]) if geo else locs[:, :2], phase_type, vel, max_km, tol)
    segments, stack = [], [np.arange(len(data))]
    while stack:
        seg = stack.pop()
        sub = graph[seg][:, seg]
        labels = DBSCAN(eps=tol, min_samples=min_samples, metric="precomputed").fit(sub).labels_
        groups = [seg[labels == k] for k in np.unique(labels[labels >= 0])]
        if len(groups) > 1:
            stack += groups
        elif len(groups) == 1 and len(groups[0]) > max_picks:
            g = groups[0]
            u = focused_origin_time(data[g], local_projection(locs[g])[0] if geo else locs[g], phase_type[g], vel, bw)
            g, us = g[np.argsort(u)], np.sort(u)
            m = len(us) // 10  # never cut off a sliver
            mid = (us[m : len(us) - m - 1] + us[m + 1 : len(us) - m]) / 2
            density = np.searchsorted(us, mid + bw) - np.searchsorted(us, mid - bw)
            k = m + 1 + np.lexsort((-np.diff(us)[m : len(us) - m - 1], density))[0]
            stack += [g[:k], g[k:]]
        elif len(groups) == 1:
            segments.append(groups[0])
    return segments


def translate_old_config(config):
    """Rename settings from before 1.5 / 1.7 / 1.8 in place, so old configs keep running; one warning lists every
    change. A new key, when also set, wins. Since 1.8 the names follow ADLoc."""
    done = []

    def rename(d, old, new, where="config", note=""):
        if old in d:
            d.setdefault(new, d.pop(old))
            done.append(f"{where}['{old}'] -> {where}['{new}']{note}")

    looser = " (since 1.5 it limits the residual after each event is relocated, so old values are looser)"
    rename(config, "max_sigma11", "max_residual_time", note=looser)
    rename(config, "max_sigma22", "max_residual_amplitude", note=looser)
    for old, new in [("min_picks_per_eq", "min_picks"), ("min_p_picks_per_eq", "min_p_picks"),
                     ("min_s_picks_per_eq", "min_s_picks"), ("x(km)", "xlim_km"), ("y(km)", "ylim_km"),
                     ("z(km)", "zlim_km")]:
        rename(config, old, new)
    for old in [k for k in config.get("vel", {}) if k.islower()]:
        rename(config["vel"], old, old.upper(), "config['vel']")
    if config.get("eikonal") is not None:
        for old in ("xlim", "ylim", "zlim"):
            rename(config["eikonal"], old, old + "_km", "config['eikonal']")
        for old in [k for k in config["eikonal"].get("vel", {}) if k.islower()]:
            rename(config["eikonal"]["vel"], old, old.upper(), "config['eikonal']['vel']")
    if "max_sigma12" in config:
        config.pop("max_sigma12")
        done.append("config['max_sigma12'] is ignored (the covariance check was removed in 1.5)")
    if "use_dbscan" in config:
        config.setdefault("segment", "link" if config.pop("use_dbscan") else None)
        done.append(f"config['use_dbscan'] -> config['segment'] = {config['segment']!r} ('link', 'dbscan' or None)")
    if any("(km)" in d for d in config.get("dims", [])):
        done.append(f"config['dims'] {config['dims']}: use 'x_km', 'y_km', 'z_km'; events keep the names given in dims")
    if done:
        msg = "GaMMA uses ADLoc's names since 1.8; old settings translated:\n  " + "\n  ".join(done)
        warnings.warn(msg, stacklevel=3)


def translate_old_columns(picks, stations, dims):
    """Rename columns from before 1.8 to ADLoc's names on copies, with a visible warning."""
    old = {"id": "station_id", "timestamp": "phase_time", "type": "phase_type", "prob": "phase_score",
           "amp": "phase_amplitude"}
    found = {o: n for o, n in old.items() if o in picks and n not in picks}
    if found:
        picks = picks.rename(columns=found)
        warnings.warn(f"picks columns {list(found)} are renamed {list(found.values())} (as ADLoc)", stacklevel=3)
    alias = {"x_km": "x(km)", "y_km": "y(km)", "z_km": "z(km)", "x(km)": "x_km", "y(km)": "y_km", "z(km)": "z_km"}
    found = {"id": "station_id"} if ("id" in stations and "station_id" not in stations) else {}
    found.update({alias[d]: d for d in dims if d not in stations and alias.get(d) in stations})
    if found:
        stations = stations.rename(columns=found)
        warnings.warn(f"stations columns {list(found)} are renamed {list(found.values())}", stacklevel=3)
    return picks, stations


def association(picks, stations, config, event_idx0=0, method="BGMM", **kwargs):
    translate_old_config(config)
    picks, stations = translate_old_columns(picks, stations, config["dims"])

    data, locs, phase_type, phase_weight, pick_idx, pick_station_id, timestamp0 = convert_picks_csv(
        picks, stations, config
    )

    if len(data) < config["min_picks"]:
        return [], []

    v = config.get("vel", {"P": 6.0, "S": 6.0 / 1.73})
    vel = {"p": v["P"], "s": v["S"]}  # internally phases are lowercase

    # cut the picks into segments for the mixture. "link" (default): groups of picks one event could produce, at most
    # segment_max_picks each; "dbscan": the hierarchical DBSCAN cut; None: all picks in one segment
    segment = config.get("segment", "link")
    if segment not in ("link", "dbscan", None):
        raise ValueError(f"config['segment'] = {segment!r}; use 'link', 'dbscan' or None")
    # dims (longitude, latitude, z_km): each segment is projected around its own stations, as ADLoc does per event,
    # so one run can cover any area, up to the whole globe
    geo = config["dims"][:2] == ["longitude", "latitude"]
    if geo and (segment == "dbscan" or len(config["dims"]) != 3):
        raise ValueError("dims (longitude, latitude, z_km) work with segment 'link' or None")
    if segment == "link":
        segments = segment_picks(data, locs, phase_type, vel, max_picks=config.get("segment_max_picks", 200), geo=geo)
        labels = np.full(len(data), -1)
        for k, seg in enumerate(sorted(segments, key=lambda i: data[i, 0].min())):  # numbered in time order
            labels[seg] = k
        unique_labels = set(labels).difference([-1])
    elif segment == "dbscan":
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

    projs, bounds = {}, {}
    if geo:  # each segment's picks in km on its own projection; its events stay within its stations +- 300 km
        locs = locs.copy()
        z_bounds = (config["zlim_km"][0], config["zlim_km"][1] + 1)
        for k in unique_labels:
            idx = labels == k
            locs[idx], projs[k] = local_projection(locs[idx])
            lo, hi = locs[idx, :2].min(0) - 300.0, locs[idx, :2].max(0) + 300.0
            bounds[k] = ((lo[0], hi[0]), (lo[1], hi[1]), z_bounds, (None, None))

    if ("eikonal" not in config) or (config["eikonal"] is None):
        config["eikonal"] = None
    else:
        if geo:  # the travel-time table only has to reach across the largest segment's bounds
            rmax = max([np.hypot(b[0][1] - b[0][0], b[1][1] - b[1][0]) for b in bounds.values()], default=1.0)
            config["eikonal"] = {**config["eikonal"], "xlim_km": (0.0, rmax), "ylim_km": (0.0, 0.0)}
        config["eikonal"] = initialize_eikonal(config["eikonal"])

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
                {**config, "bfgs_bounds": bounds[unique_label]} if geo else config,
                timestamp0,
                vel,
                method,
                event_idx,
            )
            if geo:
                to_lonlat(events_, projs[unique_label])
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
                        (k, data[idx], locs[idx], phase_type[idx], phase_weight[idx], pick_idx[idx], pick_station_id[idx],
                         bounds.get(k))
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
            if geo:
                to_lonlat(each_events, projs[k])
            for event in each_events:
                event["event_index"] += event_idx
            events.extend(each_events)
            assignment.extend((pi, ei + event_idx, pr) for pi, ei, pr in each_assignment)
            event_idx += len(each_events)

    print(f"Associated {len(events)} events with {len(assignment)} picks")
    return events, assignment  # , event_idx.value


def to_lonlat(events, proj):
    """Events located in a segment's projected km back to longitude, latitude (geographic dims)."""
    for event in events:
        event["longitude"], event["latitude"] = proj(event["longitude"], event["latitude"], inverse=True)


_SHARED = None  # (config, timestamp0, vel, method), set once per worker


def _init_worker(shared):
    global _SHARED
    config, timestamp0, vel, method, tables = shared
    if tables is not None:  # the eikonal tables, shared through the page cache rather than copied
        config["eikonal"]["tt"] = np.load(tables, mmap_mode="r")
    _SHARED = (config, timestamp0, vel, method)


def _associate_cluster(task):
    """associate() on one cluster's picks; its events are numbered from 0 and renumbered by the caller"""
    k, data, locs, phase_type, phase_weight, pick_idx, pick_station_id, bounds = task
    config, timestamp0, vel, method = _SHARED
    if bounds is not None:  # this segment's own location bounds (geographic dims)
        config = {**config, "bfgs_bounds": bounds}
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

    if len(pick_idx_) < max(3, config["min_picks"]):
        return [], []

    # time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)
    # if config["use_amplitude"]:
    #     amp_range = max(data_[:, 1].max() - data_[:, 1].min(), 1)

    # ## initialization with [1,1,1] horizontal points and N time points
    centers_init = init_centers(config, data_, locs_, phase_type_, vel, max_num_event)

    ## run clustering
    # Wishart prior on each event's residual covariance (sklearn's convention): nu0 pseudo-picks of spread sigma, i.e.
    # covariance_prior Psi = nu0 * sigma^2, with sigma = (1.5 s, 0.5 log10). The time spread is how wide an event may
    # be (real pick scatter is ~0.3 s, but a prior that tight splits events); real amplitude scatter is ~0.3.
    nu0 = config.get("degrees_of_freedom_prior", 5.0)
    prior = config.get("covariance_prior", nu0 * np.array([1.5, 0.5]) ** 2)  # [time (s^2), amplitude (log10^2)]
    if config["use_amplitude"]:
        covariance_prior = np.diag(np.asarray(prior[:2], dtype=float))
    else:
        covariance_prior = np.array([[float(prior[0])]])
        data_ = data_[:, 0:1]

    random_state = 42
    if method == "BGMM":
        gmm = BayesianGaussianMixture(
            n_components=max_num_event,
            # weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / max_num_event,
            covariance_prior=covariance_prior,
            degrees_of_freedom_prior=nu0,
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
        if len(idx) < config["min_picks"]:
            continue
        keep, center, res = refit_event(
            gmm.centers_[i], data_[idx], locs_[idx], phase_type_[idx], phase_weight_[idx], config, vel
        )
        idx, res, ph = idx[keep], res[keep], phase_type_[idx[keep]]
        if (
            len(idx) < config["min_picks"]
            or np.sum(ph == "p") < config.get("min_p_picks", 0)
            or np.sum(ph == "s") < config.get("min_s_picks", 0)
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
            **({"depth_km": center[2]} if D == 3 else {}),  # as ADLoc
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


def init_centers(config, data_, locs_, type_, vel, max_num_event=1):
    """
    max_num_event: maximum number of events at one station
    """

    # t_init = np.random.choice(data_[:, 0], max_num_event, replace=False)
    # x_init = np.random.uniform(config["x(km)"][0], config["x(km)"][1], max_num_event)
    # y_init = np.random.uniform(config["y(km)"][0], config["y(km)"][1], max_num_event)
    # z_init = np.ones(max_num_event) * (config["z(km)"][0] + config["z(km)"][1]) / 2.0

    # index = np.argsort(data_[:, 0])[:: max(len(data_) // max_num_event, 1)][:max_num_event]
    # use p first, then s
    index = np.argsort(data_[:, 0])
    is_p = type_[index] == "p"  # in time order, like index
    p_index, s_index = index[is_p], index[~is_p]
    if len(p_index) >= max_num_event:
        index = p_index[:: max(len(p_index) // max_num_event, 1)][:max_num_event]
    else:
        num = max_num_event - len(p_index)
        index = np.concatenate([p_index, s_index[:: max(len(s_index) // num, 1)][:num]])

    # each event starts at its seed pick's station at mid depth, with origin time = the seed's arrival minus the travel
    # time from there (straight ray); the arrival time itself would start every predicted arrival late
    x_init, y_init = locs_[index, 0], locs_[index, 1]
    z_init = np.ones(max_num_event) * (config["zlim_km"][0] + config["zlim_km"][1]) / 2.0
    xyz = np.column_stack([x_init, y_init, z_init])[:, : locs_.shape[1]]
    v = np.where(type_[index] == "p", vel["p"], vel["s"])
    t_init = data_[index, 0] - np.linalg.norm(xyz - locs_[index], axis=1) / v

    centers_init = np.vstack([*xyz.T, t_init]).T  # location in the first len(dims) coordinates, then origin time

    if config["use_amplitude"]:
        centers_init = np.hstack([centers_init, 1.0 * np.ones((len(centers_init), 1))])  # init magnitude to 1.0

    return centers_init
