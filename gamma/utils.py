import multiprocessing as mp
from contextlib import contextmanager
import os
import pickle
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
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture
from .seismic_ops import calc_amp, calc_loc_batch, calc_mag_batch, calc_time, initialize_eikonal, traveltime_grad


def random_seed():
    np.random.seed(42)
    random.seed(42)


def estimate_eps(stations, vp, sigma=2.0):
    """DBSCAN's eps (s) from the station spacing: the mean + sigma std of each station's second-nearest neighbour
    distance, over vp, times 1.5. stations: a DataFrame with x_km, y_km, z_km (or x(km), ...), or an (n, 3) array."""
    if isinstance(stations, pd.DataFrame):
        stations = stations[["x_km", "y_km", "z_km"] if "x_km" in stations else ["x(km)", "y(km)", "z(km)"]].values
    X = np.asarray(stations, dtype=float)
    D = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=-1))
    D[D == 0] = np.inf
    dist = np.sort(D, axis=1)[:, 1]
    return (np.mean(dist) + sigma * np.std(dist)) / vp * 1.5


def convert_picks_csv(picks, stations, config, columns=("longitude", "latitude", "z_km")):
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
    locs = meta[list(columns)].to_numpy(dtype=float)
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
    """DBSCAN in (time, station position / vp), weighted by phase_weight. phase_loc: each pick's station in km, every
    column used: projected (x, y), or Earth-centred (x, y, z) for global data. A cluster of at least min_cluster_size
    picks that lasts longer than max_time_space_ratio times its crossing time is clustered again, eps shrunk by 1.2 per
    round. A re-clustering is kept only if it splits the cluster, and new labels never reuse an old one.
    Returns one label per pick, -1 for noise."""
    X = np.hstack([data[:, :1], phase_loc / vel["p"]])  # time, position (s)
    w = np.squeeze(phase_weight, axis=-1)

    def cluster(idx, e):
        return DBSCAN(eps=e, min_samples=min_samples, n_jobs=-1).fit(X[idx], sample_weight=w[idx]).labels_

    labels = cluster(np.arange(len(X)), eps)
    for k in range(1, 51):
        pending = False  # a cluster still too large and too long
        for label in np.unique(labels[labels >= 0]):
            idx = np.flatnonzero(labels == label)
            if len(idx) < min_cluster_size:
                continue
            t, xy = X[idx, 0], phase_loc[idx]
            if t.max() - t.min() < np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)) / vel["p"] * max_time_space_ratio:
                continue
            pending = True
            sub = cluster(idx, eps / 1.2**k)
            if len(np.unique(sub[sub >= 0])) >= 2:  # split: keep it; else keep the cluster and retry at a smaller eps
                labels[idx] = np.where(sub >= 0, sub + labels.max() + 1, -1)
        if not pending:
            break
    return labels


_GEOCENTRIC = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


def geocentric(lonlat):
    """(n, 2) longitude, latitude in degrees -> (n, 3) Earth-centred x, y, z in km on the WGS84 ellipsoid. Straight-line
    distances there equal surface distances to 0.001 km within 100 km, anywhere on Earth."""
    return np.column_stack(_GEOCENTRIC.transform(lonlat[:, 0], lonlat[:, 1], np.zeros(len(lonlat)))) / 1e3


NEIGHBOR_STATIONS = 8  # an event's neighbourhood: its nearest stations, as many as 8 fully reliable ones
LARGE_EVENT_PICKS = 15  # events with this many picks are real (chance does not make them): they measure reliability


def keep_supported(ev_xy, sta_xy, pick_event, pick_station, min_support):
    """Keep the events their nearest reliable stations picked; drop picks that lined up by chance (scattered
    stations, the ones between them silent).

    ev_xy (E, d), sta_xy (S, d) in km; pick_event, pick_station (P,): the event and station of each assigned pick.
    1. Reliability r of each station: the share of the large events (LARGE_EVENT_PICKS picks, or half the largest
       event's when fewer) that picked it when it was among their NEIGHBOR_STATIONS nearest, pulled to 0.5 by one
       pseudo-count. Often-silent stations (offline, noisy, skipped by analysts) then count little either way.
    2. Neighbourhood of an event: its nearest stations until sum r reaches NEIGHBOR_STATIONS.
    3. Support: observed / expected picking stations there. A station of a real event picks with probability r, so
       support = sum r picked / sum r^2: ~1 for a real event, ~0 for a chance one.
    4. Keep if support >= min_support, eased in proportion for events picked at more than NEIGHBOR_STATIONS stations.
    """
    E, S, K = len(ev_xy), len(sta_xy), NEIGHBOR_STATIONS
    near = cKDTree(sta_xy).query(ev_xy, k=min(S, 100))[1].reshape(E, -1)  # (E, k) stations by distance
    pairs = np.unique(pick_event * S + pick_station)  # one per (event, picking station)
    picked = np.isin(np.arange(E)[:, None] * S + near, pairs)  # (E, k)
    n_stations = np.bincount(pairs // S, minlength=E)  # (E,)
    num_picks = np.bincount(pick_event, minlength=E)  # (E,)

    large = num_picks >= min(LARGE_EVENT_PICKS, num_picks.max() / 2)  # (E,)
    tested = near[large, :K]  # (L, K)
    r = (np.bincount(tested[picked[large, :K]], minlength=S) + 0.5) / (np.bincount(tested.ravel(), minlength=S) + 1)

    r = r[near]  # (E, k)
    inside = np.cumsum(r, 1) - r < K  # (E, k) the neighbourhood
    support = (r * picked * inside).sum(1) / (r**2 * inside).sum(1)
    return support >= min_support * np.minimum(1, K / n_stations)


def local_projection(locs):
    """Azimuthal equidistant projection centred on the mean of the stations of locs = (longitude, latitude, z), as ADLoc
    builds each event's frame. Longitude is averaged on the circle, so stations across +-180 deg stay together.
    Returns locs with longitude, latitude replaced by x, y in km, and the projection to map events back."""
    lon, lat = np.radians(np.unique(locs[:, :2], axis=0)).T
    lon0, lat0 = np.degrees(np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())), np.degrees(lat.mean())
    proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=km")
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
        config.setdefault("segment", "dbscan" if config.pop("use_dbscan") else None)
        done.append(f"config['use_dbscan'] -> config['segment'] = {config['segment']!r} ('link', 'dbscan' or None)")
    if "dims" in config:
        if config["dims"][:2] == ["longitude", "latitude"]:
            config.pop("dims")  # the stations' longitude, latitude: the global mode, now the default without proj
            done.append("config['dims'] is no longer needed: without config['proj'], each segment has its own frame")
        else:
            done.append(f"config['dims'] {config['dims']} is deprecated: set config['proj'] (a pyproj Proj) and GaMMA "
                        "projects the stations' longitude, latitude itself; events then carry longitude, latitude too")
    if done:
        msg = "GaMMA uses ADLoc's names since 1.8; old settings translated:\n  " + "\n  ".join(done)
        warnings.warn(msg, stacklevel=3)


def translate_old_columns(picks, stations, dims=None):
    """Rename columns from before 1.8 to ADLoc's names on copies, with a visible warning."""
    old = {"id": "station_id", "timestamp": "phase_time", "type": "phase_type", "prob": "phase_score",
           "amp": "phase_amplitude"}
    found = {o: n for o, n in old.items() if o in picks and n not in picks}
    if found:
        picks = picks.rename(columns=found)
        warnings.warn(f"picks columns {list(found)} are renamed {list(found.values())} (as ADLoc)", stacklevel=3)
    alias = {"x_km": "x(km)", "y_km": "y(km)", "z_km": "z(km)", "x(km)": "x_km", "y(km)": "y_km", "z(km)": "z_km"}
    found = {"id": "station_id"} if ("id" in stations and "station_id" not in stations) else {}
    found.update({alias[d]: d for d in (dims or []) if d not in stations and alias.get(d) in stations})
    if "z_km" not in stations and "z(km)" in stations and "z_km" not in found.values() and "z(km)" not in (dims or []):
        found["z(km)"] = "z_km"
    if found:
        stations = stations.rename(columns=found)
        warnings.warn(f"stations columns {list(found)} are renamed {list(found.values())}", stacklevel=3)
    return picks, stations


def association(picks, stations, config, event_idx0=0, method="BGMM", **kwargs):
    # work on copies: the caller's config and picks stay as they were (GaMMA adds its tables, bounds and defaults)
    config = {**config, **{k: {**config[k]} for k in ("vel", "eikonal") if isinstance(config.get(k), dict)}}
    if isinstance(config.get("eikonal"), dict) and isinstance(config["eikonal"].get("vel"), dict):
        config["eikonal"]["vel"] = {**config["eikonal"]["vel"]}
    picks = picks.copy()
    translate_old_config(config)
    picks, stations = translate_old_columns(picks, stations, config.get("dims"))

    # the frame, as in ADLoc. config["proj"] (a pyproj Proj): one projection for every station (local).
    # Without it, each segment is projected around its own stations (global), as ADLoc does per event.
    # Deprecated: config["dims"] with the stations' own projected columns, as before 1.13 (local, events in those names)
    legacy = config.get("proj") is None and "dims" in config
    geo = config.get("proj") is None and not legacy
    data, locs, phase_type, phase_weight, pick_idx, pick_station_id, timestamp0 = convert_picks_csv(
        picks, stations, config, columns=config["dims"] if legacy else ("longitude", "latitude", "z_km")
    )
    if config.get("proj") is not None:
        x, y = config["proj"](locs[:, 0], locs[:, 1])
        locs = np.column_stack([x, y, locs[:, 2]])

    if len(data) < config["min_picks"]:
        return [], []

    v = config.get("vel", {"P": 6.0, "S": 6.0 / 1.73})
    vel = {"p": v["P"], "s": v["S"]}  # internally phases are lowercase

    # cut the picks into segments for the mixture. "dbscan" (default): the hierarchical DBSCAN cut; "link": groups of
    # picks one event could produce, at most segment_max_picks each; None: one segment
    segment = config.get("segment", "dbscan")
    if segment not in ("link", "dbscan", None):
        raise ValueError(f"config['segment'] = {segment!r}; use 'link', 'dbscan' or None")
    if geo and segment is None:
        raise ValueError("without config['proj'], segment must be 'dbscan' or 'link', which keep each segment local")
    if segment == "link":
        segments = segment_picks(data, locs, phase_type, vel, max_picks=config.get("segment_max_picks", 200), geo=geo)
        labels = np.full(len(data), -1)
        for k, seg in enumerate(sorted(segments, key=lambda i: data[i, 0].min())):  # numbered in time order
            labels[seg] = k
        unique_labels = set(labels).difference([-1])
    elif segment == "dbscan":
        # stations in km: projected x, y, or Earth-centred x, y, z without a projection (distances on the globe)
        pos = geocentric(locs[:, :2]) if geo else locs[:, :2]
        # eps (s) from the spacing of the stations with picks, unless given
        sta = geocentric(np.unique(locs[:, :2], axis=0)) if geo else np.unique(locs, axis=0)
        eps = config["dbscan_eps"] if "dbscan_eps" in config else estimate_eps(sta, vel["p"])
        labels = hierarchical_dbscan_clustering(
            data,
            pos,
            phase_type,
            phase_weight,
            vel,
            eps=eps,
            min_samples=config.get("dbscan_min_samples", 3),
            min_cluster_size=config.get("dbscan_min_cluster_size", 500),
            max_time_space_ratio=config.get("dbscan_max_time_space_ratio", 10),
        )
        unique_labels = set(labels).difference([-1])
    else:
        labels = np.zeros(len(data))
        unique_labels = [0]

    # where events can be, as in ADLoc: bfgs_bounds when given, else the region xlim_km, ylim_km, zlim_km
    if not geo and "bfgs_bounds" not in config:
        region = (tuple(config["xlim_km"]), tuple(config["ylim_km"]), tuple(config["zlim_km"]))
        config["bfgs_bounds"] = (*region, (None, None))

    # one task per segment: its picks in km, and without a projection its own bounds and the projection that takes its
    # events back to degrees. Those bounds, in the segment's frame (centred on its stations): bfgs_bounds when given,
    # else xlim_km, ylim_km (or the stations' box widened by 300 km) and zlim_km, its top no higher than the segment's
    # highest station, so one high station elsewhere on the globe does not lift every event
    tasks, projs = [], {}
    for k in unique_labels:
        idx = labels == k
        xyz, bounds = locs[idx], None
        if geo:
            xyz, projs[k] = local_projection(xyz)
            if "bfgs_bounds" in config:
                bounds = tuple(config["bfgs_bounds"])
            else:
                if "xlim_km" in config and "ylim_km" in config:  # the region's size, around the segment's stations
                    xlim, ylim = tuple(config["xlim_km"]), tuple(config["ylim_km"])
                else:  # the stations' box, widened by 300 km
                    lo, hi = xyz[:, :2].min(axis=0) - 300.0, xyz[:, :2].max(axis=0) + 300.0
                    xlim, ylim = (lo[0], hi[0]), (lo[1], hi[1])
                z0, z1 = config["zlim_km"]
                bounds = (xlim, ylim, (max(z0, xyz[:, 2].min()), z1), (None, None))
        tasks.append(
            (k, data[idx], xyz, phase_type[idx], phase_weight[idx], pick_idx[idx], pick_station_id[idx], bounds)
        )

    if ("eikonal" not in config) or (config["eikonal"] is None):
        config["eikonal"] = None
    else:
        # how far the tables must reach: across the largest segment's bounds, or the region and the stations together
        if geo:
            rmax = max([np.hypot(b[0][1] - b[0][0], b[1][1] - b[1][0]) for *_, b in tasks], default=1.0)
        else:
            (x0, x1), (y0, y1) = config["bfgs_bounds"][:2]
            dx = max(x1, locs[:, 0].max()) - min(x0, locs[:, 0].min())
            dy = max(y1, locs[:, 1].max()) - min(y0, locs[:, 1].min())
            rmax = np.hypot(dx, dy)
        if "tt" in config["eikonal"]:
            # tables already built (by GaMMA or ADLoc's init_eikonal2d, the same format): used as they are
            eik = config["eikonal"]
            (z0, z1), r1 = config["zlim_km"], eik["rgrid"][-1]
            if eik["zgrid"][0] > z0 or eik["zgrid"][-1] < z1 or r1 < rmax:
                warnings.warn(
                    f"the given travel-time tables (z {eik['zgrid'][0]:.1f} to {eik['zgrid'][-1]:.1f} km, r to "
                    f"{r1:.0f} km) do not cover zlim_km = {tuple(config['zlim_km'])} and {rmax:.0f} km; beyond them "
                    "times are extrapolated",
                    stacklevel=2,
                )
        else:
            # one table per station depth (station z_km, absolute, positive down), as in ADLoc; as deep as the events
            # unless eikonal zlim_km is set (e.g. (0, 60), below the Moho, so far first arrivals are modelled)
            eik = {"zlim_km": config["zlim_km"], **config["eikonal"], "station_z_km": locs[:, 2]}
            (ez0, ez1), (z0, z1) = eik["zlim_km"], config["zlim_km"]
            if ez0 > z0 or ez1 < z1:  # the tables must cover every depth an event can take
                eik["zlim_km"] = (min(ez0, z0), max(ez1, z1))
                warnings.warn(
                    f"config['eikonal']['zlim_km'] = {(ez0, ez1)} does not cover the events' config['zlim_km'] = "
                    f"{(z0, z1)}; the tables use {eik['zlim_km']}",
                    stacklevel=2,
                )
            if geo or "xlim_km" not in eik:
                eik["rmax_km"] = rmax
            eik = initialize_eikonal(eik)
        config["eikonal"] = eik

    if "ncpu" not in config:
        config["ncpu"] = max(1, min(len(unique_labels) // 4, min(32, mp.cpu_count() - 1)))
    else:
        config["ncpu"] = min(mp.cpu_count(), config["ncpu"])

    if config["ncpu"] == 1:
        _init_worker((config, timestamp0, vel, method, None))
        progress = tqdm(tasks, desc=f"Associating {len(data)} picks (1 CPU)", unit="cluster")
        results = {task[0]: _associate_cluster(task) for task in progress}
    else:
        # the following sort and shuffle is to make sure jobs are distributed evenly
        counter = Counter(labels)
        unique_labels = sorted(unique_labels, key=lambda x: counter[x], reverse=True)
        np.random.RandomState(42).shuffle(unique_labels)  # its own generator: the same order every run
        by_label = {task[0]: task for task in tasks}

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
            shared_config, tables = {**config, "proj": None}, None  # workers see km only; the projection stays here
            if config["eikonal"] is not None:
                tables = os.path.join(tmp, "eikonal_tt.npy")
                np.save(tables, config["eikonal"]["tt"])
                eikonal = {k: v for k, v in config["eikonal"].items() if k != "tt"}
                shared_config = {**shared_config, "eikonal": eikonal}
            shared = (shared_config, timestamp0, vel, method, tables)
            assert len(pickle.dumps(shared)) < 2**16, "spawn blocks on initializer data above the 64 KB pipe buffer"
            with _one_thread_env():
                p = mp.get_context(context).Pool(config["ncpu"], initializer=_init_pool_worker, initargs=(shared,))
            with p:
                args = [by_label[k] for k in unique_labels]
                # imap (in order, like starmap) so a progress bar can follow the clusters as they finish
                progress = tqdm(
                    p.imap(_associate_cluster, args, chunksize=chunk_size),
                    total=len(args),
                    desc=f"Associating {len(data)} picks ({config['ncpu']} CPUs)",
                    unit="cluster",
                )
                results = dict(zip(unique_labels, progress))

    # number events in cluster order from event_idx0, not in the order clusters finish. Every event gets longitude and
    # latitude; with config["proj"] it keeps x_km, y_km in that frame, without one those are dropped (each segment had
    # its own frame); the deprecated dims keep their names
    names = dict(zip(["x_km", "y_km", "z_km"], config["dims"])) if legacy else {}
    events, assignment, event_idx = [], [], event_idx0
    for k in sorted(results):
        each_events, each_assignment = results[k]
        for event in each_events:
            if k in projs:
                event["longitude"], event["latitude"] = projs[k](event.pop("x_km"), event.pop("y_km"), inverse=True)
            elif config.get("proj") is not None:
                event["longitude"], event["latitude"] = config["proj"](event["x_km"], event["y_km"], inverse=True)
            event["event_index"] += event_idx
            events.append({names.get(key, key): value for key, value in event.items()})
        assignment.extend((pi, ei + event_idx, pr) for pi, ei, pr in each_assignment)
        event_idx += len(each_events)

    # drop events their reliable nearest stations did not pick (picks that lined up by chance); 0 keeps every event
    min_station_support = config.get("min_station_support", 0.3)
    if min_station_support > 0 and len(events) > 0:
        _, first, station = np.unique(picks.loc[pick_idx, "station_id"], return_index=True, return_inverse=True)
        sta_xy = locs[first, :2]  # (S, 2) every station with a pick (co-located channels stay separate stations)
        xy = ("longitude", "latitude") if geo else (names.get("x_km", "x_km"), names.get("y_km", "y_km"))
        ev_xy = np.array([[e[xy[0]], e[xy[1]]] for e in events])  # (E, 2) in the frame of locs
        if geo:  # distances in km on the globe
            sta_xy, ev_xy = geocentric(sta_xy), geocentric(ev_xy)
        pick, event = np.array([(pi, ei - event_idx0) for pi, ei, _ in assignment]).T
        pick_station = pd.Series(station, index=pick_idx).loc[pick].values
        keep = keep_supported(ev_xy, sta_xy, event, pick_station, min_station_support)
        new = np.cumsum(keep) - 1 + event_idx0  # kept events renumbered from event_idx0
        events = [{**e, "event_index": int(new[i])} for i, e in enumerate(events) if keep[i]]
        assignment = [(pi, int(new[ei - event_idx0]), pr) for pi, ei, pr in assignment if keep[ei - event_idx0]]

    print(f"Associated {len(events)} events with {len(assignment)} picks")
    return events, assignment  # , event_idx.value


@contextmanager
def _one_thread_env():
    """One thread per math library (BLAS, OpenMP) in workers spawned inside: they read these at import, before
    threadpool_limits could reach them. The workers are the parallelism; threads inside them only contend."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(dict.fromkeys(keys, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_SHARED = None  # (config, timestamp0, vel, method), set once per worker


def _init_pool_worker(shared):
    threadpool_limits(1)  # as _one_thread_env, for libraries already loaded (fork)
    _init_worker(shared)


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
    D = locs_.shape[1]  # 3 (x, y, z); fewer only with the deprecated dims
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
            **dict(zip(["x_km", "y_km", "z_km"], center[:D])),  # to longitude, latitude by association()
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
