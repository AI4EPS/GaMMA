import copy
import warnings

import numpy as np
import pandas as pd
from pyproj import Proj

from gamma.seismic_ops import initialize_eikonal
from gamma.utils import association

LON0, LAT0 = -122.8, 38.8
PROJ = Proj(f"+proj=aeqd +lon_0={LON0} +lat_0={LAT0} +units=km")
VEL = {"Z": [0.0, 5.0, 5.0, 30.0], "P": [5.5, 5.5, 6.3, 6.3], "S": [5.5 / 1.73, 5.5 / 1.73, 6.3 / 1.73, 6.3 / 1.73]}


def synthetic():
    """12 stations (some above sea level) and 4 events 20 s apart, P and S times from the same layered tables."""
    rng = np.random.default_rng(0)
    sx, sy = rng.uniform(-20, 20, 12), rng.uniform(-20, 20, 12)
    sz = -rng.uniform(0.0, 1.0, 12)  # km below sea level: stations up to 1 km high
    lon, lat = PROJ(sx, sy, inverse=True)
    stations = pd.DataFrame({"station_id": [f"XX.S{i:02d}..HH" for i in range(12)], "longitude": lon, "latitude": lat,
                             "z_km": sz, "x_km": sx, "y_km": sy})
    eik = initialize_eikonal({"vel": VEL, "h": 0.5, "xlim_km": (-30, 30), "ylim_km": (-30, 30), "zlim_km": (-1, 15),
                              "station_z_km": sz})
    from gamma.seismic_ops import traveltime_grad
    events = np.array([[-5.0, 3.0, 4.0], [8.0, -6.0, 6.0], [0.0, 0.0, 3.0], [-10.0, -8.0, 8.0]])
    rows, t0 = [], pd.Timestamp("2026-05-07T00:00:00")
    for k, ev in enumerate(events):
        for ph in ("p", "s"):
            t, _ = traveltime_grad(np.repeat(ev[None], 12, 0), np.c_[sx, sy, sz], np.array([ph] * 12), None, eik)
            for i in range(12):
                rows.append({"station_id": stations.station_id[i], "phase_type": ph.upper(), "phase_score": 0.9,
                             "phase_amplitude": 1e-5, "phase_time": (t0 + pd.Timedelta(seconds=20 * k + t[i])).isoformat()})
    return pd.DataFrame(rows), stations, events


def base_config():
    return {"use_amplitude": False, "method": "BGMM", "oversample_factor": 5, "vel": {"P": 5.5, "S": 5.5 / 1.73},
            "xlim_km": (-30, 30), "ylim_km": (-30, 30), "zlim_km": (-1, 15), "min_picks": 5, "ncpu": 1,
            "eikonal": {"vel": VEL, "h": 0.5}}


def located(events, frame):
    lon = np.array([e["longitude"] for e in events]); lat = np.array([e["latitude"] for e in events])
    x, y = PROJ(lon, lat)
    z = np.array([e["depth_km"] for e in events])
    return np.c_[x, y, z][np.argsort([e["time"] for e in events])]


def test_config_not_changed_and_reusable():
    picks, stations, _ = synthetic()
    config = {**base_config(), "proj": PROJ}
    before, picks_before = copy.deepcopy({k: v for k, v in config.items() if k != "proj"}), picks.copy()
    a, _ = association(picks, stations, config)
    assert {k: v for k, v in config.items() if k != "proj"} == before and config["proj"] is PROJ
    assert picks.equals(picks_before)
    b, _ = association(picks, stations, config)  # the same config again: before 1.13, KeyError 'vel'
    assert len(a) == len(b) == 4


def test_prebuilt_tables_and_modes():
    picks, stations, truth = synthetic()
    local, _ = association(picks, stations, {**base_config(), "proj": PROJ})
    tables = initialize_eikonal({**base_config()["eikonal"], "xlim_km": (-30, 30), "ylim_km": (-30, 30),
                                 "zlim_km": (-1, 15), "station_z_km": stations.z_km.values})
    prebuilt, _ = association(picks, stations, {**base_config(), "proj": PROJ, "eikonal": tables})
    glob, _ = association(picks, stations, base_config())  # no proj: each segment in its own frame
    assert len(local) == len(prebuilt) == len(glob) == 4
    assert np.allclose(located(local, PROJ), located(prebuilt, PROJ))
    assert np.abs(located(local, PROJ) - truth).max() < 0.5  # km
    assert np.abs(located(glob, PROJ) - located(local, PROJ)).max() < 0.5
    assert all("x_km" in e for e in local) and not any("x_km" in e for e in glob)


def test_old_dims_still_run():
    picks, stations, truth = synthetic()
    config = {**base_config(), "dims": ["x_km", "y_km", "z_km"]}  # before 1.13: the stations' own projected columns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        events, _ = association(picks, stations, config)
    assert any("dims" in str(x.message) for x in w)
    xyz = np.array([[e["x_km"], e["y_km"], e["z_km"]] for e in sorted(events, key=lambda e: e["time"])])
    assert len(events) == 4 and np.abs(xyz - truth).max() < 0.5


if __name__ == "__main__":
    test_config_not_changed_and_reusable()
    test_prebuilt_tables_and_modes()
    test_old_dims_still_run()
    print("ok")
