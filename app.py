import pandas as pd
from fastapi import FastAPI
from pyproj import Proj

from gamma.utils import association

app = FastAPI()


@app.get("/")
def greet_json():
    return {"Hello": "GaMMA!"}


@app.post("/predict/")
def predict(picks: dict, stations: dict, config: dict):
    picks = picks["data"]
    stations = stations["data"]
    picks = pd.DataFrame(picks)
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    stations = pd.DataFrame(stations)
    events_, picks_ = run_gamma(picks, stations, config)
    if events_ is None:
        return {"events": None, "picks": picks_}
    events_ = events_.to_dict(orient="records")
    picks_ = picks_.to_dict(orient="records")

    return {"events": events_, "picks": picks_}


def set_config(region="ridgecrest"):

    config = {
        "min_picks": 8,
        "min_picks_ratio": 0.2,
        "max_residual_time": 1.0,
        "max_residual_amplitude": 1.0,
        "min_score": 0.6,
        "min_s_picks": 2,
        "min_p_picks": 2,
        "use_amplitude": False,
    }

    # ## Domain
    if region.lower() == "ridgecrest":
        config.update(
            {
                "region": "ridgecrest",
                "minlongitude": -118.004,
                "maxlongitude": -117.004,
                "minlatitude": 35.205,
                "maxlatitude": 36.205,
                "mindepth_km": 0.0,
                "maxdepth_km": 41.0,
            }
        )

    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
    xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
    zmin, zmax = config["mindepth_km"], config["maxdepth_km"]
    xlim_km = (xmin, xmax)
    ylim_km = (ymin, ymax)
    zlim_km = (zmin, zmax)

    config.update(
        {
            "xlim_km": xlim_km,
            "ylim_km": ylim_km,
            "zlim_km": zlim_km,
            "z(km)": zlim_km,
            "proj": proj,
        }
    )

    config.update(
        {
            "min_picks_per_eq": 5,
            "min_p_picks_per_eq": 0,
            "min_s_picks_per_eq": 0,
            "max_sigma11": 3.0,
            "max_sigma22": 1.0,
            "max_sigma12": 1.0,
        }
    )

    config["use_dbscan"] = False
    config["use_amplitude"] = True
    config["oversample_factor"] = 8.0
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    config["method"] = "BGMM"
    config["ncpu"] = 1
    vel = {"p": 6.0, "s": 6.0 / 1.75}
    config["vel"] = vel

    config["bfgs_bounds"] = (
        (xlim_km[0] - 1, xlim_km[1] + 1),  # x
        (ylim_km[0] - 1, ylim_km[1] + 1),  # y
        (0, zlim_km[1] + 1),  # z
        (None, None),  # t
    )

    config["event_index"] = 0

    return config


config = set_config()


def run_gamma(picks, stations, config_):

    # %%
    config.update(config_)

    proj = config["proj"]

    picks = picks.rename(
        columns={
            "station_id": "id",
            "phase_time": "timestamp",
            "phase_type": "type",
            "phase_score": "prob",
            "phase_amplitude": "amp",
        }
    )
    stations = stations.rename(columns={"station_id": "id"})
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    events, assignments = association(picks, stations, config, 0, config["method"])

    if events is None:
        return None, None

    events = pd.DataFrame(events)
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z(km)"]
    events.drop(columns=["x(km)", "y(km)", "z(km)"], inplace=True, errors="ignore")
    picks = picks.rename(
        columns={
            "id": "station_id",
            "timestamp": "phase_time",
            "type": "phase_type",
            "prob": "phase_score",
            "amp": "phase_amplitude",
        }
    )

    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({"event_index": int})

    return events, picks
