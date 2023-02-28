import os
from json import dumps
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from kafka import KafkaProducer
from pydantic import BaseModel
from pyproj import Proj

from gamma.utils import association

# Kafak producer
use_kafka = False

try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    # BROKER_URL = "34.83.137.139:9094"
    producer = KafkaProducer(
        bootstrap_servers=[BROKER_URL],
        key_serializer=lambda x: dumps(x).encode('utf-8'),
        value_serializer=lambda x: dumps(x).encode('utf-8'),
    )
    use_kafka = True
    print('k8s kafka connection success!')
except BaseException:
    print('k8s Kafka connection error')
    try:
        print('Connecting to local kafka')
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            key_serializer=lambda x: dumps(x).encode('utf-8'),
            value_serializer=lambda x: dumps(x).encode('utf-8'),
        )
        use_kafka = True
        print('local kafka connection success!')
    except BaseException:
        print('local Kafka connection error')
print(f"Kafka status: {use_kafka}")

app = FastAPI()

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations_hawaii.csv")
# STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations.csv")  ## ridgecrest


def default_config(config):
    if "degree2km" not in config:
        config["degree2km"] = 111.195
    if "use_amplitude" not in config:
        config["use_amplitude"] = True
    if "use_dbscan" not in config:
        config["use_dbscan"] = True
    if "dbscan_eps" not in config:
        config["dbscan_eps"] = 30.0
    if "dbscan_min_samples" not in config:
        config["dbscan_min_samples"] = 3
    if "method" not in config:
        config["method"] = "BGMM"
    if "oversample_factor" not in config:
        config["oversample_factor"] = 5
    if "min_picks_per_eq" not in config:
        config["min_picks_per_eq"] = 10
    if "max_sigma11" not in config:
        config["max_sigma11"] = 2.0
    if "max_sigma22" not in config:
        config["max_sigma22"] = 1.0
    if "max_sigma12" not in config:
        config["max_sigma12"] = 1.0
    if "dims" not in config:
        config["dims"] = ["x(km)", "y(km)", "z(km)"]
    return config


## set config
config = {'xlim_degree': [-156.32, -154.32], 'ylim_degree': [18.39, 20.39], "z(km)": [0, 41]}  ## hawaii
# config = {'xlim_degree': [-118.004, -117.004], 'ylim_degree': [35.205, 36.205], "z(km)": [0, 41]}  ## ridgecrest

config = default_config(config)
config["center"] = [np.mean(config["xlim_degree"]), np.mean(config["ylim_degree"])]
config["x(km)"] = (np.array(config["xlim_degree"]) - config["center"][0]) * config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"]) - config["center"][1]) * config["degree2km"]
config["bfgs_bounds"] = [list(config[x]) for x in config["dims"]] + [[None, None]]

for k, v in config.items():
    print(f"{k}: {v}")

## read stations
stations = pd.read_csv(STATION_CSV, delimiter="\t")
stations = stations.rename(columns={"station": "id"})
stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0]) * config["degree2km"])
stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1]) * config["degree2km"])
stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

print(stations)


class Data(BaseModel):
    picks: List[Dict[str, Union[float, str]]]
    stations: List[Dict[str, Union[float, str]]]
    config: Dict[str, Union[List[float], List[int], List[str], float, int, str]]


class Pick(BaseModel):
    picks: List[Dict[str, Union[float, str]]]


def run_gamma(picks, config, stations):

    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")

    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x/1e3)

    catalogs, assignments = association(picks, stations, config, 0, config["method"])

    catalogs = pd.DataFrame(catalogs, columns=["time"]+config["dims"]+["magnitude", "sigma_time", "sigma_amp", "cov_time_amp",  "event_index", "gamma_score"])
    catalogs[["longitude","latitude"]] = catalogs.apply(lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1)
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x*1e3)

    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks_gamma = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({'event_index': int})

    return catalogs, picks_gamma


@app.post('/predict_stream')
def predict(data: Pick):

    picks =  pd.DataFrame(data.picks)
    if len(picks) == 0:
        return {"catalog": [], "picks": []}

    catalogs, picks_gamma = run_gamma(data, config, stations)

    if use_kafka:
        print("Push events to kafka...")
        for event in catalogs.to_dict(orient="records"):
            producer.send('gmma_events', key=event["time"], value=event)
    
    return {"catalog": catalogs.to_dict(orient="records"), "picks": picks_gamma.to_dict(orient="records")}


@app.post('/predict')
def predict(data: Data):

    picks =  pd.DataFrame(data.picks)
    if len(picks) == 0:
        return {"catalog": [], "picks": []}

    stations = pd.DataFrame(data.stations)
    if len(stations) == 0:
        return {"catalog": [], "picks": []}

    assert "latitude" in stations
    assert "longitude" in stations
    assert "elevation(m)" in stations

    config = data.config
    config = default_config(config)

    if "xlim_degree" not in config:
        config["xlim_degree"] = (stations["longitude"].min(), stations["longitude"].max())
    if "ylim_degree" not in config:
        config["ylim_degree"] = (stations["latitude"].min(), stations["latitude"].max())
    if "center" not in config:
        config["center"] = [np.mean(config["xlim_degree"]), np.mean(config["ylim_degree"])]
    if "x(km)" not in config:
        config["x(km)"] = (np.array(config["xlim_degree"]) - config["center"][0]) * config["degree2km"] * np.cos(np.deg2rad(config["center"][1]))
    if "y(km)" not in config:
        config["y(km)"] = (np.array(config["ylim_degree"]) - config["center"][1]) * config["degree2km"]
    if "z(km)" not in config:
        config["z(km)"] = (0, 41)
    if "bfgs_bounds" not in config:
        config["bfgs_bounds"] = [list(config[x]) for x in config["dims"]] + [[None, None]]

    catalogs, picks_gamma = run_gamma(picks, config, stations)

    if use_kafka:
        print("Push events to kafka...")
        for event in catalogs.to_dict(orient="records"):
            producer.send('gamma_events', key=event["time"], value=event)

    return {"catalog": catalogs.to_dict(orient="records"), "picks": picks_gamma.to_dict(orient="records")}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}