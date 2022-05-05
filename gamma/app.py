import os
import pickle
from datetime import datetime
from json import dumps
from typing import Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from kafka import KafkaProducer
from pydantic import BaseModel

from gamma import BayesianGaussianMixture, GaussianMixture
from gamma.utils import convert_picks_csv, association, from_seconds

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


def run_gamma(data, config, stations):
    picks = pd.DataFrame(data.picks)
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

    event_idx0 = 0  ## current earthquake index
    assignments = []
    if (len(picks) > 0) and (len(picks) < 5000):
        catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        for hour in sorted(list(set(picks["time_idx"]))):
            picks_ = picks[picks["time_idx"] == hour]
            if len(picks_) == 0:
                continue
            catalog, assign = association(picks_, stations, config, event_idx0, config["method"])
            event_idx0 += len(catalog)
            catalogs.extend(catalog)
            assignments.extend(assign)

    ## create catalog
    print(catalogs)
    catalogs = pd.DataFrame(catalogs, columns=["time(s)"] + config["dims"] + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp",  "event_idx", "prob_gamma"])
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x / config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x / config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)
    catalogs = catalogs[['time', 'magnitude', 'longitude', 'latitude', 'depth(m)', 'sigma_time', 'sigma_amp', 'prob_gamma', "event_idx"]]

    ## add assignment to picks
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks_gamma = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({'event_idx': int})
    picks_gamma["timestamp"] = picks_gamma["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
    if "time_idx" in picks_gamma:
        picks_gamma.drop(columns=["time_idx"], inplace=True)
    return catalogs,picks_gamma


@app.post('/predict_stream')
def predict(data: Pick):

    if len(data.picks) == 0:
        return {"catalog": [], "picks": []}

    catalogs, picks_gamma = run_gamma(data, config, stations)

    if use_kafka:
        print("Push events to kafka...")
        for event in catalogs.to_dict(orient="records"):
            producer.send('gmma_events', key=event["time"], value=event)
    
    return {"catalog": catalogs.to_dict(orient="records"), "picks": picks_gamma.to_dict(orient="records")}


@app.post('/predict')
def predict(data: Data):

    if len(data.picks) == 0:
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
        config["x(km)"] = (np.array(config["xlim_degree"]) - config["center"][0]) * config["degree2km"]
    if "y(km)" not in config:
        config["y(km)"] = (np.array(config["ylim_degree"]) - config["center"][1]) * config["degree2km"]
    if "z(km)" not in config:
        config["z(km)"] = (0, 41)
    if "bfgs_bounds" not in config:
        config["bfgs_bounds"] = [list(config[x]) for x in config["dims"]] + [[None, None]]

    stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0]) * config["degree2km"])
    stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1]) * config["degree2km"])
    stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

    catalogs, picks_gamma = run_gamma(data, config, stations)

    if use_kafka:
        print("Push events to kafka...")
        for event in catalogs.to_dict(orient="records"):
            producer.send('gamma_events', key=event["time"], value=event)

    return {"catalog": catalogs.to_dict(orient="records"), "picks": picks_gamma.to_dict(orient="records")}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}