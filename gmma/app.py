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

from gmma.association import association, convert_picks_csv, from_seconds, to_seconds

try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    producer = KafkaProducer(
        bootstrap_servers=[BROKER_URL],
        key_serializer=lambda x: dumps(x).encode('utf-8'),
        value_serializer=lambda x: dumps(x).encode('utf-8'),
    )
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
        print('local kafka connection success!')
    except BaseException:
        print('local Kafka connection error')

app = FastAPI()

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PKL = os.path.join(PROJECT_ROOT, "tests/config_hawaii.pkl")
STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations_hawaii.csv")

with open(CONFIG_PKL, "rb") as fp:
    config = pickle.load(fp)
## read stations
stations = pd.read_csv(STATION_CSV, delimiter="\t")
stations = stations.rename(columns={"station": "id"})
stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0]) * config["degree2km"])
stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1]) * config["degree2km"])
stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
## setting GMMA configs
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = True
config["use_amplitude"] = True
config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 40)
# DBSCAN
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),
)  # t
config["dbscan_eps"] = min(
    np.sqrt(
        (stations["x(km)"].max() - stations["x(km)"].min()) ** 2
        + (stations["y(km)"].max() - stations["y(km)"].min()) ** 2
    )
    / (6.0 / 1.75),
    10,
)  # s
config["dbscan_min_samples"] = min(len(stations), 3)
# Filtering
config["min_picks_per_eq"] = min(len(stations) // 2, 10)
config["oversample_factor"] = min(len(stations) // 2, 10)
for k, v in config.items():
    print(f"{k}: {v}")


class Pick(BaseModel):
    picks: List[Dict[str, Union[float, str]]]


@app.get('/predict')
def predict(data: Pick):

    picks = data.picks
    if len(picks) == 0:
        return []

    # picks = pd.read_json(picks)
    picks = pd.DataFrame(picks)
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    picks["time_idx"] = picks["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H"))  ## process by hours

    event_idx0 = 0
    ## run GMMA association
    if (len(picks) > 0) and (len(picks) < 5000):
        data, locs, phase_type, phase_weight = convert_picks_csv(picks, stations, config)
        catalogs, assignments = association(
            data, locs, phase_type, phase_weight, len(stations), picks.index.to_numpy(), event_idx0, config, stations
        )
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        for i, hour in enumerate(sorted(list(set(picks["time_idx"])))):
            picks_ = picks[picks["time_idx"] == hour]
            if len(picks_) == 0:
                continue
            data, locs, phase_type, phase_weight = convert_picks_csv(picks_, stations, config)
            catalog, assign = association(
                data,
                locs,
                phase_type,
                phase_weight,
                len(stations),
                picks_.index.to_numpy(),
                event_idx0,
                config,
                stations,
            )
            event_idx0 += len(catalog)
            catalogs.extend(catalog)

    ### create catalog
    catalogs = pd.DataFrame(catalogs, columns=["time(s)"] + config["dims"] + ["magnitude", "sigma"])
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x / config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x / config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)
    # catalogs["event_idx"] = range(event_idx0)
    if config["use_amplitude"]:
        catalogs["covariance"] = catalogs["sigma"].apply(lambda x: f"{x[0][0]:.3f},{x[1][1]:.3f},{x[0][1]:.3f}")
    else:
        catalogs["covariance"] = catalogs["sigma"].apply(lambda x: f"{x[0][0]:.3f}")

    catalogs = catalogs[['time', 'magnitude', 'longitude', 'latitude', 'depth(m)', 'covariance']]
    catalogs = catalogs.to_dict(orient='records')
    print("GMMA:", catalogs)
    for event in catalogs:
        producer.send('gmma_events', key=event["time"], value=event)

    return catalogs
