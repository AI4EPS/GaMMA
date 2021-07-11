import pandas as pd
from datetime import datetime, timedelta
from gmma import mixture
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, NamedTuple, Union
from fastapi import FastAPI
from json import dumps
import pickle
from kafka import KafkaProducer
import os

try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    producer = KafkaProducer(bootstrap_servers=[BROKER_URL],
                             key_serializer=lambda x: dumps(x).encode('utf-8'),
                             value_serializer=lambda x: dumps(x).encode('utf-8'))
    print('k8s kafka connection success!')
except BaseException:
    print('k8s Kafka connection error')

    try:
        print('Connecting to local kafka')
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                 key_serializer=lambda x: dumps(x).encode('utf-8'),
                                 value_serializer=lambda x: dumps(x).encode('utf-8'))
        print('local kafka connection success!')
    except BaseException:
        print('local Kafka connection error')

app = FastAPI()

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# STATION_CSV = "stations.csv"
# STATION_CSV = "stations_iris.csv"

CONFIG_PKL = os.path.join(PROJECT_ROOT, "tests/config_hawaii.pkl")
STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations_hawaii.csv")
with open(CONFIG_PKL, "rb") as fp:
    config = pickle.load(fp)
## read stations
stations = pd.read_csv(STATION_CSV, delimiter="\t")
stations = stations.rename(columns={"station":"id"})
stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0])*config["degree2km"])
stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1])*config["degree2km"])
stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x/1e3)
## setting GMMA configs
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = True
config["use_amplitude"] = True
dx = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]
dy = (np.array(config["ylim_degree"])-np.array(config["center"][1]))*config["degree2km"]
dz = 21
config["bfgs_bounds"] = ((dx[0]-1, dx[1]+1), #x
                            (dy[0]-1, dy[1]+1), #y
                            (0, dz), #x
                            (None, None)) #t
config["dbscan_eps"] = min(np.sqrt((stations["x(km)"].max()-stations["x(km)"].min())**2 +
                                    (stations["y(km)"].max()-stations["y(km)"].min())**2)/(6.0/1.75), 10)
config["dbscan_min_samples"] = min(len(stations), 5)
config["min_picks_per_eq"] = min(len(stations)//2, 5)
config["oversample_factor"] = min(len(stations)//2, 5)
print("Config: ", config)

class Pick(BaseModel):
    picks: List[Dict[str, Union[float, str]]]


to_seconds = lambda t: t.timestamp(tz="UTC")
from_seconds = lambda t: pd.Timestamp.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
# to_seconds = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
# from_seconds = lambda t: [datetime.utcfromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for x in t]

def convert_picks_csv(picks, stations, config):
    t = picks["timestamp"].apply(lambda x: x.timestamp()).to_numpy()
    a = picks["amp"].apply(lambda x: np.log10(x*1e2)).to_numpy()
    data = np.stack([t, a]).T
    meta = pd.merge(stations, picks["id"], on="id")
    locs = meta[config["dims"]].to_numpy()
    phase_type = picks["type"].apply(lambda x: x.lower()).to_numpy()
    phase_weight = picks["prob"].to_numpy()[:,np.newaxis]
    return data, locs, phase_type, phase_weight

def association(data, locs, phase_type, phase_weight, num_sta, pick_idx, event_idx0, config, pbar=None):

    db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(np.hstack([data[:,0:1], locs[:,:2]/6.0]))#.fit(data[:,0:1])
    labels = db.labels_
    unique_labels = set(labels)
    events = []
    preds = []
    probs = []
    
    assignment = []
    for k in unique_labels:
        if k == -1:
            continue
        
        class_mask = (labels == k)
        data_ = data[class_mask]
        locs_ = locs[class_mask]
        phase_type_ = phase_type[class_mask]
        phase_weight_ = phase_weight[class_mask]
        pick_idx_ = pick_idx[class_mask]
        
        if pbar is not None:
            pbar.set_description(f"Process {len(data_)} picks")

        num_event_ = min(max(int(len(data_)/min(num_sta,10)*config["oversample_factor"]), 1), len(data_))
        t_range = max(data_[:,0].max() - data_[:,0].min(), 1)
        centers_init = np.vstack([np.ones(num_event_)*np.mean(stations["x(km)"]),
                                    np.ones(num_event_)*np.mean(stations["y(km)"]),
                                    np.zeros(num_event_),
                                    np.linspace(data_[:,0].min()-0.1*t_range, data_[:,0].max()+0.1*t_range, num_event_)]).T # n_eve, n_dim(x, y, z) + 1(t)

        if config["use_amplitude"]:
            covariance_prior = np.array([[1,0],[0,1]]) * 3
        else:
            covariance_prior = np.array([[1]])
            data = data[:,0:1]
            
        gmm = mixture.BayesianGaussianMixture(n_components=num_event_, 
                                                weight_concentration_prior=1000/num_event_,
                                                mean_precision_prior=0.3/t_range,
                                                covariance_prior=covariance_prior,
                                                init_params="centers",
                                                centers_init=centers_init, 
                                                station_locs=locs_, 
                                                phase_type=phase_type_, 
                                                phase_weight=phase_weight_,
                                                loss_type="l1",
                                                bounds=config["bfgs_bounds"],
                                                max_covar=10.0,
                                                reg_covar=0.1,
                                                ).fit(data_) 

        pred = gmm.predict(data_) 
        prob_matrix = gmm.predict_proba(data_)
        prob_eq = prob_matrix.mean(axis=0)
#             prob = prob_matrix[range(len(data_)), pred]
#             score = gmm.score(data_)
#             score_sample = gmm.score_samples(data_)
        prob = np.exp(gmm.score_samples(data_))

        idx = np.array([True if len(data_[pred==i, 0]) >= config["min_picks_per_eq"] else False for i in range(len(prob_eq))]) #& (prob_eq > 1/num_event) #& (sigma_eq[:, 0,0] < 40)

        time = gmm.centers_[idx, len(config["dims"])]
        loc = gmm.centers_[idx, :len(config["dims"])]
        if config["use_amplitude"]:
            mag = gmm.centers_[idx, len(config["dims"])+1]
        sigma_eq = gmm.covariances_[idx,...]

        for i in range(len(time)):
            tmp = {"time(s)": time[i],
                    "magnitude": mag[i],
                    "sigma": sigma_eq[i].tolist()}
            for j, k in enumerate(config["dims"]):
                tmp[k] = loc[i][j]
            events.append(tmp)
            
        for i in range(len(pick_idx_)):
            assignment.append((pick_idx_[i], pred[i]+event_idx0, prob[i]))
        
        event_idx0 += len(time)
        
    return events, assignment


@app.get('/predict')
def predict(data: Pick):

    picks = data.picks
    if len(picks) == 0:
        return []

    # picks = pd.read_json(picks)
    picks = pd.DataFrame(picks)
    # print(picks)
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    picks["time_idx"] = picks["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H")) ## process by hours
    
    ## run GMMA association
    if len(picks) < 500:
        data, locs, phase_type, phase_weight = convert_picks_csv(picks, stations, config)
        catalogs, _ = association(data, locs, phase_type, phase_weight, len(stations), picks.index.to_numpy(), 0, config)
    else:
        catalogs = []
        for i, hour in enumerate(sorted(list(set(picks["time_idx"])))):
            picks_ = picks[picks["time_idx"] == hour]
            data, locs, phase_type, phase_weight = convert_picks_csv(picks_, stations, config)
            catalog, _ = association(data, locs, phase_type, phase_weight, len(stations), picks_.index.to_numpy(), 0, config)
            catalogs.extend(catalog)
    
    ### create catalog
    catalogs = pd.DataFrame(catalogs, columns=["time(s)"]+config["dims"]+["magnitude", "sigma"])
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: np.round(x/config["degree2km"] + config["center"][0], decimals=3))
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: np.round(x/config["degree2km"] + config["center"][1], decimals=3))
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: np.round(x*1e3, decimals=3))
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
