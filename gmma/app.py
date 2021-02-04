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

app = FastAPI()

stations = pd.read_csv("../../stations.csv", delimiter="\t", index_col="station")
num_sta = len(stations)
dims = ['x(km)', 'y(km)', 'z(km)']
bounds = ((-1, 111),(-1, 111),(0, 20), (None, None))
use_dbscan = True
use_amplitude = True
dbscan_eps = 111/(6.0/1.75)/2
dbscan_min_samples = int(16 * 0.8)
min_picks_per_eq = int(16 * 0.6)
oversample_factor = 5.0
verbose = 1

class Pick(BaseModel):
    picks: List[Dict[str, Union[float, str]]]


to_seconds = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
from_seconds = lambda t: [datetime.fromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for x in t]

def convert_picks(picks, stations):
    data, locs, phase_type, phase_weight = ([],[],[],[])
    for pick in picks:
        data.append([to_seconds(pick["timestamp"]), np.log10(pick["amp"]*1e2)])
        locs.append(stations.loc[pick["id"]][dims].values.astype("float"))
        phase_type.append(pick["type"].lower())
        phase_weight.append(pick["prob"])
    data = np.array(data)
    locs = np.array(locs)
    phase_weight = np.array(phase_weight)[:,np.newaxis]
    return data, locs, phase_type, phase_weight


def association(data, locs, phase_type, phase_weight):
    
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(data)
    labels = db.labels_
    unique_labels = set(labels)

    events = []
    for k in unique_labels:
        if k == -1:
            continue

        class_mask = (labels == k)
        data_ = data[class_mask]
        locs_ = locs[class_mask]
        phase_type_ = np.array(phase_type)[class_mask]
        phase_weight_ = phase_weight[class_mask]

        num_event_ = min(max(int(len(data_)/num_sta*oversample_factor), 1), len(data_))
        t0 = data_[:,0].min()
        t_range = max(data_[:,0].max() - data_[:,0].min(), 1)
        centers_init = np.vstack([np.ones(num_event_)*np.mean(stations["x(km)"]),
                                  np.ones(num_event_)*np.mean(stations["y(km)"]),
                                  np.zeros(num_event_),
                                  np.linspace(data_[:,0].min()-0.1*t_range, data_[:,0].max()+0.1*t_range, num_event_)]).T # n_eve, n_dim(x, y, z) + 1(t)
        

        if use_amplitude:
            covariance_prior = np.array([[1,0],[0,1]]) * 3
        else:
            covariance_prior = np.array([[1]])
            data = data[:,0:1]
        gmm = mixture.BayesianGaussianMixture(n_components=num_event_, 
                                              weight_concentration_prior=1000/num_event_,
                                              mean_precision_prior = 0.3/t_range,
                                              covariance_prior = covariance_prior,
                                              init_params="centers",
                                              centers_init=centers_init, 
                                              station_locs=locs_, 
                                              phase_type=phase_type_, 
                                              phase_weight=phase_weight_,
                                              loss_type="l1",
                                              bounds=bounds,
                                              max_covar=10.0,
                                              reg_covar=0.1,
                                              ).fit(data_) 
        
        pred = gmm.predict(data_) 
        prob = gmm.predict_proba(data_)
        prob_eq = prob.mean(axis=0)
        prob_data = prob[range(len(data_)), pred]
        score_data = gmm.score_samples(data_)

        idx = np.array([True if len(data_[pred==i, 0]) >= max(num_sta*0.6, 4) else False for i in range(len(prob_eq))]) #& (prob_eq > 1/num_event) #& (std_eq[:, 0,0] < 40)
        eq_idx = np.arange(len(idx))[idx]

        time = from_seconds(gmm.centers_[idx, len(dims)])
        loc = gmm.centers_[idx, :len(dims)]
        if use_amplitude:
            mag = gmm.centers_[idx, len(dims)+1]
        std_eq = gmm.covariances_[idx,...]

        for i in range(len(time)):
            events.append({"time": time[i],
                           "location": loc[i].tolist(),
                           "magnitude": mag[i],
                           "std": std_eq[i].tolist()})

        print(events)
    

    return events

@app.get('/predict')
def predict(data: Pick):
    
    picks = data.picks
    data, locs, phase_type, phase_weight = convert_picks(picks, stations)
    event_log = association(data, locs, phase_type, phase_weight)

    return event_log

