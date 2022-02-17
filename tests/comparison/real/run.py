#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:09:51 2022

@author: mliu
"""
import datetime
from obspy import UTCDateTime
import pandas as pd
import os
import time
import shutil


start=datetime.datetime(2019, 1, 1, 0, 0)
end=datetime.datetime(2019, 1, 2, 0, 0)
#make pick inputfile
# picks = pd.read_csv('synthetic_picks.csv',header = None, sep = '\t', names=['timestamp', 'type','jk1','jk2','sta','jk3'])
picks = pd.read_csv('../synthetic_picks.csv',sep = '\t')
picks.rename(columns={"station_id": "sta"}, inplace=True)
picks['date'] = picks['timestamp']
for i in range(len(picks['timestamp'])):
    picks['date'][i] = datetime.datetime.strptime(picks['timestamp'][i],"%Y-%m-%dT%H:%M:%S.%f").strftime("%Y%m%d")
picks_by_date = picks.groupby('date').groups
for i in range((end-start).days):
    dateid=(start + datetime.timedelta(i)).strftime("%Y%m%d")
    if os.path.exists(dateid):
        shutil.rmtree(dateid)
    os.mkdir(dateid)
    picks_date = picks_by_date[dateid]
    for ii in picks_date:
        pick = picks.iloc[ii]
        phase_type = pick['type']
        phase_arrival = UTCDateTime(pick['timestamp']) - (UTCDateTime(start) + i * 86400)
        if phase_arrival< 1000:
            if phase_type == 'p':
                output_name = dateid+"/"+pick['sta'].split('.')[0] + '.' + pick['sta'].split('.')[1] + ".P.txt"
            else:
                output_name = dateid+"/"+pick['sta'].split('.')[0] + '.' + pick['sta'].split('.')[1] + ".S.txt"
            with open(output_name, 'a') as file:
                file.write('{} 1.0 0\n'.format(phase_arrival))

#make station inputfile
# stations = pd.read_csv('stations.csv', header = None, sep = ',', names = ['jk1', 'sta', 'stlo', 'stla', 'stev', 'jk2', 'jk3', 'jk4', 'jk5', 'jk6','jk7','jk8','jk9'])
stations = pd.read_csv('../synthetic_stations.csv', sep="\t")
stations.rename(columns={"station": "sta", "latitude": "stla", "longitude": "stlo", "elevation(m)": "stev"}, inplace=True)
with open('staton.dat', 'w') as f:
    for i in range(len(stations)):
                   f.write('{} {} {} {} HHZ {}\n'.format(stations['stlo'][i],stations['stla'][i],stations['sta'][i].split('.')[0],stations['sta'][i].split('.')[1],stations['stev'][i]/1000))


#run real
start_time = time.time()
for i in range((end-start).days):
    D = (start + datetime.timedelta(i)).strftime("%Y/%m/%d")+"/35.705"
    R = "0.6/20/0.03/2/5"
    G = "1/20/0.01/1"
    V = "6/3.428"
    S = "0/0/10/0/1/0/2/0.2"
    command_line = "./REAL -D" + D + " -R" + R + " -S" + S + " -G" + G + " -V" + V  + " staton.dat ./" + (start + datetime.timedelta(i)).strftime("%Y%m%d") + " ./tt_db/ttdb.txt" 
    os.system(command_line)
print(f"--- {time.time() - start_time} seconds ---")