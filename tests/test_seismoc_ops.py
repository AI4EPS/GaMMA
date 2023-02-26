# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from gamma.seismic_ops import linloc, eikoloc, eikonal_solve, calc_time

# %%
np.random.seed(11)
vp = 6.0
vp_vs_ratio = 1.75
vs = vp/vp_vs_ratio
num_event = 3
num_station = 10
xmax = 100
rmax = 100
zmax = 30
# station_loc = np.random.uniform(low=[-xmax, -xmax, 0], high=[xmax, xmax, xmax], size=(num_station,3))
# station_loc = np.array([[10, 10, 0],
#                         [10, -10, 0],
#                         [-10, 10, 0],
#                         [-10, -10, 0],
#                         [5, 5, 0],
#                         [5, -5, 0],
#                         [-5, 5, 0],
#                         [-5, -5, 0],])
station_loc = np.array([[1, 0, 0],])
event_loc = np.random.uniform(low=0, high=xmax, size=(num_event,3))
event_t = np.random.uniform(low=0, high=xmax/vp, size=(num_event,1))
event_loc = np.array([[0, 0, 10]])
event_t = np.array([[0]])


# %%
phase_time = []
phase_type = []
phase_loc = []
phase_weight = []

for ev_loc, ev_t in zip(event_loc, event_t):
    
    for st_loc in station_loc:
        
        for type, v in zip(["P", "S"], [vp, vs]):
            
            dist = np.linalg.norm(st_loc - ev_loc)
            t = dist / v + ev_t
            phase_time.append(t.item())
            phase_type.append(type.lower())
            phase_loc.append(st_loc.tolist())
            phase_weight.append(1.0)

    phase_time = np.array(phase_time)[:,np.newaxis]
    phase_type = np.array(phase_type)
    phase_loc = np.array(phase_loc)
    phase_weight = np.array(phase_weight)[:,np.newaxis]
    break

# %%
# rlim = [-xmax, xmax]
rlim = [0, rmax]
zlim = [0, zmax]
h = 0.3
edge_grids = 0

rgrid = np.arange(rlim[0]-edge_grids*h, rlim[1], h)
zgrid = np.arange(zlim[0]-edge_grids*h, zlim[1], h)
m = len(rgrid)
n = len(zgrid)

zz = [0.0, zlim[1]]
vp = [6.0, 6.0]
vp1d = np.interp(zgrid, zz, vp)
vs1d = vp1d / vp_vs_ratio
vp = np.ones((m, n)) * vp1d
vs = np.ones((m, n)) * vs1d

up = 1000 * np.ones((m, n))
up[edge_grids, edge_grids] = 0.0
up = eikonal_solve(up, vp, h)

us = 1000 * np.ones((m, n))
us[edge_grids, edge_grids] = 0.0
us = eikonal_solve(us, vs, h)

up = torch.tensor(up, dtype=torch.float32)
us = torch.tensor(us, dtype=torch.float32)
rgrid = torch.tensor(rgrid, dtype=torch.float32)
zgrid = torch.tensor(zgrid, dtype=torch.float32)
rgrid, zgrid = torch.meshgrid(rgrid, zgrid, indexing="ij")

eikonal = {"up": up, "us": us, "rgrid": rgrid, "zgrid": zgrid, "h": h}

# %%
loc0 = np.array([[0, 0, 5, 0]]) # x, y, t
# loc0 = np.array([[0, 0, 0, 0]]) # x, y, t
# loc0 = np.array([[0, 0, -30, 0]]) # x, y, t
# loc0 = np.array([[20, -10, 0, -5]]) # x, y, t
inv_eikoloc, loss = eikoloc(loc0, phase_time, phase_type, phase_loc, phase_weight, up=up, us=us, rgrid=rgrid, zgrid=zgrid, h=h, max_iter=1000)
print(f"true loc: {ev_loc}, true t0: {ev_t}")
print(f"inv loc (eikoloc): {inv_eikoloc}; loss: {loss}")

inv_linloc, loss = linloc(loc0, phase_time, phase_type, phase_loc, phase_weight, max_iter=1000)
# inv_eikoloc, loss = eikoloc(loc0, phase_time, phase_type, phase_loc, phase_weight, eikonal=eikonal)
print(f"true loc: {ev_loc}; true t0: {ev_t}")
print(f"inv loc (linloc): {[round(x, 3) for x in inv_linloc[0].tolist()]}, loss: {loss}")
# print(f"inv loc: {inv_eikoloc}")

# %%
loc0 = np.array([[0, 0, 10, 0]]) # x, y, t
loc0 = np.array([[0, 0, 9, 0]]) # x, y, t
traveltime = calc_time(loc0, phase_loc, phase_type, eikonal=eikonal)
print("True", phase_time[:5])
print("Predicted (eikoloc)", traveltime[:5])


traveltime = calc_time(loc0, phase_loc, phase_type)
# print("True", phase_time[:5])
print("Predicted (linloc)", traveltime[:5])

# %%
