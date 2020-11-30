import itertools
import numpy as np
from numpy.core.defchararray import center
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from GMMA import mixture

figure_dir = "figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

np.random.seed(123)
vp = 6.0
vs = vp/1.75
num_true_event = 3
num_station = 35
station_loc = np.random.uniform(low=[0,0,0], high=[100,100,20], size=(num_station, 3))
event_loc = np.random.uniform(low=[0,0,0], high=[100,100,20], size=(num_true_event, 1, 3))
# event_t0 = np.random.uniform(low=0, high=100/vp, size=(num_true_event, 1))
event_t0 = np.linspace(0, 100/vp, num_true_event)[:,np.newaxis]
dist = np.linalg.norm(station_loc - event_loc, axis=-1) # n_sta, n_eve, n_dim(x, y, z)

num_event = num_true_event
centers_init = np.hstack([np.random.uniform(low=[0,0,0], high=[100,100,0], size=[num_event, 3]),
                          np.random.uniform(low=0, high=100/vp, size=[num_event, 1])]) # n_eve, n_dim(x, y, z) + 1(t)

tp = dist / vp + event_t0
ts = dist / vs + event_t0
phase_time = np.hstack([tp, ts]) # n_eve, n_phase
station_loc = np.vstack([station_loc, station_loc]) # n_sta, n_dim(x, y, z)

phase_err = 3.0
phase_type = ['p']*tp.size + ['s']*ts.size #n_phase
locs = np.zeros([phase_time.size, station_loc.shape[-1]]) #n_phase, n_dim(x, y, z)
data = np.zeros([phase_time.size, 1]) #n_phase
for i in range(phase_time.shape[0]):
    for j in range(phase_time.shape[1]):
        locs[i + j*phase_time.shape[0], :] = station_loc[j, :]
        data[i + j*phase_time.shape[0], -1] = phase_time[i, j] + np.random.uniform(low=-phase_err, high=phase_err)

plt.figure(figsize=(15,4))
plt.subplot(131)
for i in range(num_true_event):
    plt.plot(event_t0[i, 0], event_loc[i, 0, 0], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
for i in range(num_true_event): 
    plt.plot(data[i::num_true_event, 0], locs[i::num_true_event, 0], "x", c=f"C{i}", label=f"P/S-phases #{i+1}")
plt.ylim([0, 100])
plt.xlim(left=-3)
plt.ylabel("X (km)")
plt.xlabel("Time (s)")
# plt.colorbar()
plt.subplot(132)
for i in range(num_true_event):
    plt.plot(event_t0[i, 0], event_loc[i, 0, 1], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
for i in range(num_true_event): 
    plt.plot(data[i::num_true_event, 0], locs[i::num_true_event, 1],  "x", c=f"C{i}", label=f"P/S-phases #{i+1}")
plt.ylim([0, 100])
plt.xlim(left=-3)
plt.ylabel("Y (km)")
plt.xlabel("Time (s)")
# plt.colorbar()
plt.subplot(133)
for i in range(num_true_event):
    plt.plot(event_t0[i, 0], event_loc[i, 0, 2], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
for i in range(num_true_event): 
    plt.plot(data[i::num_true_event, 0], locs[i::num_true_event, 2], "x", c=f"C{i}", label=f"P/S-phases #{i+1}")
plt.ylim([0, 20])
plt.xlim(left=-3)
plt.ylabel("Z (km)")
plt.xlabel("Time (s)")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
plt.savefig(os.path.join(figure_dir, f"data_2D_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")

# plt.figure()
# ax = plt.gca(projection='3d')
# # for i in range(num_true_event):
# #     ax.plot(event_loc[i, 0, 0], event_loc[i, 0, 1], event_t0[i], label=f"True Epicenter #{i+1}", markersize=10)
# for i in range(num_true_event): 
#     ax.scatter(locs[i::num_true_event, 0], locs[i::num_true_event, 1], data[i::num_true_event, 0])
# # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# # plt.tight_layout()
# plt.xlim([0,100])
# # plt.ylim(bottom=-3)
# plt.ylim([0, 100])
# plt.xlabel("X (km)")
# plt.ylabel("Time (s)")
# # plt.axis("scaled")
# # plt.colorbar()
# # plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
# plt.savefig(os.path.join(figure_dir, f"data2d_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")


# plt.figure()
# for i in range(num_true_event):
#     plt.plot(event_loc[i, 0, 0], event_loc[i, 0, 1], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
# for i in range(num_true_event): 
#     plt.scatter(locs[i::num_true_event, 0], locs[i::num_true_event, 1], c=data[i::num_true_event, 0], label=f"P/S-phases #{i+1}")
# # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# # plt.tight_layout()
# plt.xlim([0,100])
# # plt.ylim(bottom=-3)
# plt.ylim([0, 100])
# plt.xlabel("X (km)")
# plt.ylabel("Time (s)")
# plt.axis("scaled")
# plt.colorbar()
# # plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
# plt.savefig(os.path.join(figure_dir, f"data2d_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")


# Fit a Gaussian mixture with EM 
gmm = mixture.GaussianMixture(n_components=num_event, covariance_type='full', 
                              centers_init=centers_init.copy(), station_locs=locs, phases=phase_type).fit(data)

pred = gmm.predict(data) 
prob = gmm.predict_proba(data)

idx = np.argsort(gmm.centers_[:, -1])

plt.figure(figsize=(15,4))
plt.subplot(131)
for i in range(num_event):
    plt.plot(gmm.centers_[i, -1], gmm.centers_[i, 0], "P", c=f"C{idx[i]}", label=f"Estimated Epicenter #{i+1}", markersize=10)
for i in range(num_event): 
    plt.scatter(data[pred==i, 0], locs[pred==i, 0],  c=f"C{idx[i]}", s=prob[pred==i, i]*10, label=f"P/S-phases #{i+1}")
plt.ylim([0, 100])
plt.xlim(left=-3)
plt.ylabel("X (km)")
plt.xlabel("Time (s)")
# plt.colorbar()
plt.subplot(132)
for i in range(num_event):
    plt.plot(gmm.centers_[i, -1], gmm.centers_[i, 1], "P", c=f"C{idx[i]}", label=f"Estimated Epicenter #{i+1}", markersize=10)
for i in range(num_event): 
    plt.scatter(data[pred==i, 0], locs[pred==i, 1],  c=f"C{idx[i]}", s=prob[pred==i, i]*10,label=f"P/S-phases #{i+1}")
plt.ylim([0, 100])
plt.xlim(left=-3)
plt.ylabel("Y (km)")
plt.xlabel("Time (s)")
# plt.colorbar()
plt.subplot(133)
for i in range(num_event):
    plt.plot(gmm.centers_[i, -1], gmm.centers_[i, 2], "P", c=f"C{idx[i]}", label=f"Estimated Epicenter #{i+1}", markersize=10)
for i in range(num_event): 
    plt.scatter(data[pred==i, 0], locs[pred==i, 2],  c=f"C{idx[i]}", s=prob[pred==i, i]*10, label=f"P/S-phases #{i+1}")
plt.ylim([0, 20])
plt.xlim(left=-3)
plt.ylabel("Z (km)")
plt.xlabel("Time (s)")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
plt.savefig(os.path.join(figure_dir, f"result_2D_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")


# plt.figure(figsize=(6,8))
# for i in range(num_event): 
#     # if len(pred[pred==i]) > 0:
#     plt.scatter(locs[pred==i, 0],  data[pred==i, 0], c=f"C{i}", s=prob[pred==i, i]*20, label=f"P/S-phases #{i+1}")
# for i in range(num_event):
#     # if len(pred[pred==i]) > 0:
#     plt.plot(gmm.centers_[i, 0], gmm.centers_[i, -1], "P", c=f"C{i}", markersize=10, label=f"Estimated Epicenter #{i+1}")
# plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# # plt.tight_layout()
# plt.xlim([0,100])
# plt.ylim(bottom=-3)
# plt.xlabel("X (km)")
# plt.ylabel("Time (s)") 
# # plt.savefig("result.png")
# # plt.savefig(os.path.join(figure_dir, f"result_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
# plt.savefig(os.path.join(figure_dir, f"test_result_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")


# plt.figure()
# for i in range(num_true_event):
#     plt.scatter(locs[pred==i, 0], locs[pred==i, 1], c= data[pred==i, 0], label=f"P/S-phases #{i+1}")
#     # break
# # for i in range(num_true_event): 
# #     plt.plot(gmm.centers_[i, 0], gmm.centers_[i, 1], "P", c=f"C{i}", markersize=10, label=f"Estimated Epicenter #{i+1}")
# # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# # plt.tight_layout()
# plt.xlim([0,100])
# # plt.ylim(bottom=-3)
# plt.ylim([0, 100])
# plt.xlabel("X (km)")
# plt.ylabel("Time (s)")
# plt.axis("scaled")
# plt.colorbar()
# # plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.png"), bbox_inches="tight")
# plt.savefig(os.path.join(figure_dir, f"result2d_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}.pdf"), bbox_inches="tight")


# plt.figure()
# plt.plot(event_loc[..., 0], event_loc[..., 1], 'x', label="Events")
# plt.plot(gmm.centers_[..., 0], gmm.centers_[..., 1], 'o', label="Predicted Events")
# plt.scatter(locs[np.array(phase_type)=="p",0], locs[np.array(phase_type)=="p",1], c=data[np.array(phase_type)=="p",0]-gmm.means_[0,np.array(phase_type)=="p",0], label="p")
# # plt.scatter(locs[np.array(phase_type)=="s",0], locs[np.array(phase_type)=="s",1], c=data[np.array(phase_type)=="s",0], label="s")
# plt.legend()
# plt.colorbar()
# plt.savefig("result2d.png")

# dpgmm = mixture.BayesianGaussianMixture(n_components=4, covariance_type='full',
#                                         centers_init=centers_init.copy(), station_locs=locs, phases=phase_type).fit(data)


# plt.figure()
# plt.plot(locs[:,0], data[:,0], 'k.', label="Data")
# for i in range(len(dpgmm.means_)):
#     plt.plot(locs[:,0], dpgmm.means_[i,:,0], 'x', label=f"EQ{i}")
# plt.legend()
# plt.savefig("result_dpgmm.png")

# bic = []
# lowest_bic = np.infty
# n_components_range = range(2, 7)
# for n_components in n_components_range:
#     centers_init = np.zeros([n_components, event_loc.shape[-1]+1])
#     gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', 
#                                   centers_init=centers_init.copy(), station_locs=locs).fit(data)
#     bic.append(gmm.bic(data))
#     if bic[-1] < lowest_bic:
#         lowest_bic = bic[-1]
#         best_gmm = gmm

# plt.figure()
# plt.plot(n_components_range, bic)
# plt.xticks(n_components_range)
# plt.savefig("BIC.png")
