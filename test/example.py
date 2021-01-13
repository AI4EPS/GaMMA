import numpy as np
import matplotlib.pyplot as plt
import os
from gmma import mixture
from collections import defaultdict
import time

figure_dir = "figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

np.random.seed(123)
use_amplitude = False
vp = 6.0
vs = vp/1.75
num_true_event = 5
num_station = 10
# station_loc = np.random.uniform(low=0.0+10.0, high=100.0-10.0, size=(num_station, 1))
station_loc = np.linspace(0.0+10.0, 100.0-10.0, num_station)[:, np.newaxis]
event_loc = np.random.uniform(low=0, high=100, size=(num_true_event, 1, 1))
event_mag = np.random.uniform(low=1, high=5, size=(num_true_event, 1)) 
# event_t0 = np.random.uniform(low=0, high=100/vp, size=(num_true_event, 1))
event_t0 = np.linspace(0, 100/vp, num_true_event)[:,np.newaxis]
dist = np.linalg.norm(station_loc - event_loc, axis=-1) # n_sta, n_eve, n_dim(x, y, z)

num_event = num_true_event * 4

tp = dist / vp + event_t0
ts = dist / vs + event_t0
logA = event_mag + 2.48 - 2.76 * np.log10(dist)
phase_time = np.hstack([tp, ts]) # n_eve, n_phase
station_loc = np.vstack([station_loc, station_loc]) # n_sta, n_dim(x, y, z)

phase_type = ['p']*tp.size + ['s']*ts.size #n_phase
locs = np.zeros([phase_time.size, station_loc.shape[-1]]) #n_phase, n_dim(x, y, z)
station_idx = np.zeros([phase_time.size, 1]) #n_phase, n_dim(x, y, z)
data = np.zeros([phase_time.size, 2]) #n_phase

phase_err = 0.0
for i in range(phase_time.shape[0]): #num_true_event
    for j in range(phase_time.shape[1]): #num_station
        locs[i + j*phase_time.shape[0], :] = station_loc[j, :]
        station_idx[i + j*phase_time.shape[0], :] = j
        data[i + j*phase_time.shape[0], 0] = phase_time[i, j] + np.random.uniform(low=-phase_err, high=phase_err)
        data[i + j*phase_time.shape[0], 1] = logA[i, np.mod(j, logA.shape[1])] + np.random.uniform(low=-phase_err, high=phase_err)

phase_fp = 1.0 # false positive
n_noise = int(num_true_event * num_station * phase_fp)
locs_noise = np.zeros([n_noise, station_loc.shape[-1]]) #n_phase, n_dim(x, y, z)
station_idx_noise = np.zeros([n_noise, 1])
data_noise = np.zeros([n_noise, 2]) #n_phase
phase_type_noise = []

for i in range(n_noise):
    tmp = np.random.randint(len(station_loc))
    locs_noise[i, :] = station_loc[tmp, :]
    station_idx_noise[i, 0] = tmp
    data_noise[i, 0] = np.random.uniform(low=0, high=np.max(ts))
    data_noise[i, 1] = np.random.uniform(low=np.min(logA), high=np.max(logA))
    phase_type_noise.append(np.random.choice(["p", "s"]))

plt.figure(figsize=(10,6))
for i in range(num_true_event):
    plt.plot(event_t0[i, 0], event_loc[i, 0], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
for i in range(num_true_event): 
    plt.plot(data[i::num_true_event, 0], locs[i::num_true_event, 0],  'x', c=f"C{i}", label=f"P/S-phases #{i+1}")
    # plt.plot(data[i::num_true_event, 0], locs[i::num_true_event, 0],  'x', c="gray", label=f"P/S-phases #{i+1}")
plt.plot(data_noise[:, 0], locs_noise[:, 0], 'x', c="gray", label=f"False Positives")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.ylim([0,100])
plt.xlim(left=-3)
plt.ylabel("X (km)")
plt.xlabel("Time (s)")
plt.savefig("test_time.png")
# plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}_{phase_fp:.1f}.png"), bbox_inches="tight")
# plt.savefig(os.path.join(figure_dir, f"data_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}_{phase_fp:.1f}.pdf"), bbox_inches="tight")

# plt.figure()
# for i in range(num_true_event):
#     plt.plot(event_mag[i, 0], event_loc[i, 0], "P", c=f"C{i}", label=f"True Epicenter #{i+1}", markersize=10)
# for i in range(num_true_event): 
#     plt.plot(data[i::num_true_event, 1], locs[i::num_true_event, 0],  'x', c=f"C{i}", label=f"P/S-phases #{i+1}")
# plt.plot(data_noise[:, 0], locs_noise[:, 0], 'x', c="gray", label=f"False Positives")
# plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
# # plt.tight_layout()
# plt.ylim([0,100])
# plt.xlim(left=-3)
# plt.ylabel("X (km)")
# plt.xlabel("Time (s)")
# plt.savefig("test_mag.png")

locs = np.vstack([locs, locs_noise])
station_idx = np.vstack([station_idx, station_idx_noise])
data = np.vstack([data, data_noise])
phase_type = phase_type + phase_type_noise


if not use_amplitude:
    data = data[:,0:1]

centers_init = np.vstack([np.ones(num_event)*np.mean(station_loc[:,0]),
                        #   np.ones(num_event)*np.mean(station_loc[:,1]),
                         np.linspace(data[:,0].min(), data[:,0].max(), num_event)]).T # n_eve, n_dim(x, y, z) + 1(t)


# Fit a Gaussian mixture with EM 
dummy_prob = 1/((2*np.pi)**(data.shape[-1]/2) * 2)
# dummy_prob = 1/30
print(f"dummy_prob = {dummy_prob}")

t_start = time.time()
gmm = mixture.GaussianMixture(n_components=num_event, #covariance_type='full', 
                            #   reg_covar=0.1, 
                              station_locs=locs, #centers_init=centers_init.copy(), 
                              phase_type=phase_type,
                              dummy_comp=False, 
                              dummy_prob=dummy_prob,
                              loss_type="l1"
                              ).fit(data)
t_end = time.time()
print(f"GMMA time = {t_end - t_start}")

plt.figure(figsize=(10,6))
pred = gmm.predict(data) 
prob = gmm.predict_proba(data)
prob_eq = prob.mean(axis=0)
score = gmm.score_samples(data)
colors = defaultdict(int)
idx = np.argsort(gmm.centers_[:, 1])
dum = 0
min_score = np.log(0.3/num_event)
min_num = num_station
for i in idx:
    if len(pred[pred==i]) > min_num:
    # if len(score[(pred==i) & (score>min_score)]) > 2:
        colors[i] = f"C{dum}"
        dum += 1
for i in idx:
    if len(pred[pred==i]) <= min_num:
    # if len(score[(pred==i) & (score>min_score)]) <= 2:
        colors[i] = f"C{dum}"
        dum += 1

for i in range(num_event-1): 
    # if len(pred[pred==i]) > 0:
    plt.scatter(data[(pred==i), 0], locs[(pred==i), 0],  c=colors[i], s=prob[(pred==i), i]*20, label=f"P/S-phases #{i+1}")
    # plt.scatter(data[pred==i, 0], locs[pred==i, 0],  c=colors[i], s=np.maximum(score[pred==i], min_score*np.ones_like(score[pred==i])-min_score+0.5)*5, label=f"P/S-phases #{i+1}")

for i in range(num_event-1):
    if len(pred[pred==i]) > min_num:
    # if len(score[(pred==i) & (score>min_score)]) > 2:
        plt.plot(gmm.centers_[i, 1], gmm.centers_[i, 0], "P", c=colors[i], markersize=10, label=f"Estimated Epicenter #{i+1}")

plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.ylim([0,100])
plt.xlim(left=-3)
plt.ylabel("X (km)")
plt.xlabel("Time (s)") 
# plt.savefig("result.png")
plt.savefig(os.path.join(figure_dir, f"result_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}_{phase_fp:.1f}.png"), bbox_inches="tight")
plt.savefig(os.path.join(figure_dir, f"result_{num_true_event:02d}_{num_event:02d}_{phase_err:01.1f}_{phase_fp:01.1f}.pdf"), bbox_inches="tight")
plt.show()



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
