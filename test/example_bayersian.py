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

#################### Data ##############################
## network setting
use_amplitude = True
vp = 6.0
vs = vp/1.75
num_true_event = 5
num_station = 10
event_loc = np.random.uniform(low=0+10.0, high=100.0-10.0, size=(num_true_event, 1))
event_mag = np.random.uniform(low=1, high=5, size=(num_true_event, 1)) 
event_t0 = np.linspace(0, 110/vp, num_true_event)[:,np.newaxis]
station_loc = np.linspace(0.0+5.0, 100.0-5.0, num_station)[:, np.newaxis] + np.random.uniform(low=-4, high=4, size=(num_station,1))

## synthetic picks
dist = np.linalg.norm(station_loc - event_loc[:,np.newaxis,:], axis=-1) # n_sta, n_eve, n_dim(x, y, z)
tp = dist / vp + event_t0
ts = dist / vs + event_t0
# logA = event_mag + 2.48 - 2.76 * np.log10(dist)
## Picozzi et al. (2018) A rapid response magnitude scale...
c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
logA = c0 + c1*(event_mag-3.5) + c3*np.log10(dist)
## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
# c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
# logA = c0 + c1*event_mag + c3*np.log10(dist)

phase_time = np.hstack([tp, ts]) # n_eve, n_phase
station_loc = np.vstack([station_loc, station_loc]) # n_sta, n_dim(x, y, z)

phase_type = ['p']*tp.size + ['s']*ts.size #n_phase
phase_type = np.array(phase_type)
locs = np.zeros([phase_time.size, station_loc.shape[-1]]) #n_phase, n_dim(x, y, z)
data = np.zeros([phase_time.size, 2]) #n_phase
label = np.tile(np.arange(num_true_event), num_station*2)

## add phase time error
phase_err = 1.0
for i in range(phase_time.shape[0]): #num_true_event
    for j in range(phase_time.shape[1]): #num_station
        locs[i + j*phase_time.shape[0], :] = station_loc[j, :]
        data[i + j*phase_time.shape[0], 0] = phase_time[i, j] + np.random.uniform(low=-phase_err, high=phase_err)
        # data[i + j*phase_time.shape[0], 1] = np.abs(logA[i, np.mod(j, logA.shape[1])] + np.random.uniform(low=-phase_err, high=phase_err))
        data[i + j*phase_time.shape[0], 1] = logA[i, np.mod(j, logA.shape[1])] + np.random.uniform(low=-phase_err, high=phase_err)

num_picks = len(phase_type)
idx = np.random.choice(range(num_picks), size=int(num_picks*0.7), replace=False)
locs = locs[idx]
data = data[idx]
phase_type = phase_type[idx]
label = label[idx]

## add false positive picks
phase_fp = 1.0
n_noise = int(len(data) * phase_fp)
locs_noise = np.zeros([n_noise, station_loc.shape[-1]]) #n_phase, n_dim(x, y, z)
data_noise = np.zeros([n_noise, 2]) #n_phase
phase_type_noise = []
for i in range(n_noise):
    tmp = np.mod(i, len(station_loc))
    locs_noise[i, :] = station_loc[tmp, :]
    data_noise[i, 1] = np.random.uniform(low=np.min(logA), high=np.max(logA))
    data_noise[i, 0] = np.random.uniform(low=-2, high=np.max(data[:,0]))
    phase_type_noise.append(np.random.choice(["p", "s"]))
phase_type_noise = np.array(phase_type_noise)

plt.figure(figsize=(12,4))
box = dict(boxstyle='round', facecolor='white', alpha=1)
text_loc = [0.05, 0.95]
marker_size = 3
plt.subplot(131)
if not use_amplitude:
    plt.scatter(data[:, 0], locs[:, 0], c="gray", s=12, marker='x')
    plt.scatter(data_noise[:, 0], locs_noise[:, 0], c="gray", s=12, marker="x")
else:
    plt.scatter(data[:, 0], locs[:, 0], c="gray", s=(data[:, 1]+marker_size)*10, marker='x')
    plt.scatter(data_noise[:, 0], locs_noise[:, 0], c="gray", s=(data_noise[:, 1]+marker_size)*10, marker="x")
plt.ylim([0,100])
plt.xlim(left=-3)
plt.ylabel("Distance (km)")
plt.xlabel("Time (s)")
# plt.tight_layout()
plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='left', verticalalignment='top',
        transform=plt.gca().transAxes, fontsize="medium", fontweight="normal", bbox=box)

######
plt.subplot(133)
for i in range(num_true_event):
    if not use_amplitude:
        plt.plot(event_t0[i, 0], event_loc[i, 0], "P", c=f"C{i}", label=f"Earthquake #{i+1}", markersize=8)
    else:
        plt.plot(event_t0[i, 0], event_loc[i, 0], "P", c=f"C{i}", label=f"Earthquake #{i+1}", markersize=event_mag[i]*3)
if not use_amplitude:
    plt.scatter(data[:, 0], locs[:, 0], c=[f"C{label[i]}" for i in range(len(label))], s=12, marker='x')#, label=f"P/S-phases #{i+1}")
    plt.scatter(data_noise[:, 0], locs_noise[:, 0], c="gray", s=12, marker="x")#, label=f"False Positives")
else:
    plt.scatter(data[:, 0], locs[:, 0], c=[f"C{label[i]}" for i in range(len(label))], s=(data[:, 1]+marker_size)*10, marker='x')#, label=f"P/S-phases #{i+1}")
    plt.scatter(data_noise[:, 0], locs_noise[:, 0], c="gray", s=(data_noise[:, 1]+marker_size)*10, marker="x")#, label=f"False Positives")
plt.legend(bbox_to_anchor=(1, 0.8), loc="upper right", fontsize="x-small")
plt.ylim([0,100])
plt.xlim(left=-3)
plt.gca().yaxis.set_ticklabels([])
plt.xlabel("Time (s)")
plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='left', verticalalignment='top',
        transform=plt.gca().transAxes, fontsize="medium", fontweight="normal", bbox=box)

locs = np.vstack([locs, locs_noise])
data = np.vstack([data, data_noise])
phase_type = np.hstack([phase_type, phase_type_noise])
delta_t = data[:,0].max() - data[:,0].min()
delta_amp = data[:,1].max() - data[:,1].min()
if not use_amplitude:
    data = data[:,0:1]

#################### GMMA ##############################
num_event = int(len(data)/num_station * 4)
centers_init = np.vstack([np.ones(num_event)*np.mean(station_loc[:,0]),
                        #   np.ones(num_event)*np.mean(station_loc[:,1]),
                         np.linspace(data[:,0].min(), data[:,0].max(), num_event)]).T # n_eve, n_dim(x, y, z) + 1(t)


# Fit a Gaussian mixture with EM 
dummy_prob = 1/((2*np.pi)**(data.shape[-1]/2) * 2)
# dummy_prob = 1/30
# print(f"dummy_prob = {dummy_prob}")

t_start = time.time()
mean_precision_prior = 0.1/delta_t
if not use_amplitude:
    covariance_prior = np.array([[1]])
else:
    covariance_prior = np.array([[1,0],[0,1]]) 
print(f"time range: {delta_t:.3f}, amplitude range: {delta_amp:.3f}, mean precision prior: {mean_precision_prior:.3f}")

gmm = mixture.BayesianGaussianMixture(n_components=num_event, 
                                      station_locs=locs, 
                                      phase_type=phase_type,
                                      weight_concentration_prior = 10000/num_event,
                                      mean_precision_prior = mean_precision_prior,
                                      covariance_prior = covariance_prior,
                                      # covariance_type='full', 
                                      # reg_covar=0.1, 
                                      init_params="centers",
                                      centers_init=centers_init.copy(),
                                      # dummy_comp=False, 
                                      # dummy_prob=dummy_prob,
                                      loss_type="l1",
                                    #   max_covar=30.0,
                                    #   reg_covar=0.1,
                                      ).fit(data)

# gmm = mixture.GaussianMixture(n_components=num_event,
#                               station_locs=locs, 
#                               phase_type=phase_type, 
#                               # covariance_type='full', 
#                               # reg_covar=0.1, 
#                               init_params="centers",
#                               centers_init=centers_init.copy(), 
#                               # dummy_comp=False, 
#                               # dummy_prob=dummy_prob,
#                               loss_type="l1"
#                               ).fit(data)

t_end = time.time()
print(f"GMMA time = {t_end - t_start}")

pred = gmm.predict(data) 
prob = gmm.predict_proba(data)
score = gmm.score_samples(data)
prob_eq = prob.mean(axis=0)
std_eq = gmm.covariances_[:,0,0]

min_picks = 10
filtered_idx = np.array([True if len(data[pred==i, 0]) >= min_picks else False for i in range(len(prob_eq))]) \
                & (std_eq < 10) #& (prob_eq > 1/num_event)
filtered_idx = np.arange(len(prob_eq))[filtered_idx]

plt.subplot(132)
colors = defaultdict(int)
idx = np.argsort(gmm.centers_[:, 1])
dum = 0
for i in idx:
    if i in filtered_idx:
    # if len(pred[pred==i]) > min_picks:
        if not use_amplitude:
            plt.scatter(data[(pred==i), 0], locs[(pred==i), 0], c=f"C{dum}", s=12, marker='x')
            plt.plot(gmm.centers_[i, 1], gmm.centers_[i, 0], "P", c=f"C{dum}", markersize=8, label=f"Associated #{dum+1}")
        else:
            plt.scatter(data[(pred==i), 0], locs[(pred==i), 0], c=f"C{dum}", s=(data[(pred==i), 1]+marker_size)*10, marker='x')
            plt.plot(gmm.centers_[i, 1], gmm.centers_[i, 0], "P", c=f"C{dum}", markersize=gmm.centers_[i, 2]*3, label=f"Associated #{dum+1}")
        print(prob_eq[i], std_eq[i])
        dum += 1

for i in idx:
    if i not in filtered_idx:
    # if len(pred[pred==i]) <= min_picks:
        if not use_amplitude:
            plt.scatter(data[(pred==i), 0], locs[(pred==i), 0],  c="gray", s=12, marker='x')
        else:
            plt.scatter(data[(pred==i), 0], locs[(pred==i), 0],  c="gray", s=(data[(pred==i), 1]+marker_size)*10, marker='x')
        dum += 1

plt.legend(bbox_to_anchor=(1, 0.8), loc="upper right", fontsize="x-small")
plt.tight_layout()
plt.ylim([0,100])
plt.xlim(left=-3)
# plt.ylabel("X (km)")
plt.gca().yaxis.set_ticklabels([])
plt.xlabel("Time (s)") 
plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='left', verticalalignment='top',
        transform=plt.gca().transAxes, fontsize="medium", fontweight="normal", bbox=box)
plt.savefig(os.path.join(figure_dir, f"result_eq{num_true_event:02d}_err{phase_err:01.1f}_fp{phase_fp:.1f}_amp{use_amplitude:d}.png"), bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(figure_dir, f"result_eq{num_true_event:02d}_err{phase_err:01.1f}_fp{phase_fp:01.1f}_amp{use_amplitude:d}.pdf"), bbox_inches="tight")
plt.show()