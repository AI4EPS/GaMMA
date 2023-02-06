import numpy as np
from scipy import linalg, optimize

def calc_time(center, station_locs, phase_type, vel={"p":6.0, "s":6.0/1.75}):
    """
    center: (loc, t)
    """
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    t = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True) / v + center[:,-1:]
    return t

def calc_mag(data, center, station_locs, weight, min=-2, max=8):
    """
    center: (loc, t)
    data: (n_sample, amp)
    """
    dist = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True)
    # mag_ = ( data - 2.48 + 2.76 * np.log10(dist) )
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (data - c0 - c3*np.log10(np.maximum(dist, 0.1)))/c1 + 3.5
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # mag_ = (data - c0 - c3*np.log10(dist))/c1
    #mag = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    mu = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    std = np.sqrt(np.sum((mag_-mu)**2 * weight) / (np.sum(weight)+1e-12))
    mask = (np.abs(mag_ - mu) <= 2*std)
    mag = np.sum(mag_[mask] * weight[mask]) / (np.sum(weight[mask])+1e-6)
    mag = np.clip(mag, min, max)
    return mag

def calc_amp(mag, center, station_locs):
    """
    center: (loc, t)
    """
    dist = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True)
    # logA = mag + 2.48 - 2.76 * np.log10(dist)
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    logA = c0 + c1*(mag-3.5) + c3*np.log10(np.maximum(dist, 0.1))
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # logA = c0 + c1*mag + c3*np.log10(dist)
    return logA


## L2 norm
def diff_and_grad(vars, data, station_locs, phase_type, vel={"p":6.0, "s":6.0/1.75}):
    """
    data: (n_sample, t)
    """
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    # loc, t = vars[:,:-1], vars[:,-1:]
    dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
    y = dist/v - (data - vars[:,-1:])
    J = np.zeros([data.shape[0], vars.shape[1]])
    J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
    J[:, -1] = 1
    return y, J

def newton_method(vars, data, station_locs, phase_type, weight, max_iter=20, convergence=1, vel={"p":6.0, "s":6.0/1.75}):
    for i in range(max_iter): 
        prev = vars.copy()
        y, J = diff_and_grad(vars, data, station_locs, phase_type, vel=vel)
        JTJ = np.dot(J.T, weight * J)
        I = np.zeros_like(JTJ)
        np.fill_diagonal(I, 1e-3)
        vars -= np.dot(np.linalg.inv(JTJ + I) , np.dot(J.T, y * weight)).T
        if (np.sum(np.abs(vars - prev))) < convergence:
            return vars
    return vars

## l1 norm
# def loss_and_grad(vars, data, station_locs, phase_type, weight, vel={"p":6.0, "s":6.0/1.75}):
    
#     v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
#     vars = vars[np.newaxis, :]
#     dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
#     J = np.zeros([data.shape[0], vars.shape[1]])
#     J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
#     J[:, -1] = 1

#     loss = np.sum(np.abs(dist/v - (data[:,-1:] - vars[:,-1:])) * weight)
#     J = np.sum(np.sign(dist/v - (data[:,-1:] - vars[:,-1:])) * weight * J, axis=0, keepdims=True)

#     return loss, J

## Huber loss
def loss_and_grad(vars, data, station_locs, phase_type, weight, sigma=1, vel={"p":6.0, "s":6.0/1.75}):

    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    vars = vars[np.newaxis, :]
    dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
    J = np.zeros([data.shape[0], vars.shape[1]])
    J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
    J[:, -1] = 1
    
    y = dist/v - (data[:,-1:] - vars[:,-1:])
    std = np.sqrt(np.sum(y**2 * weight) / (np.sum(weight)+1e-12))
    # mask = (np.abs(y) <= 2*std)
    mask = True
    l1 = np.squeeze((np.abs(y) > sigma) & mask)
    l2 = np.squeeze((np.abs(y) <= sigma) & mask)

    loss = np.sum( (sigma*np.abs(y[l1]) - 0.5*sigma**2) * weight[l1] ) \
           + np.sum( 0.5*y[l2]**2 * weight[l2] )
    J_ = np.sum( sigma*np.sign(y[l1]) * J[l1] * weight[l1], axis=0, keepdims=True ) \
        + np.sum( y[l2] * J[l2] * weight[l2], axis=0, keepdims=True )  

    return loss, J_


def l1_bfgs(vars, data, station_locs, phase_type, weight, max_iter=10, convergence=1e-3, bounds=None, vel={"p":6.0, "s":6.0/1.75}): 

    opt = optimize.minimize(loss_and_grad, np.squeeze(vars), method="L-BFGS-B", jac=True,
                            args=(data, station_locs, phase_type, weight, 1, vel),
                            options={"maxiter": max_iter, "gtol": convergence, "iprint": -1},
                            bounds=bounds)

    return opt.x[np.newaxis, :]


def initialize_centers(X, phase_type, centers_init, station_locs, random_state):

    n_samples, n_features = X.shape
    n_components, _ = centers_init.shape
    centers = centers_init.copy()

    means = np.zeros([n_components, n_samples, n_features])
    for i in range(n_components):
        if n_features == 1: #(time,)
            means[i, :, :] = calc_time(centers_init[i:i+1, :], station_locs, phase_type)
        elif n_features == 2: #(time, amp)
            means[i, :, 0:1] = calc_time(centers_init[i:i+1, :-1], station_locs, phase_type)
            means[i, :, 1:2] = X[:,1:2]
            # means[i, :, 1:2] = calc_amp(self.centers_init[i, -1:], self.centers_init[i:i+1, :-1], self.station_locs)
        else:
            raise ValueError(f"n_features = {n_features} > 2!")

    ## performance is not good
    # resp = np.zeros((n_samples, self.n_components))
    # dist = np.sum(np.abs(means - X), axis=-1).T # (n_components, n_samples, n_features) -> (n_samples, n_components)
    # resp[np.arange(n_samples), np.argmax(dist, axis=1)] = 1.0

    ## performance is ok
    # sigma = np.array([1.0,1.0])
    # prob = np.sum(1.0/sigma * np.exp( - (means - X) ** 2 / (2 * sigma**2)), axis=-1).T # (n_components, n_samples, n_features) -> (n_samples, n_components)
    # prob_sum = np.sum(prob, axis=1, keepdims=True)
    # prob_sum[prob_sum == 0] = 1.0
    # resp = prob / prob_sum

    dist = np.linalg.norm(means - X, axis=-1).T # (n_components, n_samples, n_features) -> (n_samples, n_components)
    resp = np.exp(-dist)
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp_sum[resp_sum == 0] = 1.0
    resp = resp / resp_sum

    if n_features == 2:
        for i in range(n_components): 
            centers[i, -1:] = calc_mag(X[:,1:2], centers_init[i:i+1,:-1], station_locs, resp[:, i:i+1])

    return resp, centers, means