import itertools
import time
from pathlib import Path

import numpy as np
import scipy.optimize
from numba import njit
from numba.typed import List

# import shelve

###################################### Eikonal Solver ######################################
# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min([a, b]) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


@njit
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    # for i, j in itertools.product(I, J):
    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = min([u[i - 1, j], u[i + 1, j]])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = min([u[i, j - 1], u[i, j + 1]])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = min([u_new, u[i, j]])

    return u


@njit
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    # I = list(range(m))
    # I = List()
    # [I.append(i) for i in range(m)]
    I = np.arange(m)
    iI = I[::-1]
    # J = list(range(n))
    # J = List()
    # [J.append(j) for j in range(n)]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, f, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iter {i}, error = {err:.3f}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
@njit
def _get_index(ir, iz, nr, nz, order="C"):
    assert np.all(ir >= 0) and np.all(ir < nr)
    assert np.all(iz >= 0) and np.all(iz < nz)
    if order == "C":
        return ir * nz + iz
    elif order == "F":
        return iz * nr + ir
    else:
        raise ValueError("order must be either C or F")


def test_get_index():
    vr, vz = np.meshgrid(np.arange(10), np.arange(20), indexing="ij")
    vr = vr.flatten()
    vz = vz.flatten()
    nr = 10
    nz = 20
    for ir in range(nr):
        for iz in range(nz):
            assert vr[_get_index(ir, iz, nr, nz)] == ir
            assert vz[_get_index(ir, iz, nr, nz)] == iz


@njit
def _interp(time_table, r, z, rgrid0, zgrid0, nr, nz, h):
    ir0 = np.floor((r - rgrid0) / h).clip(0, nr - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid0) / h).clip(0, nz - 2).astype(np.int64)
    r = r.clip(rgrid0, rgrid0 + (nr - 1) * h)
    z = z.clip(zgrid0, zgrid0 + (nz - 1) * h)
    # ir0 = ((r - rgrid0) / h).astype(np.int64)
    # iz0 = ((z - zgrid0) / h).astype(np.int64)
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid0
    x2 = ir1 * h + rgrid0
    z1 = iz0 * h + zgrid0
    z2 = iz1 * h + zgrid0

    Q11 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q12 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q21 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q22 = time_table[_get_index(ir1, iz1, nr, nz)]

    t = (
        1
        / (x2 - x1)
        / (z2 - z1)
        * (
            Q11 * (x2 - r) * (z2 - z)
            + Q21 * (r - x1) * (z2 - z)
            + Q12 * (x2 - r) * (z - z1)
            + Q22 * (r - x1) * (z - z1)
        )
    )

    return t


def traveltime(event_loc, station_loc, phase_type, eikonal):
    r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    z = event_loc[:, 2] - station_loc[:, 2]

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    p_index = phase_type == "p"
    s_index = phase_type == "s"
    tt = np.zeros(len(phase_type), dtype=np.float32)
    tt[phase_type == "p"] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    tt[phase_type == "s"] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    tt = tt[:, np.newaxis]

    return tt


def grad_traveltime(event_loc, station_loc, phase_type, eikonal):
    r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    z = event_loc[:, 2] - station_loc[:, 2]

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    p_index = phase_type == "p"
    s_index = phase_type == "s"
    dt_dr = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))
    dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    dr_dxy = (event_loc[:, :2] - station_loc[:, :2]) / (r[:, np.newaxis] + 1e-6)
    dt_dxy = dt_dr[:, np.newaxis] * dr_dxy

    grad = np.column_stack((dt_dxy, dt_dz[:, np.newaxis]))

    return grad


############################################# Seismic Ops for GaMMA #####################################################################


def calc_time(event_loc, station_loc, phase_type, vel={"p": 6.0, "s": 6.0 / 1.75}, eikonal=None, **kwargs):
    ev_loc = event_loc[:, :-1]
    ev_t = event_loc[:, -1:]

    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])[:, np.newaxis]
        tt = np.linalg.norm(ev_loc - station_loc, axis=-1, keepdims=True) / v + ev_t
    else:
        tt = traveltime(event_loc, station_loc, phase_type, eikonal) + ev_t
    return tt


def calc_mag(data, event_loc, station_loc, weight, min=-2, max=8):
    dist = np.linalg.norm(event_loc[:, :-1] - station_loc, axis=-1, keepdims=True)
    # mag_ = ( data - 2.48 + 2.76 * np.log10(dist) )
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (data - c0 - c3 * np.log10(np.maximum(dist, 0.1))) / c1 + 3.5
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # mag_ = (data - c0 - c3*np.log10(dist))/c1
    # mag = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    # (Watanabe, 1971) https://www.jstage.jst.go.jp/article/zisin1948/24/3/24_3_189/_pdf/-char/ja
    # mag_ = 1.0/0.85 * (data + 1.73 * np.log10(np.maximum(dist, 0.1)) + 2.50)
    mu = np.sum(mag_ * weight) / (np.sum(weight) + 1e-6)
    std = np.sqrt(np.sum((mag_ - mu) ** 2 * weight) / (np.sum(weight) + 1e-12))
    mask = np.abs(mag_ - mu) <= 2 * std
    mag = np.sum(mag_[mask] * weight[mask]) / (np.sum(weight[mask]) + 1e-6)
    mag = np.clip(mag, min, max)
    return mag


def calc_amp(mag, event_loc, station_loc):
    dist = np.linalg.norm(event_loc[:, :-1] - station_loc, axis=-1, keepdims=True)
    # logA = mag + 2.48 - 2.76 * np.log10(dist)
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    logA = c0 + c1 * (mag - 3.5) + c3 * np.log10(np.maximum(dist, 0.1))
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # logA = c0 + c1*mag + c3*np.log10(dist)
    # (Watanabe, 1971) https://www.jstage.jst.go.jp/article/zisin1948/24/3/24_3_189/_pdf/-char/ja
    # logA = 0.85 * mag - 2.50 - 1.73 * np.log10(np.maximum(dist, 0.1))
    return logA


################################################ Earthquake Location ################################################


def huber_loss_grad(
    event_loc, phase_time, phase_type, station_loc, weight, vel={"p": 6.0, "s": 6.0 / 1.75}, sigma=1, eikonal=None
):
    event_loc = event_loc[np.newaxis, :]
    predict_time = calc_time(event_loc, station_loc, phase_type, vel, eikonal)
    t_diff = predict_time - phase_time

    l1 = np.squeeze((np.abs(t_diff) > sigma))
    l2 = np.squeeze((np.abs(t_diff) <= sigma))

    # loss
    loss = np.sum((sigma * np.abs(t_diff[l1]) - 0.5 * sigma**2) * weight[l1]) + np.sum(
        0.5 * t_diff[l2] ** 2 * weight[l2]
    )
    J = np.zeros([phase_time.shape[0], event_loc.shape[1]])

    # gradient
    if eikonal is None:
        v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
        dist = np.linalg.norm(event_loc[:, :-1] - station_loc, axis=-1, keepdims=True)
        J[:, :-1] = (event_loc[:, :-1] - station_loc) / (dist + 1e-6) / v
    else:
        grad = grad_traveltime(event_loc, station_loc, phase_type, eikonal)
        J[:, :-1] = grad
    J[:, -1] = 1

    J_ = np.sum(sigma * np.sign(t_diff[l1]) * J[l1] * weight[l1], axis=0, keepdims=True) + np.sum(
        t_diff[l2] * J[l2] * weight[l2], axis=0, keepdims=True
    )

    return loss, J_


def calc_loc(
    phase_time,
    phase_type,
    station_loc,
    weight,
    event_loc0,
    eikonal=None,
    vel={"p": 6.0, "s": 6.0 / 1.75},
    bounds=None,
    max_iter=100,
    convergence=1e-6,
):

    opt = scipy.optimize.minimize(
        huber_loss_grad,
        np.squeeze(event_loc0),
        method="L-BFGS-B",
        jac=True,
        args=(phase_time, phase_type, station_loc, weight, vel, 1, eikonal),
        bounds=bounds,
        options={"maxiter": max_iter, "gtol": convergence, "iprint": -1},
    )

    return opt.x[np.newaxis, :], opt.fun


def initialize_eikonal(config):
    path = Path("./eikonal")
    path.mkdir(exist_ok=True)
    rlim = [0, np.sqrt((config["xlim"][1] - config["xlim"][0]) ** 2 + (config["ylim"][1] - config["ylim"][0]) ** 2)]
    zlim = config["zlim"]
    h = config["h"]

    # filename = f"timetable_{rlim[0]:.0f}_{rlim[1]:.0f}_{zlim[0]:.0f}_{zlim[1]:.0f}_{h:.3f}"
    # if (path / (filename + ".dir")).is_file():
    #     print("Loading precomputed timetable...")
    #     with shelve.open(str(path / filename)) as db:
    #         up = db["up"]
    #         us = db["us"]
    #         grad_up = db["grad_up"]
    #         grad_us = db["grad_us"]
    #         rgrid = db["rgrid"]
    #         zgrid = db["zgrid"]
    #         nr = db["nr"]
    #         nz = db["nz"]
    #         h = db["h"]
    # else:

    rgrid = np.arange(rlim[0], rlim[1], h)
    zgrid = np.arange(zlim[0], zlim[1], h)
    nr, nz = len(rgrid), len(zgrid)

    vel = config["vel"]
    zz, vp, vs = vel["z"], vel["p"], vel["s"]
    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
    vp = np.ones((nr, nz)) * vp1d
    vs = np.ones((nr, nz)) * vs1d

    ir0 = np.around((0 - rlim[0]) / h).astype(np.int64)
    iz0 = np.around((0 - zlim[0]) / h).astype(np.int64)
    up = 1000.0 * np.ones((nr, nz))
    up[ir0, iz0] = 0.0
    up = eikonal_solve(up, vp, h)

    grad_up = np.gradient(up, h)

    us = 1000.0 * np.ones((nr, nz))
    us[ir0, iz0] = 0.0
    us = eikonal_solve(us, vs, h)

    grad_us = np.gradient(us, h)

    # with shelve.open(str(path / filename)) as db:
    #     db["up"] = up
    #     db["us"] = us
    #     db["grad_up"] = grad_up
    #     db["grad_us"] = grad_us
    #     db["rgrid"] = rgrid
    #     db["zgrid"] = zgrid
    #     db["nr"] = nr
    #     db["nz"] = nz
    #     db["h"] = h

    up = up.flatten()
    us = us.flatten()
    grad_up = np.array([grad_up[0].flatten(), grad_up[1].flatten()])
    grad_us = np.array([grad_us[0].flatten(), grad_us[1].flatten()])
    config.update(
        {
            "up": up,
            "us": us,
            "grad_up": grad_up,
            "grad_us": grad_us,
            "rgrid": rgrid,
            "zgrid": zgrid,
            "nr": nr,
            "nz": nz,
            "h": h,
        }
    )

    return config


def initialize_centers(X, phase_type, centers_init, station_locs, random_state):
    n_samples, n_features = X.shape
    n_components, _ = centers_init.shape
    centers = centers_init.copy()

    means = np.zeros([n_components, n_samples, n_features])
    for i in range(n_components):
        if n_features == 1:  # (time,)
            means[i, :, :] = calc_time(centers_init[i : i + 1, :], station_locs, phase_type)
        elif n_features == 2:  # (time, amp)
            means[i, :, 0:1] = calc_time(centers_init[i : i + 1, :-1], station_locs, phase_type)
            means[i, :, 1:2] = X[:, 1:2]
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

    dist = np.linalg.norm(means - X, axis=-1).T  # (n_components, n_samples, n_features) -> (n_samples, n_components)
    resp = np.exp(-dist)
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp_sum[resp_sum == 0] = 1.0
    resp = resp / resp_sum

    # dist = np.linalg.norm(means - X, axis=-1) # (n_components, n_samples, n_features) -> (n_components, n_samples)
    # resp = np.exp(-dist/np.median(dist, axis=0, keepdims=True)).T
    # resp /= np.sum(resp, axis=1, keepdims=True) # (n_components, n_samples)

    if n_features == 2:
        for i in range(n_components):
            centers[i, -1:] = calc_mag(X[:, 1:2], centers_init[i : i + 1, :-1], station_locs, resp[:, i : i + 1])

    return resp, centers, means


#########################################################################################################################
## L2 norm
def diff_and_grad(vars, data, station_locs, phase_type, vel={"p": 6.0, "s": 6.0 / 1.75}):
    """
    data: (n_sample, t)
    """
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    # loc, t = vars[:,:-1], vars[:,-1:]
    dist = np.sqrt(np.sum((station_locs - vars[:, :-1]) ** 2, axis=1, keepdims=True))
    y = dist / v - (data - vars[:, -1:])
    J = np.zeros([data.shape[0], vars.shape[1]])
    J[:, :-1] = (vars[:, :-1] - station_locs) / (dist + 1e-6) / v
    J[:, -1] = 1
    return y, J


def newton_method(
    vars, data, station_locs, phase_type, weight, max_iter=20, convergence=1, vel={"p": 6.0, "s": 6.0 / 1.75}
):
    for i in range(max_iter):
        prev = vars.copy()
        y, J = diff_and_grad(vars, data, station_locs, phase_type, vel=vel)
        JTJ = np.dot(J.T, weight * J)
        I = np.zeros_like(JTJ)
        np.fill_diagonal(I, 1e-3)
        vars -= np.dot(np.linalg.inv(JTJ + I), np.dot(J.T, y * weight)).T
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
