import time
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.sparse
from numba import njit

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
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        if err < 1e-6:
            break
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
@njit
def _bilinear(tt, iph, r, z, r0, z0, h):
    """Time and its exact derivatives (dt/dr, dt/dz) of the bilinear interpolant of tables tt (phase, nr, nz), so the
    optimizer sees a consistent pair (as in ADLoc). Beyond the grid the edge cell's bilinear form continues, so the
    gradient never vanishes there."""
    n = r.shape[0]
    nr, nz = tt.shape[1], tt.shape[2]
    t, tr, tz = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        p = iph[k]
        x, y = (r[k] - r0) / h, (z[k] - z0) / h
        i = min(max(int(np.floor(x)), 0), nr - 2)
        j = min(max(int(np.floor(y)), 0), nz - 2)
        fx, fy = x - i, y - j
        t00, t10, t01, t11 = tt[p, i, j], tt[p, i + 1, j], tt[p, i, j + 1], tt[p, i + 1, j + 1]
        c = t11 - t10 - t01 + t00
        t[k] = t00 + (t10 - t00) * fx + (t01 - t00) * fy + c * fx * fy
        tr[k] = ((t10 - t00) + c * fy) / h
        tz[k] = ((t01 - t00) + c * fx) / h
    return t, tr, tz


def traveltime_grad(event_loc, station_loc, phase_type, vel={"p": 6.0, "s": 6.0 / 1.75}, eikonal=None):
    """Travel times (n,) and their gradients w.r.t. the event location (n, ndim) for paired (or broadcast) event and
    station locations (n, ndim): straight rays at vel, or the eikonal tables (3-D locations only)."""
    d = event_loc - station_loc  # (n, ndim)
    is_s = np.asarray(phase_type) == "s"
    if eikonal is None:
        v = np.where(is_s, vel["s"], vel["p"])
        dist = np.linalg.norm(d, axis=-1)
        return dist / v, d / (dist + 1e-6)[:, None] / v[:, None]
    r = np.hypot(d[:, 0], d[:, 1])
    t, tr, tz = _bilinear(eikonal["tt"], is_s.astype(np.int64), r, d[:, 2], eikonal["rgrid"][0], eikonal["zgrid"][0], eikonal["h"])
    return t, np.column_stack([tr * d[:, 0] / (r + 1e-6), tr * d[:, 1] / (r + 1e-6), tz])


def traveltime(event_loc, station_loc, phase_type, eikonal):
    return traveltime_grad(event_loc[:, :3], station_loc, phase_type, eikonal=eikonal)[0][:, np.newaxis]


def grad_traveltime(event_loc, station_loc, phase_type, eikonal):
    return traveltime_grad(event_loc[:, :3], station_loc, phase_type, eikonal=eikonal)[1]


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
    dist = np.linalg.norm(event_loc[..., :-1] - station_loc, axis=-1, keepdims=True)
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
        options={"maxiter": max_iter, "gtol": convergence},
    )

    return opt.x[np.newaxis, :], opt.fun


def calc_loc_batch(
    phase_time,
    phase_type,
    station_loc,
    weight,
    centers,
    vel={"p": 6.0, "s": 6.0 / 1.75},
    eikonal=None,
    bounds=None,
    sigma=1.0,
    max_iter=50,
    tol=1e-3,
    min_weight=1e-4,
):
    """calc_loc for all K components at once: the same Huber loss, minimized by projected Levenberg-Marquardt on the
    IRLS Gauss-Newton system, warm-started at centers. A step is kept only if it lowers that component's loss, and a
    coordinate at its bound with the gradient pushing outward is held fixed (an active set, as in L-BFGS-B).

    phase_time (N, 1), station_loc (N, D), weight (N, K) = responsibility x pick weight, centers (K, D + 1) as
    (location, origin time). Only pairs with weight > min_weight enter, so the cost is O(N), not O(N K)."""
    K, D1 = centers.shape
    nn, kk = np.nonzero(weight > min_weight)  # active (pick, component) pairs
    w, t_obs, sta, ph = weight[nn, kk], phase_time[nn, 0], station_loc[nn], np.asarray(phase_type)[nn]
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds]) if bounds is not None else np.full(D1, -np.inf)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds]) if bounds is not None else np.full(D1, np.inf)
    eye = np.eye(D1)

    def huber(x):  # loss (K,), residuals (P,), Jacobian (P, D + 1)
        t, g = traveltime_grad(x[kk, :-1], sta, ph, vel, eikonal)
        r = t + x[kk, -1] - t_obs
        a = np.abs(r)
        loss = np.bincount(kk, w * np.where(a <= sigma, 0.5 * r**2, sigma * a - 0.5 * sigma**2), minlength=K)
        return loss, r, np.column_stack([g, np.ones(len(r))])

    S = scipy.sparse.csr_matrix((np.ones(len(kk)), (kk, np.arange(len(kk)))), shape=(K, len(kk)))  # component <- pair

    def per_component(v):  # sum over the pairs of each component: (P, ...) -> (K, ...)
        return (S @ v.reshape(len(v), -1)).reshape(K, *v.shape[1:])

    x = np.clip(centers, lo, hi)
    loss, r, J = huber(x)
    lam = np.full(K, 1e-2)
    for _ in range(max_iter):
        omega = w * np.minimum(1.0, sigma / np.maximum(np.abs(r), 1e-12))  # IRLS weights of the Huber loss
        H = per_component(omega[:, None, None] * J[:, :, None] * J[:, None, :])  # (K, D1, D1) Gauss-Newton Hessian
        g = per_component((omega * r)[:, None] * J)  # (K, D1) gradient
        fixed = ((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0))
        g = np.where(fixed, 0.0, g)
        H = np.where(fixed[:, :, None] | fixed[:, None, :], 0.0, H) + fixed[:, :, None] * eye
        H = H + (lam[:, None] * np.einsum("kii->ki", H) + 1e-9)[:, :, None] * eye  # Marquardt damping
        step = -np.linalg.solve(H, g[..., None])[..., 0]
        x_new = np.clip(x + step, lo, hi)
        loss_new = huber(x_new)[0]
        ok = loss_new <= loss
        gain = (loss - loss_new) / np.maximum(loss, 1e-12)  # relative decrease of an accepted step
        x = np.where(ok[:, None], x_new, x)
        lam = np.where(ok, lam / 3, lam * 4)
        # converged: the proposed step is below tol (accepted or not), damping has made the step negligible, or the
        # loss no longer decreases (a flat direction of a poorly constrained component; scipy's ftol)
        if np.all((np.abs(step).max(axis=1) < tol) | (lam > 1e6) | (ok & (gain < 1e-7))):
            break
        loss, r, J = huber(x)
    return x, huber(x)[0]


def calc_mag_batch(amp, event_loc, station_loc, weight, min=-2, max=8):
    """calc_mag for all K components: amp (N, 1), event_loc (K, D), station_loc (N, D), weight (N, K) -> (K,)"""
    dist = np.linalg.norm(event_loc[:, None, :] - station_loc[None, :, :], axis=-1)  # (K, N)
    c0, c1, c3 = 1.08, 0.93, -1.68  # Picozzi et al. (2018), as in calc_mag
    mag_ = (amp[:, 0][None, :] - c0 - c3 * np.log10(np.maximum(dist, 0.1))) / c1 + 3.5
    w = weight.T
    mu = (mag_ * w).sum(1, keepdims=True) / (w.sum(1, keepdims=True) + 1e-6)
    std = np.sqrt(((mag_ - mu) ** 2 * w).sum(1, keepdims=True) / (w.sum(1, keepdims=True) + 1e-12))
    w = w * (np.abs(mag_ - mu) <= 2 * std)
    return np.clip((mag_ * w).sum(1) / (w.sum(1) + 1e-6), min, max)


def initialize_eikonal(config):
    path = Path("./eikonal")
    path.mkdir(exist_ok=True)
    if "rmax_km" in config:  # the largest event-station distance needed
        rlim = [0, config["rmax_km"]]
    else:  # the region's diagonal
        xlim, ylim = config["xlim_km"], config["ylim_km"]
        rlim = [0, np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2)]
    zlim = config["zlim_km"]
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
    zz, vp, vs = vel["Z"], vel["P"], vel["S"]  # as ADLoc
    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
    vp = np.ones((nr, nz)) * vp1d
    vs = np.ones((nr, nz)) * vs1d

    t0 = time.time()
    ir0 = np.around((0 - rlim[0]) / h).astype(np.int64)
    iz0 = np.around((0 - zlim[0]) / h).astype(np.int64)
    up = 1000.0 * np.ones((nr, nz))
    up[ir0, iz0] = 0.0
    up = eikonal_solve(up, vp, h)

    us = 1000.0 * np.ones((nr, nz))
    us[ir0, iz0] = 0.0
    us = eikonal_solve(us, vs, h)

    config.update(
        {
            "tt": np.stack([up, us]),  # (phase, nr, nz), read by traveltime_grad
            "up": up.flatten(),
            "us": us.flatten(),
            "rgrid": rgrid,
            "zgrid": zgrid,
            "nr": nr,
            "nz": nz,
            "h": h,
        }
    )
    print(f"Eikonal: P and S tables of {nr} x {nz} nodes in {time.time() - t0:.1f} s")

    return config


def initialize_centers(
    X,
    phase_type,
    phase_weight,
    centers_init,
    station_locs,
    sigma=1.0,
    random_state=None,
    vel={"p": 6.0, "s": 6.0 / 1.75},
    eikonal=None,
):
    n_samples, n_features = X.shape
    n_components, _ = centers_init.shape
    centers = centers_init.copy()

    means = np.zeros([n_components, n_samples, n_features])
    for i in range(n_components):
        if n_features == 1:  # (time,)
            means[i, :, :] = calc_time(centers_init[i : i + 1, :], station_locs, phase_type, vel=vel, eikonal=eikonal)
        elif n_features == 2:  # (time, amp)
            means[i, :, 0:1] = calc_time(centers_init[i : i + 1, :-1], station_locs, phase_type, vel=vel, eikonal=eikonal)
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
    resp = 1 / sigma * np.exp(-(dist**2) / (2 * sigma**2)) * phase_weight[:, np.newaxis]
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp_sum[resp_sum == 0] = 1.0
    resp = resp / resp_sum

    # dist = np.linalg.norm(means - X, axis=-1) # (n_components, n_samples, n_features) -> (n_components, n_samples)
    # resp = np.exp(-dist/np.median(dist, axis=0, keepdims=True)).T
    # resp /= np.sum(resp, axis=1, keepdims=True) # (n_components, n_samples)

    if n_features == 2:
        for i in range(n_components):
            centers[i, -1:] = calc_mag(
                X[:, 1:2], centers_init[i : i + 1, :-1], station_locs, resp[:, i : i + 1] * phase_weight[:, np.newaxis]
            )

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
