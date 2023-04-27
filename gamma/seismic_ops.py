import itertools
import numpy as np
import scipy.optimize
try:
    import torch
    import torch.nn.functional as F
    import torch.optim
except:
    pass


###################################### Eikonal Solver ######################################
# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min(a, b) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    for i, j in itertools.product(I, J):
        if i == 0:
            uxmin = u[i + 1, j]
        elif i == m - 1:
            uxmin = u[i - 1, j]
        else:
            uxmin = np.min([u[i - 1, j], u[i + 1, j]])

        if j == 0:
            uymin = u[i, j + 1]
        elif j == n - 1:
            uymin = u[i, j - 1]
        else:
            uymin = np.min([u[i, j - 1], u[i, j + 1]])

        u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

        u[i, j] = np.min([u_new, u[i, j]])

    return u


def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = list(range(m))
    iI = I[::-1]
    J = list(range(n))
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, f, h):
    print("Eikonal Solver: ")
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"iteration {i}, error = {err}")
        if err < 1e-6:
            break

    return u


###################################### Traveltime based on Eikonal Timetable ######################################


def _interp(time_table, r, z, rgrid, zgrid, h):
    rgrid00 = rgrid[0, 0]
    zgrid00 = zgrid[0, 0]

    ir0 = (r - rgrid00).div(h, rounding_mode='floor').clamp(0, rgrid.shape[0] - 2).long()
    iz0 = (z - zgrid00).div(h, rounding_mode='floor').clamp(0, zgrid.shape[1] - 2).long()
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid00
    x2 = ir1 * h + rgrid00
    y1 = iz0 * h + zgrid00
    y2 = iz1 * h + zgrid00

    Q11 = time_table[ir0, iz0]
    Q12 = time_table[ir0, iz1]
    Q21 = time_table[ir1, iz0]
    Q22 = time_table[ir1, iz1]

    t = (
        1
        / (x2 - x1)
        / (y2 - y1)
        * (
            Q11 * (x2 - r) * (y2 - z)
            + Q21 * (r - x1) * (y2 - z)
            + Q12 * (x2 - r) * (z - y1)
            + Q22 * (r - x1) * (z - y1)
        )
    )

    return t


def traveltime(event_loc, station_loc, time_table, rgrid, zgrid, h, **kwargs):
    r = torch.sqrt(torch.sum((event_loc[:, :2] - station_loc[:, :2]) ** 2, dim=-1, keepdims=True))
    z = event_loc[:, 2:] - station_loc[:, 2:]
    if (event_loc[:, 2:] < 0).any():
        print(f"Warning: depth is defined as positive down: {event_loc[:, 2:].detach().numpy()}")

    tt = _interp(time_table, r, z, rgrid, zgrid, h)

    return tt


##################################################################################################################


def calc_time(event_loc, station_loc, phase_type, vel={"p": 6.0, "s": 6.0 / 1.75}, eikonal=None, **kwargs):

    ev_loc = event_loc[:, :-1]
    ev_t = event_loc[:, -1:]

    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])[:, np.newaxis]
        tt = np.linalg.norm(ev_loc - station_loc, axis=-1, keepdims=True) / v + ev_t
    else:
        ev_loc = torch.from_numpy(ev_loc).float()
        station_locs = torch.from_numpy(station_loc).float()

        tp = traveltime(
            ev_loc,
            station_locs[phase_type == "p"],
            eikonal["up"],
            eikonal["rgrid"],
            eikonal["zgrid"],
            eikonal["h"],
            **kwargs,
        )
        ts = traveltime(
            ev_loc,
            station_locs[phase_type == "s"],
            eikonal["us"],
            eikonal["rgrid"],
            eikonal["zgrid"],
            eikonal["h"],
            **kwargs,
        )

        tt = np.zeros(len(phase_type), dtype=np.float32)[:, np.newaxis]
        tt[phase_type == "p"] = tp.numpy()
        tt[phase_type == "s"] = ts.numpy()
        tt = tt + ev_t

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
    return logA


## Huber loss
def loss_and_grad(event_loc, phase_time, phase_type, station_loc, weight, vel={"p": 6.0, "s": 6.0 / 1.75}, sigma=1):
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    event_loc = event_loc[np.newaxis, :]
    dist = np.sqrt(np.sum((station_loc - event_loc[:, :-1]) ** 2, axis=1, keepdims=True))
    J = np.zeros([phase_time.shape[0], event_loc.shape[1]])
    J[:, :-1] = (event_loc[:, :-1] - station_loc) / (dist + 1e-6) / v
    J[:, -1] = 1

    y = dist / v - (phase_time - event_loc[:, -1:])

    # std = np.sqrt(np.sum(y**2 * weight) / (np.sum(weight)+1e-12))
    # mask = (np.abs(y) <= 2*std)
    # l1 = np.squeeze((np.abs(y) > sigma) & mask)
    # l2 = np.squeeze((np.abs(y) <= sigma) & mask)

    l1 = np.squeeze((np.abs(y) > sigma))
    l2 = np.squeeze((np.abs(y) <= sigma))

    loss = np.sum((sigma * np.abs(y[l1]) - 0.5 * sigma**2) * weight[l1]) + np.sum(0.5 * y[l2] ** 2 * weight[l2])
    J_ = np.sum(sigma * np.sign(y[l1]) * J[l1] * weight[l1], axis=0, keepdims=True) + np.sum(
        y[l2] * J[l2] * weight[l2], axis=0, keepdims=True
    )

    return loss, J_


def linloc(
    event_loc0,
    phase_time,
    phase_type,
    station_loc,
    weight,
    max_iter=10,
    convergence=1e-3,
    bounds=None,
    vel={"p": 6.0, "s": 6.0 / 1.75},
):
    opt = scipy.optimize.minimize(
        loss_and_grad,
        np.squeeze(event_loc0),
        method="L-BFGS-B",
        jac=True,
        args=(phase_time, phase_type, station_loc, weight, vel, 1),
        bounds=bounds,
        options={"maxiter": max_iter, "gtol": convergence, "iprint": -1},
    )

    return opt.x[np.newaxis, :], opt.fun


def eikoloc(
    event_loc0,
    phase_time,
    phase_type,
    station_loc,
    weight,
    up,
    us,
    rgrid,
    zgrid,
    h,
    bounds=None,
    device="cpu",
    add_eqt=False,
    gamma=0.1,
    max_iter=1000,
    convergence=1e-9,
):
    event_loc = torch.tensor(event_loc0, dtype=torch.float32, requires_grad=True, device=device)
    if bounds is not None:
        bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
    p_index = torch.arange(len(phase_type), device=device)[phase_type == "p"]
    s_index = torch.arange(len(phase_type), device=device)[phase_type == "s"]
    time = torch.tensor(phase_time, dtype=torch.float32, device=device)
    loc = torch.tensor(station_loc, dtype=torch.float32, device=device)
    weight = torch.tensor(weight, dtype=torch.float32, device=device)
    obs_p = time[p_index]
    obs_s = time[s_index]
    loc_p = loc[p_index]
    loc_s = loc[s_index]
    weight_p = weight[p_index]
    weight_s = weight[s_index]

    # %% optimization
    optimizer = torch.optim.LBFGS(params=[event_loc], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_change=convergence)

    def closure():
        optimizer.zero_grad()
        if bounds is not None:
            loc0_ = torch.max(torch.min(event_loc[:, :-1], bounds[:, 1]), bounds[:, 0])
        else:
            loc0_ = event_loc[:, :-1]
        loc0_ = torch.nan_to_num(loc0_, nan=0)
        t0_ = event_loc[:, -1:]
        if len(p_index) > 0:
            tt_p = traveltime(loc0_, loc_p, up, rgrid, zgrid, h, sigma=1)
            pred_p = t0_ + tt_p
            loss_p = torch.mean(F.huber_loss(obs_p, pred_p, reduction="none") * weight_p)
            if add_eqt:
                dd_tt_p = tt_p.unsqueeze(-1) - tt_p.unsqueeze(-2)
                dd_time_p = obs_p.unsqueeze(-1) - obs_p.unsqueeze(-2)
                loss_p += gamma * torch.mean(
                    F.huber_loss(dd_tt_p, dd_time_p, reduction="none") * weight_p.unsqueeze(-1) * weight_p.unsqueeze(-2)
                )
            # loss_p = F.mse_loss(time_p, tt_p)
        else:
            loss_p = 0
        if len(s_index) > 0:
            tt_s = traveltime(loc0_, loc_s, us, rgrid, zgrid, h, sigma=1)
            pred_s = t0_ + tt_s
            loss_s = torch.mean(F.huber_loss(obs_s, pred_s, reduction="none") * weight_s)
            if add_eqt:
                dd_tt_s = tt_s.unsqueeze(-1) - tt_s.unsqueeze(-2)
                dd_time_s = obs_s.unsqueeze(-1) - obs_s.unsqueeze(-2)
                loss_s += gamma * torch.mean(
                    F.huber_loss(dd_tt_s, dd_time_s, reduction="none") * weight_s.unsqueeze(-1) * weight_s.unsqueeze(-2)
                )
            # loss_s = F.mse_loss(time_s, tt_s)
        else:
            loss_s = 0
        loss = loss_p + loss_s
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure().item()

    event_loc = event_loc.detach().cpu()
    if bounds is not None:
        event_loc[:, :-1] = torch.max(torch.min(event_loc[:, :-1], bounds[:, 1]), bounds[:, 0])

    return event_loc, loss


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
    if eikonal is None:
        event_loc, loss = linloc(
            event_loc0,
            phase_time,
            phase_type,
            station_loc,
            weight,
            vel=vel,
            bounds=bounds,
            max_iter=max_iter,
            convergence=convergence,
        )
    else:
        event_loc, loss = eikoloc(
            event_loc0,
            phase_time,
            phase_type,
            station_loc,
            weight,
            up=eikonal["up"],
            us=eikonal["us"],
            rgrid=eikonal["rgrid"],
            zgrid=eikonal["zgrid"],
            h=eikonal["h"],
            bounds=bounds[:-1],
            max_iter=max_iter,
            convergence=convergence,
        )

    return event_loc, loss


def initialize_eikonal(config):
    rlim = [0, np.sqrt((config["xlim"][1] - config["xlim"][0]) ** 2 + (config["ylim"][1] - config["ylim"][0]) ** 2)]
    zlim = config["zlim"]
    edge_grids = 3
    # zlim = [0, config["zlim"][1] - config["zlim"][0]]
    # edge_grids = 0
    h = config["h"]

    rgrid = np.arange(rlim[0] - edge_grids * h, rlim[1], h)
    zgrid = np.arange(zlim[0] - edge_grids * h, zlim[1], h)
    m, n = len(rgrid), len(zgrid)

    vel = config["vel"]
    zz, vp, vs = vel["z"], vel["p"], vel["s"]
    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
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

    config.update({"up": up, "us": us, "rgrid": rgrid, "zgrid": zgrid, "h": h})

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

    # dist = np.linalg.norm(means - X, axis=-1).T  # (n_components, n_samples, n_features) -> (n_samples, n_components)
    # resp = np.exp(-dist)
    # resp_sum = resp.sum(axis=1, keepdims=True)
    # resp_sum[resp_sum == 0] = 1.0
    # resp = resp / resp_sum

    # proposed by Yaqi
    dist = np.linalg.norm(means - X, axis=-1) # (n_components, n_samples, n_features) -> (n_components, n_samples)
    resp = np.exp(-dist/(np.median(dist, axis=0, keepdims=True)/10.0)).T 
    resp /= np.sum(resp, axis=1, keepdims=True) # (n_components, n_samples)

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
