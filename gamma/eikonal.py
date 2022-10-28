import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

############################################################################################################
# |\nabla u| = f

# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


def calculate_unique_solution(a, b, f, h):

    d = abs(a - b)
    if d >= f * h:
        return min(a, b) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


def sweeping_over_I_J_K(u, I, J, f, h):
    # print("Sweeping start...")
    m = len(I)
    n = len(J)
    for i in I:
        for j in J:
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

    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iteration {i}, Error = {err}")
        if err < 1e-6:
            break

    return u


def normalize(vars_, bounds):
    mean = torch.tensor(
        [
            [
                (bounds[0][0] + bounds[0][1]) / 2,
                (bounds[1][0] + bounds[1][1]) / 2,
                (bounds[2][0] + bounds[2][1]) / 2,
            ],
        ],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [
            [
                (bounds[0][1] - bounds[0][0]) / 2,
                (bounds[1][1] - bounds[1][0]) / 2,
                (bounds[2][1] - bounds[2][0]) / 2,
            ]
        ],
        dtype=torch.float32,
    )
    vars = (vars_ - mean) / std
    vars = torch.tanh(vars)
    vars = (vars * std) + mean

    return vars


def traveltime(vars_, station_locs, phase_type, up, us, rgrid, zgrid, sigma=1, bounds=None):

    if bounds is not None:
        vars = normalize(vars_, bounds)
    else:
        vars = vars_

    r = torch.sqrt(torch.sum((vars[0, :2] - station_locs[:, :2]) ** 2, dim=-1))
    z = torch.abs(vars[0, 2] - station_locs[:, 2])

    r = r.unsqueeze(-1).unsqueeze(-1)
    z = z.unsqueeze(-1).unsqueeze(-1)

    magn = (
        1.0
        / (2.0 * np.pi * sigma)
        * torch.exp(-(((rgrid - r) / (np.sqrt(2 * sigma) * h)) ** 2 + ((zgrid - z) / (np.sqrt(2 * sigma) * h)) ** 2))
    )
    tp = torch.sum(up * magn, dim=(-1, -2))
    ts = torch.sum(us * magn, dim=(-1, -2))

    tt = torch.cat([tp, ts], dim=0)

    return tt


def invert_location(
    data, event_t0, event_locs, station_locs, phase_type, weight, up, us, rgrid, zgrid, sigma=1, bounds=None
):
    t0_ = torch.tensor(event_t0, dtype=torch.float32, requires_grad=True)
    loc_ = torch.tensor(event_locs, dtype=torch.float32, requires_grad=True)
    if bounds is not None:
        loc = normalize(loc_, bounds)
        t0 = t0_
    else:
        loc = loc_
        t0 = t0_

    print("Initial:", t0_.data, loc_.data)

    station_locs = torch.tensor(station_locs[:4, :], dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)
    rgrid = torch.tensor(rgrid, dtype=torch.float32)
    zgrid = torch.tensor(zgrid, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.float32)

    p_index = None
    # optimizer = optim.LBFGS(params=[t0_, loc_], max_iter=1000, line_search_fn="strong_wolfe")
    optimizer = optim.LBFGS(params=[loc_], max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        tt = t0_[0] + traveltime(loc_, station_locs, phase_type, up, us, rgrid, zgrid, sigma, bounds=bounds)
        # loss = F.mse_loss(data, tt)
        loss = F.huber_loss(data, tt)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)

    if bounds is not None:
        loc = normalize(loc_, bounds)
        t0 = t0_
    else:
        loc = loc_
        t0 = t0_

    print("Inverted:", t0.data, loc.data)
    return loc


if __name__ == "__main__":

    xlim = [0, 30]
    ylim = [0, 30]
    zlim = [0, 20]  ## depth
    h = 0.3

    rlim = [0, ((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2) ** 0.5]

    rx = np.arange(rlim[0], rlim[1] + h, h)
    zx = np.arange(zlim[0], zlim[1] + h, h)
    m = len(rx)
    n = len(zx)
    dr = h
    dz = h

    vp = np.ones((m, n)) * 6.0
    vs = np.ones((m, n)) * (6.0 / 1.75)

    up = 1000 * np.ones((m, n))
    up[0, 0] = 0.0
    up = eikonal_solve(up, vp, h)

    us = 1000 * np.ones((m, n))
    us[0, 0] = 0.0
    us = eikonal_solve(us, vs, h)

    ############################## Check eikonal ##################################
    rgrid, zgrid = np.meshgrid(rx, zx, indexing="ij")
    up_true = np.sqrt((rgrid - 0) ** 2 + (zgrid - 0) ** 2) / np.mean(vp)
    us_true = np.sqrt((rgrid - 0) ** 2 + (zgrid - 0) ** 2) / np.mean(vs)

    fig, axes = plt.subplots(2, 1, figsize=(16, 16 * n / m * 2))
    im0 = axes[0].pcolormesh(rx, zx, up.T)
    axes[0].axis("scaled")
    axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(rx, zx, (up - up_true).T, vmax=up.max(), vmin=up.min())
    axes[1].axis("scaled")
    axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1])

    fig.savefig("test_vp.png")

    fig, axes = plt.subplots(2, 1, figsize=(16, 16 * n / m * 2))
    im0 = axes[0].pcolormesh(rx, zx, us.T)
    axes[0].axis("scaled")
    axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(rx, zx, (us - us_true).T, vmax=up.max(), vmin=up.min())
    axes[1].axis("scaled")
    axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1])

    fig.savefig("test_vs.png")

    ############################# Check traveltime extraction ###################################
    sigma = 0.8
    event_locs = np.array([[15, 15, 18]])  # (1, 3)
    # station_locs = np.array([[10, 10, 0], [20, 10, 0], [10, 20, 0], [20, 20, 0]])  # (nsta, 3)
    station_locs = np.array([[10, 10, 0], [20, 10, 0], [10, 20, 0], [20, 20, 0]])  # (nsta, 3)
    r = np.sqrt(np.sum((event_locs[0, :2] - station_locs[:, :2]) ** 2, axis=-1))  # (nsta, 3)
    z = np.abs(event_locs[0:1, 2] - station_locs[:, 2])  # (nsta, 1)
    # r = np.array([rlim[-1] - 5 * h])
    # z = np.array([zlim[-1] - 5 * h])

    r = r[:, np.newaxis, np.newaxis]
    z = z[:, np.newaxis, np.newaxis]
    rgrid, zgrid = np.meshgrid(rx, zx, indexing="ij")

    magn = (
        1.0
        / (2.0 * np.pi * sigma)
        * np.exp(-(((rgrid - r) / (np.sqrt(2 * sigma) * h)) ** 2 + ((zgrid - z) / (np.sqrt(2 * sigma) * h)) ** 2))
    )

    tp = np.sum(up * magn, axis=(-1, -2))
    ts = np.sum(us * magn, axis=(-1, -2))

    # print(up.shape, magn.shape)

    tp_true = (r[:, 0, 0] ** 2 + z[:, 0, 0] ** 2) ** 0.5 / np.mean(vp)
    ts_true = (r[:, 0, 0] ** 2 + z[:, 0, 0] ** 2) ** 0.5 / np.mean(vs)

    print(f"tp = {tp}, tp_true = {tp_true}")
    print(f"ts = {ts}, ts_true = {ts_true}")

    ############################# Inverting earthquake locations ###################################

    assert len(tp) == len(ts)
    data = np.concatenate([tp, ts])
    # data = np.concatenate([tp_true, ts_true])
    weight = np.concatenate([np.ones_like(tp), np.ones_like(ts)])
    phase_type = ["P"] * len(tp) + ["S"] * len(ts)
    station_locs = np.concatenate([station_locs, station_locs])
    bounds = [[0, 30], [0, 30], [0, 20]]
    event_t0 = np.array([[0.0]])
    event_locs = np.array(
        [
            [
                (bounds[0][0] + bounds[0][1]) / 2,
                (bounds[1][0] + bounds[1][1]) / 2,
                (bounds[2][0] + bounds[2][1]) / 2,
            ]
        ]
    )
    # event_locs = np.array([[15, 15, 18]])
    # bounds = None
    print(event_t0.shape, event_locs.shape, data.shape)
    invert_location(
        data, event_t0, event_locs, station_locs, phase_type, weight, up, us, rgrid, zgrid, sigma=0.6, bounds=bounds
    )
