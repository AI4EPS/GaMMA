import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

############################################################################################################
# |\nabla u| = f

# ((u - a1)^+)^2 + ((u - a2)^+)^2 = f^2 h^2


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


def sweeping(u, f, h):

    s = 1.0 / f  ## slowness

    m, n = u.shape
    I = list(range(m))
    iI = I[::-1]
    J = list(range(n))
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, s, h)
    u = sweeping_over_I_J_K(u, iI, J, s, h)
    u = sweeping_over_I_J_K(u, iI, iJ, s, h)
    u = sweeping_over_I_J_K(u, I, iJ, s, h)

    return u


def eikonal_solve(u, f, h):

    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))

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


def traveltime(vars_, station_locs, phase_type, up, us, h, rgrid, zgrid, bounds=None):

    def interp(X, h, input):
        # the origin is (0,0)
        ir0 = torch.floor(input[0].div(h)).type(torch.long)
        ir1 = ir0 + 1
        iz0 = torch.floor(input[1].div(h)).type(torch.long)
        iz1 = iz0 + 1
        if iz0 >= zgrid.shape[1]:
            iz0 = zgrid.shape[1] - 1
        if iz1 >= zgrid.shape[1]:
            iz1 = zgrid.shape[1] - 1
        r0 = ir0 * h
        z0 = iz0 * h

        Ia = X[ir0, iz1]
        Ib = X[ir1, iz1]
        Ic = X[ir0, iz0]
        Id = X[ir1, iz0]

        return ((Ib-Ia) * (input[0]-r0)/h + Ia - (Id-Ic) * (input[0]-r0)/h - Ic) * (input[1]-z0)/h + (Id-Ic) * (input[0]-r0)/h + Ic

    if bounds is not None:
        vars = normalize(vars_, bounds)
    else:
        vars = vars_

    r = torch.sqrt(torch.sum((vars[0, :2] - station_locs[:, :2]) ** 2, dim=-1)).unsqueeze(1)
    z = torch.abs(vars[0, 2] - station_locs[:, 2]).unsqueeze(1)
    t = torch.cat([interp(up, h, torch.cat([r[i], z[i]])).unsqueeze(0) if phase_type[i]=='p' else interp(us, h, torch.cat([r[i], z[i]])).unsqueeze(0) for i, _ in enumerate(phase_type)], 0)
    return t


def invert_location(
    data, event_t0, event_locs, station_locs, phase_type, weight, up, us, h, rgrid, zgrid, bounds=None
):
    t0_ = torch.tensor(event_t0, dtype=torch.float32, requires_grad=True)
    loc_ = torch.tensor(event_locs, dtype=torch.float32, requires_grad=True)
    if bounds is not None:
        loc = normalize(loc_, bounds)
        t0 = t0_
    else:
        loc = loc_
        t0 = t0_
    station_locs = torch.tensor(station_locs, dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32).squeeze(1)
    data = torch.tensor(data, dtype=torch.float32)
    rgrid = torch.tensor(rgrid, dtype=torch.float32)
    zgrid = torch.tensor(zgrid, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.float32)
    optimizer = optim.LBFGS(params=[t0_, loc_], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        tt = t0_[0] + traveltime(loc_, station_locs, phase_type, up, us, h, rgrid, zgrid, bounds=bounds)
        loss = F.huber_loss(data, tt, reduction='none') * weight
        loss = loss.sum() / (weight.sum() + torch.tensor(1e-6))
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)

    if bounds is not None:
        loc = normalize(loc_, bounds)
        t0 = t0_
    else:
        loc = loc_
        t0 = t0_
        
    return torch.cat((loc.squeeze(0), t0), 0).detach().numpy()