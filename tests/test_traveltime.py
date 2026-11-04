import numpy as np

from gamma.seismic_ops import initialize_eikonal, traveltime_grad

VEL = {"Z": [0.0, 5.5, 5.5, 16.0, 16.0, 32.0, 32.0], "P": [5.5, 5.5, 6.3, 6.3, 6.7, 6.7, 7.8]}  # Hadley-Kanamori
VEL["S"] = [v / 1.73 for v in VEL["P"]]
EIK = initialize_eikonal({"vel": VEL, "h": 0.3, "xlim_km": (-20, 20), "ylim_km": (-20, 20), "zlim_km": (-1, 15),
                          "station_z_km": [0.0]})
STATION = np.zeros((1, 3))


def grad_and_fd(q):
    ph = np.array(["p"] * len(q))
    g = traveltime_grad(q, STATION, ph, eikonal=EIK)[1]
    fd = np.column_stack([(traveltime_grad(q + 1e-6 * d, STATION, ph, eikonal=EIK)[0]
                           - traveltime_grad(q - 1e-6 * d, STATION, ph, eikonal=EIK)[0]) / 2e-6 for d in np.eye(3)])
    return g, fd


def test_gradient_is_the_derivative_inside_and_outside():
    """Inside the grid and beyond each edge and corner (r > rmax, above the top, below the bottom)."""
    rmax, ztop, zbot = EIK["rgrid"][-1], EIK["zgrid"][0], EIK["zgrid"][-1]
    rng = np.random.default_rng(0)
    k = 100
    r_in, z_in = rng.uniform(1, rmax - 1, k), rng.uniform(ztop + 1, zbot - 1, k)
    r_out, z_top, z_bot = rmax + rng.uniform(0.1, 5, k), ztop - rng.uniform(0.1, 5, k), zbot + rng.uniform(0.1, 5, k)
    az = rng.uniform(0, 2 * np.pi, k)
    cases = [(r_in, z_in), (r_out, z_in), (r_in, z_top), (r_in, z_bot), (r_out, z_top), (r_out, z_bot)]
    q = np.vstack([np.c_[r * np.cos(az), r * np.sin(az), z] for r, z in cases])
    g, fd = grad_and_fd(q)
    assert np.linalg.norm(g - fd, axis=1).max() < 1e-6
    assert (np.linalg.norm(g[k:], axis=1) > 0.05).all()  # never zero outside: an event pushed out is pulled back


def test_gradient_continuous_across_nodes_and_edges():
    """dT/dz is the same on both sides of a node (a bilinear time jumps there) and of each edge of the grid."""
    rmax, ztop, zbot = EIK["rgrid"][-1], EIK["zgrid"][0], EIK["zgrid"][-1]
    for a, b in [([20, 0, 5.4 - 1e-6], [20, 0, 5.4 + 1e-6]), ([rmax - 1e-6, 0, 10], [rmax + 1e-6, 0, 10]),
                 ([5, 5, ztop + 1e-6], [5, 5, ztop - 1e-6]), ([5, 5, zbot - 1e-6], [5, 5, zbot + 1e-6])]:
        t, g = traveltime_grad(np.array([a, b], float), STATION, np.array(["p", "p"]), eikonal=EIK)
        assert abs(t[1] - t[0]) < 1e-5 and np.abs(g[1] - g[0]).max() < 1e-4
