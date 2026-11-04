import numpy as np

from gamma.utils import hierarchical_dbscan_clustering


def test_split_keeps_clusters_apart():
    """All picks at one place, so the DBSCAN distance is |dt|. The first split round uses eps / 1.2 = 1.75 s.
    D: small, never split. B: one chain at 2.1 s, all noise at 1.75 s. C: two bursts 2.0 s apart, two at 1.75 s.
    Before 1.12, B's all-noise result reset the next label to 0, so C's first burst joined D, and a cluster that
    re-clustered into one kept shrinking until it was all noise."""
    D = np.arange(0, 10, 0.5)
    B = np.arange(100, 400, 2.0)
    C1, C2 = np.arange(1000, 1050, 0.5), np.arange(1052, 1100, 0.5)
    t = np.r_[D, B, C1, C2][:, None]
    labels = hierarchical_dbscan_clustering(
        t, np.zeros((len(t), 2)), np.array(["p"] * len(t)), np.ones((len(t), 1)), {"p": 6.0},
        eps=2.1, min_samples=3, min_cluster_size=100, max_time_space_ratio=1,
    )
    parts = np.split(labels, np.cumsum([len(D), len(B), len(C1)]))
    d, b, c1, c2 = (set(p) for p in parts)
    assert len(d) == 1 and len(b) == 1 and len(c1) == 1 and len(c2) == 1  # each part stays whole
    assert -1 not in d | c1 | c2  # no cluster dissolved into noise
    assert len(d | b | c1 | c2) == 4  # four distinct labels: nothing merged


def test_geocentric_matches_projected():
    """Picks of 20 events on 30 stations within 40 km: DBSCAN on Earth-centred positions (global data) groups them as
    on projected x, y, since chord and surface distances agree to 0.001 km at this range."""
    from pyproj import Proj
    from gamma.utils import geocentric

    rng = np.random.default_rng(0)
    lon0, lat0 = -122.8, 38.8
    proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=km +ellps=WGS84")
    sta = rng.uniform(-40, 40, (30, 2))
    t, xy = [], []
    for k in range(20):
        src = rng.uniform(-30, 30, 2)
        dist = np.linalg.norm(sta - src, axis=1)
        t += list(k * 60 + dist / 6.0) + list(k * 60 + dist / 3.4)  # P and S
        xy += [sta, sta]
    t, xy = np.array(t)[:, None], np.vstack(xy)
    lon, lat = proj(xy[:, 0], xy[:, 1], inverse=True)
    w, ph, vel = np.ones((len(t), 1)), np.array(["p"] * len(t)), {"p": 6.0}
    a = hierarchical_dbscan_clustering(t, xy, ph, w, vel, eps=10.0)
    b = hierarchical_dbscan_clustering(t, geocentric(np.column_stack([lon, lat])), ph, w, vel, eps=10.0)
    assert len(set(a)) == 20 and (a == b).all()


if __name__ == "__main__":
    test_split_keeps_clusters_apart()
    test_geocentric_matches_projected()
    print("ok")
