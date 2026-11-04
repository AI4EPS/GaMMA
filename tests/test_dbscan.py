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


if __name__ == "__main__":
    test_split_keeps_clusters_apart()
    print("ok")
