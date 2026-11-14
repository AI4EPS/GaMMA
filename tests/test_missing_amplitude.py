"""Picks without amplitude (NaN): associated by their time alone, the others keep using amplitude."""
import numpy as np
from scipy.stats import multivariate_normal, norm

from gamma._gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_covariances_full
from gamma._gaussian_mixture import _estimate_log_gaussian_prob
from gamma.seismic_ops import calc_amp
from gamma.utils import association
from test_association import PROJ, base_config, synthetic


def mixture(n=40, k=3):
    """n picks (time, log amplitude), their means under k components (k, n, 2), covariances and responsibilities."""
    rng = np.random.default_rng(0)
    X, means = rng.normal(size=(n, 2)), rng.normal(size=(k, n, 2))
    A = rng.normal(size=(k, 2, 2))
    cov = A @ A.transpose(0, 2, 1) + 0.1 * np.eye(2)
    resp = rng.dirichlet(np.ones(k), size=n)
    return X, means, cov, resp


def rng_noise(n):
    return 0.01 * np.random.default_rng(1).normal(size=(n, 2))


def test_no_nan_is_the_full_gaussian():
    X, means, cov, resp = mixture()
    log_prob = _estimate_log_gaussian_prob(X, means, _compute_precision_cholesky(cov, "full"), "full")
    full = np.stack([multivariate_normal.logpdf(X - means[k], cov=cov[k]) for k in range(len(cov))], axis=1)
    assert np.allclose(log_prob, full, rtol=0, atol=1e-12)
    nk = resp.sum(axis=0)
    S = _estimate_gaussian_covariances_full(resp, X, nk, means, 1e-6)
    for k in range(len(cov)):  # the formula before missing amplitudes, bit for bit
        diff = X - means[k]
        assert np.array_equal(S[k], np.dot(resp[:, k] * diff.T, diff) / nk[k] + 1e-6 * np.eye(2))


def test_nan_amplitude_is_the_time_marginal():
    X, means, cov, resp = mixture()
    chol = _compute_precision_cholesky(cov, "full")
    full = _estimate_log_gaussian_prob(X, means, chol, "full")
    nan = np.arange(len(X)) % 3 == 0
    Xn = np.where(nan[:, None] & [False, True], np.nan, X)
    log_prob = _estimate_log_gaussian_prob(Xn, means, chol, "full")
    marginal = norm.logpdf(X[:, :1], means[:, :, 0].T, np.sqrt(cov[:, 0, 0]))  # (n, k): N(time; Sigma_tt)
    assert np.allclose(log_prob[nan], marginal[nan], rtol=0, atol=1e-12)
    assert np.array_equal(log_prob[~nan], full[~nan])

    # time variance from all picks, amplitude variance and covariance from the picks with an amplitude
    nk = resp.sum(axis=0)
    S = _estimate_gaussian_covariances_full(resp, Xn, nk, means, 1e-6)
    S_all = _estimate_gaussian_covariances_full(resp, X, nk, means, 1e-6)
    S_amp = _estimate_gaussian_covariances_full(resp[~nan], X[~nan], resp[~nan].sum(axis=0), means[:, ~nan], 1e-6)
    assert np.allclose(S[:, 0, 0], S_all[:, 0, 0]) and np.allclose(S[:, 1, 1], S_amp[:, 1, 1])
    bound = np.sqrt((S[:, 0, 0] - 1e-6) * (S[:, 1, 1] - 1e-6))
    assert np.allclose(S[:, 0, 1], np.clip(S_amp[:, 0, 1], -bound, bound)) and (np.linalg.eigvalsh(S) > 0).all()
    S = _estimate_gaussian_covariances_full(resp, np.c_[X[:, 0], np.full(len(X), np.nan)], nk, means, 1e-6)
    assert np.allclose(S[:, 0, 0], S_all[:, 0, 0]) and np.allclose(S[:, 1, 1], 1e-6) and (S[:, 0, 1] == 0).all()

    # the clip at work: amplitude picks with correlated residuals, the others close to the time mean
    X, means, resp = np.zeros((40, 2)), np.zeros((1, 40, 2)), np.ones((40, 1))
    X[:8] = np.linspace(-1, 1, 8)[:, None] * [1.0, 0.5] + rng_noise(8)
    X[8:, 1] = np.nan
    S = _estimate_gaussian_covariances_full(resp, X, resp.sum(axis=0), means, 1e-6)[0]
    S_amp = _estimate_gaussian_covariances_full(resp[:8], X[:8], resp[:8].sum(axis=0), means[:, :8], 1e-6)[0]
    assert abs(S_amp[0, 1]) > S[0, 1] > 0 and np.isclose(S[0, 1] ** 2, (S[0, 0] - 1e-6) * (S[1, 1] - 1e-6))
    assert (np.linalg.eigvalsh(S) > 0).all() and np.isfinite(_compute_precision_cholesky(S[None], "full")).all()


def test_association_with_missing_amplitudes():
    picks, stations, truth = synthetic()
    mag, rng = np.array([1.0, 2.0, 1.5, 2.5]), np.random.default_rng(0)
    xyz = stations[["x_km", "y_km", "z_km"]].values
    for k in range(4):  # 24 picks per event, in the order of synthetic(): amplitude (m/s) of its magnitude, with scatter
        log_amp = np.tile(calc_amp(mag[k], np.r_[truth[k], 0.0][None], xyz)[:, 0], 2)  # log10(cm/s), P then S
        picks.loc[24 * k : 24 * k + 23, "phase_amplitude"] = 10 ** (log_amp + 0.2 * rng.normal(size=24)) / 1e2
    config = {**base_config(), "use_amplitude": True, "proj": PROJ}

    def run(picks, config=config):
        events, assignments = association(picks, stations, config)
        events = sorted(events, key=lambda e: e["time"])
        return events, np.array([e["magnitude"] for e in events]), sorted(assignments)

    _, m_full, a_full = run(picks)
    assert len(a_full) == len(picks) and np.abs(m_full - mag).max() < 0.2

    some = picks.copy()  # a third of the stations without amplitude, written as NaN, -1 or 0
    for i, value in zip((0, 3, 6, 9), (np.nan, -1, 0, np.nan)):
        some.loc[some.station_id == stations.station_id[i], "phase_amplitude"] = value
    events, m, a = run(some)
    assert len(a) == len(picks) and np.abs(m - mag).max() < 0.2  # every pick kept, magnitude from the other picks
    assert all(e["num_picks"] == 24 and 0.05 < e["sigma_amp"] < 0.4 for e in events)

    # no amplitude at all: no magnitude, and the association of the time-only model
    events, m, a = run(picks.assign(phase_amplitude=np.nan))
    events_t, _, a_t = run(picks, {**config, "use_amplitude": False})
    assert np.isnan(m).all() and all(e["sigma_amp"] == 0 and e["cov_time_amp"] == 0 for e in events)
    assert np.allclose(np.array(a), np.array(a_t), rtol=0, atol=1e-5)  # pick, event, score
    for key in ("x_km", "y_km", "z_km", "sigma_time", "gamma_score"):
        assert np.allclose([e[key] for e in events], [e[key] for e in events_t], rtol=0, atol=1e-5)
