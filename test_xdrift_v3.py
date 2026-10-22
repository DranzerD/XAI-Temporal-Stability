"""Unit tests for xdrift_v3.py statistics. Run with: python -m pytest test_xdrift_v3.py"""
import numpy as np

import xdrift_v3 as v3

RNG = lambda s=0: np.random.default_rng(s)


def test_psi_is_zero_for_identical_and_large_for_shifted():
    x = RNG(1).normal(size=20_000)
    assert v3.feature_psi(x, x) < 1e-6
    assert v3.feature_psi(x, x + 1.5) > 0.5


def test_psi_handles_low_cardinality_and_new_values():
    ref = np.array([0.36] * 1000)            # e.g. a macro feature, constant in the reference quarter
    cur = np.array([1.0] * 1000)             # a value never seen in the reference
    assert v3.feature_psi(ref, cur) > 5      # explodes, which is why a no-macro baseline exists
    assert v3.feature_psi(ref, ref) < 1e-6


def test_partial_spearman_detects_a_planted_relation():
    rng = RNG(2)
    M, K = 13, 6
    latent = rng.normal(size=(M, K))
    X = latent + 0.3 * rng.normal(size=(M, K))
    Y = latent + 0.3 * rng.normal(size=(M, K))
    r = v3.partial_spearman(X, Y, np.arange(1, K + 1), rng, n_perm=2000, n_boot=300)
    assert r["rho"] > 0.7 and r["p"] < 0.01 and r["ci_lo"] > 0.3


def test_partial_spearman_removes_a_shared_time_trend():
    """X and Y both just grow with the horizon k: raw correlation is high, partial should be ~0."""
    rng = RNG(3)
    M, K = 13, 6
    k = np.arange(1, K + 1)
    X = k * 1.0 + 0.5 * rng.normal(size=(M, K))
    Y = k * 1.0 + 0.5 * rng.normal(size=(M, K))
    raw = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
    r = v3.partial_spearman(X, Y, k, rng, n_perm=2000, n_boot=300)
    assert raw > 0.6 and abs(r["rho"]) < 0.3 and r["p"] > 0.05


def test_model_swap_permutation_is_calibrated_under_the_null():
    """Over many independent-noise panels the p-value should reject at about the nominal 5%."""
    rng = RNG(4)
    M, K = 13, 6
    p = [v3.partial_spearman(rng.normal(size=(M, K)), rng.normal(size=(M, K)), np.arange(1, K + 1),
                             rng, n_perm=300, n_boot=20)["p"] for _ in range(200)]
    assert 0.01 < np.mean(np.array(p) < 0.05) < 0.12


def test_cluster_bootstrap_auroc_perfect_and_difference():
    rng = RNG(5)
    E = rng.random((13, 5)) < 0.3
    good = E + 0.01 * rng.normal(size=E.shape)       # separates events perfectly
    noise = rng.normal(size=E.shape)
    a = v3.cluster_boot_auroc(good, E, rng, n_boot=300)
    assert a["auroc"] == 1.0
    d = v3.cluster_boot_auroc(good, E, rng, n_boot=300, S2=noise)
    assert d["auroc"] > 0.2 and d["ci_lo"] > 0.0


def test_auroc_is_nan_without_both_classes():
    assert np.isnan(v3.auroc(np.arange(5.0), np.zeros(5)))
