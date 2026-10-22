"""Unit tests for the statistics in xdrift_v2.py. Run with: python -m pytest test_xdrift_v2.py"""
import numpy as np

import xdrift_v2 as v


def _shap(seed=0, decay=(1, 0.05)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(300, 34)) * np.linspace(*decay, 34)


def test_identical_explanations_are_perfectly_stable():
    S = _shap()
    assert all(abs(val - 1.0) < 1e-9 for val in v.stability_metrics(S, S.copy()).values())


def test_reversed_importance_is_unstable():
    m = v.stability_metrics(_shap(0), _shap(1, decay=(0.05, 1)))
    assert m["xsi_abs"] < 0.4 and m["tau_only"] < 0.1 and m["jaccard10"] == 0.0


def test_circular_shift_test_finds_perfect_correlation():
    x = np.cumsum(np.random.default_rng(1).normal(size=41))
    r = v.circ_perm_spearman(x, 2 * x + 1)
    assert abs(r["rho"] - 1) < 1e-9 and abs(r["p"] - 1 / 41) < 1e-9


def test_circular_shift_test_false_positive_rate_on_random_walks():
    """Documented limitation: on strongly autocorrelated series the test is still
    anti-conservative (nominal 5%), but far better than naive Spearman (>50%)."""
    rng = np.random.default_rng(7)
    p = [v.circ_perm_spearman(np.cumsum(rng.normal(size=41)), np.cumsum(rng.normal(size=41)))["p"]
         for _ in range(200)]
    assert np.mean(np.array(p) < 0.05) < 0.35


def test_holm_matches_hand_calculation():
    assert np.allclose(v.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_h3_detects_planted_lead_and_ignores_unrelated_alerts():
    rng = np.random.default_rng(1)
    n = 40
    xsi = 0.6 + rng.normal(0, 0.01, n)
    auc = 0.72 + rng.normal(0, 0.002, n)
    for t in (15, 25, 33):
        xsi[t - 1], auc[t] = 0.2, 0.60
    alert, adef = v.eart_v2_alerts(xsi)
    event, edef = v.degradation_events(auc)
    hit = v.h3_evaluate(alert, adef, event, edef, np.ones(n, bool))
    assert hit["hit_rate"] == 1.0 and hit["p_one_sided"] < 0.05 and hit["underpowered"]
    a2, d2 = v.eart_v2_alerts(rng.permutation(xsi))
    miss = v.h3_evaluate(a2, d2, event, edef, np.ones(n, bool))
    assert miss["p_one_sided"] > 0.1


def test_alerts_have_no_lookahead():
    x = np.array([0.6] * 10 + [0.1] + [0.6] * 5)
    alert, _ = v.eart_v2_alerts(x)
    x_future = x.copy()
    x_future[12:] = 0.0  # change only the future
    alert2, _ = v.eart_v2_alerts(x_future)
    assert (alert[:12] == alert2[:12]).all()
