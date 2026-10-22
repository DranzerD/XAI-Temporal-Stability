#!/usr/bin/env python
"""
XDrift v2 -- noise-floor-calibrated explanation drift, real VIX regimes, pre-declared tests.

Implements ANALYSIS_PLAN.md. Two stages:

  compute  Rolling windows -> per-window XGBoost + TreeSHAP, plus 3 replicate models per
           window (80% subsample, independent eval draw) that define the noise floor.
           Results are cached per window, so an interrupted run resumes.
  analyze  Stability metrics, circular-shift permutation tests, early-warning evaluation,
           figures, and RESULTS.md.

Input is the processed train/test parquet written by cells 8-9 of xdrift.ipynb.

    python xdrift_v2.py all --data-dir <dir with train.parquet/test.parquet> \
        --vix data/fred/VIXCLS_cboe.csv --out results_v2 --v1 <v1 xdrift_results.csv>
"""
import argparse
import hashlib
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, kruskal, rankdata, spearmanr

# ── Settings (v1 values unless noted; see ANALYSIS_PLAN.md) ─────────────────────
WINDOW_MONTHS, STRIDE_MONTHS = 12, 3
MIN_TRAIN, MIN_EVAL = 1000, 50
SHAP_SAMPLES, BG_SAMPLES = 300, 100
TOP_K = 15
N_REPLICATES = 3
REPLICATE_FRACTION = 0.8
N_BOOT_AUC = 200
MIN_EVAL_FOR_AUC_TESTS = 10_000
EVENT_LOOKBACK, EVENT_DELTA = 4, 0.02
EART_LOOKBACK, EART_MIN_PRIOR, EART_K = 8, 6, 1.5
N_PERM_H3 = 10_000
TARGET, DATE = "default", "issue_d"
EXCLUDE = {TARGET, DATE, "issue_year", "issue_quarter"}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "n_estimators": 300,
    "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "scale_pos_weight": 3, "random_state": 42, "tree_method": "hist", "device": "cpu",
    "n_jobs": -1, "enable_categorical": False, "verbosity": 0,
}
METRICS = ["xsi_orig", "xsi_abs", "tau_only", "jaccard10"]
PRIMARY = "xsi_abs"


# ══════════════════════════════════════════════════════════════════════════════
# Stability metrics
# ══════════════════════════════════════════════════════════════════════════════
def _rank(v):
    """Rank 1 = largest value."""
    return v.argsort()[::-1].argsort() + 1


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0


def psi(a, b, n_bins=10, eps=1e-6):
    bins = np.percentile(a, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-3
    bins[-1] += 1e-3
    ca = np.histogram(a, bins=bins)[0] + eps
    cb = np.histogram(b, bins=bins)[0] + eps
    pa, pb = ca / ca.sum(), cb / cb.sum()
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def stability_metrics(Sa, Sb):
    """All four stability variants between two SHAP matrices (n_samples x n_features)."""
    abs_a, abs_b = np.abs(Sa).mean(0), np.abs(Sb).mean(0)
    # PSI component: top-K features by mean |SHAP| averaged over both windows.
    top = np.argsort((abs_a + abs_b) / 2)[::-1][:TOP_K]
    psi_comp = max(0.0, 1.0 - np.mean([min(psi(Sa[:, k], Sb[:, k]), 1.0) for k in top]))

    # v1 definition: ranks/cosine on mean *signed* SHAP.
    sa, sb = Sa.mean(0), Sb.mean(0)
    tau_o = (kendalltau(_rank(np.abs(sa)), _rank(np.abs(sb)))[0] + 1) / 2
    xsi_orig = 0.4 * tau_o + 0.4 * max(0.0, _cos(sa, sb)) + 0.2 * psi_comp

    # v2 primary: ranks/cosine on mean |SHAP| (global importance profile).
    tau_a = (kendalltau(_rank(abs_a), _rank(abs_b))[0] + 1) / 2
    xsi_abs = 0.4 * tau_a + 0.4 * max(0.0, _cos(abs_a, abs_b)) + 0.2 * psi_comp

    ta, tb = set(np.argsort(abs_a)[::-1][:10]), set(np.argsort(abs_b)[::-1][:10])
    return {"xsi_orig": float(xsi_orig), "xsi_abs": float(xsi_abs),
            "tau_only": float(tau_a), "jaccard10": len(ta & tb) / len(ta | tb)}


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════
def _corr_rank(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def circ_perm_spearman(x, y):
    """Spearman rho with an exact circular-shift permutation p-value (two-sided).
    Windows overlap by 75%, so ordinary p-values would be anti-conservative."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 8:
        return {"rho": np.nan, "p": np.nan, "n": int(n)}
    obs = _corr_rank(x, y)
    null = np.array([_corr_rank(x, np.roll(y, k)) for k in range(n)])
    return {"rho": obs, "p": float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), "n": int(n)}


def partial_circ_perm(x, y, z):
    """Partial Spearman (rank-residualised on z) with circular-shift permutation p."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 8:
        return {"rho": np.nan, "p": np.nan, "n": int(n)}
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)
    ex, ey = resid(rx, rz), resid(ry, rz)
    obs = float(np.corrcoef(ex, ey)[0, 1])
    null = np.array([np.corrcoef(ex, np.roll(ey, k))[0, 1] for k in range(n)])
    return {"rho": obs, "p": float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), "n": int(n)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def eart_v2_alerts(x):
    """Alert when x_t < mean(prior 8) - 1.5*sd(prior 8); needs >= 6 finite priors."""
    n = len(x)
    alert, defined = np.zeros(n, bool), np.zeros(n, bool)
    for t in range(n):
        prior = x[max(0, t - EART_LOOKBACK):t]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= EART_MIN_PRIOR and np.isfinite(x[t]):
            defined[t] = True
            alert[t] = x[t] < prior.mean() - EART_K * prior.std(ddof=1)
    return alert, defined


def degradation_events(auc):
    """Event at t when AUC_t <= mean(AUC_{t-4..t-1}) - 0.02."""
    n = len(auc)
    ev, defined = np.zeros(n, bool), np.zeros(n, bool)
    for t in range(EVENT_LOOKBACK, n):
        prev = auc[t - EVENT_LOOKBACK:t]
        if np.isfinite(auc[t]) and np.isfinite(prev).all():
            defined[t] = True
            ev[t] = auc[t] <= prev.mean() - EVENT_DELTA
    return ev, defined


def h3_evaluate(alert, alert_def, event, event_def, eligible, seed=0):
    """Do alerts at t-1/t-2 hit degradation events more often than shuffled alerts?"""
    n = len(alert)
    cand = [t for t in range(2, n)
            if eligible[t] and event_def[t] and alert_def[t - 1] and alert_def[t - 2]]
    out = {"n_candidate_windows": len(cand), "n_events": int(sum(event[t] for t in cand))}
    if not cand:
        return {**out, "underpowered": True}
    pool = sorted({t - k for t in cand for k in (1, 2)})
    events = [t for t in cand if event[t]]

    def hit_rate(a):
        if not events:
            return np.nan
        return float(np.mean([a[t - 1] or a[t - 2] for t in events]))
    obs = hit_rate(alert)
    alerts_in_pool = [s for s in pool if alert[s] and (s + 1) in cand and (s + 2) in cand]
    precision = (float(np.mean([event[s + 1] or event[s + 2] for s in alerts_in_pool]))
                 if alerts_in_pool else np.nan)
    rng = np.random.default_rng(seed)
    vals = alert[pool].copy()
    null = np.empty(N_PERM_H3)
    pos = {s: i for i, s in enumerate(pool)}
    for i in range(N_PERM_H3):
        rng.shuffle(vals)
        shuffled = np.zeros(n, bool)
        shuffled[pool] = vals
        null[i] = hit_rate(shuffled)
    out.update({
        "event_base_rate": len(events) / len(cand), "n_alerts_in_pool": int(alert[pool].sum()),
        "pool_size": len(pool), "hit_rate": obs, "precision": precision,
        "false_alarm_rate": (1 - precision) if np.isfinite(precision) else np.nan,
        "null_hit_rate_mean": float(np.nanmean(null)),
        "p_one_sided": float(np.mean(null >= obs - 1e-12)) if np.isfinite(obs) else np.nan,
        "underpowered": len(events) < 5,
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: compute
# ══════════════════════════════════════════════════════════════════════════════
def make_windows(d):
    min_d, max_d = d.min(), d.max()
    wins, start = [], min_d
    while True:
        end = start + pd.DateOffset(months=WINDOW_MONTHS)
        eval_end = end + pd.DateOffset(months=STRIDE_MONTHS)
        if eval_end > max_d:
            break
        wins.append({"train_start": start, "train_end": end, "eval_start": end,
                     "eval_end": eval_end, "label": end.strftime("%Y-%m")})
        start += pd.DateOffset(months=STRIDE_MONTHS)
    return wins


def load_vix(path):
    v = pd.read_csv(path)
    v["DATE"] = pd.to_datetime(v["DATE"], format="%m/%d/%Y")
    return v.set_index("DATE")["CLOSE"].sort_index()


def explain(model, X_bg, X_s):
    import shap
    ex = shap.TreeExplainer(model, data=X_bg, model_output="probability",
                            feature_perturbation="interventional")
    sv = ex.shap_values(X_s, check_additivity=False)
    return (sv[1] if isinstance(sv, list) else sv).astype(np.float32)


def boot_auc_ci(y, p, rng):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    n, vals = len(y), []
    for _ in range(N_BOOT_AUC):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def compute(args):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    tr = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    te = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
    df = pd.concat([tr, te], ignore_index=True)
    df[DATE] = pd.to_datetime(df[DATE])
    # Same (default, non-stable) sort as v1's RollingWindowManager so that .sample(random_state=42)
    # draws the same loans and v2 reproduces v1's XSI.
    df = df.sort_values(DATE).reset_index(drop=True)
    feats = [c for c in df.columns if c not in EXCLUDE]
    dates = df[DATE].values
    vix = load_vix(args.vix)
    wins = make_windows(df[DATE])
    print(f"{len(df):,} loans, {len(feats)} features, {len(wins)} windows", flush=True)
    only = {int(i) for i in args.only.split(",")} if args.only else None

    for wi, w in enumerate(wins):
        if only is not None and wi not in only:
            continue
        cache_f = os.path.join(args.out, "cache", f"w{wi:02d}.pkl")
        if os.path.exists(cache_f):
            continue
        t0 = time.time()
        i0, i1, i2 = (np.searchsorted(dates, np.datetime64(w[k]), side="left")
                      for k in ("train_start", "train_end", "eval_end"))
        trw, evw = df.iloc[i0:i1], df.iloc[i1:i2]
        if len(trw) < MIN_TRAIN or len(evw) < MIN_EVAL:
            continue
        Xtr, ytr = trw[feats], trw[TARGET].values
        Xev, yev = evw[feats], evw[TARGET].values

        rec = {"window_index": wi, "label": w["label"], "train_start": w["train_start"],
               "train_end": w["train_end"], "eval_start": w["eval_start"],
               "eval_end": w["eval_end"], "train_size": len(trw), "eval_size": len(evw),
               "train_default_rate": float(ytr.mean()), "eval_default_rate": float(yev.mean()),
               "vix_train_mean": float(vix[(vix.index >= w["train_start"]) & (vix.index < w["train_end"])].mean()),
               "vix_eval_mean": float(vix[(vix.index >= w["eval_start"]) & (vix.index < w["eval_end"])].mean())}

        # Original model (seed 42), same sampling seeds as v1.
        model = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, ytr)
        p = model.predict_proba(Xev)[:, 1]
        rec["eval_auc"] = float(roc_auc_score(yev, p)) if len(np.unique(yev)) > 1 else np.nan
        rec["auc_ci_lo"], rec["auc_ci_hi"] = boot_auc_ci(yev, p, np.random.default_rng(0))
        Xs = Xev.sample(min(SHAP_SAMPLES, len(Xev)), random_state=42)
        bg = Xtr.sample(min(BG_SAMPLES, len(Xtr)), random_state=42)
        rec["shap_orig"] = explain(model, bg, Xs)

        # Noise-floor replicates: 80% training subsample, different seed, independent eval draw.
        reps, rep_auc = [], []
        for r in range(1, N_REPLICATES + 1):
            seed = 1000 + r
            idx = np.random.default_rng(seed).choice(len(Xtr), int(REPLICATE_FRACTION * len(Xtr)),
                                                    replace=False)
            Xr, yr = Xtr.iloc[idx], ytr[idx]
            mr = xgb.XGBClassifier(**{**XGB_PARAMS, "random_state": seed}).fit(Xr, yr)
            rep_auc.append(float(roc_auc_score(yev, mr.predict_proba(Xev)[:, 1]))
                           if len(np.unique(yev)) > 1 else np.nan)
            Xs_r = Xev.sample(min(SHAP_SAMPLES, len(Xev)), random_state=seed)
            bg_r = Xr.sample(min(BG_SAMPLES, len(Xr)), random_state=seed)
            reps.append(explain(mr, bg_r, Xs_r))
        rec["shap_reps"], rec["rep_auc"] = np.stack(reps), rep_auc
        rec["features"] = feats
        with open(cache_f, "wb") as f:
            pickle.dump(rec, f)
        print(f"window {wi:02d} {w['label']} train={len(trw):,} eval={len(evw):,} "
              f"AUC={rec['eval_auc']:.3f} done in {time.time()-t0:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: analyze
# ══════════════════════════════════════════════════════════════════════════════
def load_cache(out):
    d = os.path.join(out, "cache")
    recs = [pickle.load(open(os.path.join(d, f), "rb")) for f in sorted(os.listdir(d)) if f.endswith(".pkl")]
    return sorted(recs, key=lambda r: r["window_index"])


def build_table(recs):
    rows = []
    for t, r in enumerate(recs):
        row = {k: r[k] for k in ("window_index", "label", "train_start", "train_end", "train_size",
                                 "eval_size", "train_default_rate", "eval_default_rate",
                                 "vix_train_mean", "vix_eval_mean", "eval_auc", "auc_ci_lo", "auc_ci_hi")}
        selfs = [stability_metrics(r["shap_orig"], rep) for rep in r["shap_reps"]]
        for m in METRICS:
            vals = [s[m] for s in selfs]
            row[f"{m}_self"] = float(np.mean(vals))
            row[f"{m}_self_min"], row[f"{m}_self_max"] = float(np.min(vals)), float(np.max(vals))
            if t > 0:
                row[f"{m}_consec"] = stability_metrics(recs[t - 1]["shap_orig"], r["shap_orig"])[m]
            else:
                row[f"{m}_consec"] = np.nan
            row[f"{m}_excess"] = row[f"{m}_self"] - row[f"{m}_consec"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["vix_delta_abs"] = df["vix_train_mean"].diff().abs()
    df["log_train_size"] = np.log(df["train_size"])
    df["auc_next_change"] = df["eval_auc"].shift(-1) - df["eval_auc"]
    return df


def fmt_p(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.3f}"


def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    recs = load_cache(out)
    df = build_table(recs)
    df.to_csv(os.path.join(out, "windows_v2.csv"), index=False)
    n = len(df)
    plan_hash = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "ANALYSIS_PLAN.md"), "rb").read()).hexdigest()
    big = (df["eval_size"] >= MIN_EVAL_FOR_AUC_TESTS).values

    # ── v1 reproduction check ──────────────────────────────────────────────────
    repro = None
    if args.v1 and os.path.exists(args.v1):
        v1 = pd.read_csv(args.v1)
        m = df.merge(v1[["window_label", "xsi"]], left_on="label", right_on="window_label")
        m = m.dropna(subset=["xsi", "xsi_orig_consec"])
        repro = {"n": len(m), "max_abs_diff": float((m["xsi"] - m["xsi_orig_consec"]).abs().max()),
                 "corr": float(np.corrcoef(m["xsi"], m["xsi_orig_consec"])[0, 1])}

    # ── Primary hypotheses (metric = xsi_abs) ──────────────────────────────────
    x_excess = df[f"{PRIMARY}_excess"].values
    h1 = circ_perm_spearman(x_excess, df["vix_train_mean"].values)
    sel = big & np.roll(big, -1)  # t and t+1 both have large eval sets
    sel[-1] = False
    idx = np.where(sel)[0]
    if len(idx) and (np.diff(idx) == 1).all():
        h2 = circ_perm_spearman(df[f"{PRIMARY}_consec"].values[idx], df["auc_next_change"].values[idx])
    else:
        h2 = {"rho": np.nan, "p": np.nan, "n": int(len(idx))}
    h2_all = circ_perm_spearman(df[f"{PRIMARY}_consec"].values, df["auc_next_change"].values)

    alert, alert_def = eart_v2_alerts(df[f"{PRIMARY}_consec"].values)
    event, event_def = degradation_events(df["eval_auc"].values)
    h3 = h3_evaluate(alert, alert_def, event, event_def, big)
    h3_all = h3_evaluate(alert, alert_def, event, event_def, np.ones(n, bool))

    p_adj = holm([h1["p"], h2["p"], h3.get("p_one_sided", np.nan)
                  if np.isfinite(h3.get("p_one_sided", np.nan)) else 1.0])
    df["alert_eart_v2"], df["event_degradation"] = alert, event
    df.to_csv(os.path.join(out, "windows_v2.csv"), index=False)

    # ── Secondary / exploratory ────────────────────────────────────────────────
    sec = []
    for mname in METRICS:
        r1 = circ_perm_spearman(df[f"{mname}_excess"].values, df["vix_train_mean"].values)
        r1p = partial_circ_perm(df[f"{mname}_excess"].values, df["vix_train_mean"].values, df["log_train_size"].values)
        r1v = circ_perm_spearman(df[f"{mname}_consec"].values, df["vix_train_mean"].values)
        r1d = circ_perm_spearman(df[f"{mname}_consec"].values, df["vix_delta_abs"].values)
        big_idx = np.where(big & (df["train_size"].values >= 20000))[0]
        r1big = (circ_perm_spearman(df[f"{mname}_excess"].values[big_idx], df["vix_train_mean"].values[big_idx])
                 if len(big_idx) and (np.diff(big_idx) == 1).all() else {"rho": np.nan, "p": np.nan, "n": len(big_idx)})
        r2 = circ_perm_spearman(df[f"{mname}_consec"].values[idx], df["auc_next_change"].values[idx]) \
            if len(idx) and (np.diff(idx) == 1).all() else {"rho": np.nan, "p": np.nan, "n": 0}
        sec.append({"metric": mname,
                    "H1 excess~VIX rho": r1["rho"], "H1 p": r1["p"],
                    "H1 partial(size) rho": r1p["rho"], "H1 partial p": r1p["p"],
                    "H1 train>=20k rho": r1big["rho"], "H1 train>=20k p": r1big["p"],
                    "XSI~VIX rho": r1v["rho"], "XSI~VIX p": r1v["p"],
                    "XSI~|dVIX| rho": r1d["rho"], "XSI~|dVIX| p": r1d["p"],
                    "H2 rho": r2["rho"], "H2 p": r2["p"]})
    sec = pd.DataFrame(sec)
    sec.to_csv(os.path.join(out, "secondary_tests.csv"), index=False)

    pair = df.dropna(subset=[f"{PRIMARY}_consec"]).copy()
    q1, q2 = pair["vix_train_mean"].quantile([1 / 3, 2 / 3])
    pair["vix_regime"] = np.where(pair["vix_train_mean"] <= q1, "Calm",
                                  np.where(pair["vix_train_mean"] >= q2, "Stress", "Mid"))
    terc = pair.groupby("vix_regime")[[f"{PRIMARY}_consec", f"{PRIMARY}_excess", "vix_train_mean",
                                       "eval_auc", "train_size"]].agg(["mean", "std", "count"]).round(4)
    terc.columns = ["_".join(c) for c in terc.columns]
    terc = terc.reset_index()
    terc.to_csv(os.path.join(out, "vix_tercile_table.csv"), index=False)
    kw = {}
    for col in (f"{PRIMARY}_consec", f"{PRIMARY}_excess"):
        groups = [g[col].values for _, g in pair.groupby("vix_regime")]
        kw[col] = kruskal(*groups).pvalue if len(groups) == 3 else np.nan
    noise = pair[[f"{m}_self" for m in METRICS]].mean()
    obs = pair[[f"{m}_consec" for m in METRICS]].mean()

    # ── Figures ────────────────────────────────────────────────────────────────
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    x = pd.to_datetime(df["train_end"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(x, df[f"{PRIMARY}_self_min"], df[f"{PRIMARY}_self_max"], color="#9ecae1", alpha=0.7,
                    label="Noise floor (3 replicate models, min-max)")
    ax.plot(x, df[f"{PRIMARY}_self"], color="#3182bd", lw=1.2)
    ax.plot(x, df[f"{PRIMARY}_consec"], color="#222222", lw=1.8, marker="o", ms=3, label="Consecutive-window XSI")
    if alert.any():
        ax.scatter(x[alert], df[f"{PRIMARY}_consec"][alert], marker="v", s=70, color="#d62728", zorder=5, label="EART-v2 alert")
    ax.set_ylabel("XSI (abs-profile variant)")
    ax.set_xlabel("Window end date")
    ax2 = ax.twinx()
    ax2.plot(x, df["vix_train_mean"], color="#e6850e", lw=1.2, ls="--", label="Mean VIX (training window)")
    ax2.set_ylabel("VIX", color="#e6850e")
    ax2.spines["top"].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    h2_, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h + h2_, l + l2_, loc="lower left", fontsize=8, ncol=2)
    ax.set_title("Explanation stability vs. its noise floor, with market stress (VIX)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "v2_fig1_xsi_vs_noise_floor.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = axes[0].scatter(df["vix_train_mean"], x_excess, c=np.log10(df["train_size"]), cmap="viridis", s=40)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("Mean VIX over training window")
    axes[0].set_ylabel("Excess drift = noise floor - XSI")
    axes[0].set_title(f"H1: rho={h1['rho']:.2f}, circular-shift p={fmt_p(h1['p'])}")
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("log10(train size)")
    d2 = df.iloc[idx] if len(idx) else df.iloc[:0]
    axes[1].scatter(d2[f"{PRIMARY}_consec"], d2["auc_next_change"], s=40, color="#444")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("XSI at window t")
    axes[1].set_ylabel("AUC(t+1) - AUC(t)")
    axes[1].set_title(f"H2 (eval>=10k): rho={h2['rho']:.2f}, p={fmt_p(h2['p'])}")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "v2_fig2_hypothesis_scatter.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.fill_between(x, df["auc_ci_lo"], df["auc_ci_hi"], color="#c7b9e0", alpha=0.7, label="AUC 95% bootstrap CI")
    ax.plot(x, df["eval_auc"], color="#6a3d9a", lw=1.6, label="Eval AUC")
    ev_mask = event & big
    ax.scatter(x[ev_mask], df["eval_auc"][ev_mask], marker="X", s=90, color="#d62728", zorder=5,
               label="Degradation event (>=0.02 below trailing mean)")
    ax.scatter(x[alert], np.full(alert.sum(), df["eval_auc"].min() - 0.01), marker="^", s=60,
               color="#222", label="EART-v2 alert")
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Window end date")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_title("Performance with uncertainty, degradation events, and explanation alerts")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "v2_fig3_auc_events_alerts.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (col_r, col_p, title) in zip(axes, [("H1 excess~VIX rho", "H1 p", "H1: excess drift vs VIX"),
                                                ("H2 rho", "H2 p", "H2: XSI vs next AUC change")]):
        ys = np.arange(len(sec))[::-1]
        ax.axvline(0, color="grey", lw=0.8)
        ax.scatter(sec[col_r], ys, color=["#d62728" if p < 0.05 else "#555" for p in sec[col_p]], s=60)
        for yy, rr, pp in zip(ys, sec[col_r], sec[col_p]):
            ax.text(rr + 0.02, yy + 0.15, f"p={fmt_p(pp)}", fontsize=8)
        ax.set_yticks(ys)
        ax.set_yticklabels(sec["metric"])
        ax.set_xlabel("Spearman rho")
        ax.set_title(title)
        ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "v2_fig4_metric_variants.png"), dpi=200)
    plt.close(fig)

    # ── RESULTS.md ─────────────────────────────────────────────────────────────
    def verdict(name, p_adj_val, rho, want_positive=True):
        if not np.isfinite(rho) or not np.isfinite(p_adj_val):
            return "not testable (insufficient data)"
        if p_adj_val < 0.05 and ((rho > 0) == want_positive):
            return "SUPPORTED (Holm-adjusted p < 0.05)"
        if p_adj_val < 0.05:
            return "CONTRADICTED (significant, opposite sign)"
        return "NOT SUPPORTED (no significant effect after Holm correction)"

    L = []
    L.append("# XDrift v2 — Results\n")
    L.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} · windows analysed: {n} · "
             f"Python {platform.python_version()} · analysis plan SHA-256 `{plan_hash[:16]}…`*\n")
    L.append("Plan: [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md). Everything below was computed by "
             "`xdrift_v2.py`; no variant was dropped.\n")
    if repro:
        L.append("## Sanity check: v2 reproduces the v1 XSI\n")
        L.append(f"On {repro['n']} windows, v2's `xsi_orig` for consecutive windows vs. the v1 notebook's XSI: "
                 f"correlation {repro['corr']:.4f}, max |difference| {repro['max_abs_diff']:.4f}.\n")
    L.append("## Primary hypotheses (metric `xsi_abs`, Holm-corrected across 3 tests)\n")
    L.append("| Hypothesis | n | Spearman rho | circ-shift p | Holm-adj p | Verdict |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| **H1** excess drift rises with VIX | {h1['n']} | {h1['rho']:.3f} | {fmt_p(h1['p'])} | {fmt_p(p_adj[0])} | {verdict('H1', p_adj[0], h1['rho'])} |")
    L.append(f"| **H2** low XSI precedes AUC decline (eval≥10k) | {h2['n']} | {h2['rho']:.3f} | {fmt_p(h2['p'])} | {fmt_p(p_adj[1])} | {verdict('H2', p_adj[1], h2['rho'])} |")
    h3p = h3.get("p_one_sided", np.nan)
    if h3.get("underpowered", True):
        h3v = "UNDERPOWERED (<5 events) — cannot be resolved"
    else:
        h3v = verdict("H3", p_adj[2], 1.0 if h3.get("hit_rate", 0) > h3.get("null_hit_rate_mean", 0) else -1.0)
    L.append(f"| **H3** EART-v2 alerts precede degradation events | {h3.get('n_events', 0)} events | hit rate {h3.get('hit_rate', float('nan')):.2f} vs null {h3.get('null_hit_rate_mean', float('nan')):.2f} | {fmt_p(h3p)} | {fmt_p(p_adj[2])} | {h3v} |")
    L.append(f"\nH2 using all windows (incl. small early ones): rho={h2_all['rho']:.3f}, p={fmt_p(h2_all['p'])}, n={h2_all['n']}.\n")
    L.append("### H3 detail\n")
    L.append("| quantity | eval≥10k (primary) | all windows |")
    L.append("|---|---|---|")
    for k in ("n_candidate_windows", "n_events", "event_base_rate", "n_alerts_in_pool", "hit_rate", "precision",
              "false_alarm_rate", "null_hit_rate_mean", "p_one_sided"):
        a, b = h3.get(k, np.nan), h3_all.get(k, np.nan)
        L.append(f"| {k} | {a:.3f} | {b:.3f} |" if isinstance(a, float) else f"| {k} | {a} | {b} |")
    L.append("\n## Noise floor\n")
    L.append("Mean similarity between the original model and replicate models (retraining + sampling noise only) "
             "vs. between consecutive windows:\n")
    L.append("| metric | noise floor | consecutive windows | gap |")
    L.append("|---|---|---|---|")
    for mname in METRICS:
        L.append(f"| {mname} | {noise[f'{mname}_self']:.3f} | {obs[f'{mname}_consec']:.3f} | {noise[f'{mname}_self'] - obs[f'{mname}_consec']:.3f} |")
    L.append("\n## Real-VIX regimes (terciles of mean VIX over the training window)\n")
    L.append(terc.to_markdown(index=False))
    L.append(f"\nKruskal-Wallis across terciles: XSI p={fmt_p(kw[f'{PRIMARY}_consec'])}, excess drift p={fmt_p(kw[f'{PRIMARY}_excess'])}. "
             "Note: windows in the Stress tercile are also the smallest (2008-2010), so XSI alone is confounded "
             "with sample size; excess drift and the partial correlations below address this.\n")
    L.append("## All metric variants (secondary, uncorrected; circular-shift p-values)\n")
    L.append(sec.round(3).to_markdown(index=False))
    L.append("\n## Figures\n")
    for f in ("v2_fig1_xsi_vs_noise_floor", "v2_fig2_hypothesis_scatter", "v2_fig3_auc_events_alerts", "v2_fig4_metric_variants"):
        L.append(f"![{f}](figures/{f}.png)\n")
    L.append("## Limitations\n")
    L.append("- 42 overlapping windows; effective sample size is smaller than n. The circular-shift test accounts for "
             "autocorrelation, but on synthetic independent random walks of the same length it still rejects at ~21% "
             "(nominal 5%; see `test_xdrift_v2.py`), versus >50% for a naive test. Treat p-values near 0.05 as weak evidence.\n"
             "- The secondary table reports 4 metrics x 6 tests = 24 uncorrected p-values; a few values near 0.05 are expected "
             "by chance and are **not** evidence. Only the three primary tests are corrected (Holm).\n"
             "- Windows in the VIX Stress tercile (2008-2010) are also the smallest, so stress and sample size are partly "
             "confounded even after noise-floor adjustment; partial correlations on log(train size) are reported.\n"
             "- The H3 early-warning question has very few degradation events, so it cannot be resolved with this data.\n"
             "- Windows ending before 2012 have under 25k training loans, so they are noisy; the noise floor is meant to absorb this.\n"
             "- LendingClub is one lender and one decade, with a single true stress episode (2008-09) that coincides with tiny samples.\n"
             "- Only resolved loans are kept, so recent loans are biased toward early outcomes (right-censoring).\n"
             "- The v2 design (primary metric, VIX regimes) was motivated by v1 diagnostics; v1 results are kept for comparison.\n")
    open(os.path.join(out, "..", "RESULTS.md") if args.results_md is None else args.results_md, "w", encoding="utf-8").write("\n".join(L))

    print("H1", h1, "\nH2", h2, "\nH3", h3, "\nHolm", p_adj, "\nrepro", repro)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["compute", "analyze", "all"])
    ap.add_argument("--data-dir", default=os.path.expanduser("~/xdrift_out"))
    ap.add_argument("--vix", default=os.path.join("data", "fred", "VIXCLS_cboe.csv"))
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--v1", default=None, help="v1 xdrift_results.csv for the reproduction check")
    ap.add_argument("--only", default=None, help="comma-separated window indices (smoke tests)")
    ap.add_argument("--results-md", default=None)
    args = ap.parse_args()
    if args.stage in ("compute", "all"):
        compute(args)
    if args.stage in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
