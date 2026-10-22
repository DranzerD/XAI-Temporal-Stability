#!/usr/bin/env python
"""
XDrift v3 -- frozen-model monitoring: does explanation drift lead performance decay,
and does it add anything over standard drift monitors?   (See ANALYSIS_PLAN_V3.md)

  compute  For each quarterly origin T: train one XGBoost model on the 12 months ending at T,
           freeze it, and measure on each of the next 7 quarters (quarter 0 = reference):
           AUC, TreeSHAP draws, score PSI, KS, per-feature PSI.
  analyze  Build the (model, quarter) panel, run the three pre-declared tests with
           model-swap permutation inference and cluster bootstrap, make figures and RESULTS_V3.md.

    python xdrift_v3.py all --data-dir <dir with train/test parquet> --out results_v3
"""
import argparse
import hashlib
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score

import xdrift_v2 as v2

ORIGINS = list(pd.date_range("2012-12-01", "2016-12-01", freq="3MS"))   # 17 quarterly origins
PRIMARY_LAST_ORIGIN = pd.Timestamp("2015-12-01")                          # 13 primary models
K = 6                                                                     # monitored quarters
SHAP_N, BG_N = 500, 100
EVENT_THRESHOLD = 0.02
N_PERM, N_BOOT = 10_000, 2_000
N_PERM_SEC, N_BOOT_SEC = 3_000, 1_000
MIN_EVENTS = 10
EXPL = ["xsi_abs", "xsi_orig", "tau_only", "jaccard10"]
BASELINES = ["score_psi", "feat_psi_top15", "feat_psi_all", "feat_psi_nomacro", "score_ks", "score_shift"]
MACRO_FEATURES = {"fed_funds_rate", "vix_bucket"}   # constant within a quarter -> PSI explodes
MONITORS = [f"{m}_excess" for m in EXPL] + BASELINES
PRIMARY_EXPL, PRIMARY_BASE = "xsi_abs_excess", "score_psi"


# ══════════════════════════════════════════════════════════════════════════════
# Baseline drift monitors
# ══════════════════════════════════════════════════════════════════════════════
def feature_psi(ref, cur, n_bins=10, eps=1e-4):
    """PSI of cur vs ref. Low-cardinality features use their value frequencies; the rest use
    reference deciles with open-ended outer bins."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    u = np.unique(ref)
    edges = None
    if len(u) > 10:
        edges = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if edges is None or len(edges) < 3:
        vals = np.union1d(u, np.unique(cur))
        a = np.array([(ref == x).mean() for x in vals])
        b = np.array([(cur == x).mean() for x in vals])
    else:
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(ref, bins=edges)[0] / len(ref)
        b = np.histogram(cur, bins=edges)[0] / len(cur)
    a, b = a + eps, b + eps
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════
def _partial(rx, ry, h):
    """Pearson correlation of x and y after regressing both on h (inputs already ranked)."""
    hc = h - h.mean()
    denom = (hc ** 2).sum()
    ex = rx - rx.mean() - (rx * hc).sum() / denom * hc
    ey = ry - ry.mean() - (ry * hc).sum() / denom * hc
    sx, sy = ex.std(), ey.std()
    return float(np.mean(ex * ey) / (sx * sy)) if sx > 0 and sy > 0 else np.nan


def partial_spearman(X, Y, H, rng, n_perm=N_PERM, n_boot=N_BOOT):
    """Partial Spearman of pooled (model x quarter) panels X, Y given horizon H.
    p-value: model-swap permutation (Y's rows reassigned to other models).
    CI: cluster bootstrap over models."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    M, Kk = X.shape
    Hm = np.tile(np.asarray(H, float), (M, 1))
    rx, ry = rankdata(X.ravel()).reshape(M, Kk), rankdata(Y.ravel()).reshape(M, Kk)
    obs = _partial(rx.ravel(), ry.ravel(), Hm.ravel())
    null = np.array([_partial(rx.ravel(), ry[rng.permutation(M)].ravel(), Hm.ravel()) for _ in range(n_perm)])
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, M, M)
        boots.append(_partial(rankdata(X[i].ravel()), rankdata(Y[i].ravel()), Hm[i].ravel()))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": obs, "p": p, "ci_lo": float(lo), "ci_hi": float(hi), "n_models": M}


def auroc(score, event):
    score, event = np.asarray(score).ravel(), np.asarray(event).ravel().astype(int)
    if event.min() == event.max():
        return np.nan
    return float(roc_auc_score(event, score))


def cluster_boot_auroc(S, E, rng, n_boot=N_BOOT, S2=None):
    """AUROC (and optional difference S - S2) with a cluster bootstrap over models (rows)."""
    M = S.shape[0]
    obs = auroc(S, E) if S2 is None else auroc(S, E) - auroc(S2, E)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, M, M)
        a = auroc(S[i], E[i])
        if S2 is not None:
            a = a - auroc(S2[i], E[i])
        if np.isfinite(a):
            vals.append(a)
    vals = np.array(vals)
    if len(vals) < 50:
        return {"auroc": obs, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n_boot_ok": len(vals)}
    null_ref = 0.5 if S2 is None else 0.0
    p = min(1.0, 2 * min(np.mean(vals <= null_ref), np.mean(vals >= null_ref)))
    return {"auroc": obs, "ci_lo": float(np.percentile(vals, 2.5)), "ci_hi": float(np.percentile(vals, 97.5)),
            "p": float(p), "n_boot_ok": len(vals)}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: compute
# ══════════════════════════════════════════════════════════════════════════════
def load_df(data_dir):
    tr = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    te = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    df = pd.concat([tr, te], ignore_index=True)
    df[v2.DATE] = pd.to_datetime(df[v2.DATE])
    return df.sort_values(v2.DATE).reset_index(drop=True)


def compute(args):
    import xgboost as xgb
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    df = load_df(args.data_dir)
    feats = [c for c in df.columns if c not in v2.EXCLUDE]
    dates = df[v2.DATE].values
    only = {int(i) for i in args.only.split(",")} if args.only else None
    params = {**v2.XGB_PARAMS, "n_jobs": args.n_jobs}

    def rows(a, b):
        i0, i1 = (np.searchsorted(dates, np.datetime64(x), side="left") for x in (a, b))
        return df.iloc[i0:i1]

    for oi, T in enumerate(ORIGINS):
        if only is not None and oi not in only:
            continue
        cache_f = os.path.join(args.out, "cache", f"o{oi:02d}.pkl")
        if os.path.exists(cache_f):
            continue
        t0 = time.time()
        off = lambda k: T + pd.DateOffset(months=3 * k)
        trw = rows(T - pd.DateOffset(months=12), T)
        Xtr, ytr = trw[feats], trw[v2.TARGET].values
        model = xgb.XGBClassifier(**params).fit(Xtr, ytr)
        bg = Xtr.sample(BG_N, random_state=42)

        ref = rows(off(0), off(1))
        Xref, yref = ref[feats], ref[v2.TARGET].values
        p_ref = model.predict_proba(Xref)[:, 1]
        ref_draw = Xref.sample(4 * SHAP_N, random_state=42)
        ref_shap = [v2.explain(model, bg, ref_draw.iloc[j * SHAP_N:(j + 1) * SHAP_N]) for j in range(4)]

        rec = {"origin_index": oi, "origin": T, "train_size": len(trw), "features": feats,
               "importances": model.feature_importances_.astype(float),
               "ref": {"n": len(ref), "auc": float(roc_auc_score(yref, p_ref)),
                       "default_rate": float(yref.mean())},
               "ref_shap": ref_shap, "quarters": []}
        for k in range(1, K + 1):
            q = rows(off(k), off(k + 1))
            Xq, yq = q[feats], q[v2.TARGET].values
            p_q = model.predict_proba(Xq)[:, 1]
            rec["quarters"].append({
                "k": k, "n": len(q), "default_rate": float(yq.mean()), "auc": float(roc_auc_score(yq, p_q)),
                "score_psi": feature_psi(p_ref, p_q), "score_ks": float(ks_2samp(p_ref, p_q).statistic),
                "score_shift": float(abs(p_q.mean() - p_ref.mean())),
                "feat_psi": np.array([feature_psi(Xref[c].values, Xq[c].values) for c in feats]),
                "shap": v2.explain(model, bg, Xq.sample(min(SHAP_N, len(Xq)), random_state=100 + k)),
            })
        with open(cache_f, "wb") as f:
            pickle.dump(rec, f)
        print(f"origin {oi:02d} {T:%Y-%m} train={len(trw):,} ref_auc={rec['ref']['auc']:.3f} "
              f"done in {time.time() - t0:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: analyze
# ══════════════════════════════════════════════════════════════════════════════
def build_panel(recs):
    rows = []
    for rec in recs:
        R = rec["ref_shap"]
        f1, f2 = v2.stability_metrics(R[0], R[1]), v2.stability_metrics(R[2], R[3])
        floor = {m: (f1[m] + f2[m]) / 2 for m in EXPL}
        top = np.argsort(rec["importances"])[::-1][:v2.TOP_K]
        nomacro = [i for i, c in enumerate(rec["features"]) if c not in MACRO_FEATURES]
        for q in rec["quarters"]:
            met = v2.stability_metrics(R[0], q["shap"])
            row = {"origin_index": rec["origin_index"], "origin": rec["origin"], "k": q["k"],
                   "primary": rec["origin"] <= PRIMARY_LAST_ORIGIN, "train_size": rec["train_size"],
                   "ref_n": rec["ref"]["n"], "q_n": q["n"], "auc_ref": rec["ref"]["auc"], "auc": q["auc"],
                   "decay": rec["ref"]["auc"] - q["auc"], "default_rate": q["default_rate"],
                   "score_psi": q["score_psi"], "score_ks": q["score_ks"], "score_shift": q["score_shift"],
                   "feat_psi_top15": float(np.mean(q["feat_psi"][top])), "feat_psi_all": float(np.mean(q["feat_psi"])),
                   "feat_psi_nomacro": float(np.mean(q["feat_psi"][nomacro]))}
            for m in EXPL:
                row[f"{m}_metric"], row[f"{m}_floor"] = met[m], floor[m]
                row[f"{m}_excess"] = floor[m] - met[m]
            rows.append(row)
    return pd.DataFrame(rows)


def matrices(panel, col):
    return panel.pivot(index="origin_index", columns="k", values=col).sort_index().values


def run_panel(panel, thr, seed, n_perm, n_boot, monitors):
    """All monitor statistics for one panel (set of models) at one event threshold."""
    rng = np.random.default_rng(seed)
    decay, Ev = matrices(panel, "decay"), matrices(panel, "decay") >= thr
    ks = np.arange(1, K + 1)
    out = {}
    for mon in monitors:
        X = matrices(panel, mon)
        conc = partial_spearman(X, decay, ks, rng, n_perm, n_boot)
        lead = partial_spearman(X[:, :-1], decay[:, 1:], ks[:-1], rng, n_perm, n_boot)
        a_c = cluster_boot_auroc(X, Ev, rng, n_boot)
        a_l = cluster_boot_auroc(X[:, :-1], Ev[:, 1:], rng, n_boot)
        out[mon] = {"conc": conc, "lead": lead, "auroc_conc": a_c, "auroc_lead": a_l}
    n_ev_lead = int(Ev[:, 1:].sum())
    d_auc = cluster_boot_auroc(matrices(panel, PRIMARY_EXPL)[:, :-1], Ev[:, 1:], rng, n_boot,
                               S2=matrices(panel, PRIMARY_BASE)[:, :-1])
    return {"monitors": out, "delta_auroc": d_auc, "n_events_lead": n_ev_lead, "n_events_conc": int(Ev.sum()),
            "n_cells": int(decay.size), "n_models": int(decay.shape[0])}


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    d = os.path.join(out, "cache")
    recs = sorted((pickle.load(open(os.path.join(d, f), "rb")) for f in os.listdir(d) if f.endswith(".pkl")),
                  key=lambda r: r["origin_index"])
    panel = build_panel(recs)
    panel.to_csv(os.path.join(out, "panel_v3.csv"), index=False)
    prim = panel[panel["primary"]]
    plan_hash = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "ANALYSIS_PLAN_V3.md"), "rb").read()).hexdigest()

    # ── Primary family (13 models) ─────────────────────────────────────────────
    res = run_panel(prim, EVENT_THRESHOLD, 0, N_PERM, N_BOOT, [PRIMARY_EXPL, PRIMARY_BASE])
    P1, P2 = res["monitors"][PRIMARY_EXPL]["conc"], res["monitors"][PRIMARY_EXPL]["lead"]
    P3 = res["delta_auroc"]
    p_vals = [P1["p"], P2["p"], P3["p"] if np.isfinite(P3["p"]) else 1.0]
    p_adj = v2.holm(p_vals)

    # ── Secondary: every monitor, 3 thresholds, all 17 models ─────────────────
    sec_main = run_panel(prim, EVENT_THRESHOLD, 1, N_PERM_SEC, N_BOOT_SEC, MONITORS)
    sec_all = run_panel(panel, EVENT_THRESHOLD, 2, N_PERM_SEC, N_BOOT_SEC, MONITORS)
    sens = {thr: run_panel(prim, thr, 3, N_PERM_SEC, N_BOOT_SEC, [PRIMARY_EXPL, PRIMARY_BASE]) for thr in (0.01, 0.03)}

    def table(r):
        rows = []
        for mon in MONITORS:
            m = r["monitors"][mon]
            rows.append({"monitor": mon,
                         "conc_rho": m["conc"]["rho"], "conc_p": m["conc"]["p"],
                         "conc_ci": f"[{fmt(m['conc']['ci_lo'], 2)}, {fmt(m['conc']['ci_hi'], 2)}]",
                         "lead_rho": m["lead"]["rho"], "lead_p": m["lead"]["p"],
                         "lead_ci": f"[{fmt(m['lead']['ci_lo'], 2)}, {fmt(m['lead']['ci_hi'], 2)}]",
                         "AUROC_conc": m["auroc_conc"]["auroc"],
                         "AUROC_lead": m["auroc_lead"]["auroc"],
                         "AUROC_lead_ci": f"[{fmt(m['auroc_lead']['ci_lo'], 2)}, {fmt(m['auroc_lead']['ci_hi'], 2)}]"})
        return pd.DataFrame(rows)
    t_main, t_all = table(sec_main), table(sec_all)
    t_main.to_csv(os.path.join(out, "monitors_primary_panel.csv"), index=False)
    t_all.to_csv(os.path.join(out, "monitors_all17.csv"), index=False)

    # ── Figures ────────────────────────────────────────────────────────────────
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [f"{o:%Y-%m}" for o in sorted(prim["origin"].unique())]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title, cmap in [(axes[0], "decay", "AUC decay vs. reference quarter", "RdBu_r"),
                                 (axes[1], PRIMARY_EXPL, "Explanation excess drift (noise-floor adjusted)", "viridis")]:
        Mx = matrices(prim, col)
        lim = np.nanmax(np.abs(Mx)) if col == "decay" else None
        im = ax.imshow(Mx, aspect="auto", cmap=cmap, vmin=-lim if lim else None, vmax=lim)
        ax.set_xticks(range(K)); ax.set_xticklabels(range(1, K + 1))
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Quarters since go-live (k)"); ax.set_ylabel("Model origin")
        ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "v3_fig1_decay_and_drift_heatmaps.png"), dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    ys = np.arange(len(MONITORS))[::-1]
    for ax, key, title in [(axes[0], "auroc_lead", "Flag next-quarter decay >= 0.02 (AUROC)"),
                           (axes[1], "lead", "Partial Spearman with next-quarter decay")]:
        ax.axvline(0.5 if key == "auroc_lead" else 0.0, color="grey", lw=0.8)
        for y, mon in zip(ys, MONITORS):
            r = sec_main["monitors"][mon][key]
            est = r["auroc"] if key == "auroc_lead" else r["rho"]
            if not all(np.isfinite([est, r["ci_lo"], r["ci_hi"]])):
                continue
            col ="#d62728" if mon in [f"{m}_excess" for m in EXPL] else "#1f77b4"
            ax.errorbar(est, y, xerr=[[est - r["ci_lo"]], [r["ci_hi"] - est]], fmt="o", color=col, capsize=3)
        ax.set_yticks(ys); ax.set_yticklabels(MONITORS, fontsize=8); ax.set_title(title, fontsize=10)
    axes[0].set_xlabel("AUROC (95% cluster-bootstrap CI)"); axes[1].set_xlabel("rho (95% cluster-bootstrap CI)")
    fig.suptitle("Red = explanation-drift monitors, blue = standard drift baselines", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "v3_fig2_monitor_comparison.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sc = ax.scatter(prim[PRIMARY_EXPL], prim["decay"], c=prim["k"], cmap="plasma", s=28)
    ax.axhline(0, color="grey", lw=0.8); ax.axhline(EVENT_THRESHOLD, color="#d62728", lw=0.8, ls="--")
    ax.set_xlabel("Explanation excess drift"); ax.set_ylabel("AUC decay vs. reference")
    ax.set_title(f"P1: partial rho = {fmt(P1['rho'], 2)}, p = {fmt(P1['p'])}", fontsize=10)
    fig.colorbar(sc, ax=ax).set_label("quarters since go-live")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "v3_fig3_drift_vs_decay.png"), dpi=200); plt.close(fig)

    # ── RESULTS_V3.md ──────────────────────────────────────────────────────────
    def verdict(adj_p, est, positive=True):
        if not np.isfinite(adj_p) or not np.isfinite(est):
            return "not testable"
        if adj_p < 0.05 and ((est > 0) == positive):
            return "SUPPORTED (Holm-adjusted p < 0.05)"
        if adj_p < 0.05:
            return "CONTRADICTED (significant, opposite sign)"
        return "NOT SUPPORTED (no significant effect after Holm correction)"
    under = res["n_events_lead"] < MIN_EVENTS
    L = ["# XDrift v3 — Results (frozen-model monitoring)\n",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M')} · Python {platform.python_version()} · "
         f"plan SHA-256 `{plan_hash[:16]}…`*\n",
         "Plan: [ANALYSIS_PLAN_V3.md](ANALYSIS_PLAN_V3.md). Primary panel: "
         f"{res['n_models']} frozen models × {K} quarters = {res['n_cells']} cells; "
         f"{res['n_events_conc']} decay events (≥ {EVENT_THRESHOLD} AUC) concurrent, {res['n_events_lead']} in the lead setting.\n",
         "## Primary hypotheses (Holm-corrected across 3 tests)\n",
         "| Hypothesis | estimate | 95% cluster-bootstrap CI | p | Holm-adj p | Verdict |", "|---|---|---|---|---|---|",
         f"| **P1** explanation drift tracks AUC decay (partial ρ, controlling horizon) | {fmt(P1['rho'])} | [{fmt(P1['ci_lo'], 2)}, {fmt(P1['ci_hi'], 2)}] | {fmt(P1['p'])} | {fmt(p_adj[0])} | {verdict(p_adj[0], P1['rho'])} |",
         f"| **P2** explanation drift leads next-quarter decay (partial ρ) | {fmt(P2['rho'])} | [{fmt(P2['ci_lo'], 2)}, {fmt(P2['ci_hi'], 2)}] | {fmt(P2['p'])} | {fmt(p_adj[1])} | {verdict(p_adj[1], P2['rho'])} |",
         f"| **P3** ΔAUROC: explanation drift minus score PSI, flagging next-quarter decay | {fmt(P3['auroc'])} | [{fmt(P3['ci_lo'], 2)}, {fmt(P3['ci_hi'], 2)}] | {fmt(P3['p'])} | {fmt(p_adj[2])} | "
         + ("UNDERPOWERED (<10 events)" if under else verdict(p_adj[2], P3["auroc"])) + " |", ""]
    L += ["## Every monitor vs. performance decay — primary panel (13 models)\n",
          "ρ = partial Spearman given horizon; AUROC = detecting decay ≥ 0.02 (conc = same quarter, lead = next quarter). "
          "Permutation p-values; **uncorrected**. Red/blue grouping is in Figure 2.\n",
          t_main.round(3).to_markdown(index=False), "",
          "## Sensitivity: all 17 models (adds origins 2016-03…2016-12, whose later quarters are right-censored)\n",
          t_all.round(3).to_markdown(index=False), "",
          "## Sensitivity: event threshold\n", "| threshold | events (lead) | ΔAUROC (expl − score PSI) | 95% CI |", "|---|---|---|---|"]
    for thr, r in sens.items():
        L.append(f"| {thr} | {r['n_events_lead']} | {fmt(r['delta_auroc']['auroc'])} | [{fmt(r['delta_auroc']['ci_lo'], 2)}, {fmt(r['delta_auroc']['ci_hi'], 2)}] |")
    # Post-hoc diagnostics (not in the plan; labelled as such): event clustering and within-model correlations.
    ev_by_origin = prim.assign(ev=prim["decay"] >= EVENT_THRESHOLD).groupby("origin")["ev"].sum()
    n_models_with_events = int((ev_by_origin > 0).sum())
    within = []
    for mon in [PRIMARY_EXPL, PRIMARY_BASE, "feat_psi_nomacro", "feat_psi_all"]:
        rs = np.array([pd.Series(g[mon].values).corr(pd.Series(g["decay"].values), method="spearman")
                       for _, g in prim.groupby("origin")])
        within.append({"monitor": mon, "mean_within_model_rho": np.nanmean(rs), "median": np.nanmedian(rs),
                       "models_with_positive_rho": f"{int(np.sum(rs > 0))}/{len(rs)}"})
    L += ["", "## Post-hoc diagnostics (not pre-declared)\n",
          f"**Event clustering.** The {res['n_events_conc']} decay events come from only **{n_models_with_events} of "
          f"{res['n_models']} models** ({', '.join(f'{o:%Y-%m}: {int(n)}' for o, n in ev_by_origin[ev_by_origin > 0].items())}); "
          "the other models never decay by ≥ 0.02 AUC. The number of independent events is therefore far smaller than the event count "
          "suggests, and decay is concentrated in late-period cohorts (evaluated in 2016-2017), a calendar/cohort effect rather than a drift-tracking one.\n",
          "**Within-model correlation** (Spearman across the 6 quarters of each model, then averaged over models; removes between-model differences):\n",
          pd.DataFrame(within).round(3).to_markdown(index=False), ""]
    L += ["", "## Figures\n"] + [f"![{n}](figures/{n}.png)\n" for n in
                                  ("v3_fig1_decay_and_drift_heatmaps", "v3_fig2_monitor_comparison", "v3_fig3_drift_vs_decay")]
    L += ["## Limitations\n",
          "- Only 13 independent models (cells within a model are autocorrelated); inference is at the model level, so power is limited.\n"
          "- All frozen models share calendar time, so economy-wide shocks affect many models together; the model-swap permutation test breaks that alignment deliberately, but common shocks still limit independence.\n"
          "- LendingClub only, one lender and one decade; a decay event is a ≥0.02 AUC fall relative to a single reference quarter whose own AUC is noisy.\n"
          "- Labels are right-censored for late-period loans (resolved loans only); origins reaching 2018 are held out of the primary tests.\n"
          "- The secondary tables contain many uncorrected p-values; only the three primary tests are Holm-corrected.\n"
          "- The design follows v1/v2 findings (disclosed in the plan); v1/v2 results remain in the repo.\n"]
    open(args.results_md or os.path.join(out, "..", "RESULTS_V3.md"), "w", encoding="utf-8").write("\n".join(L))
    print("P1", P1, "\nP2", P2, "\nP3", P3, "\nHolm", p_adj, "\nevents", res["n_events_conc"], res["n_events_lead"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["compute", "analyze", "all"])
    ap.add_argument("--data-dir", default=os.path.expanduser("~/xdrift_out"))
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--only", default=None, help="comma-separated origin indices (sharding / smoke tests)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--results-md", default=None)
    args = ap.parse_args()
    if args.stage in ("compute", "all"):
        compute(args)
    if args.stage in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
