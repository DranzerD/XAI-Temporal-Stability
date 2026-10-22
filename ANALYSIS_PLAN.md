# XDrift v2 — Pre-declared Analysis Plan

Written **before** running `xdrift_v2.py`. The run stamps this file's SHA-256 into
`RESULTS.md` so any later edit to the plan is visible.

## Why v2 exists (disclosure)

The v1 notebook run (`xdrift.ipynb`, real LendingClub data, 42 rolling windows) did **not**
support the original hypothesis:

| v1 finding | Value |
|---|---|
| Mean XSI by regime (Bear / Neutral / Bull) | 0.562 / 0.559 / 0.558 (Kruskal-Wallis p = 0.995) |
| Pearson r (XSI, AUC) | 0.14 (p = 0.39) |
| EART trigger rate at XSI < 0.70 | 85% of windows (35/41) → "lead time" is an artefact |
| Regime labels | HMM on window default-rate flips Neutral/Bear every quarter, incl. calm 2014-18 |
| Cosine component of XSI | mean 0.36, min 0 (mean *signed* SHAP vectors nearly cancel) |

v2 design choices below were **motivated by those diagnostics**. That is a forking-path
risk, so v1 results are reported alongside v2, every variant is reported (not just the best),
and only **one** metric and **three** tests form the primary family.

## Data and windows

Same as v1: LendingClub 2007-2018, resolved loans, 12-month training window, 3-month stride,
3-month forward evaluation window, 42 windows. Same per-window XGBoost (300 trees, v1
parameters), TreeSHAP (interventional, 300 eval loans, 100 background loans).

## Changes from v1

1. **Regimes from real market data**, not from the model's own default rate: CBOE daily VIX
   close (`VIX_History.csv`), mean over each window's 12-month training period.
   Stress = top tercile of that mean across windows; Calm = bottom tercile. Continuous
   analyses use the VIX mean directly (no thresholds).
2. **Noise floor.** For every window, 3 replicate models are trained on random 80%
   subsamples of the window's training data (different seeds) and explained on an
   *independent* 300-loan draw from the same evaluation window. The metric between the
   original model and a replicate is the "self" similarity: what instability looks like
   from retraining + sampling noise alone. Excess drift is
   `d_t = mean_r(self_t,r) - consecutive_t` (positive = more change than noise explains).
   This also removes the confound that early windows (1k-100k loans) are noisier.
3. **Metric variants** (all computed, all reported):
   - `xsi_orig`  — v1 definition (cosine on mean *signed* SHAP).
   - `xsi_abs`   — **PRIMARY.** 0.4·Kendall τ (rank of mean |SHAP|, rescaled to [0,1])
     + 0.4·cosine(mean |SHAP| vectors) + 0.2·(1 − mean PSI over the top-15 features by
     mean |SHAP| averaged across the two windows).
   - `tau_only`  — Kendall τ rescaled to [0,1], mean |SHAP| ranks.
   - `jaccard10` — Jaccard overlap of the top-10 feature sets.
4. **Performance degradation** is relative, not a fixed AUC level: event `D_t` occurs when
   `AUC_t <= mean(AUC_{t-4..t-1}) - 0.02`. AUC uncertainty is shown with 200-resample
   bootstrap 95% CIs.
5. **EART-v2** fires at t when `xsi_t < mean(xsi_{t-8..t-1}) - 1.5·sd(xsi_{t-8..t-1})`
   (needs >= 6 prior windows; no look-ahead).

## Primary hypotheses (metric `xsi_abs`; Holm correction across these 3)

- **H1** — Excess drift `d_t` rises with market stress: Spearman(d_t, mean VIX_t) > 0.
- **H2** — Low XSI precedes AUC decline: Spearman(xsi_t, AUC_{t+1} - AUC_t) > 0, using
  windows with eval_size >= 10,000 (lower AUC noise); the all-windows version is
  also reported.
- **H3** — EART-v2 alerts at t-2..t-1 hit performance-degradation events more often than
  chance: hit rate vs. a permutation null (alert timing shuffled, same number of alerts,
  10,000 draws, one-sided). Events are counted only in windows with eval_size >= 10,000
  (same reason as H2); alerts use all windows. Also reported: precision, recall, false-alarm rate, event count.
  If fewer than 5 events exist the result is labelled **underpowered**.

## Inference

Windows overlap by 75%, so ordinary p-values are anti-conservative. All correlation
p-values use an **exact circular-shift permutation test** (the series are cyclically shifted
against each other; p = share of |rho_shift| >= |rho_obs|, shift 0 included). With n ~ 41 the smallest
attainable p is 1/n ~ 0.025, so this is deliberately conservative. Two-sided.

## Secondary / exploratory (reported, not corrected)

- Spearman(xsi_t, |ΔVIX_t|); tercile Kruskal-Wallis of `d_t` and `xsi_t` by VIX regime.
- Partial Spearman controlling for log(train size).
- All four metric variants for H1 and H2.
- Restriction to windows with train_size >= 20,000.
- v1-style trigger rates for comparison.

## Reporting rule

Every number computed is written to `results_v2/`. `RESULTS.md` states which findings
support, contradict, or fail to resolve each hypothesis. Null results are reported as nulls.
