# XDrift v3 — Pre-declared Analysis Plan: frozen-model monitoring vs. standard drift monitors

Written **before** running `xdrift_v3.py`; the run stamps this file's SHA-256 into `RESULTS_V3.md`.

## Why v3 (what v1 and v2 left open)

v2 found that explanations drift beyond a retraining noise floor, but H2/H3 (does explanation
drift *lead* performance decay?) were unresolvable: only 2 degradation events, and every window
retrained a new model, which is not how monitoring works. v3 fixes both problems and adds the
question a reviewer asks first: **does explanation drift tell you anything that standard,
cheaper drift monitors (score PSI, feature PSI) do not?**

Disclosure: the design is informed by v1/v2 results (forking-path risk). v1/v2 are kept in the repo,
all variants are reported, and only three tests form the primary family.

## Design

- **Deployed models.** 17 frozen XGBoost models (same v1 parameters, 300 trees), one per quarterly
  origin T = 2012-12, 2013-03, …, 2016-12, each trained on the 12 months of LendingClub loans ending at T.
  Never retrained afterwards.
- **Monitoring quarters.** Quarter 0 after T is the **reference** (go-live baseline). Quarters 1…6 after it
  are monitored. Every origin has all 7 quarters before the data ends (2018-12).
- **Primary panel: origins T = 2012-12 … 2015-12 (13 models × 6 quarters = 78 cells).** Loans issued from 2016
  on are right-censored in this snapshot (only quickly-resolved loans are labelled), which can bias
  late-quarter AUCs, so origins whose monitoring reaches 2018 are held out of the primary tests.
  The 4 later origins (2016-03 … 2016-12) are run too and reported as a **sensitivity analysis** (all 17 models).
- **Performance decay** `decay(m,k) = AUC_ref(m) − AUC_k(m)`, using full-quarter AUC. A **decay event**
  is `decay >= 0.02`. Sensitivity thresholds 0.01 and 0.03 are reported.
- **Explanation drift.** TreeSHAP (interventional, 100 background loans from the training window) of the frozen
  model on 500-loan draws. Reference draw R1 vs. monitored-quarter draw Q_k, using the v2 metric variants
  (`xsi_abs` PRIMARY, `xsi_orig`, `tau_only`, `jaccard10`). **Noise floor**: mean metric between two disjoint
  reference draws (R1,R2) and (R3,R4) of the same quarter. `excess_drift = floor − metric(R1, Q_k)`.
- **Baseline monitors** (all label-free, computed on the full quarter, vs. the reference quarter):
  score PSI (PRIMARY baseline; reference deciles), mean PSI over the top-15 features by model gain,
  mean PSI over all 34 features, **mean PSI over the 32 non-macro features**, KS statistic on scores,
  absolute shift in mean score. *Amendment (before the full run):* the non-macro baseline was added after a
  one-origin smoke test showed `fed_funds_rate` and `vix_bucket` (constant within a quarter) inflate all-feature
  PSI to ~1, which would make a strawman baseline. The primary baseline for P3 stays score PSI.
- Direction convention: every monitor is oriented so that a larger value = more drift.

## Primary hypotheses (3 tests, Holm-corrected)

Let `d` = explanation `excess_drift` (metric `xsi_abs`), `h` = horizon index k.

- **P1 (concurrent association).** Partial Spearman between `d(m,k)` and `decay(m,k)`, controlling for `h`
  (ranks residualised on `h`, pooled panel). Tests whether explanation drift tracks performance decay beyond the
  shared time trend.
- **P2 (lead).** Same, with `d(m,k)` against `decay(m,k+1)` (k = 1…5): does explanation drift today predict
  decay next quarter beyond the trend?
- **P3 (incremental value over the standard monitor).**
  `ΔAUROC = AUROC(d at k → event at k+1) − AUROC(score PSI at k → event at k+1)` pooled over the panel, where event =
  decay >= 0.02. Positive = explanation drift flags upcoming decay better than score PSI.

## Inference

The 13 primary models (17 in the sensitivity analysis) are the independent units (their cells share calendar time). All p-values use a
**model-swap permutation**: the decay series of the models are randomly reassigned to other models (10,000 draws),
keeping each series' internal trend and autocorrelation intact; two-sided for P1/P2. For P3 and all effect sizes we
report **cluster-bootstrap CIs** (resampling models, 2,000 draws); the P3 p-value is two-sided from the bootstrap
distribution of ΔAUROC. If fewer than 10 decay events exist, P3 is labelled **underpowered**.

## Secondary / exploratory (reported, not corrected)

- Every monitor (all explanation variants and all baselines) vs. decay: partial Spearman (concurrent and lead) and
  event-detection AUROC (concurrent and lead) with cluster-bootstrap CIs.
- Event thresholds 0.01 / 0.03; reference-quarter AUC noise (bootstrap) as a check.
- A combined model is **not** fit (avoids over-flexible post-hoc modelling).

## Reporting rule

Every number is written to `results_v3/`. `RESULTS_V3.md` states for each hypothesis whether it is supported,
contradicted, or unresolved. A null (e.g. "explanation drift adds nothing over score PSI") is reported as a null.
