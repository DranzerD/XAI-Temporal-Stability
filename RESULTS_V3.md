# XDrift v3 — Results (frozen-model monitoring)

*Generated 2026-10-22 13:15 · Python 3.11.9 · plan SHA-256 `0fc4b5f63efb61a9…`*

Plan: [ANALYSIS_PLAN_V3.md](ANALYSIS_PLAN_V3.md). Primary panel: 13 frozen models × 6 quarters = 78 cells; 17 decay events (≥ 0.02 AUC) concurrent, 16 in the lead setting.

## Primary hypotheses (Holm-corrected across 3 tests)

| Hypothesis | estimate | 95% cluster-bootstrap CI | p | Holm-adj p | Verdict |
|---|---|---|---|---|---|
| **P1** explanation drift tracks AUC decay (partial ρ, controlling horizon) | 0.076 | [-0.15, 0.41] | 0.702 | 1.000 | NOT SUPPORTED (no significant effect after Holm correction) |
| **P2** explanation drift leads next-quarter decay (partial ρ) | 0.064 | [-0.21, 0.39] | 0.776 | 1.000 | NOT SUPPORTED (no significant effect after Holm correction) |
| **P3** ΔAUROC: explanation drift minus score PSI, flagging next-quarter decay | 0.013 | [-0.16, 0.18] | 0.845 | 1.000 | NOT SUPPORTED (no significant effect after Holm correction) |

## Every monitor vs. performance decay — primary panel (13 models)

ρ = partial Spearman given horizon; AUROC = detecting decay ≥ 0.02 (conc = same quarter, lead = next quarter). Permutation p-values; **uncorrected**. Red/blue grouping is in Figure 2.

| monitor          |   conc_rho |   conc_p | conc_ci       |   lead_rho |   lead_p | lead_ci       |   AUROC_conc |   AUROC_lead | AUROC_lead_ci   |
|:-----------------|-----------:|---------:|:--------------|-----------:|---------:|:--------------|-------------:|-------------:|:----------------|
| xsi_abs_excess   |      0.076 |    0.707 | [-0.15, 0.39] |      0.064 |    0.773 | [-0.21, 0.39] |        0.664 |        0.591 | [0.43, 0.78]    |
| xsi_orig_excess  |      0.002 |    0.993 | [-0.44, 0.46] |     -0.044 |    0.839 | [-0.45, 0.41] |        0.742 |        0.659 | [0.43, 0.82]    |
| tau_only_excess  |      0.024 |    0.907 | [-0.30, 0.43] |      0.004 |    0.982 | [-0.34, 0.39] |        0.616 |        0.535 | [0.33, 0.80]    |
| jaccard10_excess |     -0.206 |    0.283 | [-0.52, 0.08] |     -0.109 |    0.623 | [-0.45, 0.26] |        0.405 |        0.352 | [0.12, 0.60]    |
| score_psi        |     -0.213 |    0.284 | [-0.44, 0.11] |      0.006 |    0.977 | [-0.28, 0.31] |        0.505 |        0.578 | [0.38, 0.77]    |
| feat_psi_top15   |      0.145 |    0.506 | [-0.25, 0.51] |      0.237 |    0.318 | [-0.22, 0.59] |        0.682 |        0.68  | [0.34, 0.97]    |
| feat_psi_all     |      0.211 |    0.292 | [-0.09, 0.55] |      0.215 |    0.297 | [-0.16, 0.51] |        0.763 |        0.728 | [0.56, 0.85]    |
| feat_psi_nomacro |     -0.133 |    0.488 | [-0.45, 0.24] |     -0.09  |    0.678 | [-0.48, 0.37] |        0.595 |        0.598 | [0.44, 0.77]    |
| score_ks         |     -0.227 |    0.223 | [-0.49, 0.11] |      0.018 |    0.934 | [-0.32, 0.36] |        0.449 |        0.543 | [0.33, 0.73]    |
| score_shift      |     -0.291 |    0.151 | [-0.54, 0.05] |     -0.084 |    0.685 | [-0.37, 0.24] |        0.352 |        0.45  | [0.25, 0.64]    |

## Sensitivity: all 17 models (adds origins 2016-03…2016-12, whose later quarters are right-censored)

| monitor          |   conc_rho |   conc_p | conc_ci       |   lead_rho |   lead_p | lead_ci       |   AUROC_conc |   AUROC_lead | AUROC_lead_ci   |
|:-----------------|-----------:|---------:|:--------------|-----------:|---------:|:--------------|-------------:|-------------:|:----------------|
| xsi_abs_excess   |      0.045 |    0.799 | [-0.19, 0.33] |      0.059 |    0.753 | [-0.21, 0.38] |        0.634 |        0.587 | [0.41, 0.77]    |
| xsi_orig_excess  |     -0.044 |    0.81  | [-0.51, 0.40] |     -0.068 |    0.734 | [-0.51, 0.36] |        0.676 |        0.647 | [0.37, 0.83]    |
| tau_only_excess  |      0.036 |    0.855 | [-0.30, 0.39] |      0.038 |    0.845 | [-0.32, 0.40] |        0.601 |        0.561 | [0.35, 0.76]    |
| jaccard10_excess |     -0.19  |    0.265 | [-0.47, 0.10] |     -0.155 |    0.372 | [-0.42, 0.14] |        0.401 |        0.345 | [0.15, 0.55]    |
| score_psi        |     -0.18  |    0.277 | [-0.42, 0.08] |      0.044 |    0.806 | [-0.22, 0.29] |        0.53  |        0.583 | [0.42, 0.76]    |
| feat_psi_top15   |     -0.022 |    0.914 | [-0.41, 0.36] |      0.074 |    0.724 | [-0.33, 0.44] |        0.581 |        0.596 | [0.30, 0.88]    |
| feat_psi_all     |      0.317 |    0.05  | [0.03, 0.56]  |      0.311 |    0.08  | [-0.03, 0.54] |        0.757 |        0.729 | [0.60, 0.85]    |
| feat_psi_nomacro |     -0.162 |    0.338 | [-0.39, 0.12] |     -0.066 |    0.711 | [-0.34, 0.29] |        0.556 |        0.582 | [0.43, 0.74]    |
| score_ks         |     -0.206 |    0.195 | [-0.46, 0.08] |      0.047 |    0.782 | [-0.23, 0.34] |        0.481 |        0.555 | [0.38, 0.73]    |
| score_shift      |     -0.22  |    0.196 | [-0.49, 0.11] |     -0.016 |    0.936 | [-0.30, 0.30] |        0.412 |        0.483 | [0.31, 0.66]    |

## Sensitivity: event threshold

| threshold | events (lead) | ΔAUROC (expl − score PSI) | 95% CI |
|---|---|---|---|
| 0.01 | 17 | 0.025 | [-0.14, 0.18] |
| 0.03 | 14 | -0.004 | [-0.24, 0.19] |

## Post-hoc diagnostics (not pre-declared)

**Event clustering.** The 17 decay events come from only **5 of 13 models** (2014-12: 1, 2015-03: 2, 2015-06: 4, 2015-09: 4, 2015-12: 6); the other models never decay by ≥ 0.02 AUC. The number of independent events is therefore far smaller than the event count suggests, and decay is concentrated in late-period cohorts (evaluated in 2016-2017), a calendar/cohort effect rather than a drift-tracking one.

**Within-model correlation** (Spearman across the 6 quarters of each model, then averaged over models; removes between-model differences):

| monitor          |   mean_within_model_rho |   median | models_with_positive_rho   |
|:-----------------|------------------------:|---------:|:---------------------------|
| xsi_abs_excess   |                   0.033 |    0.029 | 7/13                       |
| score_psi        |                  -0.134 |   -0.429 | 5/13                       |
| feat_psi_nomacro |                  -0.064 |   -0.2   | 5/13                       |
| feat_psi_all     |                  -0.007 |   -0.257 | 5/13                       |


## Figures

![v3_fig1_decay_and_drift_heatmaps](figures/v3_fig1_decay_and_drift_heatmaps.png)

![v3_fig2_monitor_comparison](figures/v3_fig2_monitor_comparison.png)

![v3_fig3_drift_vs_decay](figures/v3_fig3_drift_vs_decay.png)

## Limitations

- Only 13 independent models (cells within a model are autocorrelated); inference is at the model level, so power is limited.
- All frozen models share calendar time, so economy-wide shocks affect many models together; the model-swap permutation test breaks that alignment deliberately, but common shocks still limit independence.
- LendingClub only, one lender and one decade; a decay event is a ≥0.02 AUC fall relative to a single reference quarter whose own AUC is noisy.
- Labels are right-censored for late-period loans (resolved loans only); origins reaching 2018 are held out of the primary tests.
- The secondary tables contain many uncorrected p-values; only the three primary tests are Holm-corrected.
- The design follows v1/v2 findings (disclosed in the plan); v1/v2 results remain in the repo.
