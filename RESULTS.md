# XDrift v2 — Results

*Generated 2026-10-22 12:56 · windows analysed: 42 · Python 3.11.9 · analysis plan SHA-256 `09f515a0d874d7a5…`*

Plan: [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md). Everything below was computed by `xdrift_v2.py`; no variant was dropped.

## Sanity check: v2 reproduces the v1 XSI

On 41 windows, v2's `xsi_orig` for consecutive windows vs. the v1 notebook's XSI: correlation 0.9982, max |difference| 0.0166.

## Primary hypotheses (metric `xsi_abs`, Holm-corrected across 3 tests)

| Hypothesis | n | Spearman rho | circ-shift p | Holm-adj p | Verdict |
|---|---|---|---|---|---|
| **H1** excess drift rises with VIX | 41 | 0.429 | 0.073 | 0.220 | NOT SUPPORTED (no significant effect after Holm correction) |
| **H2** low XSI precedes AUC decline (eval≥10k) | 24 | 0.170 | 0.458 | 0.458 | NOT SUPPORTED (no significant effect after Holm correction) |
| **H3** EART-v2 alerts precede degradation events | 2 events | hit rate 0.50 vs null 0.15 | 0.220 | 0.439 | UNDERPOWERED (<5 events) — cannot be resolved |

H2 using all windows (incl. small early ones): rho=-0.002, p=0.950, n=40.

### H3 detail

| quantity | eval≥10k (primary) | all windows |
|---|---|---|
| n_candidate_windows | 25 | 33 |
| n_events | 2 | 3 |
| event_base_rate | 0.080 | 0.091 |
| n_alerts_in_pool | 2 | 2 |
| hit_rate | 0.500 | 0.333 |
| precision | 0.500 | 0.500 |
| false_alarm_rate | 0.500 | 0.500 |
| null_hit_rate_mean | 0.150 | 0.119 |
| p_one_sided | 0.220 | 0.282 |

## Noise floor

Mean similarity between the original model and replicate models (retraining + sampling noise only) vs. between consecutive windows:

| metric | noise floor | consecutive windows | gap |
|---|---|---|---|
| xsi_orig | 0.662 | 0.559 | 0.103 |
| xsi_abs | 0.920 | 0.875 | 0.045 |
| tau_only | 0.939 | 0.907 | 0.032 |
| jaccard10 | 0.826 | 0.727 | 0.100 |

## Real-VIX regimes (terciles of mean VIX over the training window)

| vix_regime   |   xsi_abs_consec_mean |   xsi_abs_consec_std |   xsi_abs_consec_count |   xsi_abs_excess_mean |   xsi_abs_excess_std |   xsi_abs_excess_count |   vix_train_mean_mean |   vix_train_mean_std |   vix_train_mean_count |   eval_auc_mean |   eval_auc_std |   eval_auc_count |   train_size_mean |   train_size_std |   train_size_count |
|:-------------|----------------------:|---------------------:|-----------------------:|----------------------:|---------------------:|-----------------------:|----------------------:|---------------------:|-----------------------:|----------------:|---------------:|-----------------:|------------------:|-----------------:|-------------------:|
| Calm         |                0.8862 |               0.0336 |                     14 |                0.0362 |               0.036  |                     14 |               13.5774 |               1.0698 |                     14 |          0.7242 |         0.0157 |               14 |            188320 |          53744.5 |                 14 |
| Mid          |                0.8787 |               0.0261 |                     13 |                0.0394 |               0.0249 |                     13 |               17.7043 |               2.1645 |                     13 |          0.7032 |         0.0225 |               13 |            188496 |         160483   |                 13 |
| Stress       |                0.8617 |               0.0274 |                     14 |                0.0579 |               0.0267 |                     14 |               28.374  |               6.6243 |                     14 |          0.6493 |         0.0357 |               14 |             11254 |          11741.9 |                 14 |

Kruskal-Wallis across terciles: XSI p=0.030, excess drift p=0.090. Note: windows in the Stress tercile are also the smallest (2008-2010), so XSI alone is confounded with sample size; excess drift and the partial correlations below address this.

## All metric variants (secondary, uncorrected; circular-shift p-values)

| metric    |   H1 excess~VIX rho |   H1 p |   H1 partial(size) rho |   H1 partial p |   H1 train>=20k rho |   H1 train>=20k p |   XSI~VIX rho |   XSI~VIX p |   XSI~|dVIX| rho |   XSI~|dVIX| p |   H2 rho |   H2 p |
|:----------|--------------------:|-------:|-----------------------:|---------------:|--------------------:|------------------:|--------------:|------------:|-----------------:|---------------:|---------:|-------:|
| xsi_orig  |               0.254 |  0.171 |                  0.163 |          0.463 |              -0.065 |              0.48 |        -0.091 |       0.732 |           -0.265 |          0.171 |   -0.221 |  0.458 |
| xsi_abs   |               0.429 |  0.073 |                  0.321 |          0.146 |               0.252 |              0.08 |        -0.459 |       0.049 |           -0.243 |          0.293 |    0.17  |  0.458 |
| tau_only  |               0.503 |  0.049 |                  0.253 |          0.146 |               0.386 |              0.04 |        -0.683 |       0.024 |           -0.312 |          0.293 |    0.052 |  0.875 |
| jaccard10 |               0.496 |  0.049 |                  0.365 |          0.049 |               0.268 |              0.04 |        -0.715 |       0.024 |           -0.257 |          0.39  |    0.056 |  0.833 |

## Figures

![v2_fig1_xsi_vs_noise_floor](figures/v2_fig1_xsi_vs_noise_floor.png)

![v2_fig2_hypothesis_scatter](figures/v2_fig2_hypothesis_scatter.png)

![v2_fig3_auc_events_alerts](figures/v2_fig3_auc_events_alerts.png)

![v2_fig4_metric_variants](figures/v2_fig4_metric_variants.png)

## Limitations

- 42 overlapping windows; effective sample size is smaller than n. The circular-shift test accounts for autocorrelation, but on synthetic independent random walks of the same length it still rejects at ~21% (nominal 5%; see `test_xdrift_v2.py`), versus >50% for a naive test. Treat p-values near 0.05 as weak evidence.
- The secondary table reports 4 metrics x 6 tests = 24 uncorrected p-values; a few values near 0.05 are expected by chance and are **not** evidence. Only the three primary tests are corrected (Holm).
- Windows in the VIX Stress tercile (2008-2010) are also the smallest, so stress and sample size are partly confounded even after noise-floor adjustment; partial correlations on log(train size) are reported.
- The H3 early-warning question has very few degradation events, so it cannot be resolved with this data.
- Windows ending before 2012 have under 25k training loans, so they are noisy; the noise floor is meant to absorb this.
- LendingClub is one lender and one decade, with a single true stress episode (2008-09) that coincides with tiny samples.
- Only resolved loans are kept, so recent loans are biased toward early outcomes (right-censoring).
- The v2 design (primary metric, VIX regimes) was motivated by v1 diagnostics; v1 results are kept for comparison.
