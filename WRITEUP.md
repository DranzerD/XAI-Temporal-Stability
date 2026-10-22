# Does SHAP Explanation Drift Tell You When a Credit Model Is Decaying?
### An empirical study on LendingClub, 2007–2018

## Abstract

Regulators and model-risk teams increasingly ask for monitoring of *explanations*, not just accuracy. We test whether
the temporal stability of SHAP attributions — summarised by an Explanation Stability Index (XSI) — can warn of
performance decay in a deployed credit-default model, and whether it adds anything over standard drift monitors.
Using 1.35M resolved LendingClub loans, we run three studies with pre-declared analysis plans. Explanations do drift
beyond a retraining noise floor (consecutive-window XSI is below the entire noise band in 93% of windows), and
drift is directionally larger in high-VIX periods but not significantly so (ρ = 0.43, p = 0.07; Holm-adjusted p = 0.22).
However, in a frozen-model monitoring design (13 models × 6 quarters), explanation drift neither tracks (partial
ρ = 0.08, 95% CI [−0.15, 0.41]) nor leads (ρ = 0.06, [−0.21, 0.39]) AUC decay, and does not detect decay better than
score PSI (ΔAUROC = 0.013, [−0.16, 0.18]). We therefore find **no evidence that XSI is a useful early-warning signal
on this data**, while noting that the intervals are wide and the decay events are clustered in a few late-period models.

## 1. Motivation

Model-risk guidance (e.g. SR 11-7) and the EU AI Act's risk-management provisions motivate monitoring more than
predictive accuracy. A natural proposal is to monitor SHAP explanations and retrain when they shift. That proposal
implies two testable claims: (i) explanations drift measurably over time, beyond estimation noise, and (ii) that drift
carries information about future performance decay that cheaper monitors lack.

## 2. Data and base model

LendingClub accepted loans (2007–2018Q4), filtered to resolved outcomes (Fully Paid vs. Charged Off/Default),
1,345,350 loans with a 20% default rate; origination-time features plus quarterly macro context. A strict temporal
split (train 2007–2017, test 2018, 56k loans) gives XGBoost **AUC 0.723, Gini 0.446, KS 0.327** — a normal range for
this dataset. Explanations are exact interventional TreeSHAP values on the predicted probability.

## 3. Methods

**XSI.** For two SHAP matrices, XSI = 0.4·Kendall τ (importance ranks) + 0.4·cosine (importance vectors) +
0.2·(1 − mean PSI over the top-15 features). The primary variant uses mean |SHAP| profiles; three other variants
(signed-mean cosine, τ only, top-10 Jaccard) are reported throughout.

**Study 1 (v1) — fixed-threshold trigger.** 42 rolling windows, a fresh model per window, an HMM on window-level
macro/default-rate features for regimes, and an alert when XSI < 0.70.

**Study 2 (v2) — noise-floor calibration.** Replicate models (80% subsamples, different seeds, independent explanation
draws) quantify how much XSI moves from retraining and sampling noise alone. *Excess drift* = floor − observed XSI.
Regimes come from real CBOE VIX. Inference uses exact circular-shift permutation tests (windows overlap by 75%) and
Holm correction over three primary tests.

**Study 3 (v3) — frozen-model monitoring.** One XGBoost model per quarterly origin (2012-12 … 2016-12), never retrained.
Quarter 0 after go-live is the reference; the following six quarters are monitored. For each (model, quarter) we
compute AUC decay against the reference and a set of label-free monitors: explanation excess drift (four variants) and
six standard baselines (score PSI, top-15-feature PSI, all-feature PSI, non-macro-feature PSI, score KS, score shift).
Primary tests (Holm-corrected): **P1** partial Spearman of drift vs. decay controlling for horizon; **P2** the same one
quarter ahead; **P3** ΔAUROC of explanation drift minus score PSI for flagging next-quarter decay ≥ 0.02. Inference is
at the model level: model-swap permutation p-values and cluster-bootstrap CIs. Origins whose later quarters fall in the
right-censored 2016+ period are held out of the primary tests (13 primary models; 17 in a sensitivity analysis).

All hypotheses, metrics and tests were fixed in `ANALYSIS_PLAN.md` / `ANALYSIS_PLAN_V3.md` before the corresponding
runs; each results file stamps the plan's hash. The statistics are covered by 14 unit tests, including a
calibration check that the permutation procedure rejects at about the nominal rate under the null.

## 4. Results

**Study 1.** XSI was flat across HMM regimes (0.562 / 0.559 / 0.558; Kruskal-Wallis p = 0.995), uncorrelated with AUC
(r = 0.14), and the XSI < 0.70 rule fired on 85% of windows, so its reported 3-month lead time was an artefact of a
miscalibrated threshold. The HMM regimes flipped between "Neutral" and "Bear" nearly every quarter, including calm years.

**Study 2.** Consecutive-window XSI averages 0.875 versus a noise floor of 0.920, and falls below the whole noise band
in 93% of windows: real drift exists. Excess drift is 1.6× higher in the top VIX tercile than the bottom (0.058 vs 0.036),
but H1 (ρ = 0.43, circular-shift p = 0.073, Holm-adjusted 0.22) does not clear significance; H2 (XSI vs next-window AUC change,
ρ = 0.17, p = 0.46) and H3 (only two degradation events) are unsupported/underpowered. The stress period (2008–2010)
also coincides with the smallest training sets, a confound the noise floor only partly removes.

**Study 3.** 13 frozen models × 6 quarters = 78 cells, 17 decay events.

| Primary test | Estimate | 95% CI | Holm-adj. p |
|---|---|---|---|
| P1 drift ↔ decay (partial ρ) | 0.076 | [−0.15, 0.41] | 1.00 |
| P2 drift → next-quarter decay (partial ρ) | 0.064 | [−0.21, 0.39] | 1.00 |
| P3 ΔAUROC vs score PSI | 0.013 | [−0.16, 0.18] | 1.00 |

For flagging next-quarter decay, explanation drift (AUROC 0.59, [0.43, 0.78]) is indistinguishable from score PSI
(0.58, [0.38, 0.77]). Among all monitors only all-feature PSI has an interval excluding 0.5 (0.73, [0.56, 0.85]); this is
uncorrected and it includes two macro features that are constant within a quarter, so it largely tracks calendar time
rather than model-specific drift (with all 17 models its concurrent correlation reaches nominal significance, ρ = 0.32, p = 0.05, again uncorrected).
The explanation-drift results are unchanged with all 17 models and with decay thresholds of 0.01 and 0.03 (ΔAUROC 0.025 and −0.004).

*Post-hoc diagnostics.* All 17 events come from only 5 of 13 models (origins 2014-12 to 2015-12), evaluated in 2016–2017;
the other models never lose ≥ 0.02 AUC. Within-model correlations are also null (mean ρ = 0.03).

## 5. Discussion

The evidence supports a narrow claim and rejects a broad one. *Supported:* SHAP explanations change over time by more than
retraining noise, so explanation monitoring has something to measure. *Not supported here:* that this change predicts, or
is concurrent with, AUC decay, or that it beats a standard score-PSI monitor. The confidence intervals are wide
(a ΔAUROC between roughly −0.16 and +0.18 cannot be excluded), so these are failures to find an effect, not proof of its absence.
Plausible reasons include that decay in this dataset is a cohort effect (late-2015 originations), that reference-quarter AUC is
itself noisy, and that tree-model SHAP mostly re-encodes feature drift that PSI already captures.

## 6. Limitations

One lender and one decade, with a single true stress episode; 13 independent models and clustered events; right-censored
late-period labels; many uncorrected secondary p-values; design choices informed by earlier studies (disclosed); and a
circular-shift test that is somewhat anti-conservative on strongly autocorrelated series (measured ≈ 21% false positives at a
nominal 5% on synthetic random walks). All results, plans, code and tests are in the repository; `streamlit run app.py` explores them.

## Reproducibility

`xdrift.ipynb` (data pipeline, base model, v1) → `xdrift_v2.py` → `xdrift_v3.py`; tests via `pytest`; pinned dependencies in
`requirements.txt`. The raw dataset is the public Kaggle LendingClub file (mirrored on Hugging Face as
`codesignal/lending-club-loan-accepted`, CC0).
