# XDrift: Temporal Stability of Explainable AI in Credit Risk Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](xdrift.ipynb)

## Abstract

Machine learning models deployed in high-stakes financial domains — such as consumer credit scoring — demand not only robust predictive performance but also **interpretable and stable explanations** over time. While model monitoring traditionally focuses on accuracy degradation under distribution shift (concept drift), the temporal stability of post-hoc explanations has received limited attention.

This project presents an empirical framework, **XDrift**, for evaluating explanation drift in credit default prediction models. It introduces the **Explanation Stability Index (XSI)**, a composite metric that quantifies how SHAP feature attributions evolve across rolling time windows, and **EART (Explanation-Aware Retraining Trigger)**, a proposed monitoring rule meant to fire on explanation drift *before* predictive performance (AUC) degrades. The repo then tests those ideas on real data with a pre-declared analysis — and reports where they do and do not hold (see [Findings](#findings)).

### Core Hypothesis (tested — see Findings)

> _Explanation stability (measured by XSI) degrades before predictive performance (measured by ROC-AUC) drops under natural temporal distribution shifts — and this effect is strongest during Bear/Crisis market regimes._

### XSI Formula

For consecutive rolling windows `t-1` and `t`, with SHAP attribution vectors `shap_t`:

```
XSI_t = 0.4 · Kendall_τ(rank_t, rank_{t-1})     # rank-order stability
      + 0.4 · CosineSim(shap_t, shap_{t-1})     # directional stability
      + 0.2 · (1 − PSI_shap(t, t-1))            # distribution stability
```

`XSI ∈ [0, 1]`; `XSI = 1` means explanations are identical to the previous window.

---

## Findings

Real LendingClub data (2007–2018, 1.35M resolved loans, 42 rolling windows). Full detail in
**[RESULTS.md](RESULTS.md)**; hypotheses, metrics and tests were fixed in advance in
**[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)** (its hash is stamped into the results).

**Credit model** (train 2007–2017, test 2018 on 56k loans): AUC **0.723**, Gini 0.446, KS 0.327.

**Explanation drift is real but small.** Retraining a model on an 80% subsample and re-explaining gives a
*noise floor* for XSI. Consecutive-window XSI falls below the entire noise band in 93% of windows
(0.875 vs 0.920 for the primary variant), so explanations change more over time than retraining noise alone explains.

| Pre-declared hypothesis | Result | Verdict |
|---|---|---|
| **H1** excess drift rises with market stress (real VIX) | ρ = 0.43, circular-shift p = 0.073, Holm-adj. p = 0.22 | Directionally consistent (excess drift 1.6× higher in the high-VIX tercile), **not significant** |
| **H2** low XSI precedes AUC decline | ρ = 0.17, p = 0.46 (n = 24) | **Not supported** |
| **H3** explanation-drift alerts precede performance degradation | 2 degradation events in the usable windows | **Underpowered** — cannot be resolved |

The original v1 analysis was also negative: XSI was flat across HMM-derived regimes (0.562 / 0.559 / 0.558,
Kruskal-Wallis p = 0.995), uncorrelated with AUC (r = 0.14), and the fixed-threshold EART fired on 85% of windows, so
its "3-month lead time" was an artefact. v2 replaces the fixed threshold, the model-derived regimes and the
signed-SHAP cosine with the calibrated design above. **The early-warning claim is not supported by this data.**

Caveats: stress windows (2008–2010) are also the smallest, so stress and sample size are partly confounded;
LendingClub contains one real stress episode; the permutation test is still somewhat anti-conservative on
strongly autocorrelated series (measured false-positive rate ≈ 21% at nominal 5%). All secondary p-values are uncorrected.

![XSI vs noise floor](results_v2/figures/v2_fig1_xsi_vs_noise_floor.png)

---

## Methodology

1. **Phase 1 — Data pipeline & feature engineering.** Load LendingClub loan data, filter to resolved outcomes (Fully Paid vs. Charged Off/Default), engineer financial features, and merge quarterly macroeconomic context (Fed Funds Rate, VIX bucket).
2. **Phase 2 — Credit default model.** Train an XGBoost classifier (optionally Optuna-tuned) with `scale_pos_weight` for class imbalance.
3. **Phase 3 — XAI engine.** Generate global and local explanations with **TreeSHAP** (exact, non-sampled Shapley values) and **LIME** as a second-opinion comparator; compare their temporal stability.
4. **Phase 4 — XDrift framework (core novelty).**
   - Roll a fresh XGBoost + TreeSHAP fit across sequential time windows.
   - Compute **XSI** between consecutive windows.
   - Label each window's market regime (Bull / Neutral / Bear+Crisis) with a 3-state HMM over macroeconomic features.
   - Run **EART**: trigger a retraining alert when XSI drops below threshold, and compare its lead time against an AUC-degradation baseline trigger.
5. **Phase 5 — Results & figures.** Regime-stratified XSI/AUC summary table and the full set of paper figures (XSI timeline, XSI-by-regime violin plot, XSI component decomposition, EART vs. baseline trigger comparison, SHAP-vs-LIME agreement, AUC-vs-XSI scatter).
6. **Phase 6 — XDrift v2: calibrated, pre-declared analysis** (`xdrift_v2.py`, [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)).
   - **Noise floor:** 3 replicate models per window (80% subsample, different seed, independent eval draw) define how
     much XSI moves from retraining/sampling noise alone; *excess drift* = noise floor − observed XSI.
   - **Real market regimes:** CBOE VIX over each window replaces the HMM-on-default-rate regimes.
   - **Metric variants** (signed cosine, |SHAP|-profile cosine, Kendall τ only, top-10 Jaccard), all reported.
   - **Inference:** exact circular-shift permutation tests (windows overlap 75%), Holm correction on 3 primary tests,
     permutation null for the early-warning hit rate, bootstrap AUC intervals, relative (not fixed-level) degradation events.
   - Validated: reproduces the v1 XSI (r = 0.998) and ships unit tests (`test_xdrift_v2.py`).

## Dataset

| Dataset                   | Type                        | Years     | Source                                                                                               |
| -------------------------- | --------------------------- | --------- | ----------------------------------------------------------------------------------------------------|
| **LendingClub** (Primary) | Peer-to-peer personal loans | 2007–2018 | [Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |

> **Note:** The dataset is not included in this repository due to size and licensing. See [Getting Started](#getting-started) below.

## Repository Structure

```
├── xdrift.ipynb       # Main notebook: full XDrift pipeline (Phases 1-5)
├── xdrift_v2.py        # v2: noise-floor-calibrated XSI, VIX regimes, permutation tests, figures
├── test_xdrift_v2.py   # Unit tests for the statistics
├── ANALYSIS_PLAN.md    # Hypotheses/metrics/tests declared before the v2 run
├── RESULTS.md          # Generated v2 results (primary tests, noise floor, all variants)
├── results_v1/         # Notebook run: per-window CSV + key figures
├── results_v2/         # v2 outputs: windows_v2.csv, secondary_tests.csv, figures/
├── RUNBOOK.md          # How to run it (Kaggle/local), plus a log of bugs found & fixed
├── README.md           # This file
├── requirements.txt    # Python dependencies (local runs)
├── LICENSE             # MIT License
└── .gitignore          # Git ignore rules
```

## Getting Started

The notebook is built for Kaggle, where the dataset and most dependencies are pre-installed. See **[RUNBOOK.md](RUNBOOK.md)** for full step-by-step instructions (Kaggle and local), recommended smoke-test settings, and a log of real bugs that were found and fixed by execution.

Quick summary:

```bash
# Local setup
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/
```

The notebook uses `data/accepted_2007_to_2018Q4.csv` locally by default. If your CSV lives elsewhere, set `XDRIFT_RAW_DATA` to its path before launching Jupyter, then run all cells. Start with a low `SHAP_SAMPLES_PER_WINDOW` and `N_OPTUNA_TRIALS = 0` for a smoke test before committing to the full run — see [RUNBOOK.md](RUNBOOK.md) for details.

### Reproducing v2

Run notebook cells 1–9 once (they write `train.parquet` / `test.parquet` to the working directory), download the
[CBOE VIX history](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv) to `data/fred/VIXCLS_cboe.csv`, then:

```bash
python xdrift_v2.py all --data-dir working --vix data/fred/VIXCLS_cboe.csv --out results_v2 --v1 results_v1/xdrift_results.csv
python -m pytest test_xdrift_v2.py
```

The compute stage takes ~15 minutes on a 20-core CPU and resumes if interrupted.

## Technical Stack

| Library                                                   | Purpose                                                    |
| ----------------------------------------------------------| ------------------------------------------------------------ |
| [XGBoost](https://xgboost.readthedocs.io/)                | Credit default model (`tree_method="hist"`)                |
| [SHAP](https://shap.readthedocs.io/)                      | TreeSHAP attributions — the core input to XSI              |
| [LIME](https://github.com/marcotcr/lime)                  | Second-opinion local explanations                           |
| [hmmlearn](https://hmmlearn.readthedocs.io/)               | 3-state HMM for market regime detection                    |
| [Optuna](https://optuna.org/)                              | Hyperparameter tuning for the XGBoost model                |
| [scikit-learn](https://scikit-learn.org/)                  | Evaluation metrics (ROC-AUC, KS, Gini) and preprocessing   |
| [SciPy](https://scipy.org/) / [scikit-posthocs](https://scikit-posthocs.readthedocs.io/) | Kendall τ, statistical post-hoc tests |
| [pandas](https://pandas.pydata.org/)                       | Data manipulation and rolling-window generation            |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Paper figures                          |

## Practical Implications

XSI is designed as a **monitoring metric** for deployed credit models. On LendingClub data it detects explanation change beyond retraining noise, but this study did **not** find that it reliably leads AUC degradation (the question is underpowered with the few degradation events available), so it is best treated as a complementary audit signal rather than a proven early-warning trigger. Explanation monitoring is relevant to model-risk expectations under the **EU AI Act** (Article 9: Risk Management) and **SR 11-7** (OCC/Federal Reserve model risk management guidance).

## References

1. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. _NeurIPS_.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. _KDD_.
3. Lundberg, S. M., et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. _Nature Machine Intelligence_.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
