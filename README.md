# XDrift: Temporal Stability of Explainable AI in Credit Risk Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](xdrift.ipynb)

## Abstract

Machine learning models deployed in high-stakes financial domains — such as consumer credit scoring — demand not only robust predictive performance but also **interpretable and stable explanations** over time. While model monitoring traditionally focuses on accuracy degradation under distribution shift (concept drift), the temporal stability of post-hoc explanations has received limited attention.

This project presents an empirical framework, **XDrift**, for evaluating explanation drift in credit default prediction models. It introduces the **Explanation Stability Index (XSI)**, a composite metric that quantifies how SHAP feature attributions evolve across rolling time windows, and **EART (Explanation-Aware Retraining Trigger)**, a monitoring rule that fires on explanation drift *before* predictive performance (AUC) degrades.

### Core Hypothesis

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

## Dataset

| Dataset                   | Type                        | Years     | Source                                                                                               |
| -------------------------- | --------------------------- | --------- | ----------------------------------------------------------------------------------------------------|
| **LendingClub** (Primary) | Peer-to-peer personal loans | 2007–2018 | [Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |

> **Note:** The dataset is not included in this repository due to size and licensing. See [Getting Started](#getting-started) below.

## Repository Structure

```
├── xdrift.ipynb       # Main notebook: full XDrift pipeline (Phases 1-5)
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

Then update `RAW_DATA_FILE` in the notebook's config cell to point at the unzipped CSV, and run all cells. Start with a low `SHAP_SAMPLES_PER_WINDOW` and `N_OPTUNA_TRIALS = 0` for a smoke test before committing to the full run — see [RUNBOOK.md](RUNBOOK.md) for details.

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

XSI is designed as a **proactive monitoring metric** for deployed credit models — EART can trigger a retraining/audit before AUC-based monitoring would notice anything wrong. This is relevant for regulatory compliance under the **EU AI Act** (Article 9: Risk Management) and **SR 11-7** (OCC/Federal Reserve model risk management guidance).

## References

1. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. _NeurIPS_.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. _KDD_.
3. Lundberg, S. M., et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. _Nature Machine Intelligence_.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
