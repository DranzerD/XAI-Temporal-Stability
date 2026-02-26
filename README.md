# Temporal Stability of Explainable AI (XAI) in Financial Credit Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](XAI_TemporalStability.ipynb)

## Abstract

Machine learning models deployed in high-stakes financial domains — such as consumer credit scoring — demand not only robust predictive performance but also **interpretable and stable explanations** over time. While model monitoring traditionally focuses on accuracy degradation under distribution shift (concept drift), the temporal stability of post-hoc explanations has received limited attention.

This project presents a rigorous empirical framework for evaluating **Explanation Drift** in credit default prediction models. We introduce the **Temporal Explanation Stability Index (TESI)**, a composite metric that quantifies how feature attribution explanations evolve across sequential time windows under natural temporal distribution shifts.

### Core Hypothesis

> _Explanation stability (measured by TESI) degrades before predictive performance (measured by ROC-AUC) drops under natural temporal distribution shifts._

### TESI Formula

$$TESI_{t} = 0.5 \cdot \text{CosineSim}(\bar{E}_{base}, \bar{E}_{t}) + 0.5 \cdot \rho_s(\bar{E}_{base}, \bar{E}_{t})$$

where $\bar{E}_{base}$ is the mean attribution vector on the training distribution, $\bar{E}_{t}$ is the mean attribution vector at time window $t$, and $\rho_s$ denotes the Spearman rank correlation coefficient.

---

## Methodology

1. Load and preprocess two independent real-world credit datasets spanning multiple years
2. Train a PyTorch MLP on the earliest time window and freeze its weights
3. Generate post-hoc explanations using **Integrated Gradients** and **GradientShap** (via [Captum](https://captum.ai/))
4. Track both predictive performance (AUC, F1) and explanation stability (TESI) across future time windows
5. Demonstrate that TESI serves as an early-warning indicator of model staleness across datasets

## Datasets

| Dataset                                  | Type                        | Years     | Features                              | Source                                                                                               |
| ---------------------------------------- | --------------------------- | --------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LendingClub** (Primary)                | Peer-to-peer personal loans | 2013–2017 | 8 named financial features            | [Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| **Amex Default Prediction** (Robustness) | Credit card default         | 2017–2018 | 8 anonymized features (B, D, S, P, R) | [Kaggle: amex-default-prediction](https://www.kaggle.com/competitions/amex-default-prediction)       |

> **Note:** The datasets are not included in this repository due to size and licensing. See [Data Setup](#data-setup) for download instructions.

## Key Findings

1. **TESI degrades faster than AUC** — On LendingClub, TESI drops ~4× faster than AUC across 5 years. The same pattern emerges across quarterly windows on Amex.
2. **Cross-dataset generalizability** — The TESI degradation pattern holds across named vs. anonymized features, multi-year vs. quarterly drift, and personal loans vs. credit card portfolios.
3. **Method-agnostic drift** — Both Integrated Gradients and GradientShap exhibit consistent TESI degradation.
4. **Actionable thresholds** — TESI thresholds: stable (> 0.95), warning (0.85–0.95), critical (< 0.85).

## Repository Structure

```
├── XAI_TemporalStability.ipynb   # Main experiment notebook (end-to-end pipeline)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU works but slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/DranzerD/XAI-Temporal-Stability.git
cd XAI-Temporal-Stability

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

This project uses Kaggle datasets. You need a [Kaggle account](https://www.kaggle.com/) and API credentials.

**Option A — Run on Kaggle (Recommended):**

1. Upload the notebook to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Add the datasets via **Add Data** → search for `lending-club` (by wordsforthewise) and `amex-default-prediction`
3. Run all cells

**Option B — Run Locally:**

1. Install the Kaggle CLI: `pip install kaggle`
2. Download datasets:
   ```bash
   kaggle datasets download -d wordsforthewise/lending-club
   kaggle competitions download -c amex-default-prediction
   ```
3. Place the data files in the appropriate paths (update file paths in the notebook if needed)

### Running the Notebook

```bash
jupyter notebook XAI_TemporalStability.ipynb
```

Or open directly in VS Code with the Jupyter extension.

## Technical Stack

| Library                                                                        | Purpose                                                           |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                                                | Model construction, training, and inference                       |
| [Captum](https://captum.ai/)                                                   | Post-hoc feature attribution (Integrated Gradients, GradientShap) |
| [scikit-learn](https://scikit-learn.org/)                                      | Evaluation metrics (ROC-AUC, F1) and preprocessing                |
| [SciPy](https://scipy.org/)                                                    | Spearman rank correlation for TESI computation                    |
| [pandas](https://pandas.pydata.org/)                                           | Data manipulation and temporal windowing                          |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Publication-quality visualizations                                |

## Practical Implications

TESI can serve as a **proactive monitoring metric** for deployed models:

$$\text{If } TESI_t < 0.85 \text{ while } AUC_t > 0.70 \implies \text{Trigger retraining / audit}$$

This is relevant for regulatory compliance under the **EU AI Act** (Article 9: Risk Management) and **SR 11-7** (OCC/Federal Reserve model risk management guidance).

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. _ICML_.
2. Erion, G., et al. (2021). Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients. _Nature Machine Intelligence_.
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. _NeurIPS_.
4. Kokhlikyan, N., et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. _arXiv:2009.07896_.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

_Developed for IEEE/Springer XAI conference submission. All results are fully reproducible from fixed random seeds (SEED=42)._
