# Credit Risk Analysis: Loan Default Prediction

End-to-end machine learning pipeline for predicting consumer loan defaults on 270,887 Lending Club loans — from raw data through model deployment recommendations.

---

## Overview

Loan default prediction sits at the core of consumer lending: a missed default costs approximately 4–5× more than a false decline, so the problem requires explicit business framing, not just accuracy optimization. This project builds a complete credit risk modeling pipeline that addresses class imbalance, engineers domain-specific financial stress features, selects a model architecture through systematic comparison, and optimizes a decision threshold against a 5:1 cost asymmetry.

The final model — XGBoost with class weighting — achieves **82.4% recall** at a **30.9% precision** on a held-out test set. On a portfolio of 10,000 loans at the dataset's 22.6% default rate, it prevents an estimated $15.2M in default losses while incurring $7.9M in foregone interest from false declines, for a **net financial benefit of approximately $7.3M** versus approving all applications indiscriminately.

The project also explores Weight of Evidence (WoE) / Information Value (IV) binning as an independent validation of feature importance and as a scorecard prototype — connecting modern ML methodology to the traditional credit scoring approach still used at regulated institutions.

---

## Key Results

| Metric | Value |
|--------|-------|
| Model | XGBoost + Class Weighting |
| Decision threshold | 0.40 (tuned for 5:1 cost ratio) |
| ROC-AUC | 0.7175 |
| PR-AUC | 0.4176 |
| Recall (default class) | 82.4% |
| Precision (default class) | 30.9% |
| False alarms per 10,000 loans | 4,162 |
| Net benefit per 10,000 loans | ~$7.3M |
| WoE scorecard ROC-AUC (10 features) | 0.7045 — 98.2% of XGBoost performance |
| Training set | 216,709 loans (80%) |
| Test set | 54,178 loans (20%, held out) |

---

## Notebook Guide

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `NB01_data_exploration.ipynb` | Profile 887,379 raw Lending Club records — missing values, distributions, target imbalance, and multicollinearity candidates |
| 02 | `NB02_data_cleaning.ipynb` | Resolve data quality issues: drop leakage columns, impute missing values, cap outliers, correct encoding errors |
| 03 | `NB03_feature_engineering.ipynb` | Engineer 20 domain-specific features across 5 groups: debt stress ratios, utilization flags, delinquency severity scores, behavioral flags, and interaction terms |
| 04 | `NB04_baseline_modeling.ipynb` | Establish logistic regression baseline, compare SMOTE vs. class weighting, set the 5:1 cost ratio and initial decision threshold |
| 05 | `NB05_tree_models.ipynb` | Tune and compare Random Forest and XGBoost with RandomizedSearchCV; select final model and optimize threshold against cost function |
| 06 | `NB06_shap_interpretability.ipynb` | SHAP global and local explanations — beeswarm, bar, dependence, and waterfall plots; risk committee memo |
| 07 | `NB07_woe_iv_exploration.ipynb` | WoE/IV binning for top 10 features; IV vs. SHAP comparison; WoE logistic regression scorecard prototype |
| 08 | `NB08_business_summary.ipynb` | Executive-facing summary: business problem, methodology, model performance with dollar impact, deployment recommendations |

---

## Tech Stack

- **Python 3.9**
- **pandas**, **numpy** — data manipulation
- **scikit-learn** — preprocessing, pipelines, cross-validation, logistic regression, metrics
- **XGBoost** — final model
- **imbalanced-learn** — SMOTE implementation
- **SHAP** — model interpretability
- **optbinning** — optimal WoE/IV binning
- **matplotlib**, **seaborn** — visualisation
- **joblib** — model serialisation
- **pyarrow** — Parquet I/O

---

## Reproducing This Project

### 1. Clone the repository

```bash
git clone https://github.com/UncannyJ33/credit-risk-analysis.git
cd credit-risk-analysis
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is the [Lending Club Loan Data](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) from Kaggle. You will need a Kaggle account and API credentials (`~/.kaggle/kaggle.json`).

```bash
kaggle datasets download -d ranadeep/credit-risk-dataset -p data/raw --unzip
```

The raw file (`loan.csv`, ~421MB) should land at `data/raw/loan/loan.csv`.

### 4. Run notebooks in order

Each notebook reads from the outputs of the previous one. Execute them sequentially:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/NB01_data_exploration.ipynb

# Repeat for NB02 through NB08
```

Or open Jupyter Lab and run them interactively:

```bash
jupyter lab
```

**Expected runtimes:** NB01–04 run in under 5 minutes each. NB05 (hyperparameter tuning) takes 15–30 minutes. NB06 (SHAP) takes 3–5 minutes. NB07–08 are under 5 minutes each.

### 5. Outputs

All saved artefacts land in `outputs/`:

```
outputs/
├── best_model.joblib           # Trained XGBoost pipeline
├── model_metadata.json         # Hyperparameters and performance metrics
├── shap_plots/                 # SHAP visualisations and saved values
└── woe_plots/                  # WoE bar charts, IV ranking, model comparison
```

---

## Project Structure

```
credit-risk-analysis/
├── data/
│   ├── raw/                    # Downloaded Kaggle data (not in git)
│   ├── clean/                  # Cleaned dataset (not in git)
│   ├── engineered/             # Feature-engineered dataset (not in git)
│   └── splits/                 # Train/test splits (not in git)
├── notebooks/                  # NB01 through NB08
├── outputs/                    # Saved model, plots, metadata
├── requirements.txt            # Locked dependencies
└── README.md
```

---

---

## Development Notes

This project was built with [Claude Code](https://claude.ai/claude-code), Anthropic's agentic coding assistant. The project plan was designed using **Claude Opus**, with deliberate human-in-the-loop breakpoints inserted at key decision points — including imbalance strategy selection, threshold calibration, and model architecture comparison — to allow human review and optimization before proceeding to subsequent stages. This approach ensured that analytical judgement calls were validated rather than fully automated.

---

*Dataset source: [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset). Data files are excluded from this repository per Kaggle terms of service.*
