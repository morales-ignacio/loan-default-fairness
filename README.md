# Loan Default Prediction with Fairness Audit

> **Status:** In progress — updated as I build it out.

An end-to-end machine learning system that predicts consumer loan default probability, focused on what most credit risk projects skip:

- **Calibrated probabilities** - not just classifications
- **SHAP-based explanations** - global and for individual predictions
- **Counterfactual analysis** — what would need to change to flip a decision
- **Fairness audit** - across demographic proxies, with intersectional analysis and profit-aware mitigation
- **Temporal validation and out-of-time testing** — a final check on the most recent loan vintages, held out entirely from training, the realistic deployment scenario rather than a random holdout


## The problem

Using LendingClub historical data (2014-2018, ~2M loans), build a model that predicts loan default probability accurately *and* makes fair predictions across demographic groups (state, zip-code proxies, income brackets, age via credit history length).

The model supports the core lending decision: approve or deny an application based on predicted default risk, with the decision threshold tuned for profit rather than raw accuracy.


## Data
 
This project uses the LendingClub accepted-loans dataset, available on Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club
 
Download `accepted_2007_to_2018Q4.csv` and place it in `data/`. The analysis filters to loans originated in 2014-2018. The file is gitignored because of its size (~1.6 GB), so it is not included in this repo.


## Planned approach

1. **Foundation** — EDA, comprehensive leakage audit, temporal train/val/test split
2. **Modeling** — Baselines (approve-all, grade-only), logistic regression, LightGBM with Optuna, calibration via isotonic regression
3. **Explainability + Fairness** — SHAP global and local, counterfactual recourse analysis, Fairlearn fairness audit with intersectional analysis and profit-aware mitigation
4. **Validation** — out-of-time evaluation on held-out recent vintages, plus a model card documenting performance, calibration, and limitations


## Tech stack

Python, uv, scikit-learn, LightGBM, Optuna, SHAP, Fairlearn, pandas, numpy, matplotlib, seaborn, Jupyter Notebooks


## Project structure
```
loan-default-fairness/
├── data/         # raw data (gitignored)
├── notebooks/    # EDA, exploration
├── models/       # trained artifacts (gitignored)
└── reports/      # leakage audit, fairness audit, model card
