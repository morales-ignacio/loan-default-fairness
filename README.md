# Loan Default Prediction with Fairness Audit

> **Status:** In progress — updated as I build it out.

An end-to-end machine learning system that predicts consumer loan default probability, focused on what most credit risk projects skip:

- **Temporal validation** instead of random train/test splits
- **Calibrated probabilities**, not just classifications
- **SHAP-based explanations** for every prediction
- **Counterfactual analysis** — what would need to change to flip a decision
- **Fairness audit** across demographic proxies, with mitigation experiments and honest tradeoff analysis

## The problem

Using LendingClub historical data (2014-2018, ~2M loans), build a model that predicts loan default probability accurately *and* makes fair predictions across demographic groups (state, zip-code proxies, income brackets, age via credit history length).

The model supports two decisions: (1) approve/deny, and (2) interest rate setting based on predicted risk.

## Planned approach

1. **Foundation** — EDA, comprehensive leakage audit, temporal train/val/test split
2. **Modeling** — Baselines (approve-all, grade-only), logistic regression, LightGBM with Optuna, calibration via isotonic regression
3. **Explainability + Fairness** — SHAP global and local, DiCE counterfactuals, Fairlearn audit, mitigation experiments
4. **Deployment** — FastAPI backend, React frontend, deployed live with model card and fairness dashboard

## Tech stack

Python, pandas, scikit-learn, LightGBM, MLflow, SHAP, DiCE, Fairlearn, FastAPI, Docker, React, Tailwind

## Project structure
```
loan-default-fairness/
├── data/         # raw data (gitignored)
├── notebooks/    # EDA, exploration
├── src/          # reusable modules
├── models/       # trained artifacts (gitignored)
├── reports/      # leakage audit, fairness audit, model card
├── app/          # FastAPI backend + frontend
└── tests/
