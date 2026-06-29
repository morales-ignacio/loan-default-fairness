# Loan Default Risk Model with Fairness Audit

A deployment-oriented credit risk modeling project on the LendingClub consumer loan dataset. The work goes beyond a single accuracy number: it uses a genuine out-of-time test, a profit-based decision threshold, model explainability (SHAP and counterfactual recourse), and a full disparate-impact fairness audit with mitigation costed in dollars.

## Overview

The task is binary classification: predict the probability that a loan will default, then turn that probability into an approve or reject decision. The decision is optimized for expected profit under an explicit cost basis rather than for raw accuracy, because in lending the cost of approving a defaulter and the cost of rejecting a good borrower are very different.

The project is built as a sequence of notebooks, each producing a written report. It covers data cleaning and leakage control, exploratory analysis, four model families benchmarked against profit-aware baselines, probability calibration, profit-based threshold selection, explainability, fairness, and a final held-out evaluation on a year the model never saw.

## Key results

The model was trained on 2014 to 2015 loans, tuned and calibrated on 2016, and evaluated once on 2017 (held out for the entire project).

| Metric | Validation 2016 | Test 2017 | Change |
|---|---|---|---|
| AUC (ranking) | 0.7281 | 0.7188 | -0.0093 |
| Brier score (calibration) | 0.1570 | 0.1587 | +0.0017 |
| Profit (cost basis below) | $65.57M | $30.86M | see note |
| Profit per loan | $223.70 | $182.26 | -18.5% |
| Rejection rate | 62.6% | 63.7% | +1.1 pts |

Highlights:

- **Generalizes out-of-time.** Ranking and calibration hold across the one-year gap (AUC drop under one point, Brier change under 0.002).
- **The operating point did not drift.** The threshold chosen on 2016 (0.1592) captures 98.6 percent of the maximum achievable 2017 profit. Re-tuning on 2017 labels would recover only $0.44M (1.4 percent).
- **The watch item is economic, not statistical.** Per-loan profit fell about 18 percent even though ranking and calibration held, which is the kind of softening only profit-level monitoring detects.
- **Calibration is near-exact.** On 2017, nine of ten probability deciles fall within 0.006 of perfect calibration, so the probabilities can be used directly for pricing and provisioning.

Profit cost basis: an approved loan that is repaid returns 10 percent of principal; an approved loan that defaults loses 50 percent of principal. The break-even probability under that basis is 0.1667, and the deployed threshold of 0.1592 sits just below it.

## Approach

**Data.** LendingClub accepted loans, 2007 to 2018. Filtered to completed loans (Fully Paid or Charged Off), giving 1,345,310 loans at a 19.96 percent overall default rate.

**Temporal split (out-of-time).** Train on 2014 to 2015 (598,647 loans), validate on 2016 (293,095, 23.3 percent default), test on 2017 (169,300, 23.1 percent default). 2018 was dropped for maturity bias (long loans had not matured by the data cutoff). A chronological split makes the evaluation a real generalization test, not a random hold-out.

**Leakage control.** A column-by-column audit dropped 49 post-decision or junk fields (payment history, recoveries, hardship and settlement data, updated FICO, identifiers), keeping only information available at the moment of the lending decision.

**Features.** 79 modeling features (72 numeric, 7 categorical), expanding to 155 after one-hot encoding. Redundant and high-cardinality columns were removed, and structurally missing joint-applicant fields were dropped.

**Models.** Eight variants were benchmarked against an approve-all floor and a grade-only baseline: logistic regression (with and without the interest-rate feature), random forest (with and without class balancing), and LightGBM (default and Optuna-tuned). The champion is a **calibrated, Optuna-tuned LightGBM**, which beat the grade-only baseline by +0.046 AUC.

**Calibration and threshold.** Isotonic calibration was fit on half of the 2016 validation set and evaluated on the other half. The decision threshold was set by maximizing expected profit under the cost basis above, landing at 0.1592, very close to the theoretical break-even of 0.1667.

## Fairness

The dataset contains no legally protected attributes (race, gender, age), so the audit examines observable proxies (income tier, home ownership, employment length, state) as stratifiers, not as protected classes. A finding of disparity is a prompt for investigation with real protected-attribute data, not a legal conclusion.

Findings:

- **Disparate impact on income and home ownership.** Income tier (demographic parity ratio 0.716) and home ownership (0.777) both fall below the EEOC four-fifths threshold.
- **The intersection is the most severe.** Income by home ownership reaches a demographic parity ratio of 0.614: low-income renters face a 76.7 percent rejection rate against 47.1 percent for high-income mortgage holders.
- **This is structural, not a bug.** It is the mathematical consequence of applying a single calibrated threshold to groups with different base rates (the calibration versus equalized-odds tension).
- **Mitigation is costed.** Profit-aware per-group thresholds lift every segment above the four-fifths line at a cost of $2.4M to $5.7M. The standard Fairlearn ThresholdOptimizer is profit-blind here and collapses to approving almost everyone, so a custom profit-aware approach was used. Per-group thresholds are disparate treatment and would require legal review before any literal deployment.

## Explainability and recourse

- **SHAP** confirms the model is interpretable and credit-sensible: loan term and interest rate are the heaviest levers, every directional effect matches credit theory, and the nonlinearities (a DTI threshold-and-saturation shape, an interest-rate S-curve) are sensible.
- **Counterfactual recourse** shows a structural limitation: the decisive features (term, interest rate) are not borrower-actionable, so the levers a borrower can pull are the model's weaker ones. Borderline rejections can sometimes flip with a smaller loan; deeper rejections cannot be undone by the actionable features alone.

## Repository structure

```
loan-default-fairness/
├── notebooks/
│   ├── 01_initial_exploration.ipynb      # load, target, default-rate by year and grade
│   ├── 02_leakage_audit.ipynb            # column audit, drop leakage, save clean parquet
│   ├── 03_eda.ipynb                      # exploratory analysis
│   ├── 04_baselines.ipynb                # approve-all and grade-only baselines
│   ├── 05_logreg.ipynb                   # logistic regression, int_rate ablation
│   ├── 06_random_forest.ipynb            # random forest, class-weight ablation
│   ├── 07_lightgbm.ipynb                 # LightGBM, Optuna tuning (50 trials)
│   ├── 08_calibration_thresholds.ipynb   # isotonic calibration, profit threshold, champion
│   ├── 09_shap_explainability.ipynb      # global and local SHAP
│   ├── 10_counterfactuals.ipynb          # actionable recourse search
│   ├── 11_fairness_audit.ipynb           # disparate-impact audit
│   ├── 12_fairness_mitigation.ipynb      # profit-aware mitigation
│   └── 13_test_set_evaluation.ipynb      # final out-of-time test on 2017
├── reports/
│   ├── figures/                          # all generated plots
│   ├── leakage_audit.md
│   ├── eda_findings.md
│   ├── model_evaluation.md
│   ├── shap_analysis.md
│   ├── counterfactual_analysis.md
│   ├── fairness_analysis.md
│   ├── test_set_evaluation.md
│   ├── model_card.md
│   ├── results.json                      # benchmark of all 8 model runs
│   └── test_set_2017_results.json        # final test metrics
├── data/                                 # LendingClub CSV (not tracked, see Reproducibility)
├── models/                               # champion and artifacts (not tracked, regenerated)
└── README.md
```

## Reports

Each notebook has a companion written report. Good entry points are the model card and the fairness analysis.

- [`model_card.md`](reports/model_card.md): consolidated card with intended use, performance, fairness, limitations, and a deployment recommendation.
- [`test_set_evaluation.md`](reports/test_set_evaluation.md): the final 2017 out-of-time evaluation, including the threshold-robustness and per-loan-profit findings.
- [`fairness_analysis.md`](reports/fairness_analysis.md): the full disparate-impact audit and the costed mitigation.
- [`shap_analysis.md`](reports/shap_analysis.md): global and local explainability.
- [`counterfactual_analysis.md`](reports/counterfactual_analysis.md): actionable recourse.
- [`model_evaluation.md`](reports/model_evaluation.md): the eight-model benchmark, calibration, and threshold work.
- [`eda_findings.md`](reports/eda_findings.md): exploratory findings.
- [`leakage_audit.md`](reports/leakage_audit.md): the keep/drop decision for every raw column.

## Reproducibility

```bash
git clone https://github.com/morales-ignacio/loan-default-fairness
cd loan-default-fairness
uv sync
# Place the LendingClub file at data/accepted_2007_to_2018Q4.csv
uv run jupyter notebook
```

Run the notebooks in order. Notebooks 01 to 08 build and freeze the champion model at `models/champion_lgbm.pkl`; notebooks 09 to 13 load that frozen champion and do not retrain it.

Notes:

- The raw data and the model artifacts are not version-controlled because of their size. The data is obtained separately and the champion is regenerated by running notebooks 01 to 08.
- The Optuna search in notebook 07 is seeded (`seed=42`). LightGBM training is not guaranteed bit-identical across environments, so a regenerated champion may differ negligibly from the reference run. The reference numbers used throughout the reports are captured in `reports/results.json` and `reports/test_set_2017_results.json`.

## Tech stack

Python 3.12, uv for environment management, pandas, scikit-learn, LightGBM 4.6, Optuna (TPE sampler), SHAP, Fairlearn, matplotlib and seaborn.

## Limitations

- **No protected attributes.** The fairness audit uses proxies and is not a substitute for an audit against real protected-attribute data.
- **Source-population bias.** The data contains only loans LendingClub already chose to originate, so the model learns "which approved loans default," not "which applicants would default." A broader deployment would require recalibration.
- **Cost-basis dependence.** Profit figures are conditional on the 10 percent gain and 50 percent loss assumptions. The threshold tracks those assumptions and would shift if real economics differ.
- **Single threshold, single audit period.** Fairness metrics are computed at the deployed threshold on 2016; other thresholds or periods would differ.
