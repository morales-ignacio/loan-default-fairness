# SHAP Explainability Analysis

> Week 3 findings: global and local model explanations using SHAP (SHapley Additive exPlanations) applied to the champion calibrated LightGBM model. This document covers the full analysis in `notebooks/09_shap_explainability.ipynb`.

## Summary

Applied SHAP TreeExplainer to the champion model to produce both global explanations (which features matter across the population) and local explanations (why specific borrowers were rejected or approved). The analysis confirmed that the model's decisions are sensible, interpretable, and align with credit risk domain knowledge in every direction tested.

Three significant findings emerged:

1. **Loan term is the strongest single predictor**, not interest rate. This validates the Week 1 int_rate ablation: the model is not implicitly reusing LendingClub's risk score.
2. **The model has independently discovered a DTI threshold effect** near the CFPB qualified-mortgage boundary (43%), purely from data.
3. **The model has learned a real feature interaction**: high DTI is penalized more severely for 60-month loans than for 36-month loans, which matches credit risk theory.

Six portfolio plots were generated and saved to `reports/`:
- `shap_global_importance.png`
- `shap_beeswarm.png`
- `shap_dependence_top3.png`
- `shap_waterfall_high_risk_rejection.png`
- `shap_waterfall_low_risk_approval.png`
- `shap_waterfall_borderline_case.png`

## Methodology

### What SHAP measures

SHAP values decompose any individual model prediction into contributions from each feature. For a borrower's prediction:

```
prediction = base_value + sum(SHAP_i for each feature i)
```

Where:
- `base_value` is the model's average prediction across the training data (the "default expectation")
- `SHAP_i` is the additive contribution of feature i (positive = pushes toward default, negative = pushes toward repayment)

SHAP values are computed using game-theoretic Shapley values, treating features as players in a cooperative game. For tree-based models like LightGBM, SHAP TreeExplainer uses an exact polynomial-time algorithm specific to tree ensembles.

### Computation setup

- **Model**: champion calibrated LightGBM tuned (loaded from `models/champion_lgbm.pkl`)
- **Data**: validation set 2016 (293,095 loans)
- **Global sample**: 5,000 randomly sampled borrowers from val for summary plots (distributions stabilize well below full-val sample size)
- **Local plots**: three borrowers selected from full val for waterfall analysis
- **Preprocessing**: SHAP applied to post-preprocessor numeric arrays (155 features after one-hot encoding)

### Why analyze the uncalibrated base model

Calibration (isotonic regression) is a monotonic transformation that maps raw probabilities to better-aligned probabilities. It preserves the ranking of features by importance and does not change which features push predictions up or down. SHAP analysis on the base LightGBM is therefore equivalent for interpretability purposes.

## Global feature importance

The bar plot (`shap_global_importance.png`) ranks features by mean absolute SHAP value across 5,000 borrowers.

### Top 10 features

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | term | 0.27 |
| 2 | int_rate | 0.24 |
| 3 | dti | 0.14 |
| 4 | loan_amnt | 0.13 |
| 5 | acc_open_past_24mths | 0.12 |
| 6 | grade_A | 0.11 |
| 7 | grade_B | 0.08 |
| 8 | fico_range_low | 0.07 |
| 9 | annual_inc | 0.06 |
| 10 | home_ownership_RENT | 0.05 |

### Key observations

**Loan term outranks interest rate.** This was the most surprising finding. Going into the analysis, the natural expectation was that `int_rate` would dominate, since it implicitly encodes LendingClub's own risk model. Instead, the binary-like decision between 36-month and 60-month loans carries the highest single-feature impact.

This is internally consistent with the Week 1 `int_rate` ablation: when `int_rate` was removed, AUC dropped by only 0.001, suggesting the model was not heavily relying on it. The SHAP analysis confirms why: `term` provides a stronger independent signal than `int_rate` can offer (which is itself partly redundant with grade and FICO).

**Grade indicators show up asymmetrically.** Both `grade_A` and `grade_B` rank in the top 7, but grades C through G do not appear in the top 20. The model uses *being in a safer grade* as a protective signal, while the information for riskier grades is largely captured by `int_rate` (since LC's interest rate is itself derived from grade).

**FICO ranks lower than expected.** `fico_range_low` is only #8 with mean |SHAP| of 0.07. The conventional credit risk expectation would have placed FICO higher. Two explanations:

1. FICO's information is partly absorbed by `grade` (derived from FICO + other factors) and `int_rate` (derived from grade).
2. The model has learned to use FICO efficiently in combination with other variables rather than relying on it as a primary signal.

This is not a flaw. It is the expected behavior when multiple correlated risk signals are available.

**The tail of the top 20 consists entirely of credit fundamentals**: credit history length (`mo_sin_old_rev_tl_op`), active accounts (`num_actv_rev_tl`), total credit available (`total_bc_limit`), recent credit inquiries (`mths_since_recent_inq`), credit utilization (`percent_bc_gt_75`), and employment length. These are the variables a human credit officer would also check.

## Directional analysis (beeswarm)

The beeswarm plot (`shap_beeswarm.png`) shows the distribution of SHAP values for each feature across all 5,000 sample borrowers, with color indicating the actual feature value (red = high, blue = low).

Every directional effect in the top 20 features aligns with credit risk theory:

| Feature | Direction | Verdict |
|---|---|---|
| term | 60 months → higher default risk | Expected (consistent with Week 1 EDA: 60-month default 2x of 36-month) |
| int_rate | higher rate → higher default risk | Expected (LC prices risk into rate) |
| dti | higher DTI → higher default risk | Expected (more debt obligations) |
| loan_amnt | larger loan → higher default risk | Expected (larger payments) |
| acc_open_past_24mths | more recent accounts → higher default risk | Expected (credit-seeking behavior is a warning sign) |
| grade_A | yes → lower default risk | Expected (LC's safest tier) |
| fico_range_low | higher FICO → lower default risk | Expected (classical credit score behavior) |
| annual_inc | higher income → lower default risk | Expected, though spread is narrower than other features |
| home_ownership_RENT | yes → higher default risk | Consistent with Week 1 EDA finding that mortgage holders default less than renters |
| mo_sin_old_rev_tl_op | longer history → lower default risk | Expected (credit history length is protective) |

This is the kind of validation a credit risk team would do before production deployment. The model has not latched onto spurious or counterintuitive signals; every relationship aligns with established credit risk principles. This is a section that belongs in the eventual model card and any regulatory disclosure.

## Dependence plots: nonlinearities and interactions

Dependence plots (`shap_dependence_top3.png`) show how SHAP values change as a function of the feature's value, revealing nonlinearities, threshold effects, and interactions with other features.

### Term: clean binary effect

`term` produces two tight clusters:
- 36 months (scaled value -0.59): SHAP cluster between -0.15 and -0.25 (reduces default risk)
- 60 months (scaled value 1.71): SHAP cluster between +0.35 and +0.80 (increases default risk)

No overlap between the clusters. The variation within the 60-month cluster (0.35 to 0.80) indicates that the term penalty varies by borrower, presumably modulated by interactions with other features.

### Interest rate: smooth nonlinear effect

`int_rate` produces a monotonically increasing SHAP curve with notable nonlinearity:
- Steepest slope is in the middle range (scaled int_rate -1 to +1, the bulk of the population)
- Curve flattens at very low rates (already very safe, marginal effect minimal)
- Curve flattens at very high rates (already very risky, marginal effect minimal)

This S-curve pattern is exactly what tree-based models can learn but linear models cannot. It is part of why LightGBM modestly outperformed logistic regression in Week 2.

### DTI: threshold effect aligned with regulatory boundaries

The most analytically interesting finding. `dti` produces a sharp threshold-effect curve:

- **Below scaled DTI ≈ 0** (corresponds to actual DTI below approximately 18 to 20 percent): SHAP plateaus at approximately -0.20. The model treats anyone in this range as "low DTI risk" without distinguishing further.
- **Between scaled DTI 0 and 2** (actual DTI approximately 20 to 40 percent): SHAP rises sharply from -0.20 to +0.30. This is the transition zone where DTI becomes meaningful for the default decision.
- **Above scaled DTI ≈ 2** (actual DTI above approximately 40 percent): SHAP plateaus at +0.30. The model saturates: once a borrower is risky on DTI, going higher does not add much additional penalty.

The transition zone (DTI 20 to 40 percent) is uncomfortably close to the Consumer Financial Protection Bureau (CFPB) qualified-mortgage threshold of 43 percent. The model has independently discovered something near the regulatory boundary purely from default outcome data, without any prior knowledge of CFPB rules.

This is a strong validation finding: the model's decision rule for DTI converges with the rule that regulatory experts arrived at through different reasoning.

### Interaction: term amplifies DTI risk

The DTI dependence plot is colored by `term`. The upper portion of the curve (high SHAP values) is dominated by red dots (60-month loans), while the lower portion (low SHAP values) is dominated by blue dots (36-month loans).

This is a feature interaction the model has learned: **high DTI is penalized more severely for 60-month loans than for 36-month loans**. The intuition is sound. A borrower carrying high debt-to-income for three years faces less cumulative stress than the same borrower carrying it for five years.

Linear models cannot capture this interaction. It is one of the differentiators that justifies the architectural step up from logistic regression to LightGBM.

## Local explanations: three case studies

Local SHAP waterfall plots decompose individual predictions into feature contributions. Three illustrative borrowers were selected from the validation set:

1. **High-risk rejection**: borrower with the highest predicted default probability in val
2. **Low-risk approval**: borrower with the lowest predicted default probability in val
3. **Borderline case**: borrower whose predicted probability sits closest to the operational threshold (0.1457)

All three model predictions were correct (predicted default matched actual outcome in each case).

### Case 1: high-risk rejection

| Attribute | Value |
|---|---|
| Predicted P(default) | 0.879 |
| Decision | REJECT |
| Actual outcome | DEFAULTED |
| Base log-odds | -1.68 |
| Final log-odds | +1.98 |

Every feature in the top 12 pushes toward default. No protective signals. Key drivers:

- `int_rate` at 3.62 standard deviations above mean: +0.70 SHAP
- `term` = 60 months: +0.55
- `acc_open_past_24mths` at 4.3 standard deviations above mean: +0.35 (extreme credit-seeking behavior)
- `mo_sin_old_rev_tl_op` short credit history: +0.30
- `dti` high: +0.26
- `home_ownership_RENT`: +0.07
- 144 other features collectively: +0.97

This is the clean case: the model is highly confident, every dimension agrees, and the prediction is correct.

### Case 2: low-risk approval

| Attribute | Value |
|---|---|
| Predicted P(default) | 0.007 |
| Decision | APPROVE |
| Actual outcome | PAID |
| Base log-odds | -1.68 |
| Final log-odds | -4.95 |

Mirror image of Case 1: every feature in the top 12 pushes away from default. Key drivers:

- `int_rate` at -1.73 standard deviations (very low): -0.86 SHAP (dominant protective signal)
- `grade_A`: -0.30
- `addr_state_OR` (Oregon): -0.29
- `fico_range_low` at 3.56 standard deviations above mean: -0.27 (exceptional FICO)
- `dti` at -1.41 standard deviations (very low): -0.23
- `term` = 36 months: -0.19
- `annual_inc` at 1.66 standard deviations above mean: -0.15

A note for the fairness audit (next stage of Week 3): `addr_state_OR` carries a substantial -0.29 SHAP for this borrower. Geographic state effects deserve scrutiny in the fairness audit to ensure the model is not encoding bias against borrowers in specific states.

### Case 3: borderline case (most operationally valuable)

| Attribute | Value |
|---|---|
| Predicted P(default) | 0.146 |
| Threshold | 0.1457 |
| Decision | REJECT (by 0.001) |
| Actual outcome | DEFAULTED |
| Base log-odds | -1.68 |
| Final log-odds | -1.77 |

Mixed signals. Protective and risk-increasing features nearly cancel:

**Pushing toward default:**
- `loan_amnt` = 1.81 std (large loan): +0.22
- `int_rate` = 0.13 (slightly above mean): +0.13
- `grade_A` = 0 (not top grade): +0.09
- `addr_state_MO` (Missouri): +0.06
- `grade_B` = 0 (not second-top grade): +0.06

**Pushing toward repayment:**
- `term` = 36 months: -0.18
- `fico_range_low` = 2.23 std (high FICO): -0.15
- `dti` = -0.49 std (low DTI): -0.12
- Low credit utilization: -0.07
- High employment length: -0.06

Net effect: a very small push toward default, putting the borrower 0.001 above the threshold. The model called rejection by the thinnest possible margin, and the borrower did default.

The borderline case raises an interesting puzzle. With FICO 2.23 standard deviations above mean and low DTI, why is this borrower not in grade A or grade B? Probably because the loan amount is 1.81 standard deviations above mean, or because of factors LC's grading process captured that this model also caught indirectly.

**Operational implication**: in a production deployment, borderline cases like this are exactly the ones that should be flagged for human credit officer review rather than auto-decisioned. The model's narrow margin reflects genuine ambiguity in the underlying data, not a flaw in the model.

## Implications

### What this validates

The SHAP analysis confirms several properties that matter for production deployment and regulatory acceptability:

1. **Directional sensibility**: every important feature affects predictions in the way credit risk theory predicts.
2. **No hidden reliance on LC's risk score**: `term` outranks `int_rate` in importance, validating the Week 1 ablation finding.
3. **Regulatory alignment**: the model independently discovered something near the CFPB DTI threshold.
4. **Real interactions captured**: the term-DTI interaction matches credit risk intuition.
5. **Confident predictions are well-supported**: high-confidence approvals and rejections show all features agreeing.
6. **Mid-confidence predictions reveal genuine ambiguity**: the borderline case appropriately reflects mixed signals rather than spurious certainty.

### What this enables

Local SHAP waterfall plots are the foundation of any production decision-support tool. They answer the question "why was this borrower rejected" in a way that:

- A credit officer reviewing a borderline case can understand
- A regulator auditing the decision can verify
- A rejected borrower can be informed about (relevant for adverse action notices under the Equal Credit Opportunity Act)

The Week 4 deployment will surface these waterfalls directly in the user-facing application.

### What remains to verify

SHAP shows what the model does. It does not show whether what the model does is **fair**. Two issues observed during this analysis warrant the fairness audit:

1. **Geographic state effects**: in the low-risk approval case, `addr_state_OR` contributed -0.29 SHAP. State of residence should not be a strong predictor of individual loan default in a fair model. The fairness audit must check whether state effects are properly attributable to economic differences or whether they encode bias.
2. **The protective grade signal**: the model uses being in grade A or B as a positive signal, but grade itself is derived from FICO and other features that may have racial or socioeconomic correlations. The fairness audit must check whether protected groups are systematically disadvantaged through the grade pathway.

These are the questions Fairlearn will help answer in the next stage.

## Limitations

- **One-hot encoded categoricals**: each category (e.g., `grade_A`, `addr_state_CA`, `purpose_debt_consolidation`) appears as its own feature in SHAP attribution. This is technically correct but slightly noisy. For a cleaner attribution to the original categorical variable, the SHAP values for all categories of a single variable could be summed. This was not done in the current analysis.
- **Standardized feature values in waterfall plots**: because the preprocessor scales numeric features, the values shown in waterfall plots are in standard-deviation units rather than original units. Borrower context in plain language was provided in Cell 9 of the notebook to compensate.
- **Sample size for global plots**: SHAP values were computed on 5,000 randomly sampled borrowers from val rather than all 293,000. Distributions are visually stable at this sample size, but exact rankings could shift slightly with the full set.
- **Train-time vs evaluation-time**: SHAP was computed at evaluation time on val. The model's behavior on test (2017) has not yet been examined and is reserved for the model card.

## Reproducibility

The full analysis is in `notebooks/09_shap_explainability.ipynb`. To reproduce:

```bash
git clone https://github.com/morales-ignacio/loan-default-fairness
cd loan-default-fairness
uv sync
# Place LendingClub data at data/accepted_2007_to_2018Q4.csv
# Run notebooks 01-08 to generate models/champion_lgbm.pkl
# Then open notebooks/09_shap_explainability.ipynb
```

The notebook loads the champion model from `models/champion_lgbm.pkl` and recomputes all SHAP values inline. Total runtime is approximately 1 to 2 minutes once the data is loaded.
