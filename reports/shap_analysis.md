# SHAP Explainability Analysis

Global and local model explanations using SHAP (SHapley Additive exPlanations) applied to the champion calibrated LightGBM.

## Summary

Applied SHAP TreeExplainer to the champion model to produce both global explanations (which features matter across the population) and local explanations (why specific borrowers were rejected or approved). The analysis confirms that the model's decisions are interpretable and align with credit-risk domain knowledge in every direction tested.

Three findings worth highlighting:

1. **Loan term and interest rate are the two heaviest levers**, with term edging out int_rate (0.27 vs 0.26 mean |SHAP|). This is consistent with the int_rate ablation: removing int_rate cost only 0.001 AUC, meaning its signal is largely redundant with grade, term, and FICO, not that the model ignores it.
2. **Every directional effect is credit-sensible.** Longer terms, higher rates, higher DTI, larger loans, and more recent credit-seeking push risk up; higher FICO, grade A, and mortgages push it down. Nothing spurious or inverted.
3. **DTI shows a threshold-and-saturation shape**: risk is flat at low DTI, rises through a transition band, then plateaus once a borrower is already heavily leveraged. The model learned diminishing marginal penalty from data.

Plots generated for this analysis:
- `reports/figures/shap_global_importance.png`
- `reports/figures/shap_beeswarm.png`
- `reports/figures/shap_dependence_top3.png`
- `reports/figures/shap_waterfall_high_risk_rejection.png`
- `reports/figures/shap_waterfall_low_risk_approval.png`
- `reports/figures/shap_waterfall_borderline_case.png`

## Methodology

### What SHAP measures

SHAP values decompose any individual prediction into additive feature contributions:

```
prediction (log-odds) = base_value + sum(SHAP_i for each feature i)
```

Where `base_value` is the model's expected output over the sample, and each `SHAP_i` is the contribution of feature i (positive pushes toward default, negative toward repayment). For tree ensembles, TreeExplainer computes these exactly in polynomial time.

### Computation setup

- **Model**: champion LightGBM loaded from `models/champion_lgbm.pkl`
- **Data**: validation set 2016 (293,095 loans)
- **Global sample**: 5,000 randomly sampled borrowers (seeded) for the summary plots; distributions are stable well below full-val size
- **Local plots**: three borrowers selected from full val for waterfall analysis
- **Feature space**: SHAP applied to the post-preprocessor arrays (155 columns after one-hot encoding)
- **Base value**: -1.678 in log-odds (a baseline default probability of about 15.7%)

### Why the uncalibrated base model

SHAP is computed on the base LightGBM, not the calibrated output. Isotonic calibration is monotonic: it rescales probabilities but preserves feature ranking and direction. Global importance and directional effects are therefore identical whether or not calibration is applied, and the base model is the natural object to explain.

## Global feature importance

Ranked by mean absolute SHAP value across the 5,000-loan sample (`reports/figures/shap_global_importance.png`):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | term | 0.27 |
| 2 | int_rate | 0.26 |
| 3 | dti | 0.14 |
| 4 | loan_amnt | 0.12 |
| 5 | acc_open_past_24mths | 0.12 |
| 6 | grade_A | 0.11 |
| 7 | fico_range_low | 0.08 |
| 8 | grade_B | 0.07 |
| 9 | annual_inc | 0.06 |
| 10 | home_ownership_RENT | 0.05 |

### Key observations

**Term edges out interest rate.** Going in, the natural expectation was that `int_rate` would dominate, since it encodes LendingClub's own risk pricing. Instead the 36-vs-60-month decision carries the highest single-feature impact, with int_rate a close second.

This fits the int_rate ablation: when int_rate was dropped from the logistic-regression model, AUC fell by only 0.001. That does not mean the model ignores int_rate, SHAP shows it is the second-heaviest lever when present, but that its signal is redundant with grade, term, and FICO, so the model loses almost nothing without it. The model still reflects LendingClub's risk pricing; it simply does not depend on the rate feature specifically to do so.

**Grade indicators are asymmetric.** Both `grade_A` (#6) and `grade_B` (#8) rank in the top ten, but grades C through G do not appear in the top twenty. The model uses *being in a safer grade* as a protective signal, while the information about riskier grades is largely carried by `int_rate` (which LendingClub derives from grade).

**FICO ranks lower than its raw predictive power would suggest.** `fico_range_low` is #7 at mean |SHAP| 0.08, below int_rate, both grade indicators' combined effect, and loan_amnt. This is a redundancy effect, not the model ignoring credit score: grade and int_rate are themselves derived partly from FICO, so once they are in the model FICO's marginal contribution is largely already accounted for. The model reads the credit-score signal mostly through the rate and grade it was priced into.

**The rest of the top twenty are credit fundamentals**: credit-history length (`mo_sin_old_rev_tl_op`), active revolving accounts (`num_actv_rev_tl`), available credit (`total_bc_limit`), recent inquiries (`mths_since_recent_inq`), utilization (`percent_bc_gt_75`), and employment length, the same variables a human credit officer would check.

## Directional analysis (beeswarm)

The beeswarm plot (`reports/figures/shap_beeswarm.png`) shows the distribution of SHAP values per feature, colored by feature value (red = high, blue = low). Every directional effect in the top features aligns with credit-risk theory:

| Feature | Direction | Verdict |
|---|---|---|
| term | 60 months -> higher risk | Expected (EDA: 60-month loans default notably more than 36-month) |
| int_rate | higher rate -> higher risk | Expected (rate is priced from risk) |
| dti | higher DTI -> higher risk | Expected (more debt burden) |
| loan_amnt | larger loan -> higher risk | Expected (larger obligation) |
| acc_open_past_24mths | more recent accounts -> higher risk | Expected (credit-seeking signal) |
| grade_A | yes -> lower risk | Expected (safest tier) |
| fico_range_low | higher FICO -> lower risk | Expected (classical credit score) |
| annual_inc | higher income -> lower risk | Expected, narrower spread than the rest |
| home_ownership_RENT | yes -> higher risk | Expected (EDA: renters default more than mortgage holders) |
| mo_sin_old_rev_tl_op | longer history -> lower risk | Expected (history length is protective) |

The model has not latched onto spurious or inverted signals. This directional check is the kind of validation a credit-risk team performs before deployment and belongs in the eventual model card.

## Dependence plots

Dependence plots for the top three numeric drivers (`reports/figures/shap_dependence_top3.png`) reveal nonlinearities and interactions.

### Term: clean binary effect

`term` produces two tight, non-overlapping clusters: 36-month loans (scaled value ~ -0.59) sit around -0.15 to -0.25 SHAP (risk-reducing), 60-month loans (scaled ~ 1.71) around +0.35 to +0.75 (risk-increasing). The spread within the 60-month cluster suggests the term penalty is modulated by other features.

### Interest rate: smooth nonlinear effect

`int_rate` produces a monotonically increasing SHAP curve, steepest through the middle of the population and flattening at both extremes (very low rates already very safe, very high rates already very risky). This S-shape is exactly what tree models capture and linear models cannot, part of why LightGBM modestly outperformed logistic regression.

### DTI: threshold-and-saturation effect

`dti` produces a threshold-shaped curve:

- At low DTI, SHAP plateaus around -0.20: the model treats this whole range as "low DTI risk" without distinguishing further.
- Through a mid-range transition band, SHAP rises sharply from about -0.20 to roughly +0.30 to +0.40.
- At high DTI, the (sparse) points settle to a plateau near +0.20 to +0.25: once a borrower is risky on DTI, going higher adds little, if any, further penalty.

This diminishing-marginal-penalty shape is a sensible thing for the model to have learned from default outcomes alone. Note that the plot's x-axis is the *standardized* feature, so the transition band is not read directly in DTI percentage points here. To locate it in raw DTI and compare against benchmarks such as the CFPB qualified-mortgage threshold (43%), re-plot SHAP against unscaled DTI; that is a quick follow-up, not done in this pass.

### Term interaction check

The DTI dependence plot is colored by `term`, which would reveal an interaction if 60-month loans clustered in the high-SHAP region. Inspecting it, the two terms are interleaved across the whole curve with no clear separation, so the data here does not support a term-by-DTI interaction. A formal SHAP interaction-value computation could test this more rigorously, but the simple dependence coloring shows nothing notable.

## Local explanations: three case studies

Local SHAP waterfall plots decompose individual predictions into feature contributions (in log-odds, building from the -1.678 base value to the borrower's margin). Three borrowers were selected from val: the highest predicted-risk, the lowest, and the one closest to the operational threshold (0.1592). Exact per-feature contributions are read from the waterfall figures; the descriptions below summarize their direction.

### Case 1: high-risk rejection

| Attribute | Value |
|---|---|
| Predicted P(default) | 1.000 |
| Decision | REJECT |
| Actual outcome | DEFAULTED |
| Profile | $24,750, 60 months, 28.9% rate, grade G, $55k income, DTI 37.6%, FICO 670, RENT, credit_card |

The single highest-risk borrower in val: the calibrated probability saturates at the top of the isotonic range, so it rounds to 1.000 (this reflects the calibrator's top bin, not literal certainty). Every contribution shown points toward default, with no offsetting protective signal. The largest pushes are `int_rate` +0.73 (a 28.9% grade-G rate), `term` +0.58 (60 months), and `dti` +0.42 (37.6%), the three features that top the global ranking. Smaller positives from recent account openings (+0.21), the loan amount (+0.15), renting (+0.09), and a below-average FICO (+0.07) all add to the risk. The borrower defaulted.

### Case 2: low-risk approval

| Attribute | Value |
|---|---|
| Predicted P(default) | 0.000 |
| Decision | APPROVE |
| Actual outcome | PAID |
| Profile | $5,000, 36 months, 5.3% rate, grade A, $131k income, DTI 11.3%, FICO 800, MORTGAGE, home_improvement |

The mirror image, calibrated probability rounds to 0.000 (bottom isotonic bin). The dominant protective signal is `int_rate` -0.81 (a 5.3% rate), followed by FICO -0.30 (FICO 800), `grade_A` -0.30, the 36-month term (-0.21), and low DTI (-0.19). Almost everything pulls away from default. The one contribution pushing the other way is the borrower's state, `addr_state_NJ` at +0.11, a small geographic effect flagged for the fairness audit below. Paid, as expected.

### Case 3: borderline case (most operationally valuable)

| Attribute | Value |
|---|---|
| Predicted P(default) | 0.1579 |
| Threshold | 0.1592 |
| Decision | APPROVE (by 0.0013) |
| Actual outcome | PAID |
| Profile | $2,150, 36 months, 17.0% rate, grade D, $47.5k income, DTI 22.0%, FICO 690, MORTGAGE, other |

The borrower sits just below the cutoff and is approved by a hair. The decision is essentially a tug-of-war between two features: the very small $2,150 loan pulls hard toward repayment (`loan_amnt` -0.51), while the 17% grade-D rate pushes toward default (`int_rate` +0.32). The 36-month term adds another -0.20 downward, and everything else is individually tiny. Notably, DTI contributes almost nothing here despite sounding high at 22%, because at 0.41 standard deviations it sits near the population average, right where the DTI curve crosses zero. The net lands a fraction under the threshold. Approved, and repaid: a correct call right at the margin.

**Operational implication**: near-threshold cases like this are exactly the ones to route to a human reviewer rather than auto-decision. Here the stakes are low (a $2,150 approval the model is only marginally confident about), but the same marginal confidence on a large loan is where a second set of eyes earns its keep. The narrow margin reflects genuine ambiguity in the data, not a flaw in the model.

## Implications

### What this validates

1. **Directional sensibility**: every important feature affects predictions the way credit-risk theory predicts.
2. **No critical dependence on the rate feature**: term edges out int_rate, and the ablation shows int_rate is redundant, even though the model still reflects risk-based pricing through grade and term.
3. **Sensible nonlinearities**: the DTI threshold-and-saturation shape and the int_rate S-curve are learned from data and are the kind of structure that justified moving from logistic regression to LightGBM.
4. **Confident predictions are well-supported**, and **mid-confidence predictions reveal genuine ambiguity** rather than spurious certainty.

### What this enables

Local waterfall plots are the foundation of a decision-support tool. They answer "why was this borrower rejected" in a form a credit officer can act on, a regulator can audit, and a borrower can be told about (relevant to adverse-action notices under the Equal Credit Opportunity Act).

### What remains to verify

SHAP shows what the model does, not whether it is fair. The fairness audit should examine, in particular:

1. **The grade pathway.** The model uses being in grade A or B as a protective signal, but grade is derived from FICO and other attributes that may correlate with protected characteristics. The audit must check whether protected groups are systematically disadvantaged through grade.
2. **Geographic features.** `addr_state` is in the feature set, and it surfaced directly in one case study: the low-risk borrower's state (New Jersey) contributed +0.11. State of residence should not be a meaningful predictor of individual default in a fair model, so the audit should check whether `addr_state` effects reflect genuine regional economic differences or encode bias.

These are the questions the fairness audit will address.

## Limitations

- **One-hot categoricals**: each category (`grade_A`, `addr_state_CA`, etc.) appears as its own SHAP feature. This is correct but slightly noisy; per-variable attribution would require summing a variable's category contributions, which was not done here.
- **Standardized values in plots**: numeric features are scaled, so dependence and waterfall axes are in standard-deviation units, not original units. Plain-language borrower profiles accompany the waterfalls to compensate.
- **Sample size for global plots**: importances are from 5,000 sampled borrowers; the ranking is stable but exact values carry sampling noise.
- **Local sign surprises**: individual attributions can occasionally show a feature contributing against its global direction, due to interactions with a borrower's other extreme values. These do not generalize and are expected with correlated risk signals; read the waterfall figures directly when interpreting a specific borrower.
- **Evaluation-time only**: SHAP was computed on val. The model's behavior on test (2017) is reserved for the model card.

## Reproducibility

The full analysis is in `notebooks/09_shap.ipynb`. To reproduce:

```bash
git clone https://github.com/morales-ignacio/loan-default-fairness
cd loan-default-fairness
uv sync
# Place LendingClub data at data/accepted_2007_to_2018Q4.csv
# Run notebooks 01-08 to generate models/champion_lgbm.pkl
# Then open notebooks/09_shap.ipynb
```

The notebook loads the champion from `models/champion_lgbm.pkl` and recomputes all SHAP values inline.