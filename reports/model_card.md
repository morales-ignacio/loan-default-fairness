# Model Card: Loan Default Risk Classifier

## Model details

- **Task:** Binary classification of loan applications, predicting the probability that a loan will default.
- **Model:** Gradient-boosted decision trees (LightGBM), with isotonic calibration applied to the raw scores so that predicted probabilities match observed default rates.
- **Decision rule:** A loan is rejected when the calibrated probability of default is at or above the decision threshold of 0.1592, and approved otherwise.
- **Objective:** The threshold was chosen to maximize expected profit under an explicit cost basis: an approved loan that is repaid returns 10 percent of principal, and an approved loan that defaults loses 50 percent of principal. The break-even probability under that cost basis is 0.1667, and the deployed threshold sits just below it.
- **Artifacts:** Champion model and preprocessing pipeline serialized in models/champion_lgbm.pkl. Analysis notebooks 09 (explainability), 10 (counterfactuals), 11 (fairness audit), 12 (fairness mitigation), 13 (out-of-time evaluation).

## Intended use

The model is intended to support approve/reject decisions on consumer loan applications, as one input to an underwriting process. It produces a calibrated default probability and a profit-driven recommendation at the deployed threshold.

It is not intended to be the sole basis for a credit decision, to set interest rates, or to be applied to populations materially different from the training distribution (US consumer loans of the LendingClub type) without revalidation.

## Training and evaluation data

The data is the LendingClub consumer loan dataset. The split is chronological, which makes evaluation a genuine out-of-time test rather than a random hold-out:

- Training: loans issued before 2016.
- Validation: loans issued in 2016 (293,095 loans, 23.3 percent default rate). Used for model selection, calibration, and threshold choice.
- Test: loans issued in 2017 (169,300 loans, 23.1 percent default rate). Held out entirely during development and used only for final evaluation.

Key preprocessing: redundant and leakage-prone columns dropped; very high-cardinality and post-decision fields removed; employment length mapped to an ordinal scale; loan term parsed to an integer; sentinel and out-of-range values cleaned (debt-to-income and credit-history sentinels mapped to missing, income and revolving-utilization capped); missing values imputed with training-set medians inside the preprocessing pipeline; base-model scores calibrated with isotonic regression.

## Performance: out-of-time validation (2016 to 2017)

The central question for deployment is whether conclusions drawn on 2016 hold on a year the model never saw. They do, with minimal degradation.

| Metric | Validation 2016 | Test 2017 | Change |
|---|---|---|---|
| AUC (ranking quality) | 0.7281 | 0.7188 | -0.0093 |
| Brier score (calibration) | 0.1570 | 0.1587 | +0.0017 |
| Rejection rate | 62.6% | 63.7% | +1.1 pts |
| FPR (wrongful rejection) | 55.8% | 57.3% | +1.5 pts |
| FNR (missed default) | 15.0% | 14.8% | -0.2 pts |

Ranking quality and calibration are both essentially stable. The model continues to separate good from bad risk and to produce trustworthy probabilities a full year forward. Operating behavior (rejection rate and error rates) is stable, slightly more conservative in 2017.

**Threshold robustness.** Re-optimizing the threshold against the 2017 labels (a diagnostic that cannot be done in production, since test outcomes are unknown at decision time) yields an optimal of 0.1700, which sits essentially on the break-even point of 0.1667. The deployed 0.1592 threshold captures 98.6 percent of the maximum 2017 profit: re-tuning would add only 0.44M dollars, about 1.4 percent. The profit surface is a broad, flat plateau around the optimum, so the operating point is robust to small threshold changes. The threshold did not overfit to 2016.

**Profit and per-loan economics.** At the deployed threshold, profit is 65.57M dollars on 2016 (293,095 loans) and 30.86M dollars on 2017 (169,300 loans). These absolute figures are not directly comparable because the loan counts and dollar volumes differ. On a per-loan basis, profit is about 224 dollars per loan in 2016 and about 182 dollars per loan in 2017, a decline of roughly 18 percent. This is a notable point: the statistical metrics (AUC, calibration) held, but per-loan profitability compressed materially. Profitability should be monitored directly over time, not inferred from ranking metrics alone.

## Calibration

On the 2017 test set, calibration is near-exact. Nine of ten probability deciles fall within 0.006 of the perfect-calibration line. The only meaningful deviation is the top decile (the riskiest applicants), where the model predicts 0.543 against an observed default rate of 0.506, an over-estimate of about 3.7 points. This over-prediction is in the conservative direction for a lender (it slightly over-states the risk of the riskiest applicants). The decile that contains the decision threshold is calibrated to within 0.001, which is why the deployed operating point behaves on 2017 the way it did on 2016. The calibrated probabilities cap near 0.54, a property of the isotonic calibrator. Figure: reports/figures/calibration_test_2017.png.

## Fairness

A disparate-impact audit examined whether decisions and errors fall unevenly across observable applicant segments (income tier, home ownership, employment length, state of residence). The legally protected demographic attributes are not present in the data, so the audit speaks to these observable segments on their own terms, not to protected classes.

Summary of disparity (demographic parity ratio and equalized odds ratio; the EEOC four-fifths rule treats below 0.80 as evidence of disparate impact):

| Segment | DP ratio | EO ratio | Verdict |
|---|---|---|---|
| Income tier | 0.716 | 0.687 | Disparate impact, mitigatable |
| Home ownership | 0.777 | 0.764 | Disparate impact, mitigatable |
| Employment length | 0.821 | 0.810 | Passes, but driven by the missing-data group |
| State of residence | 0.634 | n/a (50 groups) | Large spread, mostly tracks real risk |
| Income x home ownership | 0.614 | 0.585 | Severe disparate impact, mitigatable |

The intersection of income and home ownership is the most severe: low-income renters face a 76.7 percent rejection rate against 47.1 percent for high-income mortgage holders. Across every segment, the same structure appears: higher-base-risk groups receive higher rejection and higher wrongful-rejection rates. This is a mathematical consequence of a single calibrated threshold over groups with different base rates (the calibration versus equalized-odds tension), not an idiosyncratic flaw.

This pattern is not specific to 2016. The wrongful-rejection rate (FPR) remains high in 2017 (57.3 percent), confirming the disparity is structural to the conservative operating point, not an artifact of one year.

Mitigation is feasible. Per-group thresholds that equalize selection rates lift every segment above the four-fifths threshold (mitigated ratios above 0.92) at a profit cost of 6.5 to 8.7 percent, or 4.2M to 5.7M dollars, depending on the segment. The full profit-fairness frontier is compliant: the cheapest compliant point, which gets there by accepting slightly more lending volume, costs 2.4M dollars (3.7 percent). There is no configuration that beats the baseline on both profit and fairness. Full detail in the fairness analysis document and notebooks 11 and 12.

## Limitations

- **No protected attributes.** The data omits the demographic attributes that fair-lending law protects, so the audit is limited to observable segments and is not a substitute for an audit against protected attributes.
- **Mitigation is a measurement, not a deployable control.** The per-group thresholds quantify the cost of equalizing outcomes, but setting a decision threshold by group membership is disparate treatment and would require legal review before any literal deployment. A compliant implementation would more likely adjust the model or its features.
- **Per-loan profitability drifted.** Ranking and calibration held out-of-time, but per-loan profit fell about 18 percent from 2016 to 2017. Statistical stability does not guarantee economic stability.
- **Single threshold, single audit period.** All audit metrics are computed at the 0.1592 threshold on 2016. Behavior at other thresholds or in other periods would differ.
- **Calibration ceiling.** The isotonic calibrator caps predicted probabilities near 0.54, so the model cannot express very high default probabilities; the top risk decile is over-predicted.

## Deployment recommendation

- **Threshold.** Deploy at 0.1592, or equivalently at the break-even 0.1667; both are near-optimal and within 1.4 percent of each other on 2017. The flat profit plateau means precision to the fourth decimal is not required.
- **Fairness configuration.** Choose and disclose one: hold overall volume and apply per-group thresholds (cost 4.2M to 5.7M dollars), target the income-by-home intersection directly (cost 5.7M dollars, addresses the worst-affected segment), or accept slightly more volume for the smallest fairness cost (2.4M dollars). Any per-group implementation requires legal review.
- **Adverse-action notices.** Rejections must carry principal reasons; the SHAP explanations in notebook 09 can generate them per decision.
- **Recourse.** Borderline rejections can often be approved with a smaller loan, as shown in notebook 10. Automated counter-offers are a low-cost improvement.

## Monitoring

- AUC and Brier drift against the 2016 and 2017 baselines.
- Per-loan profitability, tracked directly, given the 2017 compression.
- Calibration, especially in the threshold region.
- Approval-rate spread across regions over time, the dimension with the widest spread.

## Reproducibility

- Notebooks: 11_fairness_audit, 12_fairness_mitigation, 13_test_set_evaluation.
- Saved results: reports/test_set_2017_results.json.
- Figures: reports/figures/calibration_test_2017.png, threshold_analysis_test_2017.png, and the fairness figures referenced in the fairness analysis document.
- Profit basis: repaid approved loan returns 10 percent of principal, approved default loses 50 percent of principal.