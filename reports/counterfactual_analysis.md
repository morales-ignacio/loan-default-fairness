# Counterfactual Recourse Analysis

## Question this answers

Given a borrower the model rejects, what is the smallest change to their application that would flip the decision to approval? This is "actionable recourse." SHAP tells us which features matter for a prediction; counterfactuals tell us what specifically would need to change for the outcome to be different. The two are complementary explainability tools.

## Why we did not use DiCE

DiCE (Diverse Counterfactual Explanations) is the standard library for this task, but it assumes the model uses its default 0.5 probability threshold for the positive class. Our champion model uses a custom profit-optimized threshold of 0.1457 (chosen in Week 2 based on validation profit curves). DiCE's internal optimization could not reconcile this and consistently returned "no counterfactuals found" even for borrowers we knew were close to the decision boundary.

This is a real gap in the explainability tooling landscape: models with calibrated probabilities and profit-optimized thresholds need custom counterfactual search. We built one.

## Methodology

Hand-rolled grid search over 7 continuously varying features (loan_amnt, int_rate, dti, annual_inc, emp_length, fico_range_low, acc_open_past_24mths) on two illustrative rejected borrowers.

Two search modes:

1. **Single-feature**: for each feature independently, search 100 evenly-spaced values across the plausible range, find the smallest change that flips P(default) below the threshold.
2. **Pairwise**: 15x15 grid over each pair of features (28 pairs total), find the combination with the smallest combined normalized change that flips the decision. Normalization (change / feature range) makes changes comparable across features.

Categorical features (grade, term, purpose, home_ownership, addr_state) were held fixed. Changing them would require modeling correlated feature shifts (e.g., grade C to grade A also changes int_rate, sub_grade) which exceeds the scope of a counterfactual demonstration.

## Case study 1: "Recoverable" rejection

**Borrower idx 42**, val set 2016. Grade C, 60-month term, $12,800 loan at 14.49%, $32,000 income, DTI 24.5, FICO 750.

P(default) = 0.2092, threshold = 0.1457. Rejected, but close to the boundary.

Two single-feature counterfactuals flip the decision:

| Lever | Original | Counterfactual | Change | New P |
|---|---|---|---|---|
| loan_amnt | $12,800 | $1,788 | -$11,012 | 0.1451 |
| int_rate | 14.49% | 7.53% | -6.96 pts | 0.1388 |

**Interpretation.** This borrower could be approved either by taking a much smaller loan or by securing a much lower interest rate (which in practice would require a credit grade upgrade). Both are loan-structure changes rather than borrower-attribute changes. From a recourse standpoint this is actionable: a lender could counter-offer "we can approve $2,000 instead of $12,800" or "we can offer this at our grade A rate of 7.5%."

## Case study 2: "Systemic" rejection

**Borrower idx 0**, val set 2016. Grade C, 60-month term, $10,000 loan at 13.49%, $32,000 income, DTI 13.05, FICO 750.

P(default) = 0.3128, threshold = 0.1457. Rejected at more than double the threshold.

**No single-feature change can flip this rejection.** That is itself a meaningful finding: some borrowers cannot be saved by changing one thing.

Top 4 pairwise counterfactuals:

| Combination | int_rate change | Second feature change | New P |
|---|---|---|---|
| int_rate + annual_inc | 13.49 → 6.79 | $32,000 → $54,286 | 0.1451 |
| int_rate + dti | 13.49 → 5.00 | 13.05 → 8.57 | 0.1388 |
| int_rate + loan_amnt | 13.49 → 5.00 | $10,000 → $3,786 | 0.1451 |
| int_rate + fico_range_low | 13.49 → 5.00 | 750 → 796 | 0.1388 |

**Interpretation.** int_rate appears in every top counterfactual pair, almost always reduced to the floor of 5%. The second feature varies (income, DTI, loan size, FICO) but is secondary. This is consistent with the Week 3 SHAP finding that int_rate has high mean |SHAP| value: when the model is unsure, rate is what tips the decision. The fact that no single feature can carry the change alone shows the model has learned that risk compounds: this borrower's loan structure plus profile combination is "too far" from approval for any one lever.

## Key findings

1. **SHAP and counterfactuals converge.** Features that have high SHAP impact (int_rate, term) also appear in counterfactual recourse paths. The two methods tell consistent stories about which features matter for the model.

2. **Loan structure dominates borrower attributes in recourse.** Most counterfactual changes are to the loan itself (rate, amount) rather than to the borrower (income, FICO, employment). This is positive from a recourse standpoint: borrowers have more control over loan structure (pick a smaller loan, search for a better rate) than over underlying creditworthiness.

3. **Some rejections are systemic.** When P(default) is significantly above threshold (here, 2x), no single change can flip the decision. The model is saying "this loan structure plus this borrower profile is too risky as a combination." For lenders, this provides a useful operational signal: borderline rejections may be salvageable with a counter-offer, while deep rejections likely cannot be.

4. **Custom thresholds break standard tooling.** DiCE assumes a 0.5 threshold, which makes it incompatible with profit-optimized models. Models with calibrated probabilities and custom thresholds need custom counterfactual search.

## Limitations

- **Grid search is only pairwise.** Three-feature combinations exist (56 triples x 15^3 = 189k predictions, computationally manageable) but were not explored. The hard case might be solvable with a 3-feature change of smaller magnitude per feature.
- **Continuous features only.** Categorical changes (grade upgrade, purpose change, home ownership change) were not modeled because they require modeling correlated feature shifts. SHAP showed grade matters, but "improve grade from C to A" is not a single intervention.
- **Distance metric is feature-range normalized, not utility-weighted.** A 5-point rate reduction is much easier to achieve than a 70% income increase, but our metric treats them as equally costly if their range-normalized magnitudes are equal. A real recourse system would use a utility-weighted distance function.
- **No plausibility constraints.** The search allows int_rate = 5% for a grade C borrower, which is unrealistic in practice (rates correlate strongly with grade). Realistic recourse would constrain to plausible joint distributions.

## Reproducibility

- Notebook: `notebooks/10_counterfactuals.ipynb`
- Method: hand-rolled grid search (no DiCE dependency)
- Features varied: 7 continuous (loan_amnt, int_rate, dti, annual_inc, emp_length, fico_range_low, acc_open_past_24mths)
- Resolution: 100 steps for single-feature search, 15x15 grid for pairwise
- Threshold: 0.1457 (champion's profit-optimized threshold, set in Week 2)
- Borrowers: idx 42 (P=0.2092) and idx 0 (P=0.3128), both grade C, 60-month term
