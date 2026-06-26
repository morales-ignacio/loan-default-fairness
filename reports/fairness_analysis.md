# Fairness Audit and Mitigation

## The question

Does the champion model produce systematically different outcomes for different socioeconomic groups? If so, is the disparity explained by genuine risk differences, or by something the model has learned that a lender should not act on? And where a disparity is present, can it be reduced, and at what cost?

## A critical limitation: no protected attributes in the data

The LendingClub dataset does not contain race, gender, age, marital status, or national origin. The Equal Credit Opportunity Act (ECOA) prohibits the use of these attributes in US credit decisions, so they are absent from the data by design. A complete fairness audit against legally protected classes is therefore not possible with this dataset.

What can be audited is disparate impact across proxies that may correlate with protected attributes:

- Income tier (quartiles of annual income): a socioeconomic proxy.
- Home ownership (mortgage, own outright, rent): a wealth proxy.
- Employment length: a job-stability proxy that correlates with age.
- State of residence: a geographic proxy that correlates with race and income at the population level.

These proxies are stratifiers, not protected classes. A finding of disparity across them is a signal that warrants investigation with real protected-attribute data, not a legal determination in itself. A production system would need to obtain protected attributes separately for fairness monitoring, or estimate them using a method such as Bayesian Improved Surname Geocoding (BISG).

## Methodology

Per-group metrics were computed with Fairlearn on the 2016 validation set (293,095 loans, 23.3 percent default rate), using the champion model at its profit-optimized decision threshold of 0.1592:

- Selection rate: the proportion predicted as default, meaning rejected.
- False positive rate (FPR): the proportion of repaying borrowers wrongly rejected.
- False negative rate (FNR): the proportion of defaulters wrongly approved.
- True positive rate (TPR): the proportion of defaulters correctly caught.

Two aggregate measures summarize each dimension:

- Demographic parity ratio: how close selection rates are across groups (1.0 is identical treatment).
- Equalized odds ratio: how close error rates (FPR and FNR) are across groups (1.0 is identical error).

The EEOC four-fifths rule treats a demographic parity ratio below 0.80 as evidence of disparate impact. The same 0.80 threshold is applied to the equalized odds ratio. This audit weights the equalized odds ratio, and its FPR component in particular, more heavily than demographic parity, because the relevant harm is a creditworthy borrower being wrongly rejected, which is exactly what FPR measures.

## Summary of findings

| Dimension | DP ratio | EO ratio | Verdict |
|---|---|---|---|
| Income tier | 0.716 | 0.687 | Disparate impact, mitigatable |
| Home ownership | 0.777 | 0.764 | Disparate impact, mitigatable |
| Employment length | 0.821 | 0.810 | Passes four-fifths, but driven by the missing-data group |
| State of residence | 0.634 | n/a (50 groups) | Large spread, largely tracks real risk |
| Income x home ownership | 0.614 | 0.585 | Severe disparate impact, mitigatable |

Every dimension except employment falls below the 0.80 four-fifths threshold on at least one measure, and even employment passes only narrowly. The intersection of income and home ownership is the most severe. Mitigation (covered below) closes each gap to a ratio above 0.92, at a profit cost of 6.5 to 8.7 percent depending on the dimension.

## Income tier: the largest single-dimension disparity

| Income tier | Rejection rate | Actual default rate | FPR (wrongful rejection) | FNR (missed default) |
|---|---|---|---|---|
| Q1 (low) | 72.4% | 27.5% | 66.4% | 11.6% |
| Q2 | 65.1% | 24.4% | 58.3% | 13.7% |
| Q3 | 60.7% | 22.3% | 53.9% | 15.7% |
| Q4 (high) | 51.9% | 18.8% | 45.6% | 21.0% |

Q1 (lowest income) borrowers face a 72.4 percent rejection rate against 51.9 percent for Q4 (highest income), a gap of 20.6 percentage points. Part of that gap is defensible on risk: Q1 has an actual default rate near 27.5 percent against 18.8 percent for Q4.

The error rates show what the risk argument leaves out. Among Q1 borrowers who would have repaid, 66.4 percent are rejected anyway. Among Q4 borrowers who would have repaid, 45.6 percent are rejected. That is a 20.8 percentage point gap in wrongful rejection between the lowest and highest income groups. In the other direction, when a Q4 borrower is going to default the model misses them 21.0 percent of the time, against 11.6 percent for Q1 borrowers.

The model is more aggressive and more accurate at catching defaults among low-income borrowers, and more lenient and more error-prone among high-income borrowers. From a calibration standpoint this is internally consistent. From a disparate-impact standpoint, low-income borrowers who would have repaid carry most of the cost of the model's accuracy on their group.

## Home ownership: a renter penalty

| Home ownership | Rejection rate | Actual default rate | FPR | FNR |
|---|---|---|---|---|
| MORTGAGE | 55.2% | 19.7% | 49.0% | 19.6% |
| OWN | 64.9% | 23.4% | 58.7% | 14.9% |
| RENT | 71.1% | 27.7% | 64.2% | 11.0% |

The same shape repeats. Renters face roughly 16 percentage points more rejection than mortgage holders (71.1 against 55.2 percent), with a 15-point gap in wrongful rejection (FPR 64.2 against 49.0 percent). Mortgage holders who would have defaulted slip through nearly twice as often as renters who would have defaulted (FNR 19.6 against 11.0 percent).

The direction is not surprising: mortgage holders have already passed bank underwriting and have an asset at stake, while renters are typically younger, lower-wealth, and a more heterogeneous risk pool. The consequence is that housing status alone can move a decision, and renters bear most of the wrongful-rejection burden.

Outright owners (OWN) sit between the two, with a higher actual default rate than mortgage holders (23.4 against 19.7 percent). In a loan-applicant population, outright ownership without a mortgage skews toward borrowers with weaker income profiles, which is consistent with the elevated risk.

## Employment length: passes the rule, but the signal is in the missing data

| Employment length | Rejection rate | FPR |
|---|---|---|
| New (1 year or less) | 64.8% | 58.0% |
| Short (2 to 4 years) | 63.1% | 56.5% |
| Medium (5 to 9 years) | 63.7% | 57.0% |
| Long (10 years or more) | 58.8% | 52.2% |
| Unknown (missing) | 71.7% | 64.4% |

The demographic parity ratio is 0.821 and the equalized odds ratio is 0.810, both above the four-fifths threshold. Among borrowers who report a tenure, the model does not systematically disadvantage newer workers: the spread from "new" at 64.8 percent rejection to "long" at 58.8 percent is modest, and the ranking is not even monotonic in tenure.

The disparity that does exist is driven almost entirely by one group: borrowers whose employment length is missing, who face the highest rejection rate of any tenure group at 71.7 percent and the highest FPR at 64.4 percent. The model treats a missing employment value as a risk signal. This is the more important finding than any tenure effect: the borrowers most penalized on this dimension are those with incomplete applications, not those with short tenure. Because employment length correlates with age, the absence of a tenure effect is reassuring from an age-discrimination standpoint, with the caveat that this is a proxy and not direct age data.

## Geographic disparity: large but mostly explained

State-level rejection rates span from 45.6 percent (DC) to 72.0 percent (Nebraska), a demographic parity ratio of 0.634, well below four-fifths.

The per-state scatter (reports/figures/fairness_states_scatter.png) shows why that number overstates the concern: rejection rate correlates strongly with actual default rate. States with high rejection are states with high actual default. The model is largely doing what calibration prescribes, assigning higher rejection to higher-base-risk populations. Three observations from the scatter:

- Every state sits above the line where rejection equals actual default rate, expected given the profit-optimized threshold of 0.1592 (the model rejects far more aggressively than the base default rate).
- States fall along a roughly linear positive slope: rejection scales with actual risk.
- FPR scales with rejection, the same equalized-odds pattern seen for income and home ownership, now at the state level.

One state warrants a specific note. The explainability analysis (notebooks/09_shap.ipynb) surfaced New Jersey as the single feature pushing a very-low-risk borrower marginally toward default in one case study (addr_state_NJ contributed +0.11 against an otherwise strongly-approving profile). At the population level, that single-case signal does not translate into a systemic disparity: New Jersey ranks 19th of 50 states by rejection rate (64.7 percent, actual default 24.7 percent), squarely mid-pack. A single SHAP contribution in one case is not evidence of group-level unfairness, and the audit confirms it is not.

The geographic disparity is therefore largely explained rather than unfair in the technical sense. It remains a concern for a separate reason: state of residence correlates with race in the United States, so a model that keys on geography can produce racially disparate outcomes without race ever being an input. This is exactly the pattern that requires monitoring against the protected-attribute data the dataset does not contain.

## Intersectional analysis: income x home ownership

Disadvantage compounds. A low-income renter faces income-tier disparity and home-ownership disparity at once. Auditing the 12-cell cross-product of income tier and home ownership quantifies the combined effect.

| Group (income x home) | n loans | Actual default | Rejection rate | FPR |
|---|---|---|---|---|
| Q1_low \| RENT | 40,903 | 30.1% | 76.7% | 70.8% |
| Q1_low \| OWN | 11,387 | 25.8% | 72.8% | 67.4% |
| Q2 \| RENT | 29,811 | 28.0% | 71.1% | 64.0% |
| Q3 \| RENT | 25,016 | 26.6% | 68.2% | 60.9% |
| Q2 \| OWN | 8,494 | 24.2% | 66.2% | 59.7% |
| Q1_low \| MORTGAGE | 23,461 | 23.6% | 64.9% | 58.9% |
| Q3 \| OWN | 7,916 | 22.8% | 63.0% | 56.9% |
| Q4_high \| RENT | 18,872 | 23.8% | 62.8% | 55.8% |
| Q2 \| MORTGAGE | 33,006 | 21.4% | 59.4% | 53.1% |
| Q3 \| MORTGAGE | 40,050 | 19.5% | 55.5% | 49.4% |
| Q4_high \| OWN | 7,833 | 19.6% | 53.8% | 47.7% |
| Q4_high \| MORTGAGE | 46,283 | 16.7% | 47.1% | 41.4% |

Aggregate intersectional disparity is a demographic parity ratio of 0.614 and an equalized odds ratio of 0.585, both materially worse than either single dimension (income alone 0.716 / 0.687, home alone 0.777 / 0.764). This is the textbook compounding effect: moderate disparate impact on two dimensions independently becomes severe disparate impact on their intersection.

The most disadvantaged group is low-income renters (Q1_low | RENT): a 76.7 percent rejection rate and a 70.8 percent FPR, meaning roughly seven in ten low-income renters who would have repaid are rejected anyway. This is a population of 40,903 borrowers in the validation set, about 14 percent of all loans, not a small edge case. The least disadvantaged group is high-income mortgage holders (Q4_high | MORTGAGE) at a 47.1 percent rejection rate and a 41.4 percent FPR. The gap between the two is 29.6 points in rejection and 29.3 points in wrongful rejection.

The heatmap (reports/figures/fairness_intersectional.png) shows the gradient is smooth and monotonic across both axes, from "low-income renter is risky" to "high-income mortgage holder is safe". There are no anomalous pockets, which indicates the model is tracking the joint distribution of income and housing rather than doing something idiosyncratic. The finding does not change the audit's conclusion but sharpens it: the equalized-odds failure is concentrated on a specific, identifiable population that overlaps significantly with groups that have historically faced lending discrimination in the United States.

## The systemic pattern

Across every attribute audited, the same structure appears: the higher-base-risk group receives a higher rejection rate, a higher FPR (more wrongful rejections), and a lower FNR (fewer missed defaults) than the lower-base-risk group.

This is not a defect in the model. It is a mathematical consequence of applying a single calibrated probability threshold to populations with different base rates. The Chouldechova and Kleinberg results (2017) established this formally: a model cannot simultaneously satisfy calibration and equalized odds when base rates differ across groups, except in degenerate cases. The champion model is well-calibrated (Brier score 0.157) and was deliberately tuned to a single profit-maximizing threshold. The equalized-odds failure documented above is the cost of that design choice. There is no neutral resolution, only a choice about which fairness criterion to prioritize.

## Mitigation

The natural follow-up: can the disparities be reduced, and what does it cost? Two approaches were tested on the income dimension, and the working approach was then extended to home ownership and the intersection.

### Attempt 1: Fairlearn ThresholdOptimizer

Fairlearn's ThresholdOptimizer learns group-specific thresholds that satisfy a fairness constraint while maximizing an objective (accuracy by default). It was run with both a demographic-parity constraint and an equalized-odds constraint on income tier.

| Approach | Profit | Selection rate | DP ratio | EO ratio |
|---|---|---|---|---|
| Champion, no mitigation | $65.6M | 0.626 | 0.716 | 0.687 |
| ThresholdOptimizer, DP | -$120.5M | 0.059 | 0.996 | 0.795 |
| ThresholdOptimizer, EO | -$135.4M | 0.052 | 0.842 | 0.969 |

Both runs technically converged, and both are operationally useless. Each collapsed to approving almost everyone (a rejection rate near 5 to 6 percent), which trivially flattens selection rates across groups and satisfies the fairness constraint. The cause is the objective: accuracy is maximized by predicting the majority class (no default) for nearly everyone, and the accuracy figure of about 0.77 in both rows is simply the approve-all accuracy (1 minus the 23.3 percent base default rate). The consequence is catastrophic on profit. Each approved defaulter costs half the loan value, so approving the whole book turns a $65.6M profit into a $120M to $135M loss. The equalized-odds run is the more expensive of the two.

This is not a defect in Fairlearn. It exposes a real mismatch between the standard fairness toolchain, which assumes the objective is accuracy, and profit-driven credit decisions, where accuracy and profit point in different directions. A profit-optimized model needs profit-aware mitigation.

### Attempt 2: profit-aware per-group thresholds

The working approach is deliberately simple: for each group, choose the decision threshold that produces a target selection rate, and hold that target equal across groups at the overall baseline rate. This equalizes selection rates by construction (achieving demographic parity) while keeping overall approval volume close to the operating point the business already runs.

```python
def find_group_thresholds_for_dp(probs, groups, target_selection_rate):
    thresholds = {}
    for g in np.unique(groups):
        mask = groups == g
        thresholds[g] = np.quantile(probs[mask], 1 - target_selection_rate)
    return thresholds
```

For income tier, the resulting thresholds move in exactly the direction fair mitigation requires: looser for the already-disadvantaged groups, tighter for the already-advantaged ones.

| Income tier | Threshold | vs baseline 0.1592 | Effect |
|---|---|---|---|
| Q1 (low) | 0.1917 | +0.0325 | More lenient |
| Q2 | 0.1758 | +0.0166 | More lenient |
| Q3 | 0.1470 | -0.0122 | Slightly stricter |
| Q4 (high) | 0.1244 | -0.0348 | Stricter |

The same method applied to the intersection produces a smooth gradient of 12 thresholds, from the most lenient for low-income renters (Q1_low | RENT at 0.2088) to the strictest for high-income mortgage holders (Q4_high | MORTGAGE at 0.1171).

### Mitigation results

| Dimension | Profit | Profit change | DP ratio (base to mitigated) | EO ratio (base to mitigated) |
|---|---|---|---|---|
| Champion, no mitigation | $65.57M | (baseline) | see audit | see audit |
| Income tier | $61.23M | -$4.34M (6.6%) | 0.716 to 0.951 | 0.687 to 0.932 |
| Home ownership | $61.33M | -$4.23M (6.5%) | 0.777 to 0.982 | 0.764 to 0.973 |
| Income x home ownership | $59.84M | -$5.72M (8.7%) | 0.614 to 0.948 | 0.585 to 0.929 |

All three mitigations clear the four-fifths threshold comfortably (every mitigated ratio is above 0.92), at a profit cost between 6.5 and 8.7 percent. The cost scales with the severity of the disparity being corrected: the intersection, the worst disparity, is the most expensive to fix.

A useful side effect: equalizing selection rates for demographic parity also improves equalized odds substantially, even though equalized odds was never the optimization target (income EO rose from 0.687 to 0.932). Shifting the per-group thresholds to balance who gets through also rebalances who gets through correctly.

One honest caveat on precision. The champion's probabilities are isotonic-calibrated, which produces heavily tied (discretized) values. Quantile-based thresholds therefore do not land each group exactly on the target rate, because a threshold that falls on a tie plateau pulls a whole block of borrowers across at once. In practice the mitigated demographic parity ratio lands near 0.95 rather than exactly 1.0, and overall rejection drifts up by about two points (from 62.6 to roughly 64.3 percent). The mitigation is effective, but the reported cost is approximate rather than surgical, since part of it reflects that small volume drift.

### The profit-fairness frontier

Sweeping the target selection rate from 0.40 to 0.85 maps the full trade-off (reports/figures/fairness_mitigation_tradeoff.png). Two findings stand out.

First, the entire frontier is fair. Once per-group thresholds are applied, the demographic parity ratio stays above 0.91 (and equalized odds above 0.88) at every selection rate tested. With this method, any operating volume the business might choose is already compliant with the four-fifths rule.

Second, there is no free lunch. No point on the frontier matches the baseline profit. The most profitable fair configuration, at a target selection rate near 0.57, returns $63.2M with a demographic parity ratio of 0.94, still $2.4M (3.7 percent) below the $65.6M baseline. Fairness costs somewhere between $2.4M at the cheapest fair point and $5.7M for the full intersectional correction. The baseline single threshold remains the most profitable configuration available, and it is also the least fair. That tension is the central result: equitable treatment and maximum profit are genuinely in conflict here, and the value of this analysis is in pricing the conflict rather than pretending it away.

## What this means for deployment

The model is suitable for use with disclosure and a deliberate mitigation choice. The disparities documented here do not require abandoning the model, but they do require an explicit decision about the operating point and an ongoing monitoring plan.

Three configurations are defensible:

- Hold volume, equalize by group. Keep overall approval at the baseline rate and apply per-group thresholds for demographic parity. Cost: $4.2M to $5.7M depending on which dimension is targeted. Outcome: fairness ratios above 0.92 on the targeted dimension.
- Target the intersection directly. Apply mitigation on the income-by-home cross-product, the highest-disparity dimension. Most expensive at $5.7M, but it addresses the worst-affected population (low-income renters) head-on.
- Accept slightly more volume. A marginally more lenient overall rate (around a 0.57 target) with per-group thresholds gives the smallest fairness cost ($2.4M) while staying fully compliant, at the price of a higher accepted default rate, which carries its own regulatory and operational implications.

The choice depends on regulatory environment, risk appetite, and business strategy, and the deployed configuration should be disclosed in the model card.

Three operational requirements follow from the audit regardless of configuration:

- Adverse-action notices must be specific. ECOA requires the principal reasons for any rejection. The SHAP machinery demonstrated in notebooks/09_shap.ipynb can generate per-rejection explanations.
- Recourse should be offered on borderline rejections. As shown in the counterfactual analysis (notebooks/10_counterfactuals.ipynb), borrowers just above the threshold can often be approved with a smaller loan. Automated counter-offers for borderline cases are a low-cost improvement.
- Geographic monitoring is the highest-priority watch item. State of residence is the most likely vector for indirect racial disparity, and a production system should track approval rates by ZIP code or census tract against demographic data and flag changes over time.

## Limitations

- No protected attributes. Without race, gender, age, or other ECOA-protected data, this audit is necessarily incomplete. The proxies analyzed (income, home ownership, employment, state) correlate with protected attributes but do not substitute for them. A finding of disparity across a proxy is a prompt for investigation, not a legal conclusion.
- Per-group thresholds are disparate treatment. The mitigation explicitly uses group membership to set the decision threshold. That is effective for equalizing outcomes, but using a group attribute directly in the decision is itself legally sensitive in US lending, especially across an income-by-housing cross-product. The numbers here quantify the cost of fairness; a literal deployment of per-group thresholds would require legal review, and a compliant implementation might instead adjust the model or its features rather than apply explicit group thresholds.
- Calibration ties. Because the champion's probabilities are isotonic-calibrated and heavily tied, the mitigation lands demographic parity near 0.95 rather than exactly 1.0 and introduces a small volume drift. Reported mitigation costs are approximate.
- Single threshold, single period. All audit metrics are computed at the 0.1592 profit-optimized threshold on the 2016 validation set. Disparities at other thresholds or in other periods would differ, and disparities can shift as economic conditions change.

## Reproducibility

- Notebooks: notebooks/11_fairness_audit.ipynb (audit), notebooks/12_fairness_mitigation.ipynb (mitigation).
- Data: 2016 validation set, 293,095 loans, 23.3 percent default rate.
- Model: calibrated champion LightGBM, profit-optimized threshold 0.1592 for the baseline, per-group thresholds for mitigation.
- Profit basis: a repaid approved loan returns 10 percent of principal, an approved default loses 50 percent of principal.
- Sensitive attributes: income tier (income quartiles), home ownership, employment-length tier, state of residence.
- Library: Fairlearn.
- Figures: reports/figures/fairness_income_tier.png, fairness_home_ownership.png, fairness_states_scatter.png, fairness_intersectional.png, fairness_mitigation_tradeoff.png.