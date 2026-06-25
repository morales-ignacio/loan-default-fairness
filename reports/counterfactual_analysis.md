# Counterfactual Recourse Analysis

## What this answers

For a borrower the model rejects, what is the smallest change to their application that would flip the decision to approval? This is *actionable recourse*. SHAP explains which features drive a prediction; counterfactuals show what specifically would have to change for the outcome to differ. The two are complementary: SHAP attributes, counterfactuals prescribe.

## Scope: what counts as an actionable lever

The most important design choice here is which features the search is allowed to move. The search was restricted to four borrower-influenceable, non-circular features:

- **loan_amnt**: the borrower chooses how much to ask for.
- **dti**: can be lowered by paying down existing debt.
- **annual_inc**: can rise, though slowly.
- **fico_range_low**: can improve over time.

Three features the model uses were deliberately **excluded** because they do not make sense as recourse:

- **int_rate** is the lender's risk-based price, an *output* of the borrower's creditworthiness, not an input the borrower controls. Telling someone to "get a lower rate" is circular: the rate is the model's verdict on their risk, not a lever they can pull. Excluding it also avoids the unrealistic counterfactuals a naive search produces, such as offering a grade-C borrower a 5% rate.
- **emp_length** and **acc_open_past_24mths** are historical and slow-moving. A borrower cannot change years employed or accounts opened in the past two years in any near term.

Categorical features (grade, term, purpose, home_ownership, addr_state) were held fixed, because moving them implies correlated shifts elsewhere (changing grade also changes int_rate and sub_grade), which a single-feature counterfactual cannot represent.

This restriction ties directly back to the SHAP analysis. The model's two heaviest levers are **term** (#1) and **int_rate** (#2), and neither is available as recourse: term is categorical, int_rate is lender-set. So the features a borrower can actually act on are the lower-ranked ones (loan_amnt #4, dti #3, fico #7, annual_inc #9). That mismatch is the headline: the model's decision is dominated by factors the borrower cannot change, which is exactly why recourse turns out to be limited.

## Why a custom search rather than DiCE

DiCE is the standard counterfactual library, but it is built around a model's default 0.5 probability threshold for the positive class. The champion uses a calibrated probability with a profit-optimized threshold of **0.1592**, and the search also needed tight control over which features it may move (the actionable set above). A small hand-rolled grid search provided both: the custom threshold and the restricted lever set. The tradeoff is that it scans a coarse grid exhaustively rather than using DiCE's optimization, which is fine at this scale.

## Method

Hand-rolled grid search over the four actionable features, on two illustrative rejected borrowers:

1. **Single-feature**: for each feature, scan 100 evenly spaced values across its plausible range and find the smallest change that pushes P(default) below 0.1592.
2. **Pairwise**: a 15x15 grid over each of the six feature pairs, scored by the smallest *range-normalized* combined change that flips the decision (normalization makes a dollar change and a FICO-point change comparable).

Ranges: loan_amnt $1,000 to $40,000, dti 0 to 60, annual_inc $20,000 to $500,000, fico_range_low 600 to 850.

## Case 1: a borderline rejection

**Borrower idx 1.** Grade C, 60-month term, $21,000 loan at 14.5%, $80,000 income, DTI 9.9, FICO 735, mortgage, debt consolidation. P(default) = **0.2126**, just above the 0.1592 cutoff.

One single-feature change flips it:

| Lever | Original | Counterfactual | Change | New P |
|---|---|---|---|---|
| loan_amnt | $21,000 | $2,576 | -$18,424 | 0.1579 |

This borrower already has a low DTI, solid income, and a decent FICO, so none of those three levers moves the decision on its own. The rejection is about the *size and structure* of the loan, not the borrower's profile. The only actionable path is a much smaller loan: in practice a lender could counter-offer roughly $2,600 instead of $21,000. It is recoverable, but only by shrinking the loan by about 88%.

## Case 2: a deep rejection

**Borrower idx 19.** Grade C, 60-month term, $17,200 loan at 14.5%, $73,000 income, DTI 13.6, FICO 660, mortgage, credit card. P(default) = **0.3643**, more than double the cutoff.

- **No single-feature change** flips this rejection.
- **No pairwise change** flips it either.

Within the actionable set, flipping this decision would require **three or more simultaneous changes**. (One- and two-feature moves were tested and neither works; triples were not searched, so this is a lower bound on the number of changes needed, not a guarantee that three suffice.)

That is itself a finding: some rejections cannot be undone by the levers a borrower controls, even in combination. The grade-C, 60-month, mediocre-FICO combination sits too far from approval for the actionable features to reach, no matter how far two of them are pushed within plausible ranges.

## What this shows

1. **The decisive features are not actionable.** term and int_rate dominate the model but are off-limits as recourse (categorical and lender-set). The levers a borrower controls are the model's weaker features, which is the structural reason recourse is limited.
2. **Loan amount is the main lever that works.** The only single-feature flip in either case was reducing the loan. DTI, income, and FICO did not flip a decision on their own.
3. **Recourse difficulty scales with rejection depth.** A borderline rejection (P 0.21) flips with one change; a deep one (P 0.36) resists even two.
4. **A profit-optimized threshold plus a restricted lever set needs a custom search.** Off-the-shelf tools assume a 0.5 threshold and do not easily restrict the feature set to actionable levers.

## Limitations

- **Pairwise is the deepest search.** Three-feature combinations were not explored, so "needs 3+" for the deep case means "one and two do not work," not "three definitely do."
- **Continuous actionable features only.** Categoricals (grade, term, purpose, home ownership, state) were held fixed; some real recourse (a grade upgrade) lives there but cannot be expressed as a single-feature move.
- **Distance is range-normalized, not utility-weighted.** An $18k loan reduction and a 100-point FICO jump count as comparable if their normalized magnitudes match, even though they differ wildly in real effort. A production recourse tool would weight by how hard each change is to achieve.
- **Excluding int_rate is the right call for borrower recourse, but it means the lender-side counter-offer (re-pricing the loan) is a separate question this analysis does not model.**
- **Two illustrative borrowers**, chosen to contrast a borderline and a deep rejection, not a population-level recourse audit.

## Reproducibility

- Notebook: `notebooks/10_counterfactuals.ipynb`
- Method: hand-rolled grid search (no DiCE dependency)
- Actionable features: loan_amnt, dti, annual_inc, fico_range_low (int_rate, emp_length, acc_open_past_24mths deliberately excluded; categoricals held fixed)
- Resolution: 100 steps single-feature, 15x15 per pair (6 pairs)
- Threshold: 0.1592 (champion's profit-optimized cutoff)
- Borrowers: idx 1 (P = 0.2126) and idx 19 (P = 0.3643), both grade C, 60-month term