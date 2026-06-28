# Test Set Evaluation: 2017 Held-Out Data

## Purpose

This is the final evaluation of the champion model on completely unseen data. Throughout development, all loans from 2017 were held out and never used for training, validation, hyperparameter tuning, calibration, threshold selection, or any other modeling decision. This report documents how the model performs on that true holdout set.

The purpose of a test-set evaluation is to provide an unbiased estimate of generalization performance. Validation metrics can be optimistic, because the validation set was used implicitly to choose the model (hyperparameter tuning, calibration, threshold). Test-set metrics are the closest available proxy for deployment performance.

## Methodology

The champion model (calibrated LightGBM with a profit-optimized threshold of 0.1592) was loaded from disk and applied to the 169,300 loans issued in 2017. The same preprocessing pipeline used during training was applied: the same value cleaning (debt-to-income and credit-history sentinels mapped to missing, income and revolving utilization capped), string-to-integer conversion for term and employment length, then the saved ColumnTransformer for the remaining 79 input features. No retraining, refitting, or recalibration was performed. The model was evaluated exactly as it would be in production.

The cost basis for profit is unchanged from development: an approved loan that is repaid returns 10 percent of principal, and an approved loan that defaults loses 50 percent of principal. The break-even probability under that basis is 0.1667.

The evaluation covers four areas: headline metrics (AUC, Brier score, profit), calibration quality across deciles, decision-level outcomes (confusion matrix, FPR, FNR), and a threshold analysis (does the 2016 threshold still maximize profit on 2017 data?).

## Headline metrics

Both years are evaluated on their full sets with the identical pipeline, so the comparison is like-for-like.

| Metric | Validation 2016 | Test 2017 | Change |
|---|---|---|---|
| AUC | 0.7281 | 0.7188 | -0.0093 |
| Brier score | 0.1570 | 0.1587 | +0.0017 |
| Profit | $65.57M | $30.86M | (see note) |
| Profit per loan | $223.70 | $182.26 | -18.5% |
| Default rate | 0.233 | 0.231 | -0.002 |
| Rejection rate | 0.626 | 0.637 | +0.011 |

The AUC dropped by about 0.9 of a percentage point. This is a modest degradation, well within the range usually considered acceptable for out-of-time validation, where AUC drops under two points are treated as evidence of robust generalization. The Brier score moved by less than 0.002, indicating that probability quality was preserved across the year boundary.

Absolute profit is not directly comparable across the two years, because the loan counts and dollar volumes differ (293,095 loans in 2016 against 169,300 in 2017). The comparable figure is per-loan profit, which fell from about 224 dollars to about 182 dollars, a decline of 18.5 percent. This drop is not driven by degraded prediction (AUC and Brier are essentially unchanged) but by distributional shift in loan economics: the default rate itself was nearly identical (23.1 against 23.3 percent), so the per-loan compression reflects shifts in other dimensions such as loan-size mix, term distribution, or the dollar relationship between predicted risk and realized loss. This finding reinforces a central operational point developed later: predictive metrics alone do not surface this kind of softening, and only profit-level monitoring detects it.

## Calibration

The calibration plot (reports/figures/calibration_test_2017.png) shows near-exact alignment between predicted and observed default rates across the model's output range.

| Predicted | Observed | Difference |
|---|---|---|
| 0.0393 | 0.0397 | +0.0004 |
| 0.0898 | 0.0896 | -0.0003 |
| 0.1285 | 0.1297 | +0.0011 |
| 0.1637 | 0.1646 | +0.0009 |
| 0.2000 | 0.2017 | +0.0017 |
| 0.2273 | 0.2302 | +0.0030 |
| 0.2771 | 0.2712 | -0.0059 |
| 0.3258 | 0.3254 | -0.0005 |
| 0.3909 | 0.3943 | +0.0034 |
| 0.5430 | 0.5064 | -0.0366 |

Nine of the ten deciles show absolute calibration error below 0.006. This is exceptional calibration on out-of-time data: the isotonic calibrator fit on the pre-2016 development data continues to produce accurate probability estimates on 2017 loans.

The tenth decile (the highest-risk applicants) shows a 3.7-point over-prediction: the model predicts a 0.543 probability of default against an observed rate of 0.506. The model is slightly pessimistic in the high-risk tail. This is the less harmful direction of error: a few applicants flagged as very high risk did somewhat better than predicted, which leads to marginally more rejection in that tail rather than to surprise defaults. The calibrated probabilities cap near 0.54, a property of the isotonic calibrator.

The decile that contains the decision threshold (predicted 0.1637, just above 0.1592) is calibrated to within 0.001. The operating point therefore sits in a region of near-perfect calibration, which is why the deployed threshold behaves on 2017 as it did on 2016.

For a production lender, this calibration quality means the probabilities can be used directly for risk-based pricing, expected-loss provisioning, and economic-capital calculations without adjustment. A predicted 30 percent default probability really does correspond to roughly a 30 percent default rate, a property most production credit models do not satisfy.

## Decision-level outcomes

At the deployed threshold of 0.1592, the model's decisions on 2017 data were:

| Outcome | Count | Rate |
|---|---|---|
| True negatives (approved, repaid) | 55,597 | 32.8% |
| False positives (rejected, would have repaid) | 74,555 | 44.0% |
| False negatives (approved, defaulted) | 5,802 | 3.4% |
| True positives (rejected, would have defaulted) | 33,346 | 19.7% |

Approval rate: 36.3 percent. Rejection rate: 63.7 percent. False positive rate: 57.3 percent (of borrowers who would have repaid, 57.3 percent were rejected). False negative rate: 14.8 percent (of borrowers who would have defaulted, 14.8 percent were approved).

The high FPR is a deliberate consequence of profit-optimized thresholding under asymmetric costs. Approving a borrower who defaults loses 50 percent of principal, while approving a borrower who repays earns only 10 percent, a five-to-one asymmetry. Under that cost structure the profit-maximizing decision rejects aggressively, accepting many wrongful rejections of good borrowers to avoid the much costlier approvals of bad ones.

Among approved loans, the conditional default rate was 9.5 percent (5,802 of 61,399), well below the unconditional default rate of 23.1 percent. The model successfully filters higher-risk borrowers out of the approved pool, cutting the approved pool's default rate to under half the population rate.

## Threshold robustness

A key question for production deployment is whether the profit-optimal threshold computed on 2016 still maximizes profit on 2017. Thresholds were swept from 0.05 to 0.50 and profit was computed at each point on the 2017 test set.

| Threshold | Profit on 2017 |
|---|---|
| 0.1592 (deployed, from 2016) | $30.86M |
| 0.1700 (2017 optimal) | $31.30M |
| Difference | $0.44M (1.4%) |

The profit-optimal threshold for 2017 is 0.1700, essentially on top of the deployed 0.1592 and right at the break-even probability of 0.1667. The deployed threshold captures 98.6 percent of the maximum achievable 2017 profit: re-tuning against the 2017 labels (which is not possible in production, since outcomes are unknown at decision time) would add only 0.44M dollars. The profit curve is a broad, flat plateau around the optimum, so the operating point is robust to small threshold changes, and steep losses appear only at much higher thresholds where the model over-rejects.

The interpretation is that the threshold did not drift. The relationship between predicted probability and the optimal decision held across the year boundary: the same cutoff chosen on 2016 remains near-optimal on 2017, without ever having seen 2017 data. This is strong evidence that the operating point generalizes, not just the ranking and calibration.

## Why predictive metrics are necessary but not sufficient

The one quantity that moved materially is per-loan profitability, down 18.5 percent, while ranking (AUC), probability quality (Brier, calibration), and the operating point (threshold) all held. This is the important monitoring lesson, and it is subtle: a monitoring stack that tracked only AUC, Brier, and threshold optimality would have reported the model fully healthy, yet per-loan economics compressed by nearly a fifth. The compression comes from distributional shift in loan characteristics (size mix, the dollar relationship between risk and loss), not from any failure of the model to rank or calibrate.

The remedy is operational discipline rather than a different model. A production monitoring stack should track several independent dimensions, any one of which can signal the need for recalibration or retraining even when the others look healthy:

- Predictive metrics: AUC, Brier, calibration error.
- Decision-level metrics: selection rate, FPR, FNR, observed default rate among approved.
- Economic metrics: realized profit, profit per loan, expected against realized loss.
- Distribution monitoring: score-distribution shift (for example via population stability index or a distributional test).
- Fairness metrics: per-segment disparities from the fairness audit.

## Implications for deployment

The findings combine into a clear picture.

The model is production-ready in ranking, calibration, and operating point. AUC degradation is modest, calibration is near-exact across all deciles except the highest-risk tail (and even there the deviation is small and conservative), and the deployed threshold remains within 1.4 percent of optimal on 2017. These properties all hold out-of-time across a one-year gap, which is the evidence stakeholders need to approve deployment.

Economic monitoring is the active watch item. Predictive and operational stability did not prevent a roughly 18 percent compression in per-loan profit. That is the dimension most likely to move silently, and it must be tracked directly rather than inferred from accuracy metrics.

The model adds measurable value. Even with the deployed threshold held fixed, it generates positive profit on the holdout set while cutting the approved pool's default rate to under half the population rate. With a more lenient operating point near the break-even probability, profit improves marginally, but the deployed cutoff is already close to the best achievable.

## Conclusion

The champion model generalizes well to 2017 data. AUC degradation is modest, calibration retention is near-exact, and, in contrast to a common failure mode, the profit-optimal threshold did not drift: the deployed 0.1592 cutoff remains within 1.4 percent of the 2017 optimum. The one finding that requires attention is economic rather than predictive: per-loan profitability fell about 18 percent despite stable ranking, calibration, and operating point, which underscores that profit-level monitoring is an essential complement to standard predictive metrics in production.

The combination of robust ranking, near-exact calibration, a stable operating point, and a quantified per-loan profit shift provides a complete picture suitable for both deployment approval and the design of an operational monitoring stack.

## Reproducibility

- Notebook: 13_test_set_evaluation.
- Saved results: reports/test_set_2017_results.json.
- Figures: reports/figures/calibration_test_2017.png, reports/figures/threshold_analysis_test_2017.png.
- Data: 2017 test set, 169,300 loans, 23.1 percent default rate; compared against the full 2016 validation set, 293,095 loans, 23.3 percent default rate.
- Profit basis: repaid approved loan returns 10 percent of principal, approved default loses 50 percent of principal.