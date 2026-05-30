# Model Evaluation

> Week 2 findings: model selection, hyperparameter optimization, calibration, and threshold sensitivity analysis. This document consolidates the analytical work in `notebooks/06_random_forest.ipynb`, `notebooks/07_lightgbm.ipynb`, and `notebooks/08_calibration_thresholds.ipynb`.

## Summary

Built and evaluated 8 model variants on the temporally split LendingClub dataset (train 2014-2015, val 2016, test 2017 held out). Champion model is a calibrated LightGBM tuned via Optuna, achieving **AUC 0.7296** and **Brier 0.1573** on val_eval with an empirically optimal threshold of **0.1457**, very close to the theoretical break-even of 0.1667.

Compared to the grade-only baseline, the champion delivers:

- **+0.048 AUC** (0.682 to 0.730)
- **-5% Brier** (0.166 to 0.157), preserving the well-calibrated property
- **+$22M projected profit** on val (extrapolated from val_eval result)

The model adds genuine independent value: the int_rate ablation in Week 1 confirmed that performance does not depend on LendingClub's own risk scoring baked into the interest rate feature.

## Final scoreboard

All runs are tracked in MLflow under the `loan-default` experiment.

| Run name | AUC | PR-AUC | Brier | Profit | Notes |
|---|---|---|---|---|---|
| approve_all | 0.500 | 0.233 | 0.180 | -$207M | Do-nothing floor |
| grade_only | 0.682 | 0.359 | 0.166 | +$43M | Simple baseline |
| logreg_baseline | 0.716 | 0.425 | 0.219 | $60.7M | First real model |
| logreg_no_int_rate | 0.715 | 0.425 | 0.216 | $59.9M | Ablation (Week 1) |
| rf_baseline | 0.716 | 0.427 | 0.191 | $60.4M | Tree baseline |
| rf_no_balance | 0.716 | 0.428 | 0.161 | $60.5M | Class-weight ablation |
| lgbm_baseline | 0.725 | 0.441 | 0.159 | $63.7M | Gradient boosting |
| lgbm_tuned | 0.728 | 0.445 | 0.158 | $65.5M | Optuna (50 trials) |
| **lgbm_calibrated** | **0.730** | n/a | **0.157** | **$33.3M (val_eval)** | **Champion** |

All metrics evaluated on val (2016), except the calibrated champion which is evaluated on the held-out half of val (val_eval, 146,548 loans) since the other half was used to fit the isotonic calibrator. Projected profit on full val: approximately $66M.

## Section 1: Architecture comparison

Three architectures were evaluated: logistic regression (linear), Random Forest (parallel tree ensemble), and LightGBM (sequential gradient boosting).

**Key finding: the signal in this dataset is mostly linear.** Random Forest matched logistic regression's AUC almost exactly (0.7158 vs 0.7157), indicating that the predictive patterns are well-approximated by a weighted sum of features. Non-linear interactions exist but are not the dominant source of predictive power.

LightGBM provided a modest but real improvement (+0.013 AUC over RF), justifying the architectural step up. The gain comes from sequential error correction: each tree explicitly targets borrowers the previous trees got wrong, capturing subtle credit patterns that parallel averaging cannot.

This finding informed the rest of Week 2: extensive RF hyperparameter tuning would have low yield (no non-linear patterns to find), so optimization effort was concentrated on LightGBM.

## Section 2: The class_weight ablation

**Experiment**: trained two otherwise-identical Random Forests, one with `class_weight='balanced'` and one without.

**Result**:

| Setting | AUC | Brier | Profit | Optimal threshold |
|---|---|---|---|---|
| balanced | 0.7158 | 0.1910 | $60.4M | 0.3328 |
| None | 0.7157 | **0.1609 (-16%)** | $60.5M | **0.1409** |

**Interpretation**: class balancing shifts probability levels, not ranking. AUC and profit (which depend only on ranking) are unchanged. Brier (which depends on probability calibration) improves dramatically.

The optimal threshold also drops from 0.33 to 0.14, because the model now outputs honest probabilities. With `class_weight='balanced'` inflating predictions, the cutoff sits artificially high; with raw probabilities, the cutoff approaches the theoretical break-even rate of 0.1667.

**Generalizable insight**: `class_weight='balanced'` is appropriate when you need balanced precision/recall trade-offs at default thresholds, but actively harmful when you need calibrated probabilities for downstream decisions. For loan default prediction, where the threshold is a deliberate business choice, the unbalanced version is strictly better.

## Section 3: Hyperparameter optimization with Optuna

Used Optuna's TPE (Tree-structured Parzen Estimator) sampler for 50 trials over an 8-dimensional hyperparameter space. Each trial trained a LightGBM with early stopping on val AUC (50-round patience), with `n_estimators` capped at 1000.

### Best parameters found

```
learning_rate:     0.0216    (default: 0.10)
num_leaves:        58        (default: 31)
min_child_samples: 83        (default: 20)
feature_fraction:  0.57      (default: 1.0)
bagging_fraction:  0.67      (default: 1.0)
bagging_freq:      3
lambda_l1:         0.003
lambda_l2:         9.48      (near search boundary of 10)
```

### Pattern interpretation

Optuna's choices reveal a clear signature: **heavy regularization combined with high per-tree complexity**.

- **Slow learning** (learning_rate 0.022, less than half the default) forces incremental updates
- **Strong leaf-level regularization** (min_child_samples 83, vs default 20) prevents overfitting to rare borrower patterns
- **Heavy subsampling** (feature_fraction 0.57, bagging_fraction 0.67) adds noise to each tree
- **Maximum L2 regularization** (lambda_l2 near the search ceiling) further constrains leaf weights
- **High num_leaves** (58 vs default 31) gives each tree more flexibility within the regularized constraints

This pattern is consistent with credit risk datasets: the underlying signal is small and stable, so the model wants to learn slowly with many constraints, but allowed to explore complex per-tree structure when those constraints permit.

### Performance

- Search time: approximately 20 minutes (50 trials × 16-20 seconds each)
- Improvement over LightGBM baseline: AUC 0.7250 to 0.7279 (+0.0029)
- Actual iterations used: 930 of 1000 max (vs 501 for baseline). The slower learning rate required more trees.

The 930 iterations near the cap suggests the model would benefit from `n_estimators=2000` and longer training. Marginal improvement expected; not pursued.

## Section 4: Calibration analysis

### Methodology

Val (2016) was split 50/50 at random:

- **val_calib** (146,547 loans): used to fit isotonic regression calibrators
- **val_eval** (146,548 loans): used to evaluate calibrated model performance

This split avoids the bias of fitting and evaluating the calibrator on the same data. Test set (2017) remains untouched.

### Reliability diagrams

Pre-calibration reliability diagrams (see `reports/calibration_before.png`) show:

- **LogReg's curve bends dramatically below the diagonal.** At a predicted probability of 0.8, the actual default rate is only 0.5 — systematic over-confidence. This is the signature of `class_weight='balanced'` inflating probabilities.
- **LightGBM's curve hugs the diagonal closely.** The architecture decisions (gradient boosting, no class balancing) produced calibrated probabilities directly.

Post-calibration (see `reports/calibration_comparison.png`), both curves align with the diagonal.

### Results

| Model | Brier before | Brier after | Improvement |
|---|---|---|---|
| LogReg | 0.2189 | 0.1599 | **-27.0%** |
| LightGBM tuned | 0.1581 | 0.1573 | -0.5% |

In both cases, AUC is preserved (isotonic regression is monotonic and does not change ranking).

### Key finding

**Post-hoc calibration is powerful for repairing miscalibrated models, but no substitute for architecture choices that produce calibrated probabilities in the first place.**

The LogReg's 27% Brier reduction is dramatic. The LightGBM's 0.5% is essentially noise. The champion model didn't need calibration because the architecture decisions in Week 2 (gradient boosting, no `class_weight='balanced'`) already delivered well-calibrated probabilities.

Isotonic regression was applied to the champion anyway, both for marginal improvement and to make the model artifact production-ready: a credit officer relying on the probability output now has a stronger guarantee that "predicted 0.20 means roughly 20% chance of default."

## Section 5: Threshold optimization

### Profit curve

The profit curve for the calibrated champion (see `reports/profit_curve.png`) shows a clean inverted-U shape with a clear peak.

- Empirical optimum: threshold 0.1457, profit $33.3M on val_eval
- Theoretical break-even: 0.1667 (derived from cost matrix: GAIN_RATE / (GAIN_RATE + LOSS_RATE))

The 0.02 gap between empirical and theoretical is small, reflecting residual miscalibration. With a poorly calibrated model, this gap would exceed 0.10. The agreement between theory and empirics validates the calibration work.

### Marginal vs average default rate

The approval rate vs default rate plot (see `reports/approval_default_tradeoff.png`) reveals a subtle but important insight.

At the optimal approval rate (34.8%), the **average** default rate among approved loans is only 8.5%, well below the break-even rate of 16.7%. The optimum is not where average default rate equals break-even.

The reason: the optimum is set by the **marginal** borrower, not the average. As the threshold lowers and more loans are approved, each new approval comes from a progressively riskier borrower. The optimum is reached when the marginal borrower's default risk hits the break-even rate. Below that threshold, the approved population averages to a much lower default rate because earlier (safer) approvals dilute the riskier marginal ones.

This is a real-world banking insight: profit-maximizing approval decisions accept that the average approved borrower is much safer than the marginal one.

### Cost matrix sensitivity

The cost matrix assumptions (GAIN_RATE=0.10, LOSS_RATE=0.50) drive profit projections substantially. A sensitivity sweep across plausible ranges (see `reports/cost_sensitivity.png`) shows:

- Profit range: $4.2M (worst case: GAIN=0.05, LOSS=0.70) to $180.4M (best case: GAIN=0.20, LOSS=0.30)
- Threshold range: 0.07 to 0.40
- Profit is far more sensitive to GAIN_RATE than to LOSS_RATE (horizontal gradient dominates the vertical in the left heatmap)

**Key finding**: the model is robust across the entire sensitivity grid. Same model, same rankings, only the threshold shifts according to the formula threshold ≈ GAIN/(GAIN+LOSS).

This is the right separation of concerns:

- **The model** answers a stable question (which borrowers are more likely to default)
- **The threshold** answers a business question (where to cut, given your cost assumptions)

Anyone presenting this to executives must own the GAIN_RATE assumption. A 50% revision of GAIN_RATE more than doubles the profit projection.

## Champion model

The final champion is the **calibrated LightGBM tuned**, saved to `models/champion_lgbm.pkl` as a dictionary containing:

- `preprocessor`: fitted `ColumnTransformer` (median impute + scale for numeric, constant impute + one-hot for categorical)
- `base_model`: trained `LGBMClassifier` with Optuna best params
- `calibrator`: fitted `IsotonicRegression`
- `best_threshold`: 0.1457
- `best_params`: Optuna result dictionary
- `val_metrics`: full evaluation dictionary

### Performance summary (val_eval)

| Metric | Value |
|---|---|
| AUC | 0.7296 |
| Brier | 0.1573 |
| Optimal threshold | 0.1457 |
| Approval rate at optimum | 34.8% |
| Default rate among approved | 8.5% |
| Profit on val_eval | $33.3M |
| Projected profit on full val | approximately $66M |

### Inference example

```python
import pickle
with open('models/champion_lgbm.pkl', 'rb') as f:
    artifact = pickle.load(f)

X_processed = artifact['preprocessor'].transform(X_new)
raw_probs = artifact['base_model'].predict_proba(X_processed)[:, 1]
calibrated_probs = artifact['calibrator'].predict(raw_probs)
decisions = calibrated_probs < artifact['best_threshold']
```

## Methodology notes

### Validation discipline

- **Train (2014-2015, 598k loans)**: model fitting only
- **Val (2016, 293k loans)**: hyperparameter optimization, early stopping, threshold selection, calibrator fitting (50% subset)
- **Test (2017, 169k loans)**: untouched throughout Weeks 1-2. Reserved for final unbiased evaluation after Week 3 fairness work.

Val is used for multiple decisions (Optuna trials, early stopping, calibration, threshold). This is standard practice and acceptable because test remains held out. Final reported numbers in the model card will come from test.

2018 was dropped entirely due to maturity bias: long-term loans issued in 2018 had not matured by the dataset's end date, artificially deflating the observed default rate.

### Preprocessing consistency

All models share the same preprocessing pipeline (`build_preprocessor` function in notebooks 06-08):

- 79 features used (after dropping 6 redundant, 17 sec_app/joint, 2 high-cardinality, 1 split column)
- Numeric (72 cols): median imputation + StandardScaler
- Categorical (7 cols): constant imputation ("missing") + OneHotEncoder with `handle_unknown='ignore'`

Tree-based models do not strictly require scaling, but keeping the pipeline consistent across all models ensures apples-to-apples comparison.

### Cost matrix

GAIN_RATE = 0.10 (interest earned on fully paid loans)
LOSS_RATE = 0.50 (loss on charged-off loans)

These reflect typical LendingClub realized economics. Sensitivity analysis (Section 5) explores robustness across plausible variations.

## Limitations

- **Cost matrix assumptions matter**. Profit projections are conditional on GAIN_RATE and LOSS_RATE values. Any operational deployment must validate these against actual realized economics.
- **Val used for multiple purposes**. Optuna trials, early stopping, threshold selection, and calibrator fitting all touch val. The 50/50 val split (calib vs eval) mitigates calibration-related optimism, but val-derived metrics likely overstate true generalization by a small margin. Test set evaluation in the model card will be the unbiased measure.
- **Temporal drift is real**. Default rates rose from 19.5% (train) to 23.3% (val) to 23.1% (test). The model's absolute performance on future data depends on whether this drift continues, accelerates, or reverses.
- **Selection bias in source data**. LendingClub already filtered applicants before originating the loans in the dataset. The model learns "which of the loans LC approved end up defaulting," not "which of all applicants would default." Deployment to a broader population would require recalibration.

## Reproducibility

To reproduce all Week 2 results from a clean clone:

```bash
git clone https://github.com/morales-ignacio/loan-default-fairness
cd loan-default-fairness
uv sync
# Place LendingClub data at data/accepted_2007_to_2018Q4.csv
uv run jupyter notebook
```

Then run notebooks in order:

1. `notebooks/01_initial_exploration.ipynb` (data load, target creation)
2. `notebooks/02_leakage_audit.ipynb` (49 features dropped, clean parquet generated)
3. `notebooks/03_eda.ipynb` (exploratory findings)
4. `notebooks/04_baselines.ipynb` (approve_all + grade_only)
5. `notebooks/05_logreg.ipynb` (logistic regression + int_rate ablation)
6. `notebooks/06_random_forest.ipynb` (RF baseline + class_weight ablation)
7. `notebooks/07_lightgbm.ipynb` (LightGBM baseline + Optuna search)
8. `notebooks/08_calibration_thresholds.ipynb` (calibration + threshold sensitivity)

All MLflow runs land in `mlruns/`; view with `uv run mlflow ui --port 5050`.

The champion model artifact is generated by the final cell of notebook 08 and saved to `models/champion_lgbm.pkl`.
