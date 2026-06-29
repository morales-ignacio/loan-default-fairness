# Loan Default Prediction with Fairness Audit
## Executive summary

**Author**: Ignacio Morales
**Repository**: https://github.com/morales-ignacio/loan-default-fairness

### The project

A complete machine learning system that predicts the probability a consumer loan will default, built on 1.3 million completed LendingClub loans. The model is trained and evaluated on a chronological 2014 to 2017 split: train on 2014-2015, validate on 2016, and test on a fully held-out 2017. The system goes beyond standard credit scoring to address the concerns that separate a research model from a deployable one: well-calibrated probabilities, profit-aware decision rules, an explanation for every prediction, recourse paths for rejected applicants, fairness measurement across demographic proxies, and quantified economic drift over time.

### Headline results

**The model generalizes to unseen data.** Tested on 169,300 loans from 2017 that were withheld throughout development, it kept essentially all of its quality: ranking (AUC) fell only 0.009, from 0.728 to 0.719, and probability calibration stayed near-exact, with nine of ten deciles within 0.006 of perfect.

**The operating point proved robust, but the economics did not.** The profit-optimal threshold chosen on 2016 captured 98.6% of the maximum achievable 2017 profit; re-tuning on the held-out year would have recovered only $0.44M (1.4%). Yet per-loan profit still compressed 18.5% even as every statistical metric held steady. A monitoring stack watching only accuracy would have reported the model healthy while its economics softened. This is the finding that argues for monitoring profit directly, not inferring it from ranking metrics.

**Fairness disparities were measured and priced.** Low-income renters faced a 76.7% rejection rate versus 47.1% for high-income mortgage holders, and roughly seven in ten low-income renters who would have repaid were rejected anyway (a false-positive rate of 70.8%). A custom profit-aware mitigation brought every dimension above the regulatory four-fifths threshold at a cost of $2.4M to $5.7M. The honest result: no configuration beats the unmitigated baseline on both profit and fairness at once. The analysis prices that tension rather than asserting it away.

**The standard fairness toolchain failed, and a custom approach was needed.** Fairlearn's optimizer, built around accuracy, collapsed to approving nearly everyone and turned a $65.6M profit into a nine-figure loss. A profit-aware per-group threshold method was built to handle the asymmetric cost structure that off-the-shelf tooling does not model.

### What this demonstrates

The ability to take a prediction problem from raw data to a production-considered system, including the dimensions most projects skip: that probabilities should be trustworthy, that thresholds should reflect business economics, that predictions should be explainable, that rejections should carry recourse, that disparate impact should be measured, that mitigation should be quantified in dollars, and that drift should be monitored across time.

### Technical foundation

LightGBM gradient boosting with Bayesian (Optuna) hyperparameter search, isotonic probability calibration, profit-optimized thresholding under an asymmetric cost matrix, SHAP explanations, a custom counterfactual recourse algorithm built when standard tooling proved incompatible, multi-dimensional fairness measurement using both demographic parity and equalized odds, and held-out chronological validation.

### Detailed documentation

13 Jupyter notebooks documenting the full workflow and 8 written analysis reports. Every numerical claim is reproducible from the notebooks.
