# EDA Findings

**Dataset:** 1,345,310 completed loans (Fully Paid + Charged Off), 105 columns after leakage audit  
**Overall default rate:** 19.96%

A working record of patterns discovered during exploratory data analysis. The findings below organize into five sections: categorical relationships, numerical distributions and data quality, default rate by numeric ranges, missingness analysis, and correlation/redundancy. A modeling implications section at the end summarizes the decisions that come out of these findings.

---

## 1. Categorical features and default rate

### 1.1 Loan term: 60-month loans default at roughly 2x the rate of 36-month

```
36 months:  1,020,743 loans   15.99% default
60 months:    324,567 loans   32.45% default
```

Two compounding causes:
- **Mechanical:** 60 months is 67% more loan life, more time for adverse events.
- **Borrower self-selection:** thr longer term has a lower monthly payment, which may make it more attractive to borrowers facing tighter cash-flow constraints.

### 1.2 Home ownership: MORTGAGE holders are safer than outright owners

```
RENT      534k loans   23.22%
OWN       144k         20.62%
MORTGAGE  665k         17.21%   ← lowest
ANY, OTHER, NONE       too few loans to be meaningful (286, 144, 48)
```

**Counterintuitive finding.** Active-mortgage holders default less than outright homeowners. One possible explanation is that borrowers with active mortgages represent a financially stable and previously screened population, resulting in lower observed default rates than outright homeowners.

### 1.3 Verification status is the opposite of what it appears

```
Verified         418k loans   23.85%
Source Verified  521k         20.95%
Not Verified     406k         14.67%   ← lowest
```

Counterintuitive finding. Borrowers whose income was not verified default less than those whose income was verified. This pattern is consistent with selection bias by the lender: Lending Club appears to have used verification more frequently for borrowers perceived as higher risk. As a result, verification status may reflect the lender's prior risk assessment as much as the borrower's underlying credit quality.


### 1.4 Loan purpose: clear risk hierarchy with small business as the outlier

```
small_business      15k loans   29.71%   ← highest
moving               9k         23.35%
medical             16k         21.78%
debt_consolidation  780k        21.15%   ← largest category, 58% of all loans
other               78k         21.04%
home_improvement    88k         17.72%
credit_card        295k         16.93%
car                 15k         14.68%
wedding              2k         12.16%   ← lowest
```

Small business loans default at ~1.5x the overall rate. Wedding, car, and credit-card refinance loans exhibit the lowest default rates in the dataset. One possible explanation is that these purposes are associated with more financially stable borrowers or more predictable spending needs.

### 1.5 Joint applications default more than individual applications

```
Joint App    26k loans   24.59%
Individual   1.32M       19.87%
```

**Counterintuitive finding.** Joint applications default *more*, not less, than individual ones. One possible explanation is that borrowers apply jointly when a single applicant would not qualify on equally favorable terms, making joint applications a signal of elevated underlying credit risk rather than pooled financial strength.

---

## 2. Numerical features: data quality issues

Five issues found in the raw numerical features that required cleaning:

| Feature | Issue | Fix applied |
|---|---|---|
| `annual_inc` | Max $10,999,200 vs $250k at 99th percentile (self-reported lies) | Capped at $1,000,000 |
| `dti` | Sentinel values `-1` (couldn't compute) and `999` (out of range) mixed with real values | Replaced with NaN, then capped at 100 |
| `revol_util` | Max of 892% (impossible) | Capped at 100% |
| `credit_history_months` | Sentinel `999` (83 years of credit history) | Replaced with NaN |
| `installment` | Implausibly low min of $4.93 (lowest possible from $500 loan is ~$15) | Flagged, not blocking |

The most important point: **sentinel values disguised as real numbers** (`-1`, `999` for "couldn't compute" / "out of range") would catastrophically mislead a model. A model treats them as real numeric values and learns nonsense.

These transformations are documented here and will be re-applied in the modeling preprocessing pipeline. The canonical `loans_clean.parquet` is not modified — cleaning lives in code, not in the persisted dataset.

---

## 3. Default rate by numerical decile

Every key numerical feature shows a clean monotonic relationship with default — no kinks, no U-shapes. Good news for modeling: simple models will learn the signal cleanly.

### 3.1 `int_rate` — the dominant predictor (8.18x spread)

```
5.31-7.39%:    4.91% default
19.52-30.99%: 40.16% default
```

Although int_rate is one of the most predictive features in the dataset, it is not purely borrower information. The interest rate assigned to a loan already reflects Lending Club's internal assessment of credit risk, meaning a model that uses int_rate may be partially reproducing the lender's underwriting decisions rather than learning default risk independently. To quantify this effect, models will be trained both with and without int_rate, allowing the analysis to measure how much predictive performance comes from the lender's pricing decision versus the remaining borrower and loan characteristics.

### 3.2 `fico_range_low` — the strongest independent predictor (2.84x spread)

```
625-665:  26.28% default
740-845:   9.25% default
```

Cleanly monotonic across deciles. This is the strongest feature *not* derived from Lending Club's own risk model. FICO is the gold-standard credit signal and the data confirms it.

### 3.3 `dti` — clean and predictive (~2x spread)

```
0-7.27:    14.65%
29.78-100: 29.01%
```

Monotonic across all deciles. Higher debt-to-income → higher default, exactly as credit theory predicts.

### 3.4 `annual_inc` — weaker than expected (1.6x spread)

```
$0-34k:    24.05%
$125k+:    15.09%
```

**Counterintuitive finding.** Naive intuition says high income = safe borrower, but the data shows annual income alone is the weakest of the four key numerics. High earners can still have bad credit, debt, or financial habits. Income only becomes strongly predictive when combined with debt (which is what DTI does).

---

## 4. Missingness analysis

### 4.1 Sec_app and joint features: structural missingness

```
sec_app_* features        ~98.6% missing
*_joint features          ~98.1% missing
```

Not a data quality issue. Joint applications are 1.92% of loans, and these features only exist for joint applications. Recommendation: drop the entire `sec_app_*` block; use `application_type` as the single joint/individual indicator. Saves ~17 features that contribute nothing for 98% of rows.

### 4.2 The `mths_since_*` family: informative missingness

Heavy missingness (50-83%) in features measuring time since a negative event. Counterintuitively, this missingness is **informative**: In many cases, a NaN appears to indicate that the event never occurred, making the missingness itself informative.

Tested on six features. **All show "missing = lower default rate":**

```
mths_since_recent_inq:         -5.84%   ← strongest signal
mths_since_last_record:        -3.34%
mths_since_last_major_derog:   -2.68%
mths_since_last_delinq:        -1.41%
mths_since_recent_bc_dlq:      -1.34%
mths_since_recent_revol_delinq:-1.07%
```

**Implication for feature engineering:** because the NaN itself carries signal, median imputation does not preserve it. The pipeline median-imputes these features for simplicity; encoding the missingness explicitly with binary indicators would be a natural extension.

---

## 5. Correlation and redundancy

Twenty highest correlation pairs reveal several redundancy patterns:

### 5.1 Perfect duplicates (correlation = 1.000)

- `fico_range_low` and `fico_range_high`: bounds of the same FICO score range
- `sec_app_fico_range_low` and `sec_app_fico_range_high`: same pattern for secondary applicants
- `loan_amnt` and `funded_amnt`: Lending Club funds the requested amount almost always

### 5.2 Near-duplicates (0.99+)

- `funded_amnt` and `funded_amnt_inv`: Lending Club keeps a tiny cut from investors
- `open_acc` and `num_sats`: nearly all open accounts are satisfactory

### 5.3 Mechanical correlation (0.95)

- `loan_amnt` and `installment`: installment is computed from loan_amnt, term, and int_rate. Has to be correlated.

### 5.4 Drop list for modeling

```
fico_range_high          duplicate of fico_range_low
funded_amnt              essentially loan_amnt
funded_amnt_inv          essentially loan_amnt
num_sats                 essentially open_acc
installment              mechanically derived from loan_amnt + term + int_rate
num_rev_tl_bal_gt_0      essentially num_actv_rev_tl
sub_grade                redundant with grade, high cardinality
zip_code                 high cardinality, little signal beyond addr_state
```

Plus the entire sec_app_* and *_joint group (~17 features) for the structural missingness reason, and issue_year (used only for the temporal split, not a feature). Combined, this takes the dataset from 105 columns down to 79 modeling features (72 numeric + 7 categorical).

Note: these drops matter most for linear models (multicollinearity). Tree models handle correlated features fine, but the drops still simplify the model and clarify feature importance interpretation.

---

## Headlines 

The findings most worth featuring (in order of how interesting they are to a general reader):

1. **"Not Verified" loans default least.** A clean example of selection bias: Lending Club appears to have targeted verification toward borrowers perceived as higher risk.
2. **Joint applications default more than individual ones.** Contrary to intuition, adding a second borrower is associated with higher, not lower, default rates.
3. **MORTGAGE holders default less than outright owners.** Active mortgage holders may represent a more financially stable and previously screened population than outright homeowners.
4. **60-month loans default at roughlt 2x the rate of 36-month loans.** Both mechanical (more time, more risk) and selection (term attracts borrowers stretched on payments).
5. **Annual income is a weaker predictor than expected.** Naive intuition says high income = safe; the data shows FICO and DTI matter far more.
6. **Many missing values are *informative*.** A NaN in "months since last delinquency" doesn't mean missing data, it means the event never happened, which is a *positive* signal that naive imputation would erase.