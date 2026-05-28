# Leakage Audit: LendingClub Loan Default Model

**Purpose:** Classify every column in the raw dataset as either available at loan-decision time (keep) or only knowable after the loan is issued (drop / leakage). Using post-decision features would inflate offline performance and produce a model that is useless in real deployment.

**The test applied to every column:** *Would this information exist at the exact moment a loan officer decides whether to approve the loan?* If no, it is dropped.

---

## Summary

| Category | Count |
|---|---|
| Keep (pre-decision features) | ~95 |
| Drop: payment/recovery leakage | 16 |
| Drop: hardship program leakage | 15 |
| Drop: debt settlement leakage | 7 |
| Drop: identifiers / junk | 4 |
| Drop: high-cardinality text | 3 |
| Drop: borderline / low-signal | 1 |
| Special handling (target, dates) | 3 |

---

## DROP — Payment history, recoveries, updated FICO (post-decision leakage)

These describe money flow and credit updates that only exist after the loan is issued.

| Column | Reason for dropping |
|---|---|
| pymnt_plan | Payment plan flag, set after issuance |
| out_prncp | Outstanding principal, post-issue |
| out_prncp_inv | Outstanding principal to investors, post-issue |
| total_pymnt | Total payments received; reveals repayment outcome |
| total_pymnt_inv | Total payments to investors; reveals outcome |
| total_rec_prncp | Principal received; reveals outcome |
| total_rec_int | Interest received; reveals outcome |
| total_rec_late_fee | Late fees only charged when payments are late; strong outcome signal |
| recoveries | Gross recovery after charge-off; non-zero means default |
| collection_recovery_fee | Collection fee after charge-off; non-zero means default |
| last_pymnt_d | Date of last payment, post-issue |
| last_pymnt_amnt | Amount of last payment, post-issue |
| next_pymnt_d | Date of next scheduled payment, post-issue |
| last_credit_pull_d | Date LC last pulled credit, updated over loan life |
| last_fico_range_high | FICO updated during loan life (vs fico_range_high at application) |
| last_fico_range_low | FICO updated during loan life (vs fico_range_low at application) |

## DROP — Hardship program data (post-distress leakage)

Hardship plans are offered after a borrower hits financial trouble. None of this exists at application, and its presence is a near-perfect distress signal.

hardship_flag, hardship_type, hardship_reason, hardship_status, deferral_term, hardship_amount, hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status, orig_projected_additional_accrued_interest, hardship_payoff_balance_amount, hardship_last_payment_amount

## DROP — Debt settlement data (post-default leakage)

A settlement only happens when a borrower fails to pay in full and negotiates a reduced payoff.

debt_settlement_flag, debt_settlement_flag_date, settlement_status, settlement_date, settlement_amount, settlement_percentage, settlement_term

## DROP — Identifiers / junk

| Column | Reason |
|---|---|
| id | Loan identifier, no predictive value |
| member_id | Member identifier, mostly null |
| url | Link to listing, no value |
| policy_code | Constant value, no information |

## DROP — High-cardinality free text (deferred as potential NLP features)

| Column | Reason |
|---|---|
| emp_title | ~500k unique job titles; deferred as potential NLP feature |
| desc | Free-text loan description, mostly empty after 2014 |
| title | Redundant with `purpose`, high cardinality |

## DROP — Borderline / low signal

| Column | Reason |
|---|---|
| disbursement_method | Operational detail set at disbursement, minimal signal |

## SPECIAL HANDLING

| Column | Handling |
|---|---|
| loan_status | Source of target variable; drop after creating `default` |
| issue_d | Used for temporal split; extract year only, do not feed raw date to model |
| earliest_cr_line | Derive credit history length (issue_d − earliest_cr_line); do not use raw date |

**Edge case — int_rate:** Set by LC at approval, so technically available at decision time, but it encodes LC's own underwriting model. Kept for the baseline model, with a planned stretch experiment training with and without it to measure how much the model relies on LC's pricing versus learning default independently.

---

## KEEP — Pre-decision features (~95)

All known at application time. Grouped for readability.

**Loan terms (set at origination):** loan_amnt, term, int_rate, installment, grade, sub_grade, initial_list_status

**Borrower application info:** emp_length, home_ownership, annual_inc, verification_status, purpose, zip_code, addr_state, dti

**Credit bureau — primary applicant:** delinq_2yrs, fico_range_low, fico_range_high, inq_last_6mths, mths_since_last_delinq, mths_since_last_record, open_acc, pub_rec, revol_bal, revol_util, total_acc, collections_12_mths_ex_med, mths_since_last_major_derog, acc_now_delinq, tot_coll_amt, tot_cur_bal, open_acc_6m, open_act_il, open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il, il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util, total_rev_hi_lim, inq_fi, total_cu_tl, inq_last_12m, acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util, chargeoff_within_12_mths, delinq_amnt, mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, mths_since_recent_bc, mths_since_recent_bc_dlq, mths_since_recent_inq, mths_since_recent_revol_delinq, num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_120dpd_2m, num_tl_30dpd, num_tl_90g_dpd_24m, num_tl_op_past_12m, pct_tl_nvr_dlq, percent_bc_gt_75, pub_rec_bankruptcies, tax_liens, tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit, total_il_high_credit_limit

**Joint / secondary applicant (mostly null for individual applications, but legitimate):** application_type, annual_inc_joint, dti_joint, verification_status_joint, revol_bal_joint, sec_app_fico_range_low, sec_app_fico_range_high, sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog

**Note on missingness:** Many credit-bureau features (especially `sec_app_*` and the installment-account family) have heavy missingness. That is addressed during EDA (Day 4) as a separate concern from leakage.
