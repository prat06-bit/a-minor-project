# Synthetic experiment results

All metrics below measure agreement with `synthetic_proxy_label`, a generated heuristic target. They are not real PMIS selection accuracy.

## Clean candidate-grouped holdout

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | NDCG | Train F1 | Validation F1 | Fit seconds | Inference ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rule_baseline | 0.920 | 0.826 | 0.870 | 0.961 | 0.898 | 0.936 | 0.865 | 0.843 | 0.000 | 0.002 |
| logistic_regression | 0.721 | 0.904 | 0.802 | 0.975 | 0.921 | 0.942 | 0.853 | 0.766 | 0.182 | 0.034 |
| random_forest | 0.678 | 0.914 | 0.779 | 0.974 | 0.912 | 0.942 | 0.866 | 0.756 | 0.869 | 0.116 |
| gradient_boosting | 0.917 | 0.818 | 0.865 | 0.975 | 0.914 | 0.951 | 0.871 | 0.844 | 4.795 | 0.081 |

## Data quality by condition

| Dataset | Rows | Duplicate IDs | Missing cells | Non-numeric capacity | App/capacity >10 | Unknown categories | Out-of-range skill values | Positive proxy rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 12000 | 0 | 0 | 0 | 317 | 0 | 0 | 0.220 |
| corrupted_00pct | 12000 | 0 | 0 | 0 | 317 | 0 | 0 | 0.220 |
| cleaned_00pct | 12000 | 0 | 0 | 0 | 317 | 0 | 0 | 0.220 |
| corrupted_05pct | 11542 | 142 | 2318 | 583 | 816 | 1098 | 561 | 0.232 |
| cleaned_05pct | 11400 | 0 | 1970 | 0 | 306 | 0 | 0 | 0.232 |
| corrupted_10pct | 11070 | 270 | 4453 | 1109 | 1226 | 2017 | 1008 | 0.244 |
| cleaned_10pct | 10800 | 0 | 3673 | 0 | 277 | 0 | 0 | 0.245 |
| corrupted_20pct | 10080 | 480 | 8144 | 1997 | 1784 | 3217 | 1586 | 0.275 |
| cleaned_20pct | 9600 | 0 | 6329 | 0 | 266 | 0 | 0 | 0.275 |

## Validation notes

- Capacity allocation used `scipy.optimize.milp` with status `Optimization terminated successfully. (HiGHS Status 7: Optimal)`; 1109 assignments were made across 1109 candidates, with 0 capacity violations.
- Unseen-internship experiment: 70 source internships and 1028 held-out candidates; F1 0.793, ROC-AUC 0.9702772899069702.
- Corruption experiments include nulls, case/whitespace drift, category noise/unknowns, numeric text, out-of-range values, impossible demand ratios, conflicting locations, feature noise, duplicate deliveries/IDs, and valid-row sampling to alter class prevalence. Labels on retained rows were not changed.
- Cleaning restores fields from stable synthetic entity IDs, removes duplicate pair IDs with a log, normalizes categories, and converts out-of-range match features to missing values. Remaining missing feature values are imputed within each training fold by the sklearn pipeline.
- `App/capacity >10` is a diagnostic threshold for the injected demand-ratio stress test, not an official validity rule. The workbook's aggregate application counts already include high ratios; their semantics must be confirmed before operational use.
- Fairness metrics are not reported: no legitimate policy group data or authoritative policy rules were supplied. The fairness validation JSON records this limitation.
- No company-level real-world generalization or true applicant outcome validation is possible from the supplied source workbook.
