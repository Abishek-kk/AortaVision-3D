# Generalization analysis

## Headline to quote

Use the repeated cross-validation ROC-AUC mean and its 95% percentile interval as the primary performance estimate. Show the final holdout as a secondary, single-split result rather than choosing whichever estimate is higher. The intervals come from ten repeat-level means of five-fold CV; they describe resampling variability in this dataset, not uncertainty across hospitals or patient populations.

All repeated-CV values below use the fixed production model settings and thresholds in `model/serialized/model_metadata.json`. The repeated CV uses 5 folds × 10 repeats (seed 42). Holdout percentiles compare the seed-42 holdout AUC with the 50 individual fold AUCs and are empirical diagnostics, not calibrated probabilities or independent-test significance tests.

## Holdout compared with repeated CV

| Target | Repeated-CV ROC-AUC (mean; 95% interval) | Holdout ROC-AUC | Holdout F1 | Percentile among 50 CV fold AUCs | Above 90th percentile? |
|---|---:|---:|---:|---:|:---:|
| CAD | 0.935 [0.926, 0.942] | 0.949 | 0.868 | 68.0th | No |
| LAD | 0.836 [0.824, 0.848] | 0.906 | 0.841 | 96.0th | Yes |
| LCX | 0.819 [0.809, 0.835] | 0.832 | 0.759 | 60.0th | No |
| RCA | 0.802 [0.788, 0.817] | 0.877 | 0.809 | 94.0th | Yes |

A holdout above the 90th percentile is unusually favorable relative to the CV folds, but does not establish why that split scored well. With only 76 holdout rows, a few cases can materially affect AUC and F1.

## Holdout split-seed stability

The fixed production models and thresholds were evaluated on 20 multi-label-balanced 80/20 splits (seeds 42, 52, …, 232). These are sensitivity results, not 20 independent external test sets.

| Target | Metric | Mean | SD | Min | Max |
|---|---|---:|---:|---:|---:|
| CAD | ROC_AUC | 0.959 | 0.019 | 0.909 | 0.987 |
| CAD | F1 | 0.893 | 0.028 | 0.811 | 0.929 |
| LAD | ROC_AUC | 0.947 | 0.021 | 0.894 | 0.977 |
| LAD | F1 | 0.878 | 0.022 | 0.838 | 0.920 |
| LCX | ROC_AUC | 0.888 | 0.034 | 0.830 | 0.947 |
| LCX | F1 | 0.792 | 0.015 | 0.766 | 0.822 |
| RCA | ROC_AUC | 0.896 | 0.029 | 0.853 | 0.967 |
| RCA | F1 | 0.812 | 0.024 | 0.774 | 0.860 |

## Training-only regularization experiment

Targets with a production train-to-CV ROC-AUC gap greater than 0.10 were tested: LAD, LCX, RCA. The 76-row holdout was not used. Each configuration used identical 5-fold × 5-repeat stratified splits within the seed-42 training partition. The gap is the fold-training ROC-AUC mean minus the validation ROC-AUC mean. The 1-SD promotion gate compares each candidate's repeated-CV mean against the production baseline mean plus the production baseline SD across repeat means.

| Target | Setting | CV ROC-AUC mean ± SD | Train-CV gap | Gap shrinkage vs production | CV-AUC improvement vs production | >1 baseline SD? |
|---|---|---:|---:|---:|---:|:---:|
| LAD | production | 0.791 ± 0.015 | 0.121 | +0.000 | +0.000 | No |
| LAD | shallow_high_child_weight | 0.785 ± 0.013 | 0.074 | +0.047 | -0.005 | No |
| LAD | stronger_sampling_and_l2 | 0.787 ± 0.011 | 0.086 | +0.034 | -0.003 | No |
| LAD | stronger_l2 | 0.788 ± 0.013 | 0.111 | +0.010 | -0.003 | No |
| LCX | production | 0.812 ± 0.006 | 0.181 | +0.000 | +0.000 | No |
| LCX | shallow_high_child_weight | 0.744 ± 0.007 | 0.118 | +0.063 | -0.068 | No |
| LCX | stronger_sampling_and_l2 | 0.774 ± 0.006 | 0.132 | +0.048 | -0.038 | No |
| LCX | stronger_l2 | 0.769 ± 0.008 | 0.160 | +0.021 | -0.044 | No |
| RCA | production | 0.786 ± 0.012 | 0.143 | +0.000 | +0.000 | No |
| RCA | shallow_high_child_weight | 0.760 ± 0.013 | 0.093 | +0.050 | -0.026 | No |
| RCA | stronger_sampling_and_l2 | 0.780 ± 0.012 | 0.120 | +0.023 | -0.006 | No |
| RCA | stronger_l2 | 0.775 ± 0.012 | 0.140 | +0.003 | -0.010 | No |

No regularization candidate improved repeated-CV ROC-AUC by more than one production-baseline SD. The production models were therefore left unchanged; the configurations above are documented experiments.

## Limitations

- The dataset contains 378 observations from a single center. It is too small to support strong claims about transportability.
- The holdout contains 76 observations and is only one split; its metrics can be optimistic or pessimistic by chance.
- Repeated cross-validation reuses the same patients across folds/repeats. Its intervals are resampling summaries, not external-validation confidence intervals.
- No independent center, time period, or prospective cohort was available for external validation.
- Results reflect the documented preprocessing, label definitions, selected production configurations, and thresholds; changes to these invalidate direct comparisons.

## Reproducibility

Holdout split seed used by the final rebuild: 42. Holdout stability seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232. Regularization CV seed: 42.

The outdated seed-stability file was preserved as `metrics/cv_seed_stability_archived.csv`.
