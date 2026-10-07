# Canonical model metrics

Model performance is reported from two generated files, with distinct evaluation meanings:

- [`metrics/repeated_cv_summary.csv`](../metrics/repeated_cv_summary.csv) is the headline estimate:
  per-target repeated cross-validation means, standard deviations, and percentile 95% confidence
  intervals over repeat-level means.
- [`metrics/classification_metrics.csv`](../metrics/classification_metrics.csv) contains per-target
  results from the one fixed, single holdout split. These are report-only and are not the headline
  generalization estimate.

## Generation and provenance

| File | Producing script | Method and seed | Generated date |
|---|---|---|---|
| `metrics/classification_metrics.csv` | `model/notebooks/final_rebuild_pipeline.py` | Deterministic multi-label-balanced holdout; random state 42. Thresholds and model configurations were selected using training data only. | 2026-10-04 |
| `metrics/repeated_cv_summary.csv` | `model/notebooks/repeated_cv_evaluation.py` | RepeatedStratifiedKFold, 5 folds × 10 repeats; random state 42. The preprocessor is fitted within each fold and production thresholds/configurations are fixed. | 2026-10-07 |

The repeated-CV 95% intervals are percentile intervals over the 10 repeat-level means; each repeat
is the mean of its five fold metrics. They summarize resampling variability in this dataset and are
not external-validation intervals.

`README.md` reports repeated-CV Accuracy and ROC-AUC as mean ± SD [95% CI] and separately labels
the holdout values as a single split (n≈76). `/metrics` reads performance values from these CSVs;
the frontend renders the API response rather than embedding metric values. To check that rounded
README values remain synchronized with both CSVs, run `python scripts/check_metrics_consistency.py`
from the repository root.

Other reports and experiment outputs may be useful for analysis, but do not replace these two
canonical files for the stated repeated-CV and final-holdout results.
