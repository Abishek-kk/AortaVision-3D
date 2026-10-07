# Decision-threshold analysis

## Clinical rationale

These thresholds are intended for screening and decision support, not diagnosis. A missed stenosis can delay further evaluation, while a false alarm generally prompts review or confirmatory testing. The operating point therefore favors sensitivity (recall) over precision, accepting additional false positives to reduce missed disease. Clinical evaluation and formal diagnostic testing remain necessary.

## Selection rule and training-only comparison

The production rule maximizes F1 on five-fold out-of-fold predictions from the training split over the rebuild grid 0.20–0.70 in 0.05 increments. Ties prefer higher recall, then precision, then accuracy. There is no explicit recall floor. The broader 0.05–0.95 sweep below is descriptive and does not alter the production thresholds. Youden-J maximizes sensitivity + specificity − 1 on the same training OOF predictions. The holdout is never used to select a threshold.

| Target | Production threshold | Production OOF P / R / specificity / F1 | Default 0.50 P / R / specificity / F1 | Youden threshold | Youden OOF P / R / specificity / F1 |
|---|---:|---:|---:|---:|---:|
| CAD | 0.35 | 92.1% / 95.9% / 65.5% / 94.0% | 93.1% / 93.9% / 70.7% / 93.5% | 0.793 | 96.6% / 82.4% / 87.9% / 88.9% |
| LAD | 0.30 | 72.1% / 97.4% / 35.1% / 82.9% | 79.8% / 76.4% / 66.7% / 78.1% | 0.486 | 79.7% / 80.1% / 64.9% / 79.9% |
| LCX | 0.25 | 61.9% / 96.7% / 40.4% / 75.5% | 72.0% / 74.8% / 70.9% / 73.4% | 0.576 | 78.4% / 69.5% / 80.8% / 73.7% |
| RCA | 0.35 | 61.4% / 94.7% / 40.4% / 74.5% | 72.3% / 70.9% / 72.8% / 71.6% | 0.528 | 75.0% / 69.5% / 76.8% / 72.2% |

## Holdout report only

The following values are descriptive performance at the saved production threshold; they were not used to choose or revise thresholds.

| Target | Threshold | Precision | Recall | Specificity | False negatives / 100 | False positives / 100 |
|---|---:|---:|---:|---:|---:|---:|
| CAD | 0.35 | 78.0% | 97.9% | 55.2% | 1.3 | 17.1 |
| LAD | 0.30 | 75.0% | 95.7% | 48.3% | 2.6 | 19.7 |
| LCX | 0.25 | 67.3% | 86.8% | 57.9% | 6.6 | 21.1 |
| RCA | 0.35 | 70.6% | 94.7% | 60.5% | 2.6 | 19.7 |

## Consistency check

The documented F1 rule selects a different threshold for: CAD (saved 0.35, current rule 0.40). Saved production thresholds were left unchanged; this difference should be reconciled before changing deployed behavior.

Complete 0.05–0.95 sweep metrics and trade-off figures are in the matching `threshold_sweep_*.csv` files and `thresholds/*_tradeoff.png` figures.
