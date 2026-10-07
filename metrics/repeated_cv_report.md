# Repeated cross-validation evaluation

RepeatedStratifiedKFold on the full dataset (5 folds × 10 repeats); the preprocessor is fitted inside each fold.
The fixed production thresholds and selected per-target configurations are read from `model/serialized/model_metadata.json`; no tuning or holdout is used.
Reported 95% confidence intervals are percentile intervals over the 10 repeat-level means (each repeat is the mean of its five fold metrics).

Random state: 42

Runtime versions: Python 3.14.4, Pandas 3.0.6, NumPy 2.5.3, Scikit_Learn 1.9.1, XGBoost 3.4.1, Imbalanced_Learn 0.14.2

| Target | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | Brier_Score |
|---|---|---|---|---|---|---|---|
| CAD | 0.890 ± 0.008 [0.881, 0.902] | 0.908 ± 0.006 [0.899, 0.917] | 0.956 ± 0.005 [0.948, 0.964] | 0.931 ± 0.005 [0.925, 0.938] | 0.935 ± 0.005 [0.926, 0.942] | 0.979 ± 0.002 [0.976, 0.982] | 0.082 ± 0.005 [0.075, 0.090] |
| LAD | 0.758 ± 0.005 [0.751, 0.767] | 0.738 ± 0.004 [0.733, 0.745] | 0.957 ± 0.003 [0.954, 0.962] | 0.833 ± 0.003 [0.829, 0.839] | 0.836 ± 0.009 [0.824, 0.848] | 0.880 ± 0.007 [0.871, 0.888] | 0.167 ± 0.003 [0.164, 0.171] |
| LCX | 0.678 ± 0.009 [0.664, 0.690] | 0.620 ± 0.007 [0.608, 0.628] | 0.930 ± 0.009 [0.913, 0.941] | 0.743 ± 0.006 [0.735, 0.752] | 0.819 ± 0.009 [0.809, 0.835] | 0.816 ± 0.013 [0.792, 0.837] | 0.175 ± 0.004 [0.168, 0.180] |
| RCA | 0.703 ± 0.010 [0.689, 0.718] | 0.640 ± 0.008 [0.629, 0.652] | 0.932 ± 0.009 [0.917, 0.942] | 0.758 ± 0.008 [0.748, 0.770] | 0.802 ± 0.009 [0.788, 0.817] | 0.793 ± 0.010 [0.776, 0.808] | 0.185 ± 0.003 [0.181, 0.189] |
