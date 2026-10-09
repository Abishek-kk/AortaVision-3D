# AortaVision 3D: ML Metrics Improvement Guide

## Overview

This guide documents the nested repeated-CV framework (`model/notebooks/improve_metrics.py`) designed to safely improve ROC-AUC, F1, and accuracy for the four binary cardiovascular risk classifiers (CAD, LAD, LCX, RCA) without data leakage.

---

## Quick Start

### Prerequisites
```bash
# Install dependencies
cd D:\aortavision-3d
pip install -r model/requirements.txt
```

### Run the Full Pipeline
```bash
cd D:\aortavision-3d
python model/notebooks/improve_metrics.py --repeats 10 --inner-folds 3 --trials 50
```

**Expected runtime:** 6–12 hours  
**Output location:** `metrics/improvement_comparison.csv`, `metrics/improvement_report.md`, `metrics/improvement_folds.csv`, `metrics/calibration/*.png`

### Quick Smoke Test (Development)
```bash
python model/notebooks/improve_metrics.py --repeats 1 --inner-folds 2 --trials 0
```

**Expected runtime:** 2–5 minutes  
**Tests:** All pipeline components without Optuna tuning.

---

## Safety Guarantees

The framework enforces **no leakage** via scikit-learn Pipelines:

1. **Per-Fold Isolation:** All preprocessing, feature engineering, feature selection, SMOTE resampling, and threshold choice happen **inside** each outer-CV training fold only. Test folds are never used for fitting.

2. **Nested CV:** 
   - **Outer loop (5 folds × 10 repeats, seed 42):** Evaluation
   - **Inner loop (3 folds, stratified):** Threshold selection and model choice
   - Tuning (e.g., Optuna) happens only on inner training data

3. **Out-of-Fold Stacking (CAD):** 
   - CAD probabilities are generated via `cross_val_predict` on training data only
   - A separate CAD estimator is fit on full training fold for test-time prediction
   - No label leakage; only probabilities are used

4. **Classifier Chain (LAD/LCX/RCA):**
   - Predecessor targets (e.g., CAD, LAD for LCX) predicted out-of-fold within each fold
   - Predictions added as features; never on validation data

5. **Target Leakage Prevention:**
   - Forbidden input features: `CAD`, `LAD`, `LCX`, `RCA`, `Cath`
   - If any is present, the pipeline raises an error

---

## Baseline (Repeated-CV, Seed 42)

From `metrics/repeated_cv_summary.csv` (5 folds × 10 repeats):

| Target | ROC-AUC (mean ± SD) | F1 (mean ± SD) | Accuracy (mean ± SD) |
|--------|---------------------|----------------|----------------------|
| CAD    | 0.935 ± 0.016       | 0.905 ± 0.025  | 0.890 ± 0.028        |
| LAD    | 0.836 ± 0.035       | 0.766 ± 0.062  | 0.758 ± 0.050        |
| LCX    | 0.819 ± 0.040       | 0.665 ± 0.079  | 0.678 ± 0.063        |
| RCA    | 0.802 ± 0.033       | 0.649 ± 0.070  | 0.703 ± 0.055        |

**Data:** 378 patients, 54 raw features. Targets are binary (vessel disease: 0/1).

---

## Tasks Implemented

### A. Feature Engineering (Inside Pipeline)
Derived features added for all outer/inner folds:

- **LDL/HDL ratio** (cholesterol profile)
- **Triglyceride/HDL ratio** (lipid balance)
- **Risk-factor count:** DM + HTN + Current Smoker + Ex-Smoker + FH + DLP + Obesity
- **Age interactions:** Age × DM, Age × HTN, Age × Smoker
- **ECG abnormality score:** Q Wave + St Elevation + St Depression + T Inversion + LVH + Poor R Progression + BBB

### B. Feature Selection
Tested variants (selected inside each fold):
- **All features** (54 raw + derived)
- **Top 20 by importance** (Gini/gain from XGBoost)
- **Top 30 by importance**
- **Top 40 by importance**

Logs per-fold selections to `metrics/improvement_selections.csv`.

### C. Related-Target Information
- **CAD stacking for vessel targets:** Vessel classifiers (LAD/LCX/RCA) use out-of-fold CAD probability as an additional feature
- **Classifier chain (LAD→LCX/RCA):** For LCX, use OOF-predicted LAD as a feature; for RCA, use OOF-predicted LAD and LCX

### D. Model Comparison (Same CV Protocol)
Evaluated models:
1. **XGBoost** (current production)
2. **LightGBM**
3. **CatBoost**
4. **Random Forest**
5. **ExtraTrees**
6. **Logistic Regression** (regularized, L2)
7. **Soft-Voting Ensemble** (best 3 models by inner CV AUC)

Nested model selection chooses the best model for each outer fold based on inner CV performance.

### E. Hyperparameter Tuning
**Optuna optimization** (nested CV):
- Inner 3-fold CV to evaluate each trial
- Default 50 trials per outer fold per target
- Frozen seed (42 per repeat) for reproducibility
- Configurable via `--trials N` (0 to skip)

### F. Calibration
Tested variants (fit inside each fold):
- **No calibration** (raw model predictions)
- **Sigmoid calibration** (Platt)
- **Isotonic calibration**

Reports Brier score before/after and saves calibration plots to `metrics/calibration/*.png`.

### G. Threshold Selection
For each outer fold:
1. Generate predictions on inner training fold
2. Sweep thresholds, compute F1
3. Choose threshold that maximizes F1
4. Report precision, recall, F1, and chosen threshold at outer test time

Logs per-fold thresholds to `metrics/improvement_folds.csv`.

---

## Command-Line Options

```bash
python model/notebooks/improve_metrics.py [OPTIONS]

Options:
  --repeats N              Number of CV repeats (default: 10)
  --inner-folds N          Inner CV folds for tuning (default: 3)
  --trials N               Optuna trials per outer fit (default: 50; 0 to skip)
  --targets T1 T2 ...      Targets to evaluate (default: CAD LAD LCX RCA)
  --methods M1 M2 ...      Methods to test (default: all 13)
                           Available: xgb_baseline, xgb_engineered,
                           xgb_feature_selection_20/30/40, cad_oof_stack,
                           classifier_chain, nested_best_model,
                           nested_optuna_xgb, soft_vote_best3, calibration_sigmoid,
                           calibration_isotonic, catboost, lightgbm, rf, et,
                           logistic_regression
  --seed SEED              Random seed (default: 42)
  -v, --verbose            Enable debug logging
```

---

## Output Files

### `metrics/improvement_comparison.csv`
All-methods comparison with mean, SD, 95% CI for each metric:

```
target,method,accuracy_mean,accuracy_sd,accuracy_ci_lower,accuracy_ci_upper,
precision_mean,precision_sd,f1_mean,f1_sd,roc_auc_mean,roc_auc_sd,
roc_auc_ci_lower,roc_auc_ci_upper,pr_auc_mean,pr_auc_sd,brier_mean,brier_sd
CAD,baseline,0.890,0.028,0.833,0.947,...
```

Sorted by target and method name for easy comparison.

### `metrics/improvement_report.md`
Executive summary:
- Baseline metrics table
- Decision table: which methods improved, by how much, SD growth
- Promotion criteria: which candidates pass the gate (higher AUC mean, SD ≤ 1.1× baseline)
- Protocol notes, limitations (small dataset 378 rows, class imbalance)
- Recommendations

### `metrics/improvement_folds.csv`
Per-fold detail:
```
repeat,fold,target,method,split,auc,accuracy,precision,recall,f1,threshold,brier
```

Allows post-hoc analysis (e.g., variance across folds, outlier detection).

### `metrics/improvement_selections.csv`
Feature selection log:
```
repeat,fold,target,method,n_features_selected,selected_feature_names
```

Tracks which features were chosen in each fold (useful for debugging).

### `metrics/calibration/*.png`
Calibration plots for selected methods (if saved):
- X-axis: mean predicted probability (bins)
- Y-axis: fraction of positive samples
- Diagonal = perfect calibration
- Shows raw vs sigmoid vs isotonic overlaid

---

## Promotion Criteria

A candidate method is **promoted** (eligible for backend deployment) if:

1. **ROC-AUC improvement:** `method_auc_mean > baseline_auc_mean`
2. **Stability:** `method_auc_sd ≤ 1.1 × baseline_auc_sd` (no dramatic increase in variance)

If a target has no clear winner (tie or all losers), the baseline model is kept.

---

## Post-Run Validation (If Any Method Promotes)

After `improve_metrics.py` completes, for each promoted candidate:

```bash
# 1. Load candidate and re-run repeated-CV baseline protocol
python model/notebooks/repeated_cv_evaluation.py --model-path model/serialized_candidates/model_LAD.json --target LAD

# 2. Ensure backend still loads and explains correctly
python backend/test_backend.py
python scripts/check_metrics_consistency.py

# 3. If all pass, copy candidate to production
cp model/serialized_candidates/model_LAD.json model/serialized/model_lad.json
```

Then update:
- `model/serialized/model_metadata.json` (new thresholds, algorithm, metadata)
- `metrics/repeated_cv_summary.csv` (new baseline for future runs)
- `README.md` (performance table)

---

## Known Limitations

1. **Small Dataset:** Only 378 rows. CV-selected candidates may not generalize to a true holdout set.
2. **No External Validation:** No independent test cohort available for confirmation.
3. **Class Imbalance:** CAD is 77% positive, LAD 63%, LCX/RCA ~balanced. SMOTE inside each fold helps but is not a silver bullet.
4. **Feature Stability:** Top features vary across folds; ensemble or median selection may be more robust than fold-specific top-K.
5. **Optuna:** With only 378 rows and small data size, 50 trials may overfit. Consider reducing to 20–30 trials if overfitting is suspected in holdout tests.

---

## Reproducibility

All random seeds are fixed:
- **Outer CV splits:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)`
- **Inner CV splits:** `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- **Feature engineering:** Deterministic (no randomness in feature math)
- **Optuna:** Each repeat seeds Optuna to `42 + repeat_idx` for deterministic trial ordering

If you re-run with the same random seed and parameters, you should get identical results (within floating-point rounding).

---

## Example Workflow

```bash
# 1. Smoke test all components (quick)
python model/notebooks/improve_metrics.py --repeats 1 --inner-folds 2 --trials 0

# 2. If smoke test passes, run partial: 2 repeats, some methods
python model/notebooks/improve_metrics.py --repeats 2 --inner-folds 3 --trials 20 \
  --methods xgb_baseline xgb_engineered nested_best_model

# 3. If promising, run full pipeline
python model/notebooks/improve_metrics.py --repeats 10 --inner-folds 3 --trials 50

# 4. Review metrics/improvement_report.md to decide which candidates to promote

# 5. For each promoted candidate, re-validate with backend tests and deploy if successful
```

---

## Debugging

### Issue: Pipeline fails with "Forbidden feature ..."
**Cause:** A target column (CAD, LAD, LCX, RCA, Cath) was left in input features.  
**Fix:** Confirm `dataset.csv` excludes these from the raw features. If derived, ensure it's dropped before training.

### Issue: Optuna trials are very slow
**Cause:** 50 trials × 3 inner folds × 5 outer folds × 10 repeats = 7,500 model fits per target.  
**Fix:** Reduce `--trials` to 20–30, or `--inner-folds` to 2.

### Issue: Out-of-fold stacking produces NaNs
**Cause:** CAD classifier not fitted on full training fold before test prediction.  
**Fix:** Verify `CADProbabilityStackingClassifier.predict_proba()` calls the refitted CAD estimator, not the CV object.

### Issue: Calibration plots are blank
**Cause:** All predictions are the same value (poor separation).  
**Fix:** Check if the model is learning (compare baseline AUC to 0.5; if close, model is random).

---

## References

- **Baseline script:** `model/notebooks/repeated_cv_evaluation.py` (5×10 CV protocol)
- **Threshold analysis:** `model/notebooks/threshold_analysis.py`
- **Feature engineering:** `model/aortavision_preprocessing.py`
- **Backend:** `backend/main.py` (SHAP TreeExplainer on production models)
- **Dataset:** `dataset/dataset.csv` (378 rows, Z-Alizadeh Sani cohort)

---

## Contact & Notes

- Framework enforces nested CV with per-fold isolation to prevent data leakage
- Only tree-based models (XGBoost, LightGBM, CatBoost, RF, ExtraTrees) are SHAP-compatible with TreeExplainer
- Logistic Regression uses sklearn.inspection for feature importance (not tree-based)
- Reports include 95% confidence intervals (normal approximation on fold-wise metrics)

**Honest reporting:** If no method beats the baseline, the report will say so clearly. Small improvements within noise are not promoted.
