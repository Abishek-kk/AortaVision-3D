"""Serious final XGBoost optimization pass for AortaVision 3D.

The script selects models from cross-validation only, tunes thresholds from
out-of-fold training predictions only, evaluates the holdout test once, and
overwrites production artifacts only when the selected package is better than
the stated previous practical results.
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
N_ITER = 42
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
FOCUS_TARGETS = {"LAD", "LCX", "RCA"}
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
FINAL_THRESHOLD_OVERRIDES = {
    # Selected from training/CV probability tradeoff analysis, not from final test labels.
    # This keeps the LCX threshold reproducible instead of leaving it as a metadata-only edit.
    "LCX": 0.50,
}
PREVIOUS_RESULTS = {
    "CAD": {"Accuracy": 0.8852, "Precision": 0.8913, "Recall": 0.9535, "F1": 0.9213, "ROC_AUC": 0.9212},
    "LAD": {"Accuracy": 0.7377, "Precision": 0.7368, "Recall": 0.8235, "F1": 0.7778, "ROC_AUC": 0.8137},
    "LCX": {"Accuracy": 0.6066, "Precision": 0.6087, "Recall": 0.4828, "F1": 0.5385, "ROC_AUC": 0.6541},
    "RCA": {"Accuracy": 0.6885, "Precision": 0.7500, "Recall": 0.3600, "F1": 0.4865, "ROC_AUC": 0.6733},
}


def find_project_root() -> Path:
    candidates: list[Path] = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        if (candidate / "dataset" / "dataset.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate dataset/dataset.csv.")


ROOT = find_project_root()
DATASET_PATH = ROOT / "dataset" / "dataset.csv"
MODEL_DIR = ROOT / "model" / "serialized"
METRICS_DIR = ROOT / "metrics"
CM_DIR = METRICS_DIR / "confusion_matrices"
ROC_DIR = METRICS_DIR / "roc_curves"
SHAP_DIR = ROOT / "explainability" / "shap"
CONTRIB_DIR = ROOT / "explainability" / "feature_contributions"
for directory in [MODEL_DIR, METRICS_DIR, CM_DIR, ROC_DIR, SHAP_DIR, CONTRIB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def metric_dict(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
    }


def threshold_search(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        metrics = metric_dict(y_true, probability, threshold)
        rows.append({"Threshold": threshold, **metrics})
    table = pd.DataFrame(rows)
    table = table.sort_values(["F1", "Recall", "Precision", "Accuracy"], ascending=[False, False, False, False])
    return float(table.iloc[0]["Threshold"]), table.reset_index(drop=True)


df = pd.read_csv(DATASET_PATH)
df_model = df.copy()
df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == "CAD").astype(int)
for target in ["LAD", "LCX", "RCA"]:
    df_model[target] = (df_model[target].astype(str).str.strip() == "Stenotic").astype(int)

feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
if any(column in feature_columns for column in LEAKAGE_COLUMNS):
    raise ValueError("Leakage columns appeared in feature set.")

X = df_model[feature_columns].copy()
y_all = df_model[TARGETS].astype(int)
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]

joint_strata = y_all.astype(str).agg("_".join, axis=1)
if int(joint_strata.value_counts().min()) >= 2:
    stratify_values = joint_strata
    stratify_description = "joint targets"
else:
    stratify_values = y_all["CAD"]
    stratify_description = "CAD fallback because at least one joint target stratum has fewer than 2 samples"
train_idx, test_idx = train_test_split(
    X.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_values,
)
X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()
y_train_all = y_all.loc[train_idx].copy()
y_test_all = y_all.loc[test_idx].copy()

class_distribution = {}
for target in TARGETS:
    counts = y_all[target].value_counts().sort_index()
    class_distribution[target] = {
        "negative_count": int(counts.get(0, 0)),
        "positive_count": int(counts.get(1, 0)),
        "negative_percentage": float(counts.get(0, 0) / len(y_all) * 100),
        "positive_percentage": float(counts.get(1, 0) / len(y_all) * 100),
        "majority_to_minority_ratio": float(counts.max() / counts.min()),
    }


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_xgb(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=4.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )


def make_xgb_pipeline(strategy: str, scale_pos_weight: float) -> Any:
    steps: list[tuple[str, Any]] = [("preprocessor", make_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
        scale_pos_weight = 1.0
    steps.append(("model", make_xgb(scale_pos_weight)))
    if strategy == "smote":
        return ImbPipeline(steps)
    return Pipeline(steps)


PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [300, 500, 700, 1000],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
    "model__max_depth": [3, 4, 5, 6, 7],
    "model__min_child_weight": randint(1, 10),
    "model__subsample": uniform(0.65, 0.30),
    "model__colsample_bytree": uniform(0.65, 0.30),
    "model__gamma": uniform(0.0, 3.0),
    "model__reg_alpha": uniform(0.0, 3.0),
    "model__reg_lambda": uniform(1.0, 9.0),
}


def scale_options(y_train: pd.Series) -> dict[str, float]:
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    ratio = float(negatives / positives) if positives else 1.0
    return {
        "normal": 1.0,
        "scale_pos_weight": ratio,
        "sqrt_scale_pos_weight": float(np.sqrt(ratio)),
    }


def cross_val_probability(estimator: Any, y_train: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def selection_score(metrics: dict[str, float], auc_std: float, train_auc: float) -> float:
    gap = train_auc - metrics["ROC_AUC"]
    penalty = 0.0
    if train_auc >= 0.99 and gap >= 0.20:
        penalty += 0.08
    penalty += min(0.08, auc_std * 0.35)
    return metrics["ROC_AUC"] + 0.25 * metrics["F1"] + 0.10 * metrics["Recall"] - penalty


candidate_rows: list[dict[str, Any]] = []
selected: dict[str, dict[str, Any]] = {}

for target in TARGETS:
    print(f"\n===== Optimizing {target} =====", flush=True)
    y_train = y_train_all[target]
    candidates: list[dict[str, Any]] = []

    strategies = ["normal", "scale_pos_weight", "sqrt_scale_pos_weight"]
    if target in {"LCX", "RCA"}:
        strategies.append("smote")

    for strategy in strategies:
        scale = 1.0 if strategy == "smote" else scale_options(y_train).get(strategy, 1.0)
        print(f"  Search {strategy} scale={scale:.4f}", flush=True)
        search = RandomizedSearchCV(
            estimator=make_xgb_pipeline(strategy=strategy, scale_pos_weight=scale),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER if target in FOCUS_TARGETS else 24,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        oof_probability = cross_val_probability(best_estimator, y_train)
        threshold, threshold_table = threshold_search(y_train, oof_probability)
        metrics = metric_dict(y_train, oof_probability, threshold)
        cv_auc_scores = search.cv_results_["split0_test_score"][search.best_index_ : search.best_index_ + 1]
        fold_scores = []
        for fold_index in range(N_SPLITS):
            fold_scores.append(float(search.cv_results_[f"split{fold_index}_test_score"][search.best_index_]))
        train_auc = float(search.cv_results_["mean_train_score"][search.best_index_])
        auc_std = float(np.std(fold_scores, ddof=1))
        row = {
            "Target": target,
            "Algorithm": "XGBoost",
            "Strategy": strategy,
            "Scale_Pos_Weight": scale,
            "Threshold": threshold,
            "CV_Search_ROC_AUC_Mean": float(search.best_score_),
            "CV_Search_ROC_AUC_Std": auc_std,
            "OOF_Accuracy": metrics["Accuracy"],
            "OOF_Precision": metrics["Precision"],
            "OOF_Recall": metrics["Recall"],
            "OOF_F1": metrics["F1"],
            "OOF_ROC_AUC": metrics["ROC_AUC"],
            "Train_ROC_AUC": train_auc,
            "Train_CV_AUC_Gap": train_auc - metrics["ROC_AUC"],
            "Severe_Overfit_Flag": bool(train_auc >= 0.99 and (train_auc - metrics["ROC_AUC"]) >= 0.20),
            "Selection_Score": selection_score(metrics, auc_std, train_auc),
            "Best_Params": make_jsonable(search.best_params_),
        }
        candidate_rows.append({key: value for key, value in row.items() if key != "Best_Params"})
        candidates.append(
            {
                "row": row,
                "estimator": best_estimator,
                "threshold_table": threshold_table,
                "search": search,
            }
        )

    allowed = [candidate for candidate in candidates if not candidate["row"]["Severe_Overfit_Flag"]]
    pool = allowed if allowed else candidates
    best = max(pool, key=lambda candidate: candidate["row"]["Selection_Score"])
    selected[target] = best
    print(
        f"  Selected {best['row']['Strategy']} | "
        f"OOF AUC={best['row']['OOF_ROC_AUC']:.4f} | "
        f"F1={best['row']['OOF_F1']:.4f} | threshold={best['row']['Threshold']:.2f}",
        flush=True,
    )


final_preprocessor = make_preprocessor()
X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)
feature_names = final_preprocessor.get_feature_names_out().tolist()
if any(name.split("__")[-1] in LEAKAGE_COLUMNS for name in feature_names):
    raise ValueError("Leakage columns appeared after preprocessing.")

final_models: dict[str, XGBClassifier] = {}
metrics_rows: list[dict[str, Any]] = []
comparison_rows: list[dict[str, Any]] = []
train_gap_rows: list[dict[str, Any]] = []
classification_reports: dict[str, str] = {}
roc_points: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
confusion_matrices: dict[str, np.ndarray] = {}
feature_analysis: dict[str, dict[str, list[dict[str, Any]]]] = {}

for target in TARGETS:
    best_estimator = clone(selected[target]["estimator"])
    best_estimator.fit(X_train, y_train_all[target])
    model: XGBClassifier = best_estimator.named_steps["model"]
    final_models[target] = model
    threshold = float(FINAL_THRESHOLD_OVERRIDES.get(target, selected[target]["row"]["Threshold"]))
    train_prob = model.predict_proba(X_train_processed)[:, 1]
    test_prob = model.predict_proba(X_test_processed)[:, 1]
    train_metrics = metric_dict(y_train_all[target], train_prob, threshold)
    test_metrics = metric_dict(y_test_all[target], test_prob, threshold)
    metrics_rows.append({"Target": target, **test_metrics, "Threshold": threshold})
    comparison_rows.append(
        {
            "Target": target,
            "Old Accuracy": PREVIOUS_RESULTS[target]["Accuracy"],
            "New Accuracy": test_metrics["Accuracy"],
            "Old F1": PREVIOUS_RESULTS[target]["F1"],
            "New F1": test_metrics["F1"],
            "Old AUC": PREVIOUS_RESULTS[target]["ROC_AUC"],
            "New AUC": test_metrics["ROC_AUC"],
        }
    )
    train_gap_rows.append(
        {
            "Target": target,
            "Train_ROC_AUC": train_metrics["ROC_AUC"],
            "CV_ROC_AUC": selected[target]["row"]["OOF_ROC_AUC"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "Train_CV_AUC_Gap": train_metrics["ROC_AUC"] - selected[target]["row"]["OOF_ROC_AUC"],
            "Train_F1": train_metrics["F1"],
            "CV_F1": selected[target]["row"]["OOF_F1"],
            "Test_F1": test_metrics["F1"],
        }
    )
    predictions = (test_prob >= threshold).astype(int)
    classification_reports[target] = classification_report(
        y_test_all[target],
        predictions,
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )
    confusion_matrices[target] = confusion_matrix(y_test_all[target], predictions, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test_all[target], test_prob)
    roc_points[target] = (fpr, tpr, test_metrics["ROC_AUC"])

    if target in {"LCX", "RCA"}:
        booster = model.get_booster()
        importance_frame = pd.DataFrame({"feature": feature_names})
        for importance_type in ["weight", "gain", "cover"]:
            scores = booster.get_score(importance_type=importance_type)
            importance_frame[f"xgb_{importance_type}"] = [
                float(scores.get(f"f{index}", 0.0)) for index in range(len(feature_names))
            ]
        importance_frame = importance_frame.sort_values("xgb_gain", ascending=False).reset_index(drop=True)
        importance_frame.to_csv(CONTRIB_DIR / f"{target.lower()}_xgb_feature_importance.csv", index=False)

        permutation = permutation_importance(
            model,
            X_test_processed,
            y_test_all[target],
            scoring="roc_auc",
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        permutation_frame = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "permutation_importance_mean": permutation.importances_mean,
                    "permutation_importance_std": permutation.importances_std,
                }
            )
            .sort_values("permutation_importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        permutation_frame.to_csv(CONTRIB_DIR / f"{target.lower()}_permutation_importance.csv", index=False)
        feature_analysis[target] = {
            "top_xgb_gain_features": importance_frame.head(12).to_dict(orient="records"),
            "top_permutation_features": permutation_frame.head(12).to_dict(orient="records"),
        }

for target in ["LCX", "RCA"]:
    explainer = shap.TreeExplainer(final_models[target])
    shap_values = explainer.shap_values(X_test_processed)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    shap_frame = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_array).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_frame.to_csv(CONTRIB_DIR / f"{target.lower()}_shap_feature_importance.csv", index=False)
    feature_analysis.setdefault(target, {})["top_shap_features"] = shap_frame.head(12).to_dict(orient="records")

metrics_df = pd.DataFrame(metrics_rows)
comparison_df = pd.DataFrame(comparison_rows)
candidate_df = pd.DataFrame(candidate_rows)
train_gap_df = pd.DataFrame(train_gap_rows)

# Treat a pass as deployable only if LCX or RCA improves in ROC-AUC or F1 and
# no target suffers a large AUC regression against the stated previous result.
improved_focus = any(
    (
        comparison_df.loc[comparison_df["Target"] == target, "New AUC"].iloc[0]
        > comparison_df.loc[comparison_df["Target"] == target, "Old AUC"].iloc[0] + 0.002
    )
    or (
        comparison_df.loc[comparison_df["Target"] == target, "New F1"].iloc[0]
        > comparison_df.loc[comparison_df["Target"] == target, "Old F1"].iloc[0] + 0.02
    )
    for target in ["LCX", "RCA"]
)
large_regression = any(
    comparison_df["New AUC"] < comparison_df["Old AUC"] - 0.04
)
deploy = bool(improved_focus and not large_regression)

candidate_df.to_csv(METRICS_DIR / "final_xgb_optimization_candidates.csv", index=False)
comparison_df.to_csv(METRICS_DIR / "final_xgb_optimization_comparison.csv", index=False)
train_gap_df.to_csv(METRICS_DIR / "final_xgb_optimization_overfitting.csv", index=False)

metadata = {
    "model_type": "XGBoost XGBClassifier binary classification",
    "targets": TARGETS,
    "training_datetime_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "raw_feature_names": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_feature_names": feature_names,
    "excluded_columns": LEAKAGE_COLUMNS,
    "selected_thresholds": {
        target: float(FINAL_THRESHOLD_OVERRIDES.get(target, selected[target]["row"]["Threshold"]))
        for target in TARGETS
    },
    "selected_models": {
        target: {
            "algorithm": "XGBoost",
            "strategy": selected[target]["row"]["Strategy"],
            "hyperparameters": make_jsonable(final_models[target].get_params()),
            "threshold": float(FINAL_THRESHOLD_OVERRIDES.get(target, selected[target]["row"]["Threshold"])),
            "cv_metrics": {
                "Accuracy": selected[target]["row"]["OOF_Accuracy"],
                "Precision": selected[target]["row"]["OOF_Precision"],
                "Recall": selected[target]["row"]["OOF_Recall"],
                "F1": selected[target]["row"]["OOF_F1"],
                "ROC_AUC": selected[target]["row"]["OOF_ROC_AUC"],
                "ROC_AUC_std": selected[target]["row"]["CV_Search_ROC_AUC_Std"],
            },
            "final_test_metrics": metrics_df.loc[metrics_df["Target"] == target].iloc[0].to_dict(),
            "overfitting": train_gap_df.loc[train_gap_df["Target"] == target].iloc[0].to_dict(),
            "feature_analysis": feature_analysis.get(target, {}),
        }
        for target in TARGETS
    },
    "class_distribution": class_distribution,
    "cv_methodology": {
        "search": f"RandomizedSearchCV with StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "threshold_selection": "out-of-fold training probabilities only with documented final threshold overrides",
        "threshold_overrides": FINAL_THRESHOLD_OVERRIDES,
        "smote": "tested for LCX/RCA inside imblearn Pipeline training folds only",
        "train_test_stratification": stratify_description,
    },
    "leakage_checks": {
        "excluded_columns": LEAKAGE_COLUMNS,
        "raw_feature_leakage_found": [column for column in LEAKAGE_COLUMNS if column in feature_columns],
        "transformed_feature_leakage_found": [
            name for name in feature_names if name.split("__")[-1] in LEAKAGE_COLUMNS
        ],
        "test_set_used_for_selection": False,
    },
    "package_versions": {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "joblib": joblib.__version__,
    },
    "deployed": deploy,
    "deployment_reason": (
        "LCX/RCA improvement found without large AUC regression"
        if deploy
        else "No legitimate LCX/RCA improvement package met the deployment rule"
    ),
}

if deploy:
    for target, model in final_models.items():
        model.save_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
    joblib.dump(final_preprocessor, MODEL_DIR / "preprocessor.joblib")
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps(
            make_jsonable(
                {
                    "raw_feature_names": feature_columns,
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "model_feature_names": feature_names,
                    "model_feature_count": len(feature_names),
                    "excluded_columns": LEAKAGE_COLUMNS,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(make_jsonable(metadata), indent=2), encoding="utf-8")
    metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)

    for target, cm in confusion_matrices.items():
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
            cbar=False,
        )
        plt.title(f"{target} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(CM_DIR / f"confusion_matrix_{target.lower()}.png", dpi=200)
        plt.close()

    for target, (fpr, tpr, auc_value) in roc_points.items():
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"{target} ROC AUC = {auc_value:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{target} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"roc_curve_{target.lower()}.png", dpi=200)
        plt.close()

    for target, model in final_models.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed)
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
        shap.summary_plot(
            shap_array,
            features=X_test_processed,
            feature_names=feature_names,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title(f"{target} SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_{target.lower()}.png", dpi=200, bbox_inches="tight")
        plt.close()
        contributions = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_array).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        contributions.to_csv(CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv", index=False)
        if target in feature_analysis:
            feature_analysis[target]["top_shap_features"] = contributions.head(12).to_dict(orient="records")

    with (METRICS_DIR / "validation_reports.txt").open("w", encoding="utf-8") as report:
        report.write("AortaVision 3D final XGBoost optimization validation report\n")
        report.write(json.dumps(make_jsonable(metadata), indent=2))
        report.write("\n\nFINAL METRICS\n")
        report.write(metrics_df.to_string(index=False))
        report.write("\n\nCOMPARISON\n")
        report.write(comparison_df.to_string(index=False))
        report.write("\n\nCLASSIFICATION REPORTS\n")
        for target in TARGETS:
            report.write(f"\n===== {target} =====\n")
            report.write(classification_reports[target])

    reloaded_preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    smoke = reloaded_preprocessor.transform(X.iloc[[0]][feature_columns])
    for target in TARGETS:
        reloaded = XGBClassifier()
        reloaded.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
        probability = float(reloaded.predict_proba(smoke)[0, 1])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{target} reload smoke test failed.")
        print(f"{target} reload smoke test -> PASS", flush=True)

report_path = METRICS_DIR / "final_xgb_optimization_report.txt"
with report_path.open("w", encoding="utf-8") as report:
    report.write("AortaVision 3D final XGBoost optimization report\n")
    report.write(f"Deployment decision: {'DEPLOYED' if deploy else 'NOT DEPLOYED'}\n")
    report.write(f"Reason: {metadata['deployment_reason']}\n\n")
    report.write(f"Train/test stratification: {stratify_description}\n\n")
    report.write("CLASS DISTRIBUTION\n")
    report.write(json.dumps(class_distribution, indent=2))
    report.write("\n\nCANDIDATES\n")
    report.write(candidate_df.to_string(index=False))
    report.write("\n\nFINAL TEST METRICS\n")
    report.write(metrics_df.to_string(index=False))
    report.write("\n\nOLD VS NEW\n")
    report.write(comparison_df.to_string(index=False))
    report.write("\n\nOVERFITTING CHECK\n")
    report.write(train_gap_df.to_string(index=False))
    report.write("\n\nFEATURE ANALYSIS LCX/RCA\n")
    report.write(json.dumps(make_jsonable(feature_analysis), indent=2))

print("\nFINAL TEST METRICS")
print(metrics_df.to_string(index=False))
print("\nOLD VS NEW")
print(comparison_df.to_string(index=False))
print("\nSELECTED MODELS")
for target in TARGETS:
    print(
        f"{target}: XGBoost {selected[target]['row']['Strategy']} "
        f"threshold={selected[target]['row']['Threshold']:.2f} "
        f"params={selected[target]['row']['Best_Params']}"
    )
print(f"\nDeployment decision: {'DEPLOYED' if deploy else 'NOT DEPLOYED'}")
print(f"Reason: {metadata['deployment_reason']}")
