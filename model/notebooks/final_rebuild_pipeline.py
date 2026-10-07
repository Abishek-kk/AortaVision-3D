"""Canonical final rebuild pipeline for AortaVision 3D.

This script is intentionally self-contained and reproducible. It backs up the
raw dataset copy, performs documented cleaning, creates a deterministic
multi-target-balanced holdout, selects models/thresholds from training data only,
then saves the production artifacts and validation reports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "dataset" / "dataset.csv"
RAW_BACKUP_PATH = ROOT / "dataset" / "dataset_before_final_rebuild.csv"
QUALITY_REPORT = ROOT / "dataset" / "data_quality_report.md"
MODEL_DIR = ROOT / "model" / "serialized"
METRICS_DIR = ROOT / "metrics"
CM_DIR = METRICS_DIR / "confusion_matrices"
ROC_DIR = METRICS_DIR / "roc_curves"
SHAP_DIR = ROOT / "explainability" / "shap"
CONTRIB_DIR = ROOT / "explainability" / "feature_contributions"
DOCS_DIR = ROOT / "docs"

for directory in [MODEL_DIR, METRICS_DIR, CM_DIR, ROC_DIR, SHAP_DIR, CONTRIB_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.701, 0.05)]
THRESHOLD_OBJECTIVE = "maximize F1"
THRESHOLD_TIE_BREAKERS = ["F1", "Recall", "Precision", "Accuracy"]
THRESHOLD_RECALL_FLOOR: float | None = None

OLD_METRICS = {
    "CAD": {"Accuracy": 0.9342105263157895, "Precision": 0.9545454545454546, "Recall": 0.9692307692307692, "F1": 0.9618320610687023, "ROC_AUC": 0.9230769230769231},
    "LAD": {"Accuracy": 0.7894736842105263, "Precision": 0.796875, "Recall": 0.9444444444444444, "F1": 0.864406779661017, "ROC_AUC": 0.82996632996633},
    "LCX": {"Accuracy": 0.7631578947368421, "Precision": 0.825, "Recall": 0.75, "F1": 0.7857142857142857, "ROC_AUC": 0.8231534090909091},
    "RCA": {"Accuracy": 0.75, "Precision": 0.7169811320754716, "Recall": 0.9047619047619048, "F1": 0.8, "ROC_AUC": 0.76890756302521},
}

OLD_OVERFIT = {
    "CAD": {"Train_ROC_AUC": 0.9879482999534234, "CV_ROC_AUC": 0.9236143455985096, "Gap": 0.06433395435491374},
    "LAD": {"Train_ROC_AUC": 0.9638909358879882, "CV_ROC_AUC": 0.8487472365512159, "Gap": 0.11514369933677238},
    "LCX": {"Train_ROC_AUC": 0.9996925104326817, "CV_ROC_AUC": 0.8072479683725017, "Gap": 0.19244454206018002},
    "RCA": {"Train_ROC_AUC": 0.9904323019530392, "CV_ROC_AUC": 0.8248847926267281, "Gap": 0.16554750932631113},
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.DataFrame(index=df.index)
    y["CAD"] = (df["Cath"].astype(str).str.strip() == "CAD").astype(int)
    for target in ["LAD", "LCX", "RCA"]:
        y[target] = (df[target].astype(str).str.strip() == "Stenotic").astype(int)
    return y[TARGETS]


def deterministic_multilabel_holdout(y: pd.DataFrame, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, str]:
    n_test = int(round(len(y) * test_size))
    desired = y.sum(axis=0).to_numpy(dtype=float) * (n_test / len(y))
    rng = np.random.default_rng(random_state)
    order = rng.permutation(y.index.to_numpy())
    selected: list[int] = []
    selected_counts = np.zeros(len(TARGETS), dtype=float)

    while len(selected) < n_test:
        best_index: int | None = None
        best_score: tuple[float, float, float] | None = None
        for idx in order:
            if idx in selected:
                continue
            next_counts = selected_counts + y.loc[idx, TARGETS].to_numpy(dtype=float)
            remaining_slots = n_test - len(selected) - 1
            overshoot = np.maximum(0, next_counts - desired).sum()
            distance = np.abs(desired - next_counts).sum()
            rarity_bonus = -float(y.loc[idx, TARGETS].sum()) if remaining_slots > 0 else 0.0
            score = (float(overshoot), float(distance), rarity_bonus)
            if best_score is None or score < best_score:
                best_score = score
                best_index = int(idx)
        if best_index is None:
            break
        selected.append(best_index)
        selected_counts += y.loc[best_index, TARGETS].to_numpy(dtype=float)

    test_idx = np.array(sorted(selected), dtype=int)
    train_idx = np.array(sorted(set(y.index) - set(test_idx)), dtype=int)
    description = (
        "deterministic greedy multilabel-balanced holdout preserving each target "
        f"as closely as possible; random_state={random_state}"
    )
    return train_idx, test_idx, description


def metric_dict(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
    }


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows = []
    for threshold in THRESHOLDS:
        metrics = metric_dict(y_true, probability, threshold)
        rows.append({"Threshold": threshold, **metrics})
    table = pd.DataFrame(rows)
    table = table.sort_values(THRESHOLD_TIE_BREAKERS, ascending=[False] * len(THRESHOLD_TIE_BREAKERS))
    best = table.iloc[0].to_dict()
    threshold = float(best.pop("Threshold"))
    return threshold, {k: float(v) for k, v in best.items()}, table.reset_index(drop=True)


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            ),
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


def make_xgb(scale_pos_weight: float, variant: str) -> XGBClassifier:
    common = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )
    if variant == "very_regularized":
        return XGBClassifier(
            **common,
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.3,
            reg_alpha=2.0,
            reg_lambda=8.0,
        )
    if variant == "regularized":
        return XGBClassifier(
            **common,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=1.0,
            reg_lambda=5.0,
        )
    return XGBClassifier(
        **common,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.0,
        reg_alpha=0.5,
        reg_lambda=3.0,
    )


def candidate_pipelines(numeric_features: list[str], categorical_features: list[str], y_train: pd.Series) -> list[dict[str, Any]]:
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    scale = float(negatives / positives) if positives else 1.0
    pre = make_preprocessor(numeric_features, categorical_features)
    candidates: list[dict[str, Any]] = [
        {
            "algorithm": "LogisticRegression",
            "strategy": "balanced",
            "class_balancing": "class_weight=balanced",
            "estimator": Pipeline(
                [
                    ("preprocessor", pre),
                    ("model", LogisticRegression(class_weight="balanced", C=0.25, max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
                ]
            ),
            "deployable": False,
        },
        {
            "algorithm": "RandomForest",
            "strategy": "balanced_subsample",
            "class_balancing": "class_weight=balanced_subsample",
            "estimator": Pipeline(
                [
                    ("preprocessor", pre),
                    ("model", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)),
                ]
            ),
            "deployable": False,
        },
        {
            "algorithm": "ExtraTrees",
            "strategy": "balanced",
            "class_balancing": "class_weight=balanced",
            "estimator": Pipeline(
                [
                    ("preprocessor", pre),
                    ("model", ExtraTreesClassifier(n_estimators=400, max_depth=None, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
                ]
            ),
            "deployable": False,
        },
        {
            "algorithm": "HistGradientBoosting",
            "strategy": "regularized",
            "class_balancing": "none",
            "estimator": Pipeline(
                [
                    ("preprocessor", pre),
                    ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=2.0, random_state=RANDOM_STATE)),
                ]
            ),
            "deployable": False,
        },
        {
            "algorithm": "XGBoost",
            "strategy": "very_regularized",
            "class_balancing": "scale_pos_weight",
            "estimator": Pipeline([("preprocessor", pre), ("model", make_xgb(scale, "very_regularized"))]),
            "deployable": True,
        },
        {
            "algorithm": "XGBoost",
            "strategy": "regularized",
            "class_balancing": "scale_pos_weight",
            "estimator": Pipeline([("preprocessor", pre), ("model", make_xgb(scale, "regularized"))]),
            "deployable": True,
        },
        {
            "algorithm": "XGBoost",
            "strategy": "smote_regularized",
            "class_balancing": "SMOTE inside CV/final training pipeline",
            "estimator": ImbPipeline(
                [
                    ("preprocessor", pre),
                    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                    ("model", make_xgb(1.0, "regularized")),
                ]
            ),
            "deployable": True,
        },
    ]
    return candidates


def selection_score(row: dict[str, Any]) -> float:
    gap = row["Train_ROC_AUC"] - row["CV_ROC_AUC"]
    penalty = max(0.0, gap - 0.12) * 0.8
    if row["Algorithm"] != "XGBoost":
        penalty += 0.015  # backend currently consumes XGBoost JSON models.
    return row["CV_ROC_AUC"] + 0.25 * row["CV_F1"] + 0.08 * row["CV_Recall"] - penalty


def main() -> None:
    if not RAW_BACKUP_PATH.exists():
        shutil.copy2(DATASET_PATH, RAW_BACKUP_PATH)

    original_df = pd.read_csv(DATASET_PATH)
    df = original_df.copy()
    cleaning_actions: list[dict[str, str]] = []

    if "Fmale" in set(df["Sex"].astype(str)):
        df["Sex"] = df["Sex"].replace({"Fmale": "Female"})
        cleaning_actions.append({"item": "Sex", "action": "Fmale -> Female", "reason": "unambiguous category spelling correction"})
    if "mild" in set(df["VHD"].astype(str)):
        df["VHD"] = df["VHD"].replace({"mild": "Mild"})
        cleaning_actions.append({"item": "VHD", "action": "mild -> Mild", "reason": "unambiguous category casing correction"})

    y_all = derive_targets(df)
    inconsistent = df[(df["Cath"].eq("Normal")) & (df[["LAD", "LCX", "RCA"]].eq("Stenotic").any(axis=1))]
    tg_outliers = df[df["TG"] > 1000]

    constant_features = [
        column
        for column in df.columns
        if column not in LEAKAGE_COLUMNS and df[column].nunique(dropna=False) <= 1
    ]
    removed_features = [
        {"feature": feature, "reason": "constant feature in current training data", "decision": "removed from model inputs"}
        for feature in constant_features
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in LEAKAGE_COLUMNS and column not in constant_features
    ]
    X = df[feature_columns].copy()
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    train_idx, test_idx, split_description = deterministic_multilabel_holdout(y_all, TEST_SIZE, RANDOM_STATE)
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train_all = y_all.loc[train_idx].copy()
    y_test_all = y_all.loc[test_idx].copy()

    candidates_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for target in TARGETS:
        best_record: dict[str, Any] | None = None
        for candidate in candidate_pipelines(numeric_features, categorical_features, y_train_all[target]):
            estimator = candidate["estimator"]
            scoring = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision", "accuracy": "accuracy"}
            cv_result = cross_validate(estimator, X_train, y_train_all[target], cv=cv, scoring=scoring, return_train_score=True, n_jobs=1)
            oof_prob = cross_val_predict(estimator, X_train, y_train_all[target], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            threshold, cv_threshold_metrics, threshold_table = choose_threshold(y_train_all[target], oof_prob)
            row = {
                "Target": target,
                "Algorithm": candidate["algorithm"],
                "Strategy": candidate["strategy"],
                "Class_Balancing": candidate["class_balancing"],
                "Deployable": candidate["deployable"],
                "Threshold": threshold,
                "Train_ROC_AUC": float(np.mean(cv_result["train_roc_auc"])),
                "CV_ROC_AUC": float(np.mean(cv_result["test_roc_auc"])),
                "CV_ROC_AUC_Std": float(np.std(cv_result["test_roc_auc"], ddof=1)),
                "Train_F1": float(np.mean(cv_result["train_f1"])),
                "CV_F1": cv_threshold_metrics["F1"],
                "CV_Recall": cv_threshold_metrics["Recall"],
                "CV_Precision": cv_threshold_metrics["Precision"],
                "CV_Accuracy": cv_threshold_metrics["Accuracy"],
                "Train_CV_AUC_Gap": float(np.mean(cv_result["train_roc_auc"]) - np.mean(cv_result["test_roc_auc"])),
                "Estimator": estimator,
            }
            row["Selection_Score"] = selection_score(row)
            candidates_rows.append({k: v for k, v in row.items() if k != "Estimator"})
            if candidate["deployable"] and (best_record is None or row["Selection_Score"] > best_record["Selection_Score"]):
                best_record = row
                threshold_tables[target] = threshold_table
        if best_record is None:
            raise RuntimeError(f"No deployable model selected for {target}")
        selected[target] = best_record

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    transformed_features = preprocessor.get_feature_names_out().tolist()
    if any(name.split("__")[-1] in LEAKAGE_COLUMNS for name in transformed_features):
        raise ValueError("Leakage column appeared in transformed features.")

    final_models: dict[str, XGBClassifier] = {}
    metrics_rows: list[dict[str, Any]] = []
    overfit_rows: list[dict[str, Any]] = []
    reports: dict[str, str] = {}
    feature_contrib: dict[str, pd.DataFrame] = {}

    for target in TARGETS:
        estimator = clone(selected[target]["Estimator"])
        estimator.fit(X_train, y_train_all[target])
        model = estimator.named_steps["model"]
        final_models[target] = model
        threshold = float(selected[target]["Threshold"])
        train_prob = model.predict_proba(X_train_processed)[:, 1]
        test_prob = model.predict_proba(X_test_processed)[:, 1]
        train_metrics = metric_dict(y_train_all[target], train_prob, threshold)
        test_metrics = metric_dict(y_test_all[target], test_prob, threshold)
        metrics_rows.append({"Target": target, **test_metrics, "Threshold": threshold})
        overfit_rows.append(
            {
                "Target": target,
                "Train_ROC_AUC": train_metrics["ROC_AUC"],
                "CV_ROC_AUC": selected[target]["CV_ROC_AUC"],
                "Test_ROC_AUC": test_metrics["ROC_AUC"],
                "Train_CV_AUC_Gap": train_metrics["ROC_AUC"] - selected[target]["CV_ROC_AUC"],
                "Train_F1": train_metrics["F1"],
                "CV_F1": selected[target]["CV_F1"],
                "Test_F1": test_metrics["F1"],
            }
        )
        pred = (test_prob >= threshold).astype(int)
        reports[target] = classification_report(y_test_all[target], pred, target_names=["Negative", "Positive"], digits=4, zero_division=0)

        cm = confusion_matrix(y_test_all[target], pred, labels=[0, 1])
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"], cbar=False)
        plt.title(f"{target} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(CM_DIR / f"confusion_matrix_{target.lower()}.png", dpi=200)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test_all[target], test_prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"{target} ROC AUC = {test_metrics['ROC_AUC']:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{target} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"roc_curve_{target.lower()}.png", dpi=200)
        plt.close()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed)
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
        shap.summary_plot(shap_array, features=X_test_processed, feature_names=transformed_features, plot_type="bar", max_display=20, show=False)
        plt.title(f"{target} SHAP Feature Importance\nmodel explanation, not clinical causality")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_{target.lower()}.png", dpi=200, bbox_inches="tight")
        plt.close()
        contrib = pd.DataFrame({"feature": transformed_features, "mean_abs_shap": np.abs(shap_array).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        contrib.to_csv(CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv", index=False)
        feature_contrib[target] = contrib

    cleaned_hash = file_sha256(DATASET_PATH)
    df.to_csv(DATASET_PATH, index=False)
    cleaned_hash = file_sha256(DATASET_PATH)

    metrics_df = pd.DataFrame(metrics_rows)
    overfit_df = pd.DataFrame(overfit_rows)
    candidates_df = pd.DataFrame([{k: v for k, v in row.items() if k != "Estimator"} for row in candidates_rows])
    metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)
    overfit_df.to_csv(METRICS_DIR / "final_rebuild_overfitting.csv", index=False)
    candidates_df.to_csv(METRICS_DIR / "final_rebuild_candidates.csv", index=False)

    for target, model in final_models.items():
        model.save_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps(
            jsonable(
                {
                    "raw_feature_names": feature_columns,
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "model_feature_names": transformed_features,
                    "model_feature_count": len(transformed_features),
                    "excluded_columns": LEAKAGE_COLUMNS,
                    "removed_features": removed_features,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    class_distribution = {
        target: {
            "negative_count": int((y_all[target] == 0).sum()),
            "positive_count": int((y_all[target] == 1).sum()),
        }
        for target in TARGETS
    }
    metadata = {
        "project": "AortaVision 3D",
        "training_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_sha256": cleaned_hash,
        "random_state": RANDOM_STATE,
        "targets": TARGETS,
        "excluded_columns": LEAKAGE_COLUMNS,
        "raw_feature_names": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_feature_names": transformed_features,
        "removed_features": removed_features,
        "cleaning_actions": cleaning_actions,
        "class_distribution": class_distribution,
        "split_strategy": split_description,
        "training_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "cv_strategy": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "final_test_set_used_for_selection": False,
        "selected_thresholds": {row["Target"]: float(row["Threshold"]) for row in metrics_rows},
        "selected_models": {
            target: {
                "algorithm": "XGBoost",
                "strategy": selected[target]["Strategy"],
                "class_balancing": selected[target]["Class_Balancing"],
                "hyperparameters": jsonable(final_models[target].get_params()),
                "threshold": float(selected[target]["Threshold"]),
                "cv_metrics": {k.replace("CV_", ""): selected[target][k] for k in ["CV_ROC_AUC", "CV_F1", "CV_Recall", "CV_Precision", "CV_Accuracy"]},
                "train_cv_auc_gap": selected[target]["Train_CV_AUC_Gap"],
                "final_test_metrics": metrics_df.loc[metrics_df["Target"] == target].iloc[0].to_dict(),
                "overfitting": overfit_df.loc[overfit_df["Target"] == target].iloc[0].to_dict(),
            }
            for target in TARGETS
        },
        "preprocessing": {
            "numeric": "median imputation + RobustScaler",
            "categorical": "most-frequent imputation + OneHotEncoder(handle_unknown='ignore')",
            "fitted_on": "training data only",
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
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")

    QUALITY_REPORT.write_text(
        "\n".join(
            [
                "# Data quality report",
                "",
                f"Rows: {len(df)}",
                f"Columns: {len(df.columns)}",
                f"Exact duplicates: {int(df.duplicated().sum())}",
                f"Missing values: {int(df.isna().sum().sum())}",
                "",
                "## Cleaning actions",
                *[f"- {item['item']}: {item['action']} ({item['reason']})" for item in cleaning_actions],
                *([ "- none" ] if not cleaning_actions else []),
                "",
                "## Removed model-input features",
                *[f"- {item['feature']}: {item['reason']}" for item in removed_features],
                *([ "- none" ] if not removed_features else []),
                "",
                "## Cath/vessel inconsistency",
                f"- Rows found: {len(inconsistent)}",
                "- Decision: retained. CAD is derived from Cath by project definition; vessel labels are retained for vessel-specific targets. No correction was invented.",
                "",
                "## TG > 1000",
                f"- Rows found: {len(tg_outliers)}",
                "- Decision: retained as physiologically possible but extreme; robust scaling is used for numeric preprocessing. No target information was used.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with (METRICS_DIR / "validation_reports.txt").open("w", encoding="utf-8") as report:
        report.write("AortaVision 3D final rebuild validation report\n\n")
        report.write("The final test set was not used for model selection, hyperparameter tuning, feature selection, or threshold optimization.\n\n")
        report.write(f"Dataset rows: {len(df)}\nColumns: {len(df.columns)}\nFinal raw features: {len(feature_columns)}\nTransformed features: {len(transformed_features)}\n")
        report.write(f"Removed features: {removed_features}\n")
        report.write(f"Class distributions: {json.dumps(class_distribution)}\n\n")
        report.write(f"Split: train={len(train_idx)}, test={len(test_idx)}, random_state={RANDOM_STATE}\n{split_description}\n\n")
        report.write("FINAL TEST METRICS\n")
        report.write(metrics_df.to_string(index=False))
        report.write("\n\nOVERFITTING\n")
        report.write(overfit_df.to_string(index=False))
        report.write("\n\nSELECTED MODELS\n")
        for target in TARGETS:
            report.write(f"{target}: {selected[target]['Algorithm']} {selected[target]['Strategy']} threshold={selected[target]['Threshold']}\n")
        report.write("\nCLASSIFICATION REPORTS\n")
        for target in TARGETS:
            report.write(f"\n===== {target} =====\n")
            report.write(reports[target])

    (DOCS_DIR / "CANONICAL_METRICS.md").write_text(
        "# Canonical production metrics\n\n"
        "The canonical production metrics are the values in:\n\n"
        "- `metrics/classification_metrics.csv`\n\n"
        "and the corresponding metadata in:\n\n"
        "- `model/serialized/model_metadata.json`\n\n"
        "Auxiliary candidate/experiment files are not production results.\n",
        encoding="utf-8",
    )

    # Reload-from-disk consistency check.
    reloaded_pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    reloaded_X_test = reloaded_pre.transform(X_test)
    reloaded_meta = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    verify_rows = []
    for target in TARGETS:
        model = XGBClassifier()
        model.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
        prob = model.predict_proba(reloaded_X_test)[:, 1]
        threshold = float(reloaded_meta["selected_thresholds"][target])
        verify_rows.append({"Target": target, **metric_dict(y_test_all[target], prob, threshold), "Threshold": threshold})
    verify_df = pd.DataFrame(verify_rows)
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]:
        if not np.allclose(verify_df[metric], metrics_df[metric], atol=1e-12):
            raise RuntimeError(f"Reload verification failed for {metric}")

    print("FINAL METRICS")
    print(metrics_df.to_string(index=False))
    print("\nOVERFITTING")
    print(overfit_df.to_string(index=False))
    print("\nSelected")
    for target in TARGETS:
        print(target, selected[target]["Algorithm"], selected[target]["Strategy"], selected[target]["Threshold"])
    print("\nReload verification PASS")


if __name__ == "__main__":
    main()
