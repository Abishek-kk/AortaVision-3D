"""Leakage-safe optimization pipeline for AortaVision 3D models.

This script is intentionally auditable: all model selection is performed on the
training split with fold-contained preprocessing/feature selection, while the
holdout split is evaluated once after final choices are locked.
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

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
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
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

from model.aortavision_preprocessing import ClinicalFeatureEngineer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
STABILITY_SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.20
N_SPLITS = 5
N_ITER_SEARCH = 45
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
ENGINEERED_NUMERIC_FEATURES = [
    "Risk Factor Count",
    "Symptom Burden",
    "ECG Burden",
    "LDL HDL Ratio",
    "TG HDL Ratio",
    "Neut Lymph Ratio",
    "Platelet Lymph Ratio",
    "Age BMI Product",
    "BP PR Product",
    "Renal Load",
    "Inflammation Load",
    "EF RWMA Interaction",
]
OLD_RESULTS = {
    "CAD": {"Accuracy": 0.8361, "Precision": 0.8235, "Recall": 0.9767, "F1": 0.8936, "ROC_AUC": 0.9199},
    "LAD": {"Accuracy": 0.7541, "Precision": 0.7209, "Recall": 0.9118, "F1": 0.8052, "ROC_AUC": 0.8312},
    "LCX": {"Accuracy": 0.6393, "Precision": 0.6129, "Recall": 0.6552, "F1": 0.6333, "ROC_AUC": 0.6853},
    "RCA": {"Accuracy": 0.5738, "Precision": 0.4865, "Recall": 0.7200, "F1": 0.5806, "ROC_AUC": 0.6689},
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
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


df = pd.read_csv(DATASET_PATH)
required = {"Cath", "LAD", "LCX", "RCA"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

df_model = df.copy()
df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == "CAD").astype(int)
for target in ["LAD", "LCX", "RCA"]:
    df_model[target] = (df_model[target].astype(str).str.strip() == "Stenotic").astype(int)

feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
if any(column in feature_columns for column in LEAKAGE_COLUMNS):
    raise ValueError("Leakage column found in feature set.")

X = df_model[feature_columns].copy()
y_all = df_model[TARGETS].astype(int)
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]
preprocessor_numeric_features = numeric_features + ENGINEERED_NUMERIC_FEATURES

class_distribution: dict[str, dict[str, Any]] = {}
for target in TARGETS:
    counts = y_all[target].value_counts().sort_index()
    class_distribution[target] = {
        "positive_count": int(counts.get(1, 0)),
        "negative_count": int(counts.get(0, 0)),
        "positive_percentage": float((counts.get(1, 0) / len(y_all)) * 100),
        "negative_percentage": float((counts.get(0, 0) / len(y_all)) * 100),
        "majority_to_minority_ratio": float(counts.max() / counts.min()),
    }

joint_strata = y_all.astype(str).agg("_".join, axis=1)
stratify_values = joint_strata if int(joint_strata.value_counts().min()) >= 2 else None
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


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), preprocessor_numeric_features),
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
        n_estimators=140,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=5,
        gamma=1.0,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=6.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )


def make_search_pipeline(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            ("engineer", ClinicalFeatureEngineer()),
            ("preprocessor", make_preprocessor()),
            ("selector", SelectKBest(score_func=f_classif, k="all")),
            ("model", make_xgb(scale_pos_weight, random_state)),
        ]
    )


PARAM_DISTRIBUTIONS = {
    "selector__k": ["all", 15, 25, 35, 50, 65],
    "model__n_estimators": randint(60, 241),
    "model__max_depth": randint(1, 4),
    "model__learning_rate": uniform(0.015, 0.085),
    "model__min_child_weight": randint(3, 13),
    "model__gamma": uniform(0.0, 4.0),
    "model__subsample": uniform(0.60, 0.35),
    "model__colsample_bytree": uniform(0.55, 0.40),
    "model__reg_alpha": uniform(0.0, 5.0),
    "model__reg_lambda": uniform(2.0, 12.0),
}


def metric_dict(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
    }


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.20, 0.701, 0.01), 2):
        rows.append({"Threshold": float(threshold), **metric_dict(y_true, probabilities, float(threshold))})
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["F1", "Recall", "Precision", "Accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return float(frame.loc[0, "Threshold"]), frame


def oof_probabilities(estimator: Pipeline, x_data: pd.DataFrame, y_data: pd.Series, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(estimator, x_data, y_data, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def scale_candidates(y_train: pd.Series) -> dict[str, float]:
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    ratio = float(negatives / positives) if positives else 1.0
    sqrt_ratio = float(np.sqrt(ratio))
    return {
        "none": 1.0,
        "scale_pos_weight": ratio,
        "sqrt_scale_pos_weight": sqrt_ratio,
    }


def stability_metrics(estimator: Pipeline, y_train: pd.Series, threshold: float) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    rows = []
    for seed in STABILITY_SEEDS:
        seeded = clone(estimator)
        seeded.set_params(model__random_state=seed)
        probabilities = oof_probabilities(seeded, X_train, y_train, seed)
        rows.append({"Seed": seed, **metric_dict(y_train, probabilities, threshold)})
    frame = pd.DataFrame(rows)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    return (
        frame,
        frame[metrics].mean().to_dict(),
        frame[metrics].std(ddof=1).fillna(0).to_dict(),
        frame[metrics].min().to_dict(),
        frame[metrics].max().to_dict(),
    )


selected: dict[str, dict[str, Any]] = {}
search_rows: list[dict[str, Any]] = []
stability_rows: list[dict[str, Any]] = []

for target in TARGETS:
    print(f"\n===== Optimizing {target} =====", flush=True)
    y_train = y_train_all[target]
    best_record: dict[str, Any] | None = None

    for balance_name, scale_pos_weight in scale_candidates(y_train).items():
        print(f"  Search balance={balance_name}, scale_pos_weight={scale_pos_weight:.4f}", flush=True)
        search = RandomizedSearchCV(
            estimator=make_search_pipeline(scale_pos_weight),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        probabilities = oof_probabilities(estimator, X_train, y_train, RANDOM_STATE)
        threshold, threshold_table = select_threshold(y_train, probabilities)
        cv_metrics = metric_dict(y_train, probabilities, threshold)
        train_auc = float(search.cv_results_["mean_train_score"][search.best_index_])
        search_record = {
            "Target": target,
            "Balancing": balance_name,
            "Scale_Pos_Weight": scale_pos_weight,
            "Best_Search_ROC_AUC": float(search.best_score_),
            "Train_Search_ROC_AUC": train_auc,
            "Train_CV_AUC_Gap": train_auc - float(search.best_score_),
            "Threshold": threshold,
            **cv_metrics,
            "Best_Params": make_jsonable(search.best_params_),
        }
        search_rows.append(search_record)

        # Favor discrimination first, then a stable F1/recall balance, and apply
        # a small penalty for obvious search overfit.
        selection_score = (
            cv_metrics["ROC_AUC"]
            + 0.20 * cv_metrics["F1"]
            + 0.05 * cv_metrics["Recall"]
            - 0.15 * max(0.0, search_record["Train_CV_AUC_Gap"] - 0.10)
        )
        candidate = {
            "estimator": estimator,
            "balance_name": balance_name,
            "scale_pos_weight": scale_pos_weight,
            "threshold": threshold,
            "threshold_table": threshold_table,
            "cv_metrics_seed_42": cv_metrics,
            "selection_score": selection_score,
            "search": search,
            "search_record": search_record,
        }
        if best_record is None or candidate["selection_score"] > best_record["selection_score"]:
            best_record = candidate

    if best_record is None:
        raise RuntimeError(f"No candidate selected for {target}")

    stable_frame, stable_mean, stable_std, stable_min, stable_max = stability_metrics(
        best_record["estimator"],
        y_train,
        best_record["threshold"],
    )
    stable_frame.insert(0, "Target", target)
    stable_frame.insert(1, "Balancing", best_record["balance_name"])
    stability_rows.extend(stable_frame.to_dict(orient="records"))
    selected[target] = {
        **best_record,
        "stability_mean": stable_mean,
        "stability_std": stable_std,
        "stability_min": stable_min,
        "stability_max": stable_max,
    }
    print(
        f"  Selected {best_record['balance_name']} | "
        f"CV AUC={best_record['cv_metrics_seed_42']['ROC_AUC']:.4f} | "
        f"F1={best_record['cv_metrics_seed_42']['F1']:.4f} | "
        f"threshold={best_record['threshold']:.2f}",
        flush=True,
    )


final_preprocessor = Pipeline(
    [
        ("engineer", ClinicalFeatureEngineer()),
        ("preprocessor", make_preprocessor()),
    ]
)
X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)
all_feature_names = final_preprocessor.named_steps["preprocessor"].get_feature_names_out().tolist()
if any(name.split("__")[-1] in LEAKAGE_COLUMNS for name in all_feature_names):
    raise ValueError("Leakage column found after preprocessing.")

final_models: dict[str, XGBClassifier] = {}
target_feature_indices: dict[str, list[int]] = {}
target_feature_names: dict[str, list[str]] = {}
metrics_rows: list[dict[str, Any]] = []
comparison_rows: list[dict[str, Any]] = []
classification_reports: dict[str, str] = {}
train_gap_rows: list[dict[str, Any]] = []
confusion_matrices: dict[str, np.ndarray] = {}
roc_points: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

for target in TARGETS:
    cfg = selected[target]
    fitted_estimator: Pipeline = clone(cfg["estimator"])
    fitted_estimator.fit(X_train, y_train_all[target])
    selector: SelectKBest = fitted_estimator.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_indices = np.flatnonzero(selected_mask).astype(int).tolist()
    selected_names = [all_feature_names[index] for index in selected_indices]
    model_params = fitted_estimator.named_steps["model"].get_params()
    model = XGBClassifier(**model_params)
    model.fit(X_train_processed[:, selected_indices], y_train_all[target])
    final_models[target] = model
    target_feature_indices[target] = selected_indices
    target_feature_names[target] = selected_names

    threshold = float(cfg["threshold"])
    train_prob = model.predict_proba(X_train_processed[:, selected_indices])[:, 1]
    test_prob = model.predict_proba(X_test_processed[:, selected_indices])[:, 1]
    train_metrics = metric_dict(y_train_all[target], train_prob, threshold)
    test_metrics = metric_dict(y_test_all[target], test_prob, threshold)
    metrics_rows.append({"Target": target, **test_metrics, "Threshold": threshold})
    comparison_rows.append(
        {
            "Target": target,
            **{f"Old_{metric}": OLD_RESULTS[target][metric] for metric in OLD_RESULTS[target]},
            **{f"New_{metric}": test_metrics[metric] for metric in OLD_RESULTS[target]},
            **{f"Delta_{metric}": test_metrics[metric] - OLD_RESULTS[target][metric] for metric in OLD_RESULTS[target]},
        }
    )
    train_gap_rows.append(
        {
            "Target": target,
            "Train_F1": train_metrics["F1"],
            "CV_F1": cfg["cv_metrics_seed_42"]["F1"],
            "Test_F1": test_metrics["F1"],
            "Train_ROC_AUC": train_metrics["ROC_AUC"],
            "CV_ROC_AUC": cfg["cv_metrics_seed_42"]["ROC_AUC"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "F1_train_cv_gap": train_metrics["F1"] - cfg["cv_metrics_seed_42"]["F1"],
            "AUC_train_cv_gap": train_metrics["ROC_AUC"] - cfg["cv_metrics_seed_42"]["ROC_AUC"],
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

metrics_df = pd.DataFrame(metrics_rows)
comparison_df = pd.DataFrame(comparison_rows)
search_df = pd.DataFrame([{k: v for k, v in row.items() if k != "Best_Params"} for row in search_rows])
stability_df = pd.DataFrame(stability_rows)
train_gap_df = pd.DataFrame(train_gap_rows)

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
    indices = target_feature_indices[target]
    names = target_feature_names[target]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed[:, indices])
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    if shap_array.shape[1] != len(names):
        raise ValueError(f"SHAP feature count mismatch for {target}.")
    shap.summary_plot(
        shap_array,
        features=X_test_processed[:, indices],
        feature_names=names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title(f"{target} SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"shap_{target.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()
    contributions = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_array).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    contributions.to_csv(CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv", index=False)

model_paths: dict[str, str] = {}
for target, model in final_models.items():
    path = MODEL_DIR / f"model_{target.lower()}.json"
    model.save_model(str(path))
    model_paths[target] = str(path.relative_to(ROOT))

preprocessor_path = MODEL_DIR / "preprocessor.joblib"
joblib.dump(final_preprocessor, preprocessor_path)

feature_payload = {
    "raw_feature_names": feature_columns,
    "numeric_features": numeric_features,
    "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
    "categorical_features": categorical_features,
    "model_feature_names": all_feature_names,
    "model_feature_count": len(all_feature_names),
    "target_feature_indices": target_feature_indices,
    "target_feature_names": target_feature_names,
    "excluded_columns": LEAKAGE_COLUMNS,
}
(MODEL_DIR / "feature_names.json").write_text(json.dumps(make_jsonable(feature_payload), indent=2), encoding="utf-8")

metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)
comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
comparison_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)
search_df.to_csv(METRICS_DIR / "optimization_search_summary.csv", index=False)
stability_df.to_csv(METRICS_DIR / "cv_seed_stability.csv", index=False)

metadata_targets: dict[str, Any] = {}
for target in TARGETS:
    cfg = selected[target]
    row = metrics_df.loc[metrics_df["Target"] == target].iloc[0].to_dict()
    metadata_targets[target] = {
        "target": target,
        "hyperparameters": {
            key: value
            for key, value in final_models[target].get_params().items()
            if key
            in {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "gamma",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
                "random_state",
            }
        },
        "preprocessing_method": "ColumnTransformer: median numeric imputation, most-frequent categorical imputation, one-hot encoding",
        "feature_selection_method": "SelectKBest(f_classif) fit inside CV folds and final training split",
        "selected_feature_count": len(target_feature_indices[target]),
        "selected_feature_names": target_feature_names[target],
        "balancing_method": cfg["balance_name"],
        "scale_pos_weight": cfg["scale_pos_weight"],
        "threshold": cfg["threshold"],
        "random_state": RANDOM_STATE,
        "cv_methodology": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state=42); repeated seeds {STABILITY_SEEDS}",
        "cv_metrics_seed_42": cfg["cv_metrics_seed_42"],
        "cv_metrics_mean": cfg["stability_mean"],
        "cv_metrics_standard_deviation": cfg["stability_std"],
        "cv_metrics_min": cfg["stability_min"],
        "cv_metrics_max": cfg["stability_max"],
        "final_test_metrics": {metric: float(row[metric]) for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]},
        "search_summary": {
            "selected_score": cfg["selection_score"],
            "best_search_record": {k: v for k, v in cfg["search_record"].items() if k != "Best_Params"},
            "best_params": cfg["search_record"]["Best_Params"],
        },
        "overfitting_status": train_gap_df.loc[train_gap_df["Target"] == target].iloc[0].to_dict(),
    }

metadata = {
    "model_type": "XGBoost XGBClassifier binary classification",
    "targets": TARGETS,
    "training_datetime_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "stability_random_states": STABILITY_SEEDS,
    "test_size": TEST_SIZE,
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "raw_feature_count": len(feature_columns),
    "model_feature_count": len(all_feature_names),
    "raw_feature_names": feature_columns,
    "numeric_features": numeric_features,
    "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
    "categorical_features": categorical_features,
    "model_feature_names": all_feature_names,
    "target_feature_indices": target_feature_indices,
    "target_feature_names": target_feature_names,
    "excluded_columns": LEAKAGE_COLUMNS,
    "model_filenames": model_paths,
    "preprocessor_filename": str(preprocessor_path.relative_to(ROOT)),
    "feature_names_filename": str((MODEL_DIR / "feature_names.json").relative_to(ROOT)),
    "selected_thresholds": {target: float(selected[target]["threshold"]) for target in TARGETS},
    "cv_methodology": {
        "search": f"RandomizedSearchCV with StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "search_iterations_per_balancing_strategy": N_ITER_SEARCH,
        "search_scoring": "roc_auc",
        "balancing_strategies": ["none", "scale_pos_weight", "sqrt_scale_pos_weight"],
        "threshold_selection": "cross-validated training predictions only; F1 primary with recall/precision/accuracy tie-breaks",
        "holdout_test": "untouched until final selected models were fit",
    },
    "preprocessing": {
        "type": "sklearn.compose.ColumnTransformer",
        "numeric_imputation": "median",
        "engineered_features": ENGINEERED_NUMERIC_FEATURES,
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "cv_fit_scope": "inside each cross-validation fold via sklearn Pipeline",
        "final_fit_scope": "final training split only",
    },
    "class_distribution_full_dataset": class_distribution,
    "targets_detail": metadata_targets,
    "evaluation_metrics": metrics_df.to_dict(orient="records"),
    "old_vs_new_comparison": comparison_df.to_dict(orient="records"),
    "optimization_search_summary": search_df.to_dict(orient="records"),
    "train_cv_test_gap": train_gap_df.to_dict(orient="records"),
    "leakage_checks": {
        "excluded_columns": LEAKAGE_COLUMNS,
        "raw_feature_leakage_found": [column for column in LEAKAGE_COLUMNS if column in feature_columns],
        "transformed_feature_leakage_found": [
            name for name in all_feature_names if name.split("__")[-1] in LEAKAGE_COLUMNS
        ],
        "preprocessing_fit_scope": "CV folds for selection/search; final train split only before test evaluation",
        "feature_selection_fit_scope": "CV folds for selection/search; final train split only before test evaluation",
        "smote_used": False,
        "test_set_used_for_selection": False,
    },
    "dataset_inspection": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
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
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(make_jsonable(metadata), indent=2), encoding="utf-8")

with (METRICS_DIR / "validation_reports.txt").open("w", encoding="utf-8") as report:
    report.write("AortaVision 3D optimized validation reports\n")
    report.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
    report.write("LEAKAGE CHECK\n")
    report.write(json.dumps(metadata["leakage_checks"], indent=2))
    report.write("\n\nCLASS DISTRIBUTION\n")
    report.write(json.dumps(class_distribution, indent=2))
    report.write("\n\nOPTIMIZATION SEARCH SUMMARY\n")
    report.write(search_df.to_string(index=False))
    report.write("\n\nCV STABILITY\n")
    report.write(stability_df.to_string(index=False))
    report.write("\n\nFINAL TEST METRICS\n")
    report.write(metrics_df.to_string(index=False))
    report.write("\n\nOLD VS NEW\n")
    report.write(comparison_df.to_string(index=False))
    report.write("\n\nTRAIN/CV/TEST GAP CHECK\n")
    report.write(train_gap_df.to_string(index=False))
    report.write("\n\nCLASSIFICATION REPORTS\n")
    for target in TARGETS:
        report.write(f"\n===== {target} =====\n")
        report.write(classification_reports[target])

reloaded_preprocessor = joblib.load(preprocessor_path)
reloaded_metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
smoke_raw = X.iloc[[0]][feature_columns].copy()
smoke_processed = reloaded_preprocessor.transform(smoke_raw)
smoke_results = {}
for target in TARGETS:
    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
    indices = reloaded_metadata["target_feature_indices"][target]
    probability = float(model.predict_proba(smoke_processed[:, indices])[0, 1])
    threshold = float(reloaded_metadata["selected_thresholds"][target])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{target} reload probability outside [0, 1].")
    smoke_results[target] = {"probability": probability, "threshold": threshold, "prediction": int(probability >= threshold)}
    print(f"{target} prediction -> PASS", flush=True)

print("\nOLD RESULTS")
print(pd.DataFrame([{"Target": target, **metrics} for target, metrics in OLD_RESULTS.items()]).to_string(index=False))
print("\nNEW RESULTS")
print(metrics_df[["Target", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]].to_string(index=False))
print("\nIMPROVEMENT")
print(comparison_df[[column for column in comparison_df.columns if column == "Target" or column.startswith("Delta_")]].to_string(index=False))

mean_auc = float(metrics_df["ROC_AUC"].mean())
mean_f1 = float(metrics_df["F1"].mean())
verdict = "NEEDS IMPROVEMENT"
if mean_auc >= 0.90 and mean_f1 >= 0.88:
    verdict = "EXCELLENT"
elif mean_auc >= 0.78 or mean_f1 >= 0.72:
    verdict = "GOOD"
print(f"\nFINAL VERDICT: {verdict}")
