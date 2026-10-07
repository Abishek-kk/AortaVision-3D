"""Optimize and validate the AortaVision 3D XGBoost model pipeline."""

# %% [markdown]
# # AortaVision 3D Model Training
#
# This executable notebook trains four independent XGBoost classifiers for CAD,
# LAD, LCX, and RCA. It excludes CAD, LAD, LCX, RCA, and Cath from the input
# features, fits preprocessing only inside training/CV folds, runs a controlled
# randomized hyperparameter search, selects decision thresholds from
# cross-validated training predictions, and evaluates once on an untouched
# holdout test set.

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import argparse
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
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## 2. Configuration

# %%
RANDOM_STATE = 42
STABILITY_SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.20
N_SPLITS = 5
N_ITER_SEARCH = 30
TARGET_COLUMNS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
POSITIVE_LABELS = {
    "CAD": "CAD",
    "LAD": "Stenotic",
    "LCX": "Stenotic",
    "RCA": "Stenotic",
}
TARGET_GOALS = {
    "Accuracy": 0.92,
    "Precision": 0.92,
    "Recall": 0.93,
    "F1": 0.92,
    "ROC_AUC": 0.94,
}


def find_project_root() -> Path:
    """Find the project root from either script or notebook execution."""
    candidates: list[Path] = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        if (candidate / "dataset" / "dataset.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate dataset/dataset.csv from this context.")


ROOT = find_project_root()
parser = argparse.ArgumentParser(description="Train and validate AortaVision models.")
parser.add_argument(
    "--dataset",
    type=Path,
    help="CSV dataset path (defaults to dataset/dataset.csv).",
)
args, _ = parser.parse_known_args()
DATASET_PATH = args.dataset or Path("dataset") / "dataset.csv"
if not DATASET_PATH.is_absolute():
    DATASET_PATH = ROOT / DATASET_PATH
DATASET_PATH = DATASET_PATH.resolve()
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset file does not exist: {DATASET_PATH}")
MODEL_DIR = ROOT / "model" / "serialized"
METRICS_DIR = ROOT / "metrics"
CM_DIR = METRICS_DIR / "confusion_matrices"
ROC_DIR = METRICS_DIR / "roc_curves"
SHAP_DIR = ROOT / "explainability" / "shap"
CONTRIB_DIR = ROOT / "explainability" / "feature_contributions"

for directory in [MODEL_DIR, METRICS_DIR, CM_DIR, ROC_DIR, SHAP_DIR, CONTRIB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Project root: {ROOT}")
print(f"Primary random state: {RANDOM_STATE}")
print(f"Stability seeds: {STABILITY_SEEDS}")

# %% [markdown]
# ## 3. Load and Inspect Dataset

# %%
df = pd.read_csv(DATASET_PATH)
dataset_inspection = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "column_names": df.columns.tolist(),
    "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
    "duplicate_rows": int(df.duplicated().sum()),
}

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("Column names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())
print(f"\nDuplicate rows: {dataset_inspection['duplicate_rows']}")

# %% [markdown]
# ## 4. Target Encoding and Class Balance

# %%
missing_required = [column for column in ["LAD", "LCX", "RCA", "Cath"] if column not in df.columns]
if missing_required:
    raise ValueError(f"Dataset is missing required target/source columns: {missing_required}")

df_model = df.copy()
if "CAD" not in df_model.columns:
    df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == POSITIVE_LABELS["CAD"]).astype(int)
else:
    df_model["CAD"] = (
        df_model["CAD"].astype(str).str.strip().isin([POSITIVE_LABELS["CAD"], "1", "True", "true"])
    ).astype(int)

for target in ["LAD", "LCX", "RCA"]:
    df_model[target] = (df_model[target].astype(str).str.strip() == POSITIVE_LABELS[target]).astype(int)

y_all = df_model[TARGET_COLUMNS].astype(int)
class_distribution: dict[str, dict[str, object]] = {}
for target in TARGET_COLUMNS:
    counts = y_all[target].value_counts().sort_index()
    percentages = (counts / len(y_all) * 100).round(2)
    imbalance_ratio = float(counts.max() / counts.min())
    class_distribution[target] = {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "percentages": {str(key): float(value) for key, value in percentages.items()},
        "majority_to_minority_ratio": imbalance_ratio,
    }
    print(f"\n{target} distribution:")
    print(counts)
    print(percentages)
    print(f"Majority/minority ratio: {imbalance_ratio:.3f}")

# %% [markdown]
# ## 5. Leakage Prevention and Feature Typing

# %%
feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
leakage_found = [column for column in LEAKAGE_COLUMNS if column in feature_columns]
if leakage_found:
    raise ValueError(f"Leakage columns found in feature set: {leakage_found}")

X = df_model[feature_columns].copy()
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]

print(f"Raw feature count: {len(feature_columns)}")
print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# %% [markdown]
# ## 6. Untouched Holdout Test Split

# %%
joint_strata = y_all.astype(str).agg("_".join, axis=1)
feature_groups = pd.MultiIndex.from_frame(X).factorize(sort=True)[0]
group_targets = y_all.groupby(feature_groups).nunique()
if group_targets.gt(1).any(axis=None):
    raise ValueError("Identical feature rows have conflicting target labels.")
representative_mask = ~X.duplicated(keep="first")
unique_indices = X.index[representative_mask]
unique_group_ids = feature_groups[representative_mask]
unique_strata = joint_strata.loc[unique_indices]
strata_counts = unique_strata.value_counts()
stratify_values = unique_strata if int(strata_counts.min()) >= 2 else None

train_group_ids, test_group_ids = train_test_split(
    unique_group_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_values,
)
train_mask = np.isin(feature_groups, train_group_ids)
test_mask = np.isin(feature_groups, test_group_ids) & representative_mask
train_idx = X.index[train_mask]
test_idx = X.index[test_mask]
train_groups = feature_groups[train_mask]
X_train_raw = X.loc[train_idx].copy()
X_test_raw = X.loc[test_idx].copy()
y_train_all = y_all.loc[train_idx].copy()
y_test_all = y_all.loc[test_idx].copy()

print(f"Dataset: {DATASET_PATH}")
print(f"Unique feature records: {len(unique_indices)}")
print(f"Training rows (including oversamples): {len(train_idx)}")
print(f"Training feature groups: {len(train_group_ids)}")
print(f"Test feature groups: {len(test_group_ids)}")
print("The holdout test set is not used for search, preprocessing fit, balancing decisions, or threshold selection.")

# %% [markdown]
# ## 7. Pipeline Helpers

# %%
def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_xgb(scale_pos_weight: float = 1.0, random_state: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=140,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )


def make_pipeline(scale_pos_weight: float = 1.0, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", make_xgb(scale_pos_weight=scale_pos_weight, random_state=random_state)),
        ]
    )


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
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        metrics = metric_dict(y_true, probabilities, float(threshold))
        rows.append({"Threshold": float(threshold), **metrics})
    threshold_df = pd.DataFrame(rows)
    threshold_df = threshold_df.sort_values(
        ["F1", "Recall", "Precision", "Accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return float(threshold_df.loc[0, "Threshold"]), threshold_df


def train_cv_predictions(
    estimator: Pipeline,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(
        estimator,
        X_data,
        y_data,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


param_distributions = {
    "model__n_estimators": randint(80, 321),
    "model__max_depth": randint(2, 5),
    "model__learning_rate": uniform(0.025, 0.125),
    "model__subsample": uniform(0.70, 0.30),
    "model__colsample_bytree": uniform(0.65, 0.35),
    "model__min_child_weight": randint(1, 8),
    "model__gamma": uniform(0.0, 1.2),
    "model__reg_alpha": uniform(0.0, 1.0),
    "model__reg_lambda": uniform(1.0, 5.0),
}

# %% [markdown]
# ## 8. Baseline, Search, Threshold Selection, and Stability CV

# %%
selection_by_target: dict[str, dict[str, Any]] = {}
comparison_rows: list[dict[str, Any]] = []
stability_rows: list[dict[str, Any]] = []

for target in TARGET_COLUMNS:
    print(f"\n===== {target} =====")
    y_train = y_train_all[target]
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    scale_pos_weight = float(negatives / positives) if positives else 1.0
    baseline_estimator = make_pipeline(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)

    baseline_cv_prob = train_cv_predictions(
        baseline_estimator, X_train_raw, y_train, train_groups, RANDOM_STATE
    )
    baseline_threshold, _ = select_threshold(y_train, baseline_cv_prob)
    baseline_cv_metrics = metric_dict(y_train, baseline_cv_prob, baseline_threshold)

    search_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=make_pipeline(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=search_cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    search.fit(X_train_raw, y_train, groups=train_groups)

    optimized_estimator = search.best_estimator_
    optimized_cv_prob = train_cv_predictions(
        optimized_estimator, X_train_raw, y_train, train_groups, RANDOM_STATE
    )
    optimized_threshold, _ = select_threshold(y_train, optimized_cv_prob)
    optimized_cv_metrics = metric_dict(y_train, optimized_cv_prob, optimized_threshold)

    selected_version = "Optimized"
    selected_estimator = optimized_estimator
    selected_threshold = optimized_threshold
    selected_cv_prob = optimized_cv_prob
    selected_cv_metrics = optimized_cv_metrics
    if (
        optimized_cv_metrics["ROC_AUC"] < baseline_cv_metrics["ROC_AUC"] - 0.005
        and optimized_cv_metrics["F1"] < baseline_cv_metrics["F1"] - 0.005
    ):
        selected_version = "Baseline"
        selected_estimator = baseline_estimator
        selected_threshold = baseline_threshold
        selected_cv_prob = baseline_cv_prob
        selected_cv_metrics = baseline_cv_metrics

    seed_metrics = []
    for seed in STABILITY_SEEDS:
        seeded_estimator = clone(selected_estimator)
        seeded_estimator.set_params(model__random_state=seed)
        seed_prob = train_cv_predictions(
            seeded_estimator, X_train_raw, y_train, train_groups, seed
        )
        seed_row = {"Target": target, "Seed": seed, **metric_dict(y_train, seed_prob, selected_threshold)}
        seed_metrics.append(seed_row)
        stability_rows.append(seed_row)
    stability_target_df = pd.DataFrame(seed_metrics)
    stability_mean = stability_target_df[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].mean().to_dict()
    stability_std = stability_target_df[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].std(ddof=1).fillna(0).to_dict()

    comparison_rows.append({"Target": target, "Version": "Baseline", "Threshold": baseline_threshold, **baseline_cv_metrics})
    comparison_rows.append({"Target": target, "Version": "Optimized", "Threshold": optimized_threshold, **optimized_cv_metrics})

    selection_by_target[target] = {
        "selected_version": selected_version,
        "selected_estimator": selected_estimator,
        "selected_threshold": selected_threshold,
        "scale_pos_weight": scale_pos_weight,
        "best_search_score_roc_auc": float(search.best_score_),
        "best_params": {
            key.replace("model__", ""): value.item() if hasattr(value, "item") else value
            for key, value in search.best_params_.items()
        },
        "baseline_cv_metrics": baseline_cv_metrics,
        "optimized_cv_metrics": optimized_cv_metrics,
        "selected_cv_metrics": selected_cv_metrics,
        "cv_probability_min": float(np.min(selected_cv_prob)),
        "cv_probability_max": float(np.max(selected_cv_prob)),
        "stability_mean": {key: float(value) for key, value in stability_mean.items()},
        "stability_std": {key: float(value) for key, value in stability_std.items()},
    }
    print(f"Baseline CV: {baseline_cv_metrics}, threshold={baseline_threshold:.2f}")
    print(f"Optimized CV: {optimized_cv_metrics}, threshold={optimized_threshold:.2f}")
    print(f"Selected: {selected_version}")
    print(f"Best search ROC-AUC: {search.best_score_:.4f}")

comparison_df = pd.DataFrame(comparison_rows)
stability_df = pd.DataFrame(stability_rows)
print("\nBaseline vs optimized CV comparison:")
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## 9. Final Model Fit and Test Evaluation

# %%
final_preprocessor = make_preprocessor()
X_train_processed = final_preprocessor.fit_transform(X_train_raw)
X_test_processed = final_preprocessor.transform(X_test_raw)
feature_names = final_preprocessor.get_feature_names_out().tolist()

if any(name.split("__")[-1] in LEAKAGE_COLUMNS for name in feature_names):
    raise ValueError("A leakage column appeared in transformed feature names.")

models: dict[str, XGBClassifier] = {}
metrics_rows: list[dict[str, Any]] = []
classification_reports: dict[str, str] = {}
confusion_matrices: dict[str, np.ndarray] = {}
roc_points: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
train_gap_rows: list[dict[str, Any]] = []

for target in TARGET_COLUMNS:
    selected = selection_by_target[target]
    params = selected["selected_estimator"].named_steps["model"].get_params()
    model = XGBClassifier(**params)
    model.fit(X_train_processed, y_train_all[target])
    models[target] = model

    threshold = float(selected["selected_threshold"])
    train_prob = model.predict_proba(X_train_processed)[:, 1]
    test_prob = model.predict_proba(X_test_processed)[:, 1]
    train_metrics = metric_dict(y_train_all[target], train_prob, threshold)
    test_metrics = metric_dict(y_test_all[target], test_prob, threshold)
    metrics_rows.append({"Target": target, **test_metrics, "Threshold": threshold})
    train_gap_rows.append(
        {
            "Target": target,
            "Train_F1": train_metrics["F1"],
            "CV_F1": selected["selected_cv_metrics"]["F1"],
            "Test_F1": test_metrics["F1"],
            "Train_ROC_AUC": train_metrics["ROC_AUC"],
            "CV_ROC_AUC": selected["selected_cv_metrics"]["ROC_AUC"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "F1_train_cv_gap": train_metrics["F1"] - selected["selected_cv_metrics"]["F1"],
            "AUC_train_cv_gap": train_metrics["ROC_AUC"] - selected["selected_cv_metrics"]["ROC_AUC"],
        }
    )
    classification_reports[target] = classification_report(
        y_test_all[target],
        (test_prob >= threshold).astype(int),
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )
    confusion_matrices[target] = confusion_matrix(y_test_all[target], (test_prob >= threshold).astype(int), labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test_all[target], test_prob)
    roc_points[target] = (fpr, tpr, test_metrics["ROC_AUC"])
    print(f"{target} final test metrics: {metrics_rows[-1]}")

metrics_df = pd.DataFrame(metrics_rows)
train_gap_df = pd.DataFrame(train_gap_rows)

# %% [markdown]
# ## 10. Confusion Matrices and ROC Curves

# %%
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
    output_path = CM_DIR / f"confusion_matrix_{target.lower()}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path.relative_to(ROOT)}")

for target, (fpr, tpr, auc) in roc_points.items():
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{target} ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{target} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = ROC_DIR / f"roc_curve_{target.lower()}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path.relative_to(ROOT)}")

# %% [markdown]
# ## 11. SHAP Explainability

# %%
shap_contribution_rows: dict[str, pd.DataFrame] = {}
for target, model in models.items():
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    if shap_array.shape[1] != len(feature_names):
        raise ValueError(f"SHAP feature count mismatch for {target}.")

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
    shap_output = SHAP_DIR / f"shap_{target.lower()}.png"
    plt.savefig(shap_output, dpi=200, bbox_inches="tight")
    plt.close()

    contributions = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    contribution_output = CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv"
    contributions.to_csv(contribution_output, index=False)
    shap_contribution_rows[target] = contributions
    print(f"Saved {shap_output.relative_to(ROOT)}")
    print(f"Saved {contribution_output.relative_to(ROOT)}")

# %% [markdown]
# ## 12. Save Artifacts and Metadata

# %%
model_paths: dict[str, str] = {}
for target, model in models.items():
    model_path = MODEL_DIR / f"model_{target.lower()}.json"
    model.save_model(str(model_path))
    model_paths[target] = str(model_path.relative_to(ROOT))
    print(f"Saved {model_path.relative_to(ROOT)}")

preprocessor_path = MODEL_DIR / "preprocessor.joblib"
joblib.dump(final_preprocessor, preprocessor_path)
print(f"Saved {preprocessor_path.relative_to(ROOT)}")

feature_names_path = MODEL_DIR / "feature_names.json"
feature_payload = {
    "raw_feature_names": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_feature_names": feature_names,
    "model_feature_count": len(feature_names),
    "excluded_columns": LEAKAGE_COLUMNS,
}
feature_names_path.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")
print(f"Saved {feature_names_path.relative_to(ROOT)}")

metrics_path = METRICS_DIR / "classification_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved {metrics_path.relative_to(ROOT)}")

comparison_path = METRICS_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
stability_path = METRICS_DIR / "cv_seed_stability.csv"
stability_df.to_csv(stability_path, index=False)

target_achievement = {}
for _, row in metrics_df.iterrows():
    target_achievement[row["Target"]] = {
        metric: bool(float(row[metric]) >= goal)
        for metric, goal in TARGET_GOALS.items()
    }

leakage_checks = {
    "excluded_columns": LEAKAGE_COLUMNS,
    "raw_feature_leakage_found": leakage_found,
    "transformed_feature_leakage_found": [
        name for name in feature_names if name.split("__")[-1] in LEAKAGE_COLUMNS
    ],
    "preprocessing_fit_scope": "fit on training folds inside CV/search; final preprocessor fit only on final training split",
    "smote_used": False,
    "threshold_selection_scope": "cross-validated training predictions only",
    "holdout_test_scope": "used once for final evaluation only",
}

metadata_targets: dict[str, Any] = {}
for target in TARGET_COLUMNS:
    selected = selection_by_target[target]
    row = metrics_df.loc[metrics_df["Target"] == target].iloc[0].to_dict()
    metadata_targets[target] = {
        "selected_version": selected["selected_version"],
        "selected_hyperparameters": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in models[target].get_params().items()
            if key
            in {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "gamma",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
                "random_state",
            }
        },
        "selected_threshold": float(selected["selected_threshold"]),
        "random_state": RANDOM_STATE,
        "cv_methodology": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "class_balancing_method": "XGBoost scale_pos_weight",
        "scale_pos_weight": float(selected["scale_pos_weight"]),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "final_test_metrics": {
            key: float(row[key])
            for key in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        },
        "cross_validation_mean": selected["stability_mean"],
        "cross_validation_standard_deviation": selected["stability_std"],
        "primary_cv_metrics_seed_42": selected["selected_cv_metrics"],
        "baseline_cv_metrics_seed_42": selected["baseline_cv_metrics"],
        "optimized_cv_metrics_seed_42": selected["optimized_cv_metrics"],
        "randomized_search_best_score_roc_auc": selected["best_search_score_roc_auc"],
        "randomized_search_best_params": selected["best_params"],
    }

metadata = {
    "model_type": "XGBoost XGBClassifier binary classification",
    "targets": TARGET_COLUMNS,
    "training_datetime_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "stability_random_states": STABILITY_SEEDS,
    "test_size": TEST_SIZE,
    "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
    "dataset_rows": int(len(df)),
    "unique_feature_groups": int(len(unique_indices)),
    "duplicate_rows": int(df.duplicated().sum()),
    "training_samples": int(len(train_idx)),
    "training_feature_groups": int(len(train_group_ids)),
    "test_samples": int(len(test_idx)),
    "test_feature_groups": int(len(test_group_ids)),
    "raw_feature_count": int(len(feature_columns)),
    "model_feature_count": int(len(feature_names)),
    "raw_feature_names": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_feature_names": feature_names,
    "excluded_columns": LEAKAGE_COLUMNS,
    "model_filenames": model_paths,
    "preprocessor_filename": str(preprocessor_path.relative_to(ROOT)),
    "feature_names_filename": str(feature_names_path.relative_to(ROOT)),
    "selected_thresholds": {
        target: float(selection_by_target[target]["selected_threshold"])
        for target in TARGET_COLUMNS
    },
    "cv_methodology": {
        "search": f"RandomizedSearchCV with StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "search_iterations_per_target": N_ITER_SEARCH,
        "search_scoring": "roc_auc",
        "threshold_selection": "best training CV F1, tie-broken by recall, precision, accuracy",
        "stability": f"5-fold StratifiedGroupKFold repeated across seeds {STABILITY_SEEDS}",
    },
    "preprocessing": {
        "type": "sklearn.compose.ColumnTransformer",
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "cv_fit_scope": "inside each cross-validation fold via sklearn Pipeline",
        "final_fit_scope": "final training split only",
    },
    "class_distribution_full_dataset": class_distribution,
    "class_distribution_training": {
        target: {
            "train_counts": {
                str(key): int(value)
                for key, value in y_train_all[target].value_counts().sort_index().items()
            },
            "train_percentages": {
                str(key): float(value)
                for key, value in (y_train_all[target].value_counts().sort_index() / len(y_train_all) * 100)
                .round(2)
                .items()
            },
            "scale_pos_weight": float(selection_by_target[target]["scale_pos_weight"]),
        }
        for target in TARGET_COLUMNS
    },
    "targets_detail": metadata_targets,
    "evaluation_metrics": metrics_df.to_dict(orient="records"),
    "comparison_metrics_cv": comparison_df.to_dict(orient="records"),
    "train_cv_test_gap": train_gap_df.to_dict(orient="records"),
    "target_achievement": target_achievement,
    "leakage_checks": leakage_checks,
    "dataset_inspection": dataset_inspection,
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

metadata_path = MODEL_DIR / "model_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved {metadata_path.relative_to(ROOT)}")

# %% [markdown]
# ## 13. Validation Report and Artifact Smoke Test

# %%
validation_report_path = METRICS_DIR / "validation_reports.txt"
with validation_report_path.open("w", encoding="utf-8") as report_file:
    report_file.write("AortaVision 3D validation reports\n")
    report_file.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
    report_file.write("DATASET INSPECTION\n")
    report_file.write(f"Rows: {df.shape[0]}\n")
    report_file.write(f"Columns: {df.shape[1]}\n")
    report_file.write(f"Missing values: {int(df.isna().sum().sum())}\n")
    report_file.write(f"Duplicate rows: {dataset_inspection['duplicate_rows']}\n")
    report_file.write(f"Numerical features: {numeric_features}\n")
    report_file.write(f"Categorical features: {categorical_features}\n\n")
    report_file.write("CLASS DISTRIBUTION\n")
    report_file.write(json.dumps(class_distribution, indent=2))
    report_file.write("\n\nMODEL COMPARISON - TRAINING CV\n")
    report_file.write(comparison_df.to_string(index=False))
    report_file.write("\n\nFINAL TEST METRICS\n")
    report_file.write(metrics_df.to_string(index=False))
    report_file.write("\n\nTRAIN/CV/TEST GAP CHECK\n")
    report_file.write(train_gap_df.to_string(index=False))
    report_file.write("\n\nSEED STABILITY\n")
    report_file.write(stability_df.to_string(index=False))
    report_file.write("\n\nLEAKAGE CHECK\n")
    report_file.write(json.dumps(leakage_checks, indent=2))
    report_file.write("\n\nCLASSIFICATION REPORTS\n")
    for target in TARGET_COLUMNS:
        report_file.write(f"\n===== {target} =====\n")
        report_file.write(classification_reports[target])
        report_file.write("\n")
print(f"Saved {validation_report_path.relative_to(ROOT)}")

reloaded_preprocessor = joblib.load(preprocessor_path)
reloaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
reload_models: dict[str, XGBClassifier] = {}
for target in TARGET_COLUMNS:
    reloaded_model = XGBClassifier()
    reloaded_model.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
    reload_models[target] = reloaded_model

smoke_raw = X.iloc[[0]][feature_columns].copy()
smoke_processed = reloaded_preprocessor.transform(smoke_raw)
for target, model in reload_models.items():
    probability = float(model.predict_proba(smoke_processed)[0, 1])
    threshold = float(reloaded_metadata["selected_thresholds"][target])
    prediction = int(probability >= threshold)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{target} probability outside [0, 1]: {probability}")
    print(f"{target} probability: {probability:.4f}; threshold: {threshold:.2f}; predicted class: {prediction}")

required_files = [
    *(MODEL_DIR / f"model_{target.lower()}.json" for target in TARGET_COLUMNS),
    preprocessor_path,
    feature_names_path,
    metadata_path,
    metrics_path,
    validation_report_path,
    comparison_path,
    stability_path,
    *(CM_DIR / f"confusion_matrix_{target.lower()}.png" for target in TARGET_COLUMNS),
    *(ROC_DIR / f"roc_curve_{target.lower()}.png" for target in TARGET_COLUMNS),
    *(SHAP_DIR / f"shap_{target.lower()}.png" for target in TARGET_COLUMNS),
    *(CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv" for target in TARGET_COLUMNS),
]
missing_outputs = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file() or path.stat().st_size == 0]
if missing_outputs:
    raise FileNotFoundError(f"Missing or empty required outputs: {missing_outputs}")

# %% [markdown]
# ## 14. Final Report

# %%
print("\n========================================")
print("FINAL MODEL PERFORMANCE")
print("========================================")
print(metrics_df[["Target", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))

print("\n========================================")
print("TARGET ACHIEVEMENT")
print("========================================")
for _, row in metrics_df.iterrows():
    print(f"\n{row['Target']}")
    for metric, goal in TARGET_GOALS.items():
        status = "TARGET ACHIEVED" if float(row[metric]) >= goal else "TARGET NOT ACHIEVED"
        print(f"{metric}: {row[metric]:.4f} - {status}")

print("\n========================================")
print("MODEL QUALITY")
print("========================================")
for target in TARGET_COLUMNS:
    selected = selection_by_target[target]
    gap = train_gap_df.loc[train_gap_df["Target"] == target].iloc[0].to_dict()
    overfit_flag = gap["AUC_train_cv_gap"] > 0.12 or gap["F1_train_cv_gap"] > 0.12
    perfect_flag = all(
        abs(float(metrics_df.loc[metrics_df["Target"] == target, metric].iloc[0]) - 1.0) < 1e-12
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    )
    print(f"\n{target}")
    print(f"CV mean: {selected['stability_mean']}")
    print(f"CV standard deviation: {selected['stability_std']}")
    print(f"Final test performance: {metrics_df.loc[metrics_df['Target'] == target].iloc[0].to_dict()}")
    print(f"Overfitting check: {'possible overfitting' if overfit_flag else 'no large train/CV gap detected'}")
    print(f"Leakage check: passed; no CAD/LAD/LCX/RCA/Cath in features and preprocessing was fold-contained")
    print(f"Threshold used: {selected['selected_threshold']:.2f}")
    print(f"Class balancing method: XGBoost scale_pos_weight={selected['scale_pos_weight']:.4f}")
    print(f"Perfect-score investigation: {'required' if perfect_flag else 'not triggered'}")

print("\n========================================")
print("FINAL DECISION")
print("========================================")
for target in TARGET_COLUMNS:
    selected = selection_by_target[target]
    print(
        f"{target}: {selected['selected_version']} XGBoost, "
        f"threshold={selected['selected_threshold']:.2f}, "
        f"scale_pos_weight={selected['scale_pos_weight']:.4f}"
    )

print("\nTRAINING COMPLETED")
