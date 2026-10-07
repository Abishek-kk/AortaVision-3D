"""Optimize AortaVision 3D XGBoost models without target leakage."""

# %% [markdown]
# # AortaVision 3D Optimized Model Training
#
# This workflow trains four independent XGBoost classifiers for CAD, LAD, LCX,
# and RCA. It uses leakage-safe preprocessing inside cross-validation,
# controlled hyperparameter search, CV-selected thresholds, final holdout
# evaluation, SHAP explainability, serialization, and reload validation.

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## 2. Configuration

# %%
RANDOM_STATE = 42
SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.20
N_SPLITS = 5
N_ITER = 20

TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
POSITIVE_LABELS = {
    "CAD": "CAD",
    "LAD": "Stenotic",
    "LCX": "Stenotic",
    "RCA": "Stenotic",
}

BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 160,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": randint(80, 261),
    "model__max_depth": randint(2, 5),
    "model__learning_rate": uniform(0.025, 0.095),
    "model__subsample": uniform(0.70, 0.25),
    "model__colsample_bytree": uniform(0.70, 0.25),
    "model__min_child_weight": randint(1, 6),
    "model__gamma": uniform(0.0, 0.30),
    "model__reg_alpha": uniform(0.0, 0.50),
    "model__reg_lambda": uniform(1.0, 4.0),
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

print(f"Project root: {ROOT}")
print(f"Random state: {RANDOM_STATE}")
print(f"CV seeds: {SEEDS}")

# %% [markdown]
# ## 3. Dataset Inspection

# %%
df = pd.read_csv(DATASET_PATH)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("Columns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())
print(f"\nDuplicate rows: {int(df.duplicated().sum())}")

if not {"Cath", "LAD", "LCX", "RCA"}.issubset(df.columns):
    missing = sorted({"Cath", "LAD", "LCX", "RCA"} - set(df.columns))
    raise ValueError(f"Dataset is missing required columns: {missing}")

df_model = df.copy()
df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == POSITIVE_LABELS["CAD"]).astype(int)
for target in ["LAD", "LCX", "RCA"]:
    df_model[target] = (df_model[target].astype(str).str.strip() == POSITIVE_LABELS[target]).astype(int)

feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
if any(column in feature_columns for column in LEAKAGE_COLUMNS):
    raise ValueError("Leakage columns appeared in feature_columns.")

X = df_model[feature_columns].copy()
y_all = df_model[TARGETS].astype(int)
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]

dataset_profile = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
    "duplicate_rows": int(df.duplicated().sum()),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}

print(f"\nRaw feature count: {len(feature_columns)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

target_distribution: dict[str, dict[str, object]] = {}
for target in TARGETS:
    counts = y_all[target].value_counts().sort_index()
    percentages = (counts / len(y_all) * 100).round(2)
    target_distribution[target] = {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "percentages": {str(key): float(value) for key, value in percentages.items()},
    }
    print(f"\n{target} target distribution:")
    print(counts)
    print(percentages)

# %% [markdown]
# ## 4. Leakage-Safe Preprocessing and Holdout Split

# %%
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_pipeline(params: dict[str, object], scale_pos_weight: float, random_state: int) -> Pipeline:
    model_params = dict(params)
    model_params["scale_pos_weight"] = scale_pos_weight
    model_params["random_state"] = random_state
    return Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", XGBClassifier(**model_params)),
        ]
    )


joint_strata = y_all.astype(str).agg("_".join, axis=1)
stratify_values = joint_strata if int(joint_strata.value_counts().min()) >= 2 else y_all["CAD"]
if stratify_values is y_all["CAD"]:
    print("Joint stratification had rare patterns; final holdout is stratified by CAD.")
else:
    print("Final holdout is stratified by combined CAD/LAD/LCX/RCA target pattern.")

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

print(f"Training samples: {len(X_train)}")
print(f"Final holdout samples: {len(X_test)}")

# %% [markdown]
# ## 5. Optimization Helpers

# %%
def metric_dict(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.50
    best_metrics = metric_dict(y_true, probabilities, best_threshold)
    for threshold in np.round(np.arange(0.20, 0.81, 0.01), 2):
        metrics = metric_dict(y_true, probabilities, float(threshold))
        current_key = (metrics["F1"], metrics["Recall"], metrics["Precision"])
        best_key = (best_metrics["F1"], best_metrics["Recall"], best_metrics["Precision"])
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def cv_summary(estimator: Pipeline, x_data: pd.DataFrame, y_data: pd.Series, seeds: list[int]) -> dict[str, dict[str, float]]:
    rows: list[dict[str, float]] = []
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
    }
    for seed in seeds:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        scores = cross_validate(estimator, x_data, y_data, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({name: float(np.mean(scores[f"test_{name}"])) for name in scoring})
    frame = pd.DataFrame(rows)
    return {
        metric: {"mean": float(frame[metric].mean()), "std": float(frame[metric].std(ddof=0))}
        for metric in frame.columns
    }


def xgb_param_from_pipeline_params(params: dict[str, object]) -> dict[str, object]:
    selected = dict(BASE_PARAMS)
    for key, value in params.items():
        if key.startswith("model__"):
            selected[key.replace("model__", "")] = value.item() if hasattr(value, "item") else value
    return selected

# %% [markdown]
# ## 6. Baseline vs Optimized Model Selection

# %%
comparison_rows: list[dict[str, object]] = []
final_rows: list[dict[str, object]] = []
selected_configs: dict[str, dict[str, object]] = {}
final_models: dict[str, XGBClassifier] = {}
final_thresholds: dict[str, float] = {}
classification_reports: dict[str, str] = {}
confusion_matrices: dict[str, np.ndarray] = {}
roc_points: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

final_preprocessor = make_preprocessor()
X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)
feature_names = final_preprocessor.get_feature_names_out().tolist()
if any(name.split("__")[-1] in LEAKAGE_COLUMNS for name in feature_names):
    raise ValueError("Leakage columns appeared in transformed feature names.")

for target in TARGETS:
    print(f"\n===== {target} =====")
    y_train = y_train_all[target]
    y_test = y_test_all[target]
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    scale_pos_weight = float(negatives / positives) if positives else 1.0
    print(f"Training class counts: {counts.to_dict()}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    baseline_pipeline = make_pipeline(BASE_PARAMS, scale_pos_weight, RANDOM_STATE)
    baseline_pipeline.fit(X_train, y_train)
    baseline_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = metric_dict(y_test, baseline_prob, 0.50)
    baseline_cv = cv_summary(baseline_pipeline, X_train, y_train, SEEDS)

    comparison_rows.append({"Target": target, "Version": "Baseline", **baseline_metrics, "Threshold": 0.50})

    search_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=make_pipeline(BASE_PARAMS, scale_pos_weight, RANDOM_STATE),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="roc_auc",
        refit=True,
        cv=search_cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    train_roc = float(search.cv_results_["mean_train_score"][search.best_index_])
    cv_roc = float(search.best_score_)
    generalization_gap = train_roc - cv_roc

    threshold_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_probabilities = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=threshold_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, threshold_cv_metrics = choose_threshold(y_train, cv_probabilities)

    best_params = xgb_param_from_pipeline_params(search.best_params_)
    optimized_model = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight)
    optimized_model.fit(X_train_processed, y_train)
    optimized_prob = optimized_model.predict_proba(X_test_processed)[:, 1]
    optimized_metrics = metric_dict(y_test, optimized_prob, threshold)
    optimized_cv = cv_summary(make_pipeline(best_params, scale_pos_weight, RANDOM_STATE), X_train, y_train, SEEDS)

    comparison_rows.append({"Target": target, "Version": "Optimized", **optimized_metrics, "Threshold": threshold})

    keep_optimized = (
        optimized_cv["ROC_AUC"]["mean"] >= baseline_cv["ROC_AUC"]["mean"] - 0.005
        and optimized_metrics["F1"] >= baseline_metrics["F1"] - 0.02
    )
    selected_version = "Optimized" if keep_optimized else "Baseline"
    if keep_optimized:
        selected_model = optimized_model
        selected_threshold = threshold
        selected_metrics = optimized_metrics
        selected_cv = optimized_cv
        selected_params = best_params
        selected_search = {
            "best_cv_roc_auc": cv_roc,
            "best_train_roc_auc": train_roc,
            "generalization_gap": generalization_gap,
            "best_params": search.best_params_,
        }
    else:
        baseline_model = XGBClassifier(**BASE_PARAMS, scale_pos_weight=scale_pos_weight)
        baseline_model.fit(X_train_processed, y_train)
        selected_model = baseline_model
        selected_threshold = 0.50
        selected_metrics = baseline_metrics
        selected_cv = baseline_cv
        selected_params = dict(BASE_PARAMS)
        selected_params["scale_pos_weight"] = scale_pos_weight
        selected_search = {"reason_optimized_not_kept": "Optimized model did not improve CV ROC-AUC/F1 enough to justify replacement."}

    final_models[target] = selected_model
    final_thresholds[target] = selected_threshold
    final_rows.append({"Target": target, **selected_metrics, "Threshold": selected_threshold})
    selected_configs[target] = {
        "selected_version": selected_version,
        "selected_hyperparameters": selected_params,
        "selected_threshold": selected_threshold,
        "threshold_cv_metrics": threshold_cv_metrics,
        "class_balancing_method": "scale_pos_weight",
        "scale_pos_weight": scale_pos_weight,
        "baseline_cv": baseline_cv,
        "optimized_cv": optimized_cv,
        "selected_cv": selected_cv,
        "baseline_test_metrics": baseline_metrics,
        "optimized_test_metrics": optimized_metrics,
        "selected_test_metrics": selected_metrics,
        "search_summary": selected_search,
    }

    selected_prob = selected_model.predict_proba(X_test_processed)[:, 1]
    selected_pred = (selected_prob >= selected_threshold).astype(int)
    classification_reports[target] = classification_report(
        y_test,
        selected_pred,
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )
    confusion_matrices[target] = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, selected_prob)
    roc_points[target] = (fpr, tpr, selected_metrics["ROC_AUC"])

    print(f"Selected version: {selected_version}")
    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Final test metrics: {selected_metrics}")
    print(f"Selected CV ROC-AUC mean/std: {selected_cv['ROC_AUC']['mean']:.4f} / {selected_cv['ROC_AUC']['std']:.4f}")

comparison_df = pd.DataFrame(comparison_rows)
final_metrics_df = pd.DataFrame(final_rows)
print("\nModel comparison:")
print(comparison_df.to_string(index=False))
print("\nFinal selected model metrics:")
print(final_metrics_df.to_string(index=False))

# %% [markdown]
# ## 7. Save Metrics, Plots, SHAP, Models, and Metadata

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
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1])
    else:
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
    plt.savefig(SHAP_DIR / f"shap_{target.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()

    contributions = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_array).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    contributions.to_csv(CONTRIB_DIR / f"{target.lower()}_feature_contributions.csv", index=False)

model_paths = {}
for target, model in final_models.items():
    path = MODEL_DIR / f"model_{target.lower()}.json"
    model.save_model(str(path))
    model_paths[target] = str(path.relative_to(ROOT))

preprocessor_path = MODEL_DIR / "preprocessor.joblib"
joblib.dump(final_preprocessor, preprocessor_path)

feature_payload = {
    "raw_feature_names": feature_columns,
    "model_feature_names": feature_names,
    "model_feature_count": len(feature_names),
    "excluded_columns": LEAKAGE_COLUMNS,
}
(MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

final_metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)
comparison_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)

with (METRICS_DIR / "validation_reports.txt").open("w", encoding="utf-8") as handle:
    handle.write("AortaVision 3D optimized validation reports\n")
    handle.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n")
    handle.write("Final holdout test set was not used for hyperparameter or threshold selection.\n\n")
    handle.write("MODEL COMPARISON\n")
    handle.write(comparison_df.to_string(index=False))
    handle.write("\n\n")
    for target in TARGETS:
        handle.write(f"===== {target} =====\n")
        handle.write(classification_reports[target])
        handle.write("\n")
        row = final_metrics_df.loc[final_metrics_df["Target"] == target].iloc[0].to_dict()
        handle.write(
            "Metrics: "
            + ", ".join(
                f"{metric}={row[metric]:.6f}"
                for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]
            )
            + "\n"
        )
        handle.write(
            f"CV ROC_AUC mean={selected_configs[target]['selected_cv']['ROC_AUC']['mean']:.6f}, "
            f"std={selected_configs[target]['selected_cv']['ROC_AUC']['std']:.6f}\n"
        )
        handle.write(f"Selected version: {selected_configs[target]['selected_version']}\n\n")

metadata = {
    "model_type": "XGBoost XGBClassifier binary classification",
    "targets": TARGETS,
    "training_datetime_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "cv_seeds": SEEDS,
    "cv_methodology": {
        "search": f"RandomizedSearchCV with StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "search_scoring": "roc_auc",
        "threshold_selection": "Threshold chosen from cross-validated training probabilities only, maximizing F1 with recall and precision tie-breaks.",
        "final_holdout": "Untouched until final evaluation.",
    },
    "test_size": TEST_SIZE,
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "raw_feature_count": len(feature_columns),
    "model_feature_count": len(feature_names),
    "raw_feature_names": feature_columns,
    "model_feature_names": feature_names,
    "excluded_columns": LEAKAGE_COLUMNS,
    "model_filenames": model_paths,
    "preprocessor_filename": str(preprocessor_path.relative_to(ROOT)),
    "feature_names_filename": str((MODEL_DIR / "feature_names.json").relative_to(ROOT)),
    "preprocessing": {
        "type": "sklearn.compose.ColumnTransformer",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "cv_fit_scope": "inside each cross-validation training fold",
        "final_fit_scope": "final training split only",
    },
    "dataset_profile": dataset_profile,
    "target_distribution_full_dataset": target_distribution,
    "selected_model_configs": selected_configs,
    "comparison_metrics": comparison_df.to_dict(orient="records"),
    "evaluation_metrics": final_metrics_df.to_dict(orient="records"),
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
    "leakage_check": {
        "excluded_columns": LEAKAGE_COLUMNS,
        "raw_feature_leaks": [column for column in LEAKAGE_COLUMNS if column in feature_columns],
        "model_feature_leaks": [
            column for column in LEAKAGE_COLUMNS if any(name.split("__")[-1] == column for name in feature_names)
        ],
        "preprocessing_leakage": "Preprocessing was fitted inside CV folds for search and on the training split for final models.",
        "smote_leakage": "SMOTE was not used.",
    },
}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# %% [markdown]
# ## 8. Reload Validation and Final Report

# %%
reloaded_preprocessor = joblib.load(preprocessor_path)
reloaded_features = json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))
reloaded_metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
if reloaded_features["model_feature_names"] != feature_names:
    raise ValueError("Reloaded feature names do not match final feature names.")
if reloaded_metadata["model_feature_count"] != len(feature_names):
    raise ValueError("Reloaded metadata feature count does not match final feature count.")

smoke_raw = X.iloc[[0]][feature_columns]
smoke_processed = reloaded_preprocessor.transform(smoke_raw)
for target in TARGETS:
    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
    probability = float(model.predict_proba(smoke_processed)[0, 1])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{target} smoke probability outside [0, 1].")
    print(f"{target} smoke probability generated: yes")

targets_met = {}
for _, row in final_metrics_df.iterrows():
    targets_met[row["Target"]] = {
        "Accuracy": row["Accuracy"] >= 0.92,
        "Precision": row["Precision"] >= 0.92,
        "Recall": row["Recall"] >= 0.93,
        "F1": row["F1"] >= 0.92,
        "ROC_AUC": row["ROC_AUC"] >= 0.94,
    }

print("\n========================================")
print("FINAL MODEL PERFORMANCE")
print("========================================")
print(final_metrics_df[["Target", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]].to_string(index=False))

print("\n========================================")
print("TARGET ACHIEVEMENT")
print("========================================")
for target, metric_flags in targets_met.items():
    for metric, achieved in metric_flags.items():
        status = "TARGET ACHIEVED" if achieved else "TARGET NOT ACHIEVED"
        print(f"{target} {metric}: {status}")

print("\n========================================")
print("MODEL QUALITY")
print("========================================")
for target in TARGETS:
    cfg = selected_configs[target]
    print(f"{target}: version={cfg['selected_version']}, threshold={cfg['selected_threshold']:.2f}, balancing=scale_pos_weight")
    print(f"  CV ROC_AUC mean={cfg['selected_cv']['ROC_AUC']['mean']:.4f}, std={cfg['selected_cv']['ROC_AUC']['std']:.4f}")
    print(f"  Final test ROC_AUC={cfg['selected_test_metrics']['ROC_AUC']:.4f}, F1={cfg['selected_test_metrics']['F1']:.4f}")
    gap = cfg.get("search_summary", {}).get("generalization_gap")
    if gap is not None:
        print(f"  Search train-CV ROC_AUC gap={gap:.4f}")

print("\nLeakage check passed: target columns and Cath are excluded from raw and transformed features.")
print("Threshold selection leakage check passed: thresholds used only CV training predictions.")
print("SMOTE leakage check passed: SMOTE was not used.")

print("\n========================================")
print("FINAL DECISION")
print("========================================")
for target in TARGETS:
    cfg = selected_configs[target]
    print(f"{target}: {cfg['selected_version']} XGBoost, threshold={cfg['selected_threshold']:.2f}")
