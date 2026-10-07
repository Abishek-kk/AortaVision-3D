"""Benchmark candidate algorithms for AortaVision 3D without overwriting models."""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
CURRENT_XGB_RESULTS = {
    "CAD": {"Accuracy": 0.8361, "Precision": 0.8235, "Recall": 0.9767, "F1": 0.8936, "ROC_AUC": 0.9199},
    "LAD": {"Accuracy": 0.7541, "Precision": 0.7209, "Recall": 0.9118, "F1": 0.8052, "ROC_AUC": 0.8312},
    "LCX": {"Accuracy": 0.6393, "Precision": 0.6129, "Recall": 0.6552, "F1": 0.6333, "ROC_AUC": 0.6853},
    "RCA": {"Accuracy": 0.5738, "Precision": 0.4865, "Recall": 0.7200, "F1": 0.5806, "ROC_AUC": 0.6689},
}


def find_project_root() -> Path:
    candidates = []
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
METRICS_DIR = ROOT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def metric_dict(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
    }


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    if scale_numeric:
        categorical_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_pipeline(model: Any, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def candidate_estimators(y_train: pd.Series) -> list[dict[str, Any]]:
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    scale_pos_weight = float(negatives / positives) if positives else 1.0
    sqrt_scale_pos_weight = float(np.sqrt(scale_pos_weight))

    candidates: list[dict[str, Any]] = [
        {
            "Algorithm": "Logistic Regression",
            "Variant": "balanced",
            "Estimator": make_pipeline(
                LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear", random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
        },
        {
            "Algorithm": "Random Forest",
            "Variant": "balanced_subsample",
            "Estimator": make_pipeline(
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale_numeric=False,
            ),
        },
        {
            "Algorithm": "Extra Trees",
            "Variant": "balanced",
            "Estimator": make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale_numeric=False,
            ),
        },
        {
            "Algorithm": "XGBoost",
            "Variant": "scale_pos_weight",
            "Estimator": make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    gamma=0.2,
                    reg_alpha=0.2,
                    reg_lambda=3.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale_numeric=False,
            ),
        },
        {
            "Algorithm": "HistGradientBoosting",
            "Variant": "default",
            "Estimator": make_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=120,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=False,
            ),
        },
        {
            "Algorithm": "SVM",
            "Variant": "balanced_rbf",
            "Estimator": make_pipeline(
                SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
        },
    ]

    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        candidates.append(
            {
                "Algorithm": "HistGradientBoosting",
                "Variant": "balanced",
                "Estimator": make_pipeline(
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    scale_numeric=False,
                ),
            }
        )

    # A small tuned XGBoost candidate keeps the incumbent family represented
    # without turning this benchmark into an XGBoost-only search.
    candidates.append(
        {
            "Algorithm": "XGBoost",
            "Variant": "small_random_search",
            "Estimator": RandomizedSearchCV(
                estimator=make_pipeline(
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        tree_method="hist",
                        scale_pos_weight=sqrt_scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    scale_numeric=False,
                ),
                param_distributions={
                    "model__n_estimators": randint(80, 241),
                    "model__max_depth": randint(2, 5),
                    "model__learning_rate": uniform(0.025, 0.10),
                    "model__min_child_weight": randint(1, 7),
                    "model__subsample": uniform(0.70, 0.25),
                    "model__colsample_bytree": uniform(0.70, 0.25),
                    "model__gamma": uniform(0.0, 1.0),
                    "model__reg_alpha": uniform(0.0, 1.0),
                    "model__reg_lambda": uniform(1.0, 5.0),
                },
                n_iter=18,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
                return_train_score=True,
            ),
        }
    )
    return candidates


df = pd.read_csv(DATASET_PATH)
df_model = df.copy()
df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == "CAD").astype(int)
for target in ["LAD", "LCX", "RCA"]:
    df_model[target] = (df_model[target].astype(str).str.strip() == "Stenotic").astype(int)

feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
if any(column in feature_columns for column in LEAKAGE_COLUMNS):
    raise ValueError("Leakage columns found in feature set.")

X = df_model[feature_columns].copy()
y_all = df_model[TARGETS].astype(int)
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]

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

comparison_rows: list[dict[str, Any]] = []
overfit_rows: list[dict[str, Any]] = []
selected_by_target: dict[str, dict[str, Any]] = {}
final_rows: list[dict[str, Any]] = []

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc",
}

for target in TARGETS:
    print(f"\n===== Benchmarking {target} =====", flush=True)
    y_train = y_train_all[target]
    target_candidates = candidate_estimators(y_train)
    target_records: list[dict[str, Any]] = []

    for candidate in target_candidates:
        label = f"{candidate['Algorithm']} ({candidate['Variant']})"
        print(f"  {label}", flush=True)
        estimator = candidate["Estimator"]
        cv_result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
        )
        row = {
            "Target": target,
            "Algorithm": candidate["Algorithm"],
            "Variant": candidate["Variant"],
        }
        for metric in METRICS:
            test_values = cv_result[f"test_{metric}"]
            train_values = cv_result[f"train_{metric}"]
            row[f"{metric} Mean"] = float(np.mean(test_values))
            row[f"{metric} Std"] = float(np.std(test_values, ddof=1))
            row[f"Train {metric} Mean"] = float(np.mean(train_values))
        row["Train_Validation_ROC_AUC_Gap"] = row["Train ROC_AUC Mean"] - row["ROC_AUC Mean"]
        row["Overfit_Flag"] = bool(row["Train ROC_AUC Mean"] >= 0.98 and row["Train_Validation_ROC_AUC_Gap"] >= 0.12)
        row["Selection_Rank_Key"] = (
            row["ROC_AUC Mean"],
            row["F1 Mean"],
            row["Recall Mean"],
            row["Precision Mean"],
            row["Accuracy Mean"],
        )
        comparison_rows.append({k: v for k, v in row.items() if k != "Selection_Rank_Key"})
        target_records.append({"row": row, "estimator": estimator})
        overfit_rows.append(
            {
                "Target": target,
                "Algorithm": candidate["Algorithm"],
                "Variant": candidate["Variant"],
                "Train ROC-AUC Mean": row["Train ROC_AUC Mean"],
                "Validation ROC-AUC Mean": row["ROC_AUC Mean"],
                "Train-Validation ROC-AUC Gap": row["Train_Validation_ROC_AUC_Gap"],
                "Overfit Flag": row["Overfit_Flag"],
            }
        )

    not_flagged = [record for record in target_records if not record["row"]["Overfit_Flag"]]
    selection_pool = not_flagged if not_flagged else target_records
    best = max(selection_pool, key=lambda record: record["row"]["Selection_Rank_Key"])
    selected_by_target[target] = best

    final_estimator = clone(best["estimator"])
    final_estimator.fit(X_train, y_train)
    probability = final_estimator.predict_proba(X_test)[:, 1]
    final_metrics = metric_dict(y_test_all[target], probability)
    final_rows.append(
        {
            "Target": target,
            "Selected Algorithm": best["row"]["Algorithm"],
            "Selected Variant": best["row"]["Variant"],
            **final_metrics,
            **{f"Current_XGB_{metric}": CURRENT_XGB_RESULTS[target][metric] for metric in METRICS},
            **{f"Delta_vs_Current_XGB_{metric}": final_metrics[metric] - CURRENT_XGB_RESULTS[target][metric] for metric in METRICS},
        }
    )
    print(
        f"  BEST: {best['row']['Algorithm']} ({best['row']['Variant']}) | "
        f"CV ROC-AUC={best['row']['ROC_AUC Mean']:.4f}",
        flush=True,
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values(["Target", "ROC_AUC Mean", "F1 Mean", "Recall Mean"], ascending=[True, False, False, False])
overfit_df = pd.DataFrame(overfit_rows)
final_df = pd.DataFrame(final_rows)

comparison_path = METRICS_DIR / "algorithm_benchmark_model_comparison.csv"
overfit_path = METRICS_DIR / "algorithm_benchmark_overfitting.csv"
final_path = METRICS_DIR / "algorithm_benchmark_final_test.csv"
report_path = METRICS_DIR / "algorithm_benchmark_report.txt"
json_path = METRICS_DIR / "algorithm_benchmark_summary.json"

comparison_df.to_csv(comparison_path, index=False)
overfit_df.to_csv(overfit_path, index=False)
final_df.to_csv(final_path, index=False)

summary = {
    "leakage_excluded_columns": LEAKAGE_COLUMNS,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "cv_methodology": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
    "test_split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratify": "joint targets if all strata >= 2 else None"},
    "best_by_target": {
        target: {
            "algorithm": selected_by_target[target]["row"]["Algorithm"],
            "variant": selected_by_target[target]["row"]["Variant"],
            "cv_metrics": {
                metric: selected_by_target[target]["row"][f"{metric} Mean"]
                for metric in METRICS
            },
            "cv_roc_auc_std": selected_by_target[target]["row"]["ROC_AUC Std"],
            "overfit_flag": selected_by_target[target]["row"]["Overfit_Flag"],
        }
        for target in TARGETS
    },
    "final_test_metrics": final_df.to_dict(orient="records"),
}
json_path.write_text(json.dumps(make_jsonable(summary), indent=2), encoding="utf-8")

with report_path.open("w", encoding="utf-8") as report:
    report.write("AortaVision 3D algorithm benchmark\n")
    report.write("No serialized production models were overwritten by this benchmark.\n\n")
    report.write("Leakage prevention: CAD, LAD, LCX, RCA, and Cath were excluded from inputs.\n")
    report.write(f"CV: {summary['cv_methodology']}\n")
    report.write("Selection priority: ROC-AUC, F1, Recall, Precision, Accuracy.\n")
    report.write("SMOTE: not tested; class-weighting/scale_pos_weight variants were benchmarked instead.\n\n")
    report.write("MODEL COMPARISON\n")
    visible_columns = [
        "Target",
        "Algorithm",
        "Variant",
        "Accuracy Mean",
        "Precision Mean",
        "Recall Mean",
        "F1 Mean",
        "ROC_AUC Mean",
        "ROC_AUC Std",
    ]
    report.write(comparison_df[visible_columns].to_string(index=False))
    report.write("\n\nBEST MODELS\n")
    for target in TARGETS:
        best = selected_by_target[target]["row"]
        report.write(
            f"{target}: {best['Algorithm']} ({best['Variant']}), "
            f"CV ROC-AUC={best['ROC_AUC Mean']:.6f}, "
            f"F1={best['F1 Mean']:.6f}, Recall={best['Recall Mean']:.6f}\n"
        )
    report.write("\n\nOVERFITTING CHECK\n")
    report.write(overfit_df.to_string(index=False))
    report.write("\n\nFINAL UNTOUCHED TEST METRICS\n")
    report.write(final_df.to_string(index=False))

print("\nMODEL COMPARISON")
print(comparison_df[["Target", "Algorithm", "Variant", "Accuracy Mean", "Precision Mean", "Recall Mean", "F1 Mean", "ROC_AUC Mean", "ROC_AUC Std"]].to_string(index=False))
print("\nBEST MODELS")
for target in TARGETS:
    best = selected_by_target[target]["row"]
    print(f"{target}: {best['Algorithm']} ({best['Variant']})")
print("\nFINAL TEST")
print(final_df.to_string(index=False))
print(f"\nSaved benchmark files under {METRICS_DIR}")
