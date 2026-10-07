"""Run from the repository root with: python model/notebooks/repeated_cv_evaluation.py"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import imblearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "dataset" / "dataset.csv"
METADATA_PATH = ROOT / "model" / "serialized" / "model_metadata.json"
METRICS_DIR = ROOT / "metrics"
N_SPLITS = 5
N_REPEATS = 10

METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "Brier_Score",
]
LEAKAGE_COLUMNS = {"CAD", "LAD", "LCX", "RCA", "Cath"}


def load_metadata() -> dict[str, Any]:
    with METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required_keys = {"targets", "random_state", "raw_feature_names", "numeric_features", "categorical_features", "selected_models"}
    missing = required_keys.difference(metadata)
    if missing:
        raise ValueError(f"Model metadata is missing required keys: {sorted(missing)}")
    return metadata


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if "Sex" in cleaned.columns:
        cleaned["Sex"] = cleaned["Sex"].replace({"Fmale": "Female"})
    if "VHD" in cleaned.columns:
        cleaned["VHD"] = cleaned["VHD"].replace({"mild": "Mild"})
    return cleaned


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
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


def make_estimator(model_config: dict[str, Any], random_state: int) -> XGBClassifier:
    if model_config.get("algorithm") != "XGBoost":
        raise ValueError(f"Unsupported selected algorithm: {model_config.get('algorithm')!r}")
    parameters = {
        name: value
        for name, value in model_config["hyperparameters"].items()
        if value is not None
    }
    parameters["random_state"] = random_state
    return XGBClassifier(**parameters)


def derive_targets(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    if "Cath" not in frame.columns:
        raise ValueError("Dataset is missing the Cath column required to derive CAD.")
    missing = set(targets).difference({"CAD"}).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing target columns: {sorted(missing)}")
    labels = pd.DataFrame(index=frame.index)
    for target in targets:
        if target == "CAD":
            labels[target] = frame["Cath"].astype(str).str.strip().eq("CAD").astype(int)
        else:
            labels[target] = frame[target].astype(str).str.strip().eq("Stenotic").astype(int)
    return labels


def calculate_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(average_precision_score(y_true, probability)),
        "Brier_Score": float(brier_score_loss(y_true, probability)),
    }


def runtime_versions() -> dict[str, str]:
    return {
        "Python_Version": platform.python_version(),
        "Pandas_Version": pd.__version__,
        "NumPy_Version": np.__version__,
        "Scikit_Learn_Version": sklearn.__version__,
        "XGBoost_Version": xgboost.__version__,
        "Imbalanced_Learn_Version": imblearn.__version__,
    }


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = load_metadata()
    targets = list(metadata["targets"])
    random_state = int(metadata["random_state"])
    raw_features = list(metadata["raw_feature_names"])
    forbidden = LEAKAGE_COLUMNS.intersection(raw_features)
    if forbidden:
        raise ValueError(f"Leakage columns found in model inputs: {sorted(forbidden)}")
    missing_features = set(raw_features).difference(pd.read_csv(DATASET_PATH, nrows=0).columns)
    if missing_features:
        raise ValueError(f"Dataset is missing model input columns: {sorted(missing_features)}")

    frame = clean_dataset(pd.read_csv(DATASET_PATH))
    features = frame.loc[:, raw_features].copy()
    labels = derive_targets(frame, targets)

    numeric_features = list(metadata["numeric_features"])
    categorical_features = list(metadata["categorical_features"])
    if set(numeric_features).union(categorical_features) != set(raw_features):
        raise ValueError("Metadata numeric/categorical features do not match raw_feature_names.")
    versions = runtime_versions()
    fold_rows: list[dict[str, Any]] = []

    for target in targets:
        config = metadata["selected_models"][target]
        threshold = float(config["threshold"])
        splitter = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=random_state,
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(features, labels[target])
        ):
            repeat = fold_index // N_SPLITS + 1
            fold = fold_index % N_SPLITS + 1
            estimator = make_estimator(config, random_state)
            steps: list[tuple[str, Any]] = [
                ("preprocessor", make_preprocessor(numeric_features, categorical_features))
            ]
            if config.get("strategy") == "smote_regularized":
                steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
                pipeline: Any = ImbalancedPipeline
            else:
                pipeline = Pipeline
            steps.append(("model", estimator))
            model = pipeline(steps)

            X_train = features.iloc[train_indices]
            X_test = features.iloc[test_indices]
            y_train = labels[target].iloc[train_indices]
            y_test = labels[target].iloc[test_indices]
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_test)[:, 1]
            fold_rows.append(
                {
                    "Target": target,
                    "Repeat": repeat,
                    "Fold": fold,
                    "Random_State": random_state,
                    "Algorithm": config["algorithm"],
                    "Strategy": config["strategy"],
                    "Threshold": threshold,
                    "Train_Samples": len(train_indices),
                    "Test_Samples": len(test_indices),
                    **calculate_metrics(y_test, probability, threshold),
                    **versions,
                }
            )

    folds = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        target_folds = folds.loc[folds["Target"] == target]
        for metric in METRIC_NAMES:
            repeat_means = target_folds.groupby("Repeat", sort=True)[metric].mean()
            summary_rows.append(
                {
                    "Target": target,
                    "Metric": metric,
                    "Mean": float(repeat_means.mean()),
                    "SD": float(repeat_means.std(ddof=1)),
                    "CI_95_Lower": float(repeat_means.quantile(0.025)),
                    "CI_95_Upper": float(repeat_means.quantile(0.975)),
                    "CI_Method": "Percentile over 10 repeat-level means",
                    "N_Repeats": N_REPEATS,
                    "N_Folds_Per_Repeat": N_SPLITS,
                    "Random_State": random_state,
                    "Algorithm": metadata["selected_models"][target]["algorithm"],
                    "Strategy": metadata["selected_models"][target]["strategy"],
                    "Threshold": float(metadata["selected_models"][target]["threshold"]),
                    **versions,
                }
            )
    return pd.DataFrame(summary_rows), folds, {
        "targets": targets,
        "random_state": random_state,
        "versions": versions,
    }


def write_report(summary: pd.DataFrame, run_info: dict[str, Any]) -> None:
    table = summary.pivot(index="Target", columns="Metric", values=["Mean", "SD", "CI_95_Lower", "CI_95_Upper"])
    lines = [
        "# Repeated cross-validation evaluation",
        "",
        "RepeatedStratifiedKFold on the full dataset (5 folds × 10 repeats); the preprocessor is fitted inside each fold.",
        "The fixed production thresholds and selected per-target configurations are read from `model/serialized/model_metadata.json`; no tuning or holdout is used.",
        "Reported 95% confidence intervals are percentile intervals over the 10 repeat-level means (each repeat is the mean of its five fold metrics).",
        "",
        f"Random state: {run_info['random_state']}",
        "",
        "Runtime versions: " + ", ".join(f"{key.removesuffix('_Version')} {value}" for key, value in run_info["versions"].items()),
        "",
        "| Target | " + " | ".join(METRIC_NAMES) + " |",
        "|---|" + "|".join("---" for _ in METRIC_NAMES) + "|",
    ]
    for target in run_info["targets"]:
        cells = []
        for metric in METRIC_NAMES:
            mean = table.loc[target, ("Mean", metric)]
            sd = table.loc[target, ("SD", metric)]
            lower = table.loc[target, ("CI_95_Lower", metric)]
            upper = table.loc[target, ("CI_95_Upper", metric)]
            cells.append(f"{mean:.3f} ± {sd:.3f} [{lower:.3f}, {upper:.3f}]")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    (METRICS_DIR / "repeated_cv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary, folds, run_info = run_evaluation()
    summary.to_csv(METRICS_DIR / "repeated_cv_summary.csv", index=False)
    folds.to_csv(METRICS_DIR / "repeated_cv_folds.csv", index=False)
    write_report(summary, run_info)
    print(f"Repeated CV complete: {len(folds)} fold evaluations ({N_SPLITS} folds × {N_REPEATS} repeats per target).")
    for target in run_info["targets"]:
        row = summary.loc[(summary["Target"] == target) & (summary["Metric"] == "F1")].iloc[0]
        print(
            f"{target}: F1 {row['Mean']:.3f} ± {row['SD']:.3f} "
            f"[{row['CI_95_Lower']:.3f}, {row['CI_95_Upper']:.3f}]"
        )
    print(f"Outputs: {METRICS_DIR / 'repeated_cv_summary.csv'}, {METRICS_DIR / 'repeated_cv_folds.csv'}, "
          f"{METRICS_DIR / 'repeated_cv_report.md'}")


if __name__ == "__main__":
    main()
