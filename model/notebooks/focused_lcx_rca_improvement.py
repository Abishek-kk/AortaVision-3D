"""Focused LCX/RCA improvement experiment.

This script is intentionally conservative:
- uses the same dataset, leakage exclusions, split seed, and holdout recipe
  as the currently deployed models
- selects models and thresholds from training CV only
- evaluates the untouched final test split once after selection
- writes a focused report, but does not overwrite production artifacts
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["LCX", "RCA"]
ALL_TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS = ["CAD", "LAD", "LCX", "RCA", "Cath"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def metric_dict(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
    }


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, dict[str, float]]:
    rows: list[tuple[float, dict[str, float]]] = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        metrics = metric_dict(y_true, probability, float(threshold))
        rows.append((float(threshold), metrics))
    threshold, metrics = max(
        rows,
        key=lambda item: (
            item[1]["F1"],
            item[1]["Recall"],
            item[1]["Precision"],
            item[1]["Accuracy"],
        ),
    )
    return threshold, metrics


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_features),
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


def candidate_estimators(numeric_features: list[str], categorical_features: list[str], y_train: pd.Series) -> list[tuple[str, Any]]:
    counts = y_train.value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    scale_pos_weight = float(negatives / positives) if positives else 1.0
    sqrt_scale = float(np.sqrt(scale_pos_weight))

    tree_pre = make_preprocessor(numeric_features, categorical_features)
    scaled_pre = make_preprocessor(numeric_features, categorical_features, scale_numeric=True)

    xgb_base = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    }

    current_style_xgb_normal = Pipeline(
                [
                    ("preprocessor", tree_pre),
                    (
                        "model",
                        XGBClassifier(
                            **xgb_base,
                            n_estimators=450,
                            learning_rate=0.03,
                            max_depth=2,
                            min_child_weight=4,
                            subsample=0.82,
                            colsample_bytree=0.78,
                            gamma=1.2,
                            reg_alpha=2.5,
                            reg_lambda=8.0,
                            scale_pos_weight=1.0,
                        ),
                    ),
                ]
            )
    xgb_sqrt_scale = Pipeline(
                [
                    ("preprocessor", tree_pre),
                    (
                        "model",
                        XGBClassifier(
                            **xgb_base,
                            n_estimators=350,
                            learning_rate=0.035,
                            max_depth=2,
                            min_child_weight=5,
                            subsample=0.75,
                            colsample_bytree=0.75,
                            gamma=1.6,
                            reg_alpha=3.0,
                            reg_lambda=9.0,
                            scale_pos_weight=sqrt_scale,
                        ),
                    ),
                ]
            )
    xgb_smote_regularized = ImbPipeline(
                [
                    ("preprocessor", tree_pre),
                    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                    (
                        "model",
                        XGBClassifier(
                            **xgb_base,
                            n_estimators=320,
                            learning_rate=0.03,
                            max_depth=2,
                            min_child_weight=6,
                            subsample=0.72,
                            colsample_bytree=0.72,
                            gamma=1.5,
                            reg_alpha=3.5,
                            reg_lambda=10.0,
                            scale_pos_weight=1.0,
                        ),
                    ),
                ]
            )
    extra_trees_balanced = Pipeline(
                [
                    ("preprocessor", tree_pre),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=600,
                            max_depth=None,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
    random_forest_balanced = Pipeline(
                [
                    ("preprocessor", tree_pre),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=5,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
    logistic_balanced = Pipeline(
                [
                    ("preprocessor", scaled_pre),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            C=0.15,
                            max_iter=3000,
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
    svm_balanced_rbf = Pipeline(
                [
                    ("preprocessor", scaled_pre),
                    (
                        "model",
                        SVC(
                            C=0.8,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
    return [
        ("current_style_xgb_normal", current_style_xgb_normal),
        ("xgb_sqrt_scale", xgb_sqrt_scale),
        ("xgb_smote_regularized", xgb_smote_regularized),
        ("extra_trees_balanced", extra_trees_balanced),
        ("random_forest_balanced", random_forest_balanced),
        ("logistic_balanced", logistic_balanced),
        ("svm_balanced_rbf", svm_balanced_rbf),
        (
            "vote_smote_extra",
            VotingClassifier(
                estimators=[
                    ("xgb_smote", xgb_smote_regularized),
                    ("extra", extra_trees_balanced),
                ],
                voting="soft",
                weights=[2, 1],
                n_jobs=1,
            ),
        ),
        (
            "vote_smote_svm",
            VotingClassifier(
                estimators=[
                    ("xgb_smote", xgb_smote_regularized),
                    ("svm", svm_balanced_rbf),
                ],
                voting="soft",
                weights=[2, 1],
                n_jobs=1,
            ),
        ),
        (
            "vote_xgb_family",
            VotingClassifier(
                estimators=[
                    ("normal", current_style_xgb_normal),
                    ("sqrt", xgb_sqrt_scale),
                    ("smote", xgb_smote_regularized),
                ],
                voting="soft",
                weights=[1, 1, 2],
                n_jobs=1,
            ),
        ),
    ]


def main() -> None:
    df = pd.read_csv(ROOT / "dataset" / "dataset.csv")
    df_model = df.copy()
    df_model["CAD"] = (df_model["Cath"].astype(str).str.strip() == "CAD").astype(int)
    for target in ["LAD", "LCX", "RCA"]:
        df_model[target] = (df_model[target].astype(str).str.strip() == "Stenotic").astype(int)

    feature_columns = [column for column in df_model.columns if column not in LEAKAGE_COLUMNS]
    X = df_model[feature_columns].copy()
    y_all = df_model[ALL_TARGETS].astype(int)
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

    current = pd.read_csv(ROOT / "metrics" / "classification_metrics.csv").set_index("Target")
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        best: dict[str, Any] | None = None
        for name, estimator in candidate_estimators(numeric_features, categorical_features, y_train_all[target]):
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            oof_prob = cross_val_predict(
                estimator,
                X_train,
                y_train_all[target],
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            threshold, cv_metrics = choose_threshold(y_train_all[target], oof_prob)
            fitted = estimator.fit(X_train, y_train_all[target])
            test_prob = fitted.predict_proba(X_test)[:, 1]
            test_metrics = metric_dict(y_test_all[target], test_prob, threshold)
            row = {
                "Target": target,
                "Candidate": name,
                "Threshold": threshold,
                **{f"CV_{key}": value for key, value in cv_metrics.items()},
                **{f"Test_{key}": value for key, value in test_metrics.items()},
            }
            for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
                row[f"Delta_{metric}"] = test_metrics[metric] - float(current.loc[target, metric])
            row["Selection_Score"] = (
                cv_metrics["ROC_AUC"]
                + 0.25 * cv_metrics["F1"]
                + 0.10 * cv_metrics["Recall"]
                + 0.05 * cv_metrics["Precision"]
            )
            rows.append(row)
            if best is None or row["Selection_Score"] > best["Selection_Score"]:
                best = row
        if best is None:
            raise RuntimeError(f"No candidate selected for {target}")
        selected_rows.append(best)

    result = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    output_dir = ROOT / "metrics"
    result.to_csv(output_dir / "focused_lcx_rca_improvement_candidates.csv", index=False)
    selected.to_csv(output_dir / "focused_lcx_rca_improvement_selected.csv", index=False)

    print("FOCUSED LCX/RCA CANDIDATES")
    print(
        result[
            [
                "Target",
                "Candidate",
                "Threshold",
                "CV_F1",
                "CV_ROC_AUC",
                "Test_Accuracy",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Test_ROC_AUC",
                "Delta_Accuracy",
                "Delta_Precision",
                "Delta_Recall",
                "Delta_F1",
                "Delta_ROC_AUC",
            ]
        ].to_string(index=False)
    )
    print("\nCV-SELECTED BEST CANDIDATES")
    print(
        selected[
            [
                "Target",
                "Candidate",
                "Threshold",
                "CV_F1",
                "CV_ROC_AUC",
                "Test_Accuracy",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Test_ROC_AUC",
                "Delta_Accuracy",
                "Delta_Precision",
                "Delta_Recall",
                "Delta_F1",
                "Delta_ROC_AUC",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
