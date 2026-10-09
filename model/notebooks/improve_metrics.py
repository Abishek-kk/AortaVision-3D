"""Leakage-safe, nested repeated-CV model comparison for AortaVision 3D.

Run from the repository root:
    python model/notebooks/improve_metrics.py

The full default run is intentionally compute-intensive: 5 outer folds x 10
repeats, nested threshold/model selection, and 50 Optuna trials per outer fit.
Use --trials 0 to skip the Optuna phase while developing or smoke-testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "dataset.csv"
METRICS_DIR = ROOT / "metrics"
SERIALIZED_DIR = ROOT / "model" / "serialized"
CANDIDATE_DIR = ROOT / "model" / "serialized_candidates"
TARGETS = ("CAD", "LAD", "LCX", "RCA")
FORBIDDEN = frozenset((*TARGETS, "Cath"))
METRICS = ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier_Score")
LOG = logging.getLogger("aortavision.improve_metrics")


def _yes(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.casefold()
    return values.isin({"y", "yes", "1", "true", "mild", "moderate", "severe", "lbbb", "rbbb"}).astype(float)


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic features; no statistics are learned outside the training fold."""

    engineered_features = (
        "LDL HDL Ratio",
        "TG HDL Ratio",
        "Risk Factor Count",
        "Age DM Interaction",
        "Age HTN Interaction",
        "Age Smoker Interaction",
        "ECG Abnormality Score",
    )

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ClinicalFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        def number(column: str) -> pd.Series:
            if column not in frame:
                return pd.Series(0.0, index=frame.index)
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)

        def categorical_indicator(column: str) -> pd.Series:
            if column not in frame:
                return pd.Series(0.0, index=frame.index)
            return _yes(frame[column])

        hdl = number("HDL").replace(0.0, np.nan)
        frame["LDL HDL Ratio"] = (number("LDL") / hdl).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["TG HDL Ratio"] = (number("TG") / hdl).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        dlp = (
            pd.to_numeric(frame["DLP"], errors="coerce").fillna(0.0).astype(float)
            if "DLP" in frame and pd.api.types.is_numeric_dtype(frame["DLP"])
            else categorical_indicator("DLP")
        )
        frame["Risk Factor Count"] = (
            number("DM")
            + number("HTN")
            + number("Current Smoker")
            + number("EX-Smoker")
            + number("FH")
            + dlp
            + categorical_indicator("Obesity")
        )
        frame["Age DM Interaction"] = number("Age") * number("DM")
        frame["Age HTN Interaction"] = number("Age") * number("HTN")
        frame["Age Smoker Interaction"] = number("Age") * number("Current Smoker")
        frame["ECG Abnormality Score"] = (
            number("Q Wave")
            + number("St Elevation")
            + number("St Depression")
            + number("Tinversion")
            + categorical_indicator("LVH")
            + categorical_indicator("Poor R Progression")
            + categorical_indicator("BBB")
        )
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(self.engineered_features, dtype=object)
        return np.asarray(
            [*(str(name) for name in input_features), *self.engineered_features],
            dtype=object,
        )


def _positive_target(values: pd.Series, target: str) -> np.ndarray:
    normalized = values.astype("string").str.strip().str.casefold()
    if target == "CAD":
        positive = normalized.eq("cad")
        negative = normalized.eq("normal")
    else:
        positive = normalized.isin({"stenotic", "stenosis", "1", "yes", "true"})
        negative = normalized.isin({"normal", "0", "no", "false"})
    unknown = ~(positive | negative)
    if unknown.any():
        unknown_values = sorted(normalized[unknown].dropna().unique().tolist())
        raise ValueError(f"Unexpected labels for {target}: {unknown_values}")
    return positive.astype(np.int8).to_numpy()


def _feature_contract() -> tuple[list[str], dict[str, Any]]:
    metadata = json.loads((SERIALIZED_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    features = [str(name) for name in metadata.get("raw_feature_names", [])]
    if not features:
        features = json.loads((SERIALIZED_DIR / "feature_names.json").read_text(encoding="utf-8"))[
            "raw_feature_names"
        ]
    leaked = sorted(set(features) & FORBIDDEN)
    if leaked:
        raise ValueError(f"Target leakage in saved feature contract: {leaked}")
    return features, metadata


def _pipeline_preprocessor(
    features: list[str],
    metadata: dict[str, Any],
    engineered: bool,
    extra_numeric: tuple[str, ...] = (),
) -> tuple[Pipeline, ColumnTransformer]:
    numeric = [str(name) for name in metadata.get("numeric_features", []) if name in features]
    categorical = [str(name) for name in metadata.get("categorical_features", []) if name in features]
    if engineered:
        numeric.extend(ClinicalFeatureEngineer.engineered_features)
    numeric.extend(extra_numeric)
    numeric_transform = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())]
    )
    categorical_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("num", numeric_transform, numeric),
            ("cat", categorical_transform, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    steps: list[tuple[str, Any]] = []
    if engineered:
        steps.append(("features", ClinicalFeatureEngineer()))
    steps.append(("preprocess", transformer))
    return Pipeline(steps), transformer


def _optional_estimators(target: str, y: np.ndarray, seed: int) -> dict[str, Any]:
    from xgboost import XGBClassifier

    models: dict[str, Any] = {}
    model_metadata_path = SERIALIZED_DIR / "model_metadata.json"
    saved = json.loads(model_metadata_path.read_text(encoding="utf-8")).get("selected_models", {})
    config = saved.get(target, {}).get("hyperparameters", {})
    allowed_xgb = {
        key: config[key]
        for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "tree_method",
            "eval_metric",
        )
        if config.get(key) is not None
    }
    allowed_xgb.update(
        objective="binary:logistic",
        n_jobs=1,
        random_state=seed,
        verbosity=0,
        scale_pos_weight=1.0,
    )
    models["current_xgb"] = XGBClassifier(**allowed_xgb)
    models["xgb_engineered"] = XGBClassifier(**allowed_xgb)
    for count in (20, 30, 40):
        models[f"xgb_top{count}"] = XGBClassifier(**allowed_xgb)

    models["random_forest"] = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        n_jobs=1, random_state=seed,
    )
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        n_jobs=1, random_state=seed,
    )
    models["logistic_l2"] = LogisticRegression(
        C=1.0, penalty="l2", class_weight="balanced", max_iter=3000,
        solver="liblinear", random_state=seed,
    )
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
            class_weight="balanced", verbosity=-1, n_jobs=1, random_state=seed,
        )
    except ImportError as exc:
        raise ImportError("Install model/requirements.txt dependencies to evaluate LightGBM.") from exc
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.03, loss_function="Logloss",
            auto_class_weights="Balanced", verbose=False, thread_count=1,
            random_seed=seed, allow_writing_files=False,
        )
    except ImportError as exc:
        raise ImportError("Install model/requirements.txt dependencies to evaluate CatBoost.") from exc
    return models


def _model_pipeline(
    method: str,
    target: str,
    y: np.ndarray,
    features: list[str],
    metadata: dict[str, Any],
    seed: int,
    estimator_override: Any = None,
    extra_numeric: tuple[str, ...] = (),
) -> Pipeline:
    engineered = method in {
        "xgb_engineered", "xgb_top20", "xgb_top30", "xgb_top40",
        "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_l2",
    }
    prep, _ = _pipeline_preprocessor(features, metadata, engineered, extra_numeric)
    models = _optional_estimators(target, y, seed)
    estimator = clone(estimator_override) if estimator_override is not None else clone(models[method])
    steps: list[tuple[str, Any]] = list(prep.steps)
    if method in {"xgb_top20", "xgb_top30", "xgb_top40"}:
        top_k = int(method.removeprefix("xgb_top"))
        selector = SelectFromModel(
            ExtraTreesClassifier(
                n_estimators=160, min_samples_leaf=2, class_weight="balanced",
                n_jobs=1, random_state=seed,
            ),
            threshold=-np.inf,
            max_features=top_k,
        )
        steps.append(("select", selector))
    if method == "current_xgb" and target == "CAD":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbalancedPipeline

        steps.append(("resample", SMOTE(random_state=seed, k_neighbors=5)))
        steps.append(("classifier", estimator))
        return ImbalancedPipeline(steps)
    steps.append(("classifier", estimator))
    return Pipeline(steps)


class CADProbabilityStackingClassifier(ClassifierMixin, BaseEstimator):
    """Adds only inner-fold OOF CAD probabilities to vessel features."""

    def __init__(self, cad_estimator: Any, vessel_estimator: Any, random_state: int = 42):
        self.cad_estimator = cad_estimator
        self.vessel_estimator = vessel_estimator
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CADProbabilityStackingClassifier":
        if "__cad_training_label" not in X.columns:
            raise ValueError("Internal CAD training labels are required for the stacking fit.")
        cad_y = X["__cad_training_label"].to_numpy(dtype=np.int8)
        raw = X.drop(columns=["__cad_training_label"])
        if FORBIDDEN.intersection(raw.columns):
            raise ValueError("A target/leakage column reached the CAD stacking model.")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(
            clone(self.cad_estimator), raw, cad_y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        self.cad_estimator_ = clone(self.cad_estimator).fit(raw, cad_y)
        augmented = raw.copy()
        augmented["CAD OOF Probability"] = oof
        self.vessel_estimator_ = clone(self.vessel_estimator).fit(augmented, y)
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = raw.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = X.drop(columns=["__cad_training_label"], errors="ignore")
        if FORBIDDEN.intersection(raw.columns):
            raise ValueError("A target/leakage column reached the CAD stack prediction.")
        cad_probability = self.cad_estimator_.predict_proba(raw)[:, 1]
        augmented = raw.copy()
        augmented["CAD OOF Probability"] = cad_probability
        return self.vessel_estimator_.predict_proba(augmented)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)


class CrossFittedTargetChainClassifier(ClassifierMixin, BaseEstimator):
    """Adds preceding target probabilities trained out-of-fold, never target labels."""

    def __init__(
        self,
        base_estimator: Any,
        previous_targets: tuple[str, ...],
        random_state: int = 42,
        prior_estimator: Any = None,
    ):
        self.base_estimator = base_estimator
        self.previous_targets = previous_targets
        self.random_state = random_state
        self.prior_estimator = prior_estimator

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CrossFittedTargetChainClassifier":
        label_columns = {target: f"__chain_target_{target}" for target in self.previous_targets}
        missing = sorted(set(label_columns.values()) - set(X.columns))
        if missing:
            raise ValueError(f"Internal prior-target labels are missing: {missing}")
        helper_columns = [column for column in X if column.startswith("__chain_target_")]
        raw = X.drop(columns=helper_columns)
        if FORBIDDEN.intersection(raw.columns):
            raise ValueError("A target/leakage column reached the chain model.")
        self.previous_estimators_ = {}
        augmented = raw.copy()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        prior_estimator = self.prior_estimator or self.base_estimator
        for target, column in label_columns.items():
            previous_y = X[column].to_numpy(dtype=np.int8)
            oof_probability = cross_val_predict(
                clone(prior_estimator), raw, previous_y, cv=cv,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            augmented[f"{target} OOF Probability"] = oof_probability
            self.previous_estimators_[target] = clone(prior_estimator).fit(raw, previous_y)
        self.estimator_ = clone(self.base_estimator).fit(augmented, y)
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = raw.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        helper_columns = [column for column in X if column.startswith("__chain_target_")]
        raw = X.drop(columns=helper_columns)
        if FORBIDDEN.intersection(raw.columns):
            raise ValueError("A target/leakage column reached chain prediction.")
        augmented = raw.copy()
        for target, estimator in self.previous_estimators_.items():
            augmented[f"{target} OOF Probability"] = estimator.predict_proba(raw)[:, 1]
        return self.estimator_.predict_proba(augmented)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)


class NestedBestClassifier(ClassifierMixin, BaseEstimator):
    """Selects a model family using training-only inner CV, then refits it."""

    def __init__(
        self, candidates: tuple[tuple[str, Any], ...], random_state: int = 42, top_n: int = 1
    ):
        self.candidates = candidates
        self.random_state = random_state
        self.top_n = top_n

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NestedBestClassifier":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        ranked: list[tuple[float, str, Any]] = []
        for name, estimator in self.candidates:
            scores = cross_val_score(
                clone(estimator), X, y, cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise"
            )
            ranked.append((float(np.mean(scores)), name, estimator))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        self.selected_names_ = [row[1] for row in ranked[: self.top_n]]
        self.selected_scores_ = {row[1]: row[0] for row in ranked}
        selected = [("m" + str(i), clone(row[2])) for i, row in enumerate(ranked[: self.top_n])]
        if self.top_n == 1:
            self.estimator_ = selected[0][1].fit(X, y)
        else:
            self.estimator_ = VotingClassifier(estimators=selected, voting="soft", n_jobs=1).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)


class NestedOptunaXGBClassifier(ClassifierMixin, BaseEstimator):
    """Runs Optuna only on the current outer-training partition."""

    def __init__(
        self, features: list[str], metadata: dict[str, Any], trials: int = 50, random_state: int = 42
    ):
        self.features = features
        self.metadata = metadata
        self.trials = trials
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NestedOptunaXGBClassifier":
        import optuna
        from xgboost import XGBClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 12.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.5),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": self.random_state,
                "verbosity": 0,
            }
            estimator = _model_pipeline(
                "xgb_engineered", "CAD", y, self.features, self.metadata, self.random_state,
                estimator_override=XGBClassifier(**params),
            )
            return float(np.mean(cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=1)))

        study.optimize(objective, n_trials=self.trials, show_progress_bar=False)
        self.best_params_ = dict(study.best_params)
        final_estimator = XGBClassifier(
            **self.best_params_, objective="binary:logistic", eval_metric="auc",
            tree_method="hist", n_jobs=1, random_state=self.random_state, verbosity=0,
        )
        self.estimator_ = _model_pipeline(
            "xgb_engineered", "CAD", y, self.features, self.metadata, self.random_state,
            estimator_override=final_estimator,
        ).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)


@dataclass(frozen=True)
class FoldResult:
    target: str
    method: str
    repeat: int
    fold: int
    threshold: float
    metrics: dict[str, float]
    probabilities: tuple[float, ...]
    labels: tuple[int, ...]


def _threshold_by_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best: tuple[float, float, float, float, float] | None = None
    threshold_best = 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        predicted = (probabilities >= threshold).astype(np.int8)
        score = (
            f1_score(y_true, predicted, zero_division=0),
            recall_score(y_true, predicted, zero_division=0),
            precision_score(y_true, predicted, zero_division=0),
            accuracy_score(y_true, predicted),
            -float(threshold),
        )
        if best is None or score > best:
            best = score
            threshold_best = float(threshold)
    return threshold_best


def _score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(np.int8)
    return {
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
        "PR_AUC": float(average_precision_score(y_true, probabilities)),
        "Brier_Score": float(brier_score_loss(y_true, probabilities)),
    }


def _method_estimator(
    method: str,
    target: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cad_y: np.ndarray,
    features: list[str],
    metadata: dict[str, Any],
    seed: int,
    trials: int,
) -> Any:
    if method in {"xgb_calibrated_sigmoid", "xgb_calibrated_isotonic"}:
        base = _model_pipeline("xgb_engineered", target, y, features, metadata, seed)
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid" if method.endswith("sigmoid") else "isotonic",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
            ensemble=True,
        )
    if method == "nested_optuna_xgb":
        if trials < 1:
            raise ValueError("Optuna candidate requested with --trials 0.")
        return NestedOptunaXGBClassifier(features, metadata, trials, seed)
    if method == "nested_best_model":
        pool_names = ("xgb_engineered", "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_l2")
        pool = tuple(
            (
                name,
                _model_pipeline(name, target, y, features, metadata, seed),
            )
            for name in pool_names
        )
        return NestedBestClassifier(pool, seed, top_n=1)
    if method == "soft_vote_best3":
        pool_names = ("xgb_engineered", "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_l2")
        pool = tuple(
            (name, _model_pipeline(name, target, y, features, metadata, seed))
            for name in pool_names
        )
        return NestedBestClassifier(pool, seed, top_n=3)
    if method == "cad_oof_stack":
        if target == "CAD":
            raise ValueError("CAD stacking is defined only for LAD, LCX, and RCA.")
        cad_estimator = _model_pipeline("current_xgb", "CAD", cad_y, features, metadata, seed)
        vessel_estimator = _model_pipeline(
            "xgb_engineered", target, y, features, metadata, seed,
            extra_numeric=("CAD OOF Probability",),
        )
        return CADProbabilityStackingClassifier(cad_estimator, vessel_estimator, seed)
    if method == "classifier_chain":
        previous_targets = tuple(TARGETS[: TARGETS.index(target)])
        extra_numeric = tuple(f"{name} OOF Probability" for name in previous_targets)
        base_estimator = _model_pipeline(
            "xgb_engineered", target, y, features, metadata, seed,
            extra_numeric=extra_numeric,
        )
        prior_estimator = _model_pipeline(
            "xgb_engineered", target, y, features, metadata, seed,
        )
        return CrossFittedTargetChainClassifier(
            base_estimator, previous_targets, seed, prior_estimator,
        )
    return _model_pipeline(method, target, y, features, metadata, seed)


def _predict_positive(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"Expected binary probabilities, got shape {probabilities.shape}.")
    return probabilities[:, 1]


def _evaluate_candidate(
    X: pd.DataFrame,
    y: np.ndarray,
    labels_by_target: dict[str, np.ndarray],
    target: str,
    method: str,
    features: list[str],
    metadata: dict[str, Any],
    repeats: int,
    inner_folds: int,
    trials: int,
) -> tuple[list[FoldResult], list[dict[str, Any]]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=42
    )
    fold_rows: list[FoldResult] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        cad_train = labels_by_target["CAD"][train_idx]
        X_stack_train = X_train.copy()
        if method == "cad_oof_stack":
            X_stack_train["__cad_training_label"] = cad_train
        if method == "classifier_chain":
            for chain_target, chain_labels in labels_by_target.items():
                X_stack_train[f"__chain_target_{chain_target}"] = chain_labels[train_idx]
        train_for_fit = X_stack_train
        seed = 42 + outer_index
        estimator = _method_estimator(
            method, target, train_for_fit, y_train, cad_train,
            features, metadata, seed, trials,
        )
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        threshold_probabilities = cross_val_predict(
            clone(estimator),
            train_for_fit,
            y_train,
            cv=inner,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = _threshold_by_f1(y_train, threshold_probabilities)
        estimator.fit(train_for_fit, y_train)
        test_probabilities = _predict_positive(estimator, X_test)
        metrics = _score(y_test, test_probabilities, threshold)
        repeat = outer_index // 5 + 1
        fold = outer_index % 5 + 1
        fold_rows.append(
            FoldResult(
                target, method, repeat, fold, threshold, metrics,
                tuple(float(value) for value in test_probabilities),
                tuple(int(value) for value in y_test),
            )
        )
        chosen = getattr(estimator, "selected_names_", [])
        selection_rows.append(
            {
                "Target": target,
                "Method": method,
                "Repeat": repeat,
                "Fold": fold,
                "Threshold": threshold,
                "Selected_Models": "|".join(chosen),
                "Selected_Model_CV_AUCs": json.dumps(
                    getattr(estimator, "selected_scores_", {}), sort_keys=True
                ),
            }
        )
        LOG.info(
            "%s %s repeat %d fold %d/%d: AUC=%.3f F1=%.3f threshold=%.2f",
            target, method, repeat, fold, repeats * 5,
            metrics["ROC_AUC"], metrics["F1"], threshold,
        )
    return fold_rows, selection_rows


def _aggregate(fold_results: list[FoldResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[FoldResult]] = {}
    for result in fold_results:
        grouped.setdefault((result.target, result.method), []).append(result)
    rows: list[dict[str, Any]] = []
    for (target, method), values in grouped.items():
        repeat_means: dict[int, dict[str, float]] = {}
        for repeat in sorted({result.repeat for result in values}):
            repeat_values = [result for result in values if result.repeat == repeat]
            repeat_means[repeat] = {
                metric: float(np.mean([row.metrics[metric] for row in repeat_values]))
                for metric in METRICS
            }
        row: dict[str, Any] = {
            "Target": target,
            "Method": method,
            "N_Repeats": len(repeat_means),
            "N_Folds_Per_Repeat": 5,
            "Random_State": 42,
            "Threshold_Objective": "maximize inner-OOF F1; ties favor recall, precision, accuracy, then lower threshold",
        }
        for metric in METRICS:
            scores = np.asarray([summary[metric] for summary in repeat_means.values()], dtype=float)
            row[f"{metric}_Mean"] = float(np.mean(scores))
            row[f"{metric}_SD"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            row[f"{metric}_CI95_Lower"] = float(np.percentile(scores, 2.5))
            row[f"{metric}_CI95_Upper"] = float(np.percentile(scores, 97.5))
        row["Threshold_Mean"] = float(np.mean([result.threshold for result in values]))
        row["Threshold_SD"] = float(np.std([result.threshold for result in values], ddof=1))
        rows.append(row)
    return rows


def _load_baseline_rows() -> list[dict[str, Any]]:
    summary_path = METRICS_DIR / "repeated_cv_summary.csv"
    baseline = pd.read_csv(summary_path)
    required = {"Target", "Metric", "Mean", "SD", "CI_95_Lower", "CI_95_Upper"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Baseline summary is missing required columns: {sorted(required - set(baseline))}")
    names = {metric: metric.replace("ROC_AUC", "ROC_AUC").replace("PR_AUC", "PR_AUC") for metric in METRICS}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        row: dict[str, Any] = {
            "Target": target,
            "Method": "baseline_current_production",
            "N_Repeats": 10,
            "N_Folds_Per_Repeat": 5,
            "Random_State": 42,
            "Threshold_Objective": "existing production threshold; source is repeated_cv_summary.csv",
        }
        subset = baseline.loc[baseline["Target"] == target].set_index("Metric")
        for metric in METRICS:
            source_metric = "Brier_Score" if metric == "Brier_Score" else metric
            if source_metric not in subset.index:
                raise ValueError(f"Baseline summary lacks {target} {source_metric}.")
            values = subset.loc[source_metric]
            row[f"{metric}_Mean"] = float(values["Mean"])
            row[f"{metric}_SD"] = float(values["SD"])
            row[f"{metric}_CI95_Lower"] = float(values["CI_95_Lower"])
            row[f"{metric}_CI95_Upper"] = float(values["CI_95_Upper"])
        threshold = json.loads((SERIALIZED_DIR / "model_metadata.json").read_text(encoding="utf-8"))
        row["Threshold_Mean"] = float(threshold["selected_thresholds"][target])
        row["Threshold_SD"] = 0.0
        rows.append(row)
    return rows


def _calibration_plots(
    fold_results: list[FoldResult], summary_rows: list[dict[str, Any]]
) -> None:
    plot_dir = METRICS_DIR / "calibration"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        candidates = [row for row in summary_rows if row["Target"] == target]
        calibrated = [
            row for row in candidates
            if row["Method"] in {"xgb_calibrated_sigmoid", "xgb_calibrated_isotonic"}
        ]
        if not calibrated:
            continue
        raw_method = "xgb_engineered"
        raw_rows = [row for row in candidates if row["Method"] == raw_method]
        if not raw_rows:
            raw_rows = [row for row in candidates if row["Method"] == "current_xgb"]
        selected = min(calibrated, key=lambda row: row["Brier_Score_Mean"])
        methods = [row["Method"] for row in raw_rows[:1]] + [selected["Method"]]
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        for method in methods:
            values = [r for r in fold_results if r.target == target and r.method == method]
            probabilities = np.asarray([p for r in values for p in r.probabilities])
            labels = np.asarray([y for r in values for y in r.labels])
            if not len(probabilities):
                continue
            edges = np.linspace(0, 1, 11)
            bin_ids = np.digitize(probabilities, edges[1:-1], right=True)
            means_x, means_y = [], []
            for bin_id in range(10):
                mask = bin_ids == bin_id
                if mask.any():
                    means_x.append(float(probabilities[mask].mean()))
                    means_y.append(float(labels[mask].mean()))
            ax.plot(means_x, means_y, marker="o", label=method)
        ax.set(title=f"{target}: repeated-CV calibration", xlabel="Mean predicted probability", ylabel="Observed positive fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{target.lower()}_calibration.png", dpi=150)
        plt.close(fig)


def _write_report(
    rows: list[dict[str, Any]],
    folds: list[FoldResult],
    data_sha256: str,
    repeats: int,
    trials: int,
) -> None:
    baseline = {row["Target"]: row for row in rows if row["Method"] == "baseline_current_production"}
    new = [row for row in rows if row["Method"] != "baseline_current_production"]
    report = [
        "# Nested-CV model improvement experiment",
        "",
        f"Dataset: `dataset/dataset.csv` ({pd.read_csv(DATA_PATH).shape[0]} rows); SHA-256 `{data_sha256}`.",
        f"Protocol: RepeatedStratifiedKFold, 5 folds × {repeats} repeats, random state 42. ",
        "All transformations, feature selection, model selection, calibration and threshold selection are fit on each outer-training partition only.",
        f"Inner threshold/model selection uses stratified {3}-fold OOF predictions; threshold objective is maximum F1, with ties resolved by recall, precision, accuracy, then lower threshold.",
        "The Optuna candidate uses nested 3-fold ROC-AUC tuning on each outer-training partition; configured trials per fit: "
        f"{trials}. CAD is mapped from Cath; CAD/LAD/LCX/RCA/Cath are never predictor columns.",
        "",
        "## Results and decision rule",
        "",
        "A candidate is considered for promotion only when its repeated-CV ROC-AUC mean exceeds the current baseline and its repeat-level SD is no more than 10% above baseline SD. These internal CV comparisons are exploratory across many candidates; no external validation is available.",
        "",
        "| Target | Baseline AUC (mean ± SD) | Best candidate AUC (mean ± SD) | Best candidate F1 | Promotion decision |",
        "|---|---:|---:|---:|---|",
    ]
    for target in TARGETS:
        base = baseline[target]
        candidate_rows = [row for row in new if row["Target"] == target]
        best = max(candidate_rows, key=lambda row: row["ROC_AUC_Mean"]) if candidate_rows else None
        if best is None:
            decision, auc_label, f1_label = "No candidate result", "—", "—"
        else:
            auc_label = f"{best['ROC_AUC_Mean']:.3f} ± {best['ROC_AUC_SD']:.3f} ({best['Method']})"
            f1_label = f"{best['F1_Mean']:.3f} ± {best['F1_SD']:.3f}"
            nested_rows = [
                row for row in candidate_rows if row["Method"] == "nested_best_model"
            ]
            nested = nested_rows[0] if nested_rows else None
            improved = nested is not None and nested["ROC_AUC_Mean"] > base["ROC_AUC_Mean"]
            stable = nested is not None and nested["ROC_AUC_SD"] <= base["ROC_AUC_SD"] * 1.10
            decision = (
                "Nested model selection clears CV gate; inspect selected algorithm"
                if improved and stable
                else "Keep current model"
            )
        report.append(
            f"| {target} | {base['ROC_AUC_Mean']:.3f} ± {base['ROC_AUC_SD']:.3f} | {auc_label} | {f1_label} | {decision} |"
        )
    report.extend(
        [
            "",
            "No holdout result was used for tuning or selection. A single 20% holdout result in the repository is not used as validation evidence here.",
            "A top-scoring method among many CV candidates can be optimistic despite the nested per-fold tuning; an independent external cohort is needed before clinical or production claims.",
            "Pulse pressure was not derived because no diastolic blood-pressure field is in the saved 54-feature contract.",
            "Calibration plots compare the engineered XGBoost probabilities with the better calibrated variant by Brier score, using repeated out-of-fold predictions.",
            "",
            "## Protocol notes and limitations",
            "",
            "- The checkout contains 378 records (not the 303 stated in the supplied project summary). CAD is derived from `Cath`.",
            "- Feature engineering uses LDL/HDL, TG/HDL, a 7-factor count, three age interactions, and the requested ECG abnormality count.",
            "- Feature-selection and estimator fitting occur inside sklearn pipelines. CAD stacking generates CAD probabilities out-of-fold within each training partition; classifier-chain probabilities use earlier predicted targets, not held-out labels.",
            "- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and Brier are summarized from per-fold scores as repeat-level means ± SD; 95% intervals are percentile intervals over repeat-level means.",
            "- The dataset is small and single-source, and the target labels are imbalanced for CAD and LAD. Small changes may be sampling noise.",
            "- If no candidate clears the AUC-mean and SD gate, existing serialized models and canonical metrics remain unchanged.",
            "",
        ]
    )
    (METRICS_DIR / "improvement_report.md").write_text("\n".join(report), encoding="utf-8")


def _write_comparison(rows: list[dict[str, Any]]) -> None:
    columns = [
        "Target", "Method", "N_Repeats", "N_Folds_Per_Repeat", "Random_State",
        "Threshold_Objective", "Threshold_Mean", "Threshold_SD",
    ]
    for metric in METRICS:
        columns.extend(
            [
                f"{metric}_Mean", f"{metric}_SD",
                f"{metric}_CI95_Lower", f"{metric}_CI95_Upper",
            ]
        )
    frame = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=columns).to_csv(METRICS_DIR / "improvement_comparison.csv", index=False)


def _write_fold_checkpoints(
    folds: list[FoldResult], selections: list[dict[str, Any]], completed: list[str]
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in folds:
        record: dict[str, Any] = {
            "Target": row.target, "Method": row.method, "Repeat": row.repeat,
            "Fold": row.fold, "Threshold": row.threshold,
        }
        record.update(row.metrics)
        records.append(record)
    pd.DataFrame(records).to_csv(METRICS_DIR / "improvement_folds.csv", index=False)
    pd.DataFrame(selections).to_csv(METRICS_DIR / "improvement_selections.csv", index=False)
    (METRICS_DIR / "improvement_progress.json").write_text(
        json.dumps({"completed_methods": completed, "updated_epoch": time.time()}, indent=2),
        encoding="utf-8",
    )


def _method_list(trials: int) -> list[str]:
    methods = [
        "current_xgb", "xgb_engineered", "xgb_top20", "xgb_top30", "xgb_top40",
        "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_l2",
        "nested_best_model", "soft_vote_best3", "xgb_calibrated_sigmoid",
        "xgb_calibrated_isotonic",
    ]
    if trials > 0:
        methods.append("nested_optuna_xgb")
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10, help="Outer CV repeats (default: 10).")
    parser.add_argument("--inner-folds", type=int, default=3, help="Inner folds for OOF threshold estimates.")
    parser.add_argument("--trials", type=int, default=50, help="Nested Optuna trials per outer fit; 0 skips E.")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "current_xgb", "xgb_engineered", "xgb_top20", "xgb_top30", "xgb_top40",
            "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_l2",
            "nested_best_model", "soft_vote_best3", "xgb_calibrated_sigmoid",
            "xgb_calibrated_isotonic", "nested_optuna_xgb", "cad_oof_stack",
            "classifier_chain",
        ),
        help="Optional subset for targeted runs; omit to run every experiment.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.inner_folds < 2 or args.trials < 0:
        parser.error("--repeats and --inner-folds must be positive; --trials cannot be negative.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    data = pd.read_csv(DATA_PATH)
    features, metadata = _feature_contract()
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing input columns: {missing}")
    if FORBIDDEN.intersection(features):
        raise ValueError("A target/leakage column is present in the predictor contract.")
    X = data.loc[:, features].copy()
    if FORBIDDEN.intersection(X.columns):
        raise ValueError("Target leakage detected in X.")
    y_by_target = {
        "CAD": _positive_target(data["Cath"], "CAD"),
        **{target: _positive_target(data[target], target) for target in ("LAD", "LCX", "RCA")},
    }
    cad_y = y_by_target["CAD"]
    dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    LOG.info(
        "Starting nested repeated CV: n=%d, features=%d, repeats=%d, trials=%d, sha256=%s",
        len(data), len(features), args.repeats, args.trials, dataset_hash,
    )
    LOG.info("A: engineered ratios, risk count, age interactions and ECG score are in the feature pipeline.")
    LOG.info("B: top-20/30/40 selection is fitted inside each training fold.")
    LOG.info("C: CAD stacking will be tested in vessel targets; classifier-chain also uses nested cross-fitted targets.")
    LOG.info("D: comparing XGBoost, LightGBM, CatBoost, RF, ExtraTrees, Logistic Regression and inner-selected soft voting.")
    LOG.info("E: Optuna search is nested inside every outer training split; trials per fit=%d.", args.trials)
    LOG.info("F: sigmoid/isotonic calibration is fit within the training folds; Brier and plots are reported.")
    LOG.info("G: F1 thresholds are selected from inner-fold OOF predictions only.")

    fold_results: list[FoldResult] = []
    selections: list[dict[str, Any]] = []
    completed: list[str] = []
    baseline_rows = _load_baseline_rows()
    for target in args.targets:
        y = y_by_target[target]
        LOG.info("%s class counts: positive=%d negative=%d.", target, int(y.sum()), int(len(y) - y.sum()))
        methods = list(args.methods) if args.methods else _method_list(args.trials)
        if args.methods is None and target != "CAD":
            methods.extend(["cad_oof_stack", "classifier_chain"])
        for method in methods:
            method_folds, method_selections = _evaluate_candidate(
                X, y, y_by_target, target, method, features, metadata,
                args.repeats, args.inner_folds, args.trials,
            )
            fold_results.extend(method_folds)
            selections.extend(method_selections)
            completed.append(f"{target}:{method}")
            current_rows = baseline_rows + _aggregate(fold_results)
            _write_fold_checkpoints(fold_results, selections, completed)
            _write_comparison(current_rows)
            LOG.info("Completed %s/%s; checkpointed metrics/improvement_comparison.csv.", target, method)

    summary_rows = baseline_rows + _aggregate(fold_results)
    _write_comparison(summary_rows)
    _calibration_plots(fold_results, summary_rows)
    _write_report(summary_rows, fold_results, dataset_hash, args.repeats, args.trials)
    LOG.info(
        "Finished. Comparison: %s; report: %s; folds: %s; plots: %s",
        METRICS_DIR / "improvement_comparison.csv",
        METRICS_DIR / "improvement_report.md",
        METRICS_DIR / "improvement_folds.csv",
        METRICS_DIR / "calibration",
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
