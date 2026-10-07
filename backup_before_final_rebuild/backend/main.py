"""AortaVision 3D inference API.

The backend loads serialized XGBoost Booster models from model/serialized and
accepts only feature names that match the project dataset. If the training
notebook later defines a different final feature order, update the dataset-driven
feature list in patient_schema.py or set AORTAVISION_FEATURE_ORDER to override it.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__:
    from .schemas.patient_schema import DATASET_FEATURE_COLUMNS, PatientRequest
else:
    from schemas.patient_schema import DATASET_FEATURE_COLUMNS, PatientRequest

logger = logging.getLogger("aortavision.backend")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_configured_model_dir = os.getenv("AORTAVISION_MODEL_DIR")
MODEL_DIR = (
    Path(_configured_model_dir).expanduser()
    if _configured_model_dir
    else PROJECT_ROOT / "model" / "serialized"
)
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (PROJECT_ROOT / MODEL_DIR).resolve()

MODEL_PATHS = {
    "cad": MODEL_DIR / "model_cad.json",
    "lad": MODEL_DIR / "model_lad.json",
    "lcx": MODEL_DIR / "model_lcx.json",
    "rca": MODEL_DIR / "model_rca.json",
}
METADATA_PATH = MODEL_DIR / "model_metadata.json"

LEAKAGE_COLUMNS = frozenset({"cad", "lad", "lcx", "rca", "cath"})


def _resolve_feature_order() -> tuple[str, ...]:
    """Use the dataset schema by default while allowing a future training override."""
    env_features = tuple(
        feature.strip()
        for feature in os.getenv("AORTAVISION_FEATURE_ORDER", "").split(",")
        if feature.strip()
    )
    if env_features:
        return env_features
    if DATASET_FEATURE_COLUMNS:
        return DATASET_FEATURE_COLUMNS
    return tuple()


FEATURE_ORDER = _resolve_feature_order()

_preprocessor_setting = os.getenv("AORTAVISION_PREPROCESSOR_PATH")
PREPROCESSOR_PATH = (
    Path(_preprocessor_setting).expanduser() if _preprocessor_setting else None
)
if PREPROCESSOR_PATH is None:
    default_preprocessor = MODEL_DIR / "preprocessor.joblib"
    if default_preprocessor.is_file():
        PREPROCESSOR_PATH = default_preprocessor
if PREPROCESSOR_PATH is not None and not PREPROCESSOR_PATH.is_absolute():
    PREPROCESSOR_PATH = (PROJECT_ROOT / PREPROCESSOR_PATH).resolve()

PREPROCESSING_MODE = os.getenv(
    "AORTAVISION_PREPROCESSING_MODE", "artifact_required"
).strip().lower()

try:
    TOP_N_FEATURES = int(os.getenv("AORTAVISION_SHAP_TOP_N", "10"))
except ValueError as exc:
    raise RuntimeError("AORTAVISION_SHAP_TOP_N must be a positive integer.") from exc
if TOP_N_FEATURES < 1:
    raise RuntimeError("AORTAVISION_SHAP_TOP_N must be a positive integer.")

try:
    POSITIVE_CLASS_LABELS: dict[str, Any] = json.loads(
        os.getenv("AORTAVISION_POSITIVE_CLASS_LABELS", "{}")
    )
except json.JSONDecodeError as exc:
    raise RuntimeError(
        "AORTAVISION_POSITIVE_CLASS_LABELS must contain a JSON object."
    ) from exc
if not isinstance(POSITIVE_CLASS_LABELS, dict):
    raise RuntimeError("AORTAVISION_POSITIVE_CLASS_LABELS must contain a JSON object.")

DEFAULT_CORS_ORIGINS = ("http://localhost:5173",)
_cors_value = os.getenv("AORTAVISION_CORS_ORIGINS")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in _cors_value.split(",") if origin.strip()]
    if _cors_value is not None
    else list(DEFAULT_CORS_ORIGINS)
)


def _load_thresholds() -> dict[str, float]:
    """Load training-selected decision thresholds, defaulting to 0.50."""
    thresholds = {target: 0.5 for target in MODEL_PATHS}
    if not METADATA_PATH.is_file():
        logger.warning("Model metadata is missing; using 0.50 thresholds.")
        return thresholds

    try:
        payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read model metadata; using 0.50 thresholds.")
        return thresholds

    saved_thresholds = payload.get("selected_thresholds", {})
    if not isinstance(saved_thresholds, dict):
        return thresholds
    for target in thresholds:
        value = saved_thresholds.get(target.upper(), saved_thresholds.get(target))
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value_float <= 1.0:
            thresholds[target] = value_float
    return thresholds


def _load_target_feature_indices() -> dict[str, list[int] | None]:
    """Load optional per-target selected feature indices from metadata."""
    indices = {target: None for target in MODEL_PATHS}
    if not METADATA_PATH.is_file():
        return indices
    try:
        payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read target feature indices from metadata.")
        return indices
    saved_indices = payload.get("target_feature_indices", {})
    if not isinstance(saved_indices, dict):
        return indices
    for target in indices:
        values = saved_indices.get(target.upper(), saved_indices.get(target))
        if not isinstance(values, list):
            continue
        try:
            parsed = [int(value) for value in values]
        except (TypeError, ValueError):
            continue
        if parsed and min(parsed) >= 0:
            indices[target] = parsed
    return indices


class PredictionUnavailable(Exception):
    """Raised when real inference cannot be performed with current artifacts."""


class TargetPrediction(BaseModel):
    prediction: int = Field(description="Predicted class: 1 is configured positive.")
    probability: float = Field(
        ge=0,
        le=1,
        description="Model probability for the configured positive class.",
    )


class FeatureContribution(BaseModel):
    feature: str
    value: str | float | int | bool
    contribution: float


class PredictionResponse(BaseModel):
    cad: TargetPrediction
    lad: TargetPrediction
    lcx: TargetPrediction
    rca: TargetPrediction
    explanations: dict[str, list[FeatureContribution]]


def _load_models() -> tuple[dict[str, Any], dict[str, str]]:
    """Load each classifier once, retaining explicit errors for unavailable files."""
    models: dict[str, Any] = {}
    errors: dict[str, str] = {}

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.exception("XGBoost is required to load prediction models.")
        message = "XGBoost is not installed; install backend requirements."
        return models, {target: message for target in MODEL_PATHS}

    for target, model_path in MODEL_PATHS.items():
        if not model_path.is_file():
            errors[target] = f"Model file is missing: {model_path.name}"
            logger.error("Required model file is missing: %s", model_path)
            continue
        if model_path.stat().st_size == 0:
            errors[target] = f"Model file is empty: {model_path.name}"
            logger.error("Required model file is empty: %s", model_path)
            continue

        try:
            classifier = XGBClassifier()
            classifier.load_model(str(model_path))
            if not callable(getattr(classifier, "predict_proba", None)):
                raise TypeError("Loaded model does not support predict_proba().")
            models[target] = classifier
            logger.info("Loaded %s model from %s", target.upper(), model_path)
        except Exception:
            logger.exception("Failed to load %s model from %s", target.upper(), model_path)
            errors[target] = (
                f"Model file could not be loaded: {model_path.name}; "
                "see backend logs for details."
            )

    return models, errors


def _load_preprocessor() -> tuple[Any | None, str | None]:
    """Load only the training-produced preprocessing artifact, if configured."""
    if PREPROCESSOR_PATH is None:
        if PREPROCESSING_MODE == "identity":
            return None, None
        return (
            None,
            "Preprocessing is not configured. Set "
            "AORTAVISION_PREPROCESSOR_PATH or explicitly configure identity "
            "preprocessing after confirming it matches training.",
        )

    if PREPROCESSING_MODE == "identity":
        return None, (
            "Configure either AORTAVISION_PREPROCESSOR_PATH or "
            "AORTAVISION_PREPROCESSING_MODE=identity, not both."
        )
    if not PREPROCESSOR_PATH.is_file():
        logger.error("Configured preprocessing artifact is missing: %s", PREPROCESSOR_PATH)
        return None, f"Preprocessing artifact is missing: {PREPROCESSOR_PATH.name}"
    if PREPROCESSOR_PATH.stat().st_size == 0:
        logger.error("Configured preprocessing artifact is empty: %s", PREPROCESSOR_PATH)
        return None, f"Preprocessing artifact is empty: {PREPROCESSOR_PATH.name}"

    try:
        import joblib

        return joblib.load(PREPROCESSOR_PATH), None
    except Exception:
        logger.exception("Failed to load preprocessing artifact: %s", PREPROCESSOR_PATH)
        return None, (
            f"Preprocessing artifact could not be loaded: {PREPROCESSOR_PATH.name}; "
            "see backend logs for details."
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.models, application.state.model_errors = _load_models()
    application.state.preprocessor, application.state.preprocessor_error = (
        _load_preprocessor()
    )
    application.state.thresholds = _load_thresholds()
    application.state.target_feature_indices = _load_target_feature_indices()
    yield


app = FastAPI(
    title="AortaVision 3D API",
    description=(
        "Educational decision-support inference service. Model outputs are not a "
        "medical diagnosis and are not a substitute for professional medical "
        "evaluation or formal diagnostic imaging."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.state.models = {}
app.state.model_errors = {}
app.state.preprocessor = None
app.state.preprocessor_error = (
    "Startup has not loaded the preprocessing artifact yet."
)
app.state.thresholds = {target: 0.5 for target in MODEL_PATHS}
app.state.target_feature_indices = {target: None for target in MODEL_PATHS}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    """Identify the running backend service."""
    return {
        "project": "AortaVision 3D",
        "service": "Cardiovascular Risk Prediction API",
        "status": "running",
    }


@app.get("/health", tags=["service"])
def health(request: Request) -> dict[str, Any]:
    """Liveness check; model readiness is reported separately."""
    return {
        "status": "healthy",
        "models_loaded": len(request.app.state.models) == len(MODEL_PATHS),
    }


@app.get("/models/status", tags=["service"])
def models_status(request: Request) -> dict[str, Any]:
    """Report actual model load state and configuration blockers."""
    models = getattr(request.app.state, "models", {})
    response: dict[str, Any] = {
        target: target in models for target in MODEL_PATHS
    }
    response["errors"] = dict(getattr(request.app.state, "model_errors", {}))
    response["feature_order_configured"] = bool(FEATURE_ORDER)
    response["preprocessing_ready"] = (
        getattr(request.app.state, "preprocessor_error", None) is None
    )
    response["thresholds"] = dict(getattr(request.app.state, "thresholds", {}))
    if getattr(request.app.state, "preprocessor_error", None):
        response["preprocessing_error"] = request.app.state.preprocessor_error
    return response


def _validate_feature_contract(features: dict[str, Any]) -> None:
    if not FEATURE_ORDER:
        raise PredictionUnavailable(
            "The training pipeline has not defined a feature order for the backend."
        )
    if len(set(FEATURE_ORDER)) != len(FEATURE_ORDER):
        raise PredictionUnavailable("The configured feature order contains duplicates.")
    if any(feature.casefold() in LEAKAGE_COLUMNS for feature in FEATURE_ORDER):
        raise PredictionUnavailable(
            "The configured feature order contains a prohibited target/leakage column."
        )

    supplied = set(features)
    expected = set(FEATURE_ORDER)
    missing = [feature for feature in FEATURE_ORDER if feature not in supplied]
    unexpected = sorted(supplied - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing required features: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected features: {', '.join(unexpected)}")
        raise HTTPException(status_code=422, detail="; ".join(details))


def _prepare_features(features: dict[str, Any], preprocessor: Any | None) -> Any:
    input_frame = pd.DataFrame(
        [[features[name] for name in FEATURE_ORDER]],
        columns=list(FEATURE_ORDER),
    )
    if preprocessor is None:
        transformed = input_frame
    else:
        transform = getattr(preprocessor, "transform", None)
        if not callable(transform):
            raise PredictionUnavailable(
                "The configured preprocessing artifact has no transform() method."
            )
        transformed = transform(input_frame)

    if getattr(transformed, "shape", None) is None or len(transformed.shape) != 2:
        raise PredictionUnavailable(
            "Preprocessing must return a two-dimensional feature matrix."
        )
    if transformed.shape[0] != 1:
        raise PredictionUnavailable(
            "Preprocessing must preserve the single-patient batch size."
        )
    return transformed


def _feature_names(transformed: Any, preprocessor: Any | None) -> list[str]:
    if isinstance(transformed, pd.DataFrame):
        names = [str(name) for name in transformed.columns]
    elif preprocessor is not None and callable(
        getattr(preprocessor, "get_feature_names_out", None)
    ):
        names = [
            str(name)
            for name in preprocessor.get_feature_names_out(list(FEATURE_ORDER))
        ]
    elif transformed.shape[1] == len(FEATURE_ORDER):
        names = list(FEATURE_ORDER)
    else:
        raise PredictionUnavailable(
            "The preprocessing artifact must return named features or implement "
            "get_feature_names_out() so SHAP contributions can be labeled."
        )

    if len(names) != transformed.shape[1]:
        raise PredictionUnavailable(
            "Preprocessing feature names do not match the transformed feature count."
        )
    return names


def _slice_target_features(
    transformed: Any,
    feature_names: list[str],
    indices: list[int] | None,
) -> tuple[Any, list[str]]:
    if indices is None:
        return transformed, feature_names
    if max(indices) >= len(feature_names):
        raise PredictionUnavailable(
            "Target feature selection indices exceed transformed feature count."
        )
    selected_names = [feature_names[index] for index in indices]
    if isinstance(transformed, pd.DataFrame):
        return transformed.iloc[:, indices], selected_names
    return transformed[:, indices], selected_names


def _positive_class_index(model: Any, target: str) -> int:
    classes = list(model.classes_)
    if len(classes) != 2:
        raise PredictionUnavailable(
            f"The {target.upper()} model is not a binary classifier."
        )

    if target in POSITIVE_CLASS_LABELS:
        configured_label = POSITIVE_CLASS_LABELS[target]
        matches = [
            index
            for index, label in enumerate(classes)
            if str(label) == str(configured_label)
        ]
        if len(matches) != 1:
            raise PredictionUnavailable(
                f"Configured positive label for {target.upper()} does not match "
                "exactly one model class."
            )
        return matches[0]

    positive_matches = [
        index
        for index, label in enumerate(classes)
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool)
        and int(label) == 1
    ]
    if len(positive_matches) == 1:
        return positive_matches[0]

    raise PredictionUnavailable(
        f"Set {target!r} in AORTAVISION_POSITIVE_CLASS_LABELS to identify the "
        "positive model class."
    )


def _shap_contributions(
    model: Any,
    target: str,
    transformed: Any,
    feature_names: list[str],
    positive_index: int,
) -> list[FeatureContribution]:
    try:
        import shap
    except ImportError as exc:
        logger.exception("SHAP is required to generate prediction explanations.")
        raise PredictionUnavailable(
            "SHAP is not installed; install backend requirements."
        ) from exc

    explanation = shap.TreeExplainer(model)(transformed)
    values = explanation.values
    if isinstance(values, list):
        if positive_index >= len(values):
            raise PredictionUnavailable("SHAP returned an unexpected class shape.")
        contributions = np.asarray(values[positive_index])[0]
    else:
        values_array = np.asarray(values)
        if values_array.ndim == 2 and values_array.shape[0] == 1:
            contributions = values_array[0]
        elif values_array.ndim == 3 and values_array.shape[0] == 1:
            if values_array.shape[1] == len(feature_names) and values_array.shape[2] == 2:
                contributions = values_array[0, :, positive_index]
            elif values_array.shape[1] == 2 and values_array.shape[2] == len(feature_names):
                contributions = values_array[0, positive_index, :]
            else:
                raise PredictionUnavailable("SHAP returned an unexpected class shape.")
        else:
            raise PredictionUnavailable("SHAP returned an unexpected value shape.")

    contributions = np.asarray(contributions).reshape(-1)
    if len(contributions) != len(feature_names):
        raise PredictionUnavailable(
            "SHAP contribution count does not match the transformed features."
        )
    if not np.isfinite(contributions).all():
        raise PredictionUnavailable("SHAP returned non-finite contributions.")

    if isinstance(transformed, pd.DataFrame):
        row_values = transformed.iloc[0].tolist()
    else:
        row_values = np.asarray(transformed)[0].tolist()
    ranked_indices = np.argsort(np.abs(contributions))[::-1][:TOP_N_FEATURES]

    result = []
    for index in ranked_indices:
        value = row_values[int(index)]
        if isinstance(value, np.generic):
            value = value.item()
        result.append(
            FeatureContribution(
                feature=feature_names[int(index)],
                value=value,
                contribution=float(contributions[int(index)]),
            )
        )
    return result


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict model outputs for one patient",
    description=(
        "Returns model predictions, positive-class probabilities, and SHAP "
        "feature attributions. Outputs are model explanations, not causal "
        "medical findings or a substitute for clinical evaluation."
    ),
)
def predict(patient: PatientRequest, request: Request) -> PredictionResponse:
    """Validate features and run the four loaded models without training."""
    _validate_feature_contract(patient.features)

    models = request.app.state.models
    if len(models) != len(MODEL_PATHS):
        unavailable = sorted(set(MODEL_PATHS) - set(models))
        raise HTTPException(
            status_code=503,
            detail={
                "message": "One or more trained models are unavailable.",
                "unavailable_models": unavailable,
                "model_errors": request.app.state.model_errors,
            },
        )
    if request.app.state.preprocessor_error:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.preprocessor_error,
        )

    try:
        transformed = _prepare_features(
            patient.features, request.app.state.preprocessor
        )
        explanations: dict[str, list[FeatureContribution]] = {}
        predictions: dict[str, TargetPrediction] = {}
        feature_names = _feature_names(
            transformed, request.app.state.preprocessor
        )

        for target, model in models.items():
            try:
                target_indices = getattr(
                    request.app.state, "target_feature_indices", {}
                ).get(target)
                target_features, target_feature_names = _slice_target_features(
                    transformed,
                    feature_names,
                    target_indices,
                )
                probabilities = np.asarray(model.predict_proba(target_features), dtype=float)
                if probabilities.shape != (1, 2):
                    raise PredictionUnavailable(
                        f"The {target.upper()} model did not return binary probabilities."
                    )
                if not np.isfinite(probabilities).all() or np.any(
                    (probabilities < 0) | (probabilities > 1)
                ):
                    raise PredictionUnavailable(
                        f"The {target.upper()} model returned invalid probabilities."
                    )
                positive_index = _positive_class_index(model, target)
                threshold = float(
                    getattr(request.app.state, "thresholds", {}).get(target, 0.5)
                )
                predictions[target] = TargetPrediction(
                    prediction=int(probabilities[0, positive_index] >= threshold),
                    probability=float(probabilities[0, positive_index]),
                )
                explanations[target] = _shap_contributions(
                    model,
                    target,
                    target_features,
                    target_feature_names,
                    positive_index,
                )
            except PredictionUnavailable:
                raise
            except Exception as exc:
                logger.exception(
                    "Prediction or explanation failed for %s.", target.upper()
                )
                raise RuntimeError(
                    f"Prediction or explanation failed for {target.upper()}."
                ) from exc

        return PredictionResponse(
            cad=predictions["cad"],
            lad=predictions["lad"],
            lcx=predictions["lcx"],
            rca=predictions["rca"],
            explanations=explanations,
        )
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
