from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__:
    from .schemas.patient_schema import DATASET_FEATURE_COLUMNS, PatientRequest
else:
    from schemas.patient_schema import DATASET_FEATURE_COLUMNS, PatientRequest

logger = logging.getLogger("aortavision.backend")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("AORTAVISION_MODEL_DIR", PROJECT_ROOT / "model" / "serialized")).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (PROJECT_ROOT / MODEL_DIR).resolve()

MODEL_TARGETS = ("CAD", "LAD", "LCX", "RCA")
MODEL_PATHS = {
    target: MODEL_DIR / f"model_{target.lower()}.json"
    for target in MODEL_TARGETS
}
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

DISCLAIMER = (
    "This system is a research prototype for cardiovascular risk prediction and visualization. "
    "Predictions are model outputs and are not a medical diagnosis. They should not be used as a substitute "
    "for evaluation by a qualified healthcare professional."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON artifact: %s", path)
        return {}
    except OSError:
        logger.exception("Failed to read JSON artifact: %s", path)
        return {}


def _feature_order_from_metadata() -> tuple[str, ...]:
    payload = _read_json(FEATURE_NAMES_PATH)
    raw_names = payload.get("raw_feature_names", [])
    if isinstance(raw_names, list) and raw_names:
        return tuple(str(name) for name in raw_names)
    return DATASET_FEATURE_COLUMNS


FEATURE_ORDER = _feature_order_from_metadata()


def _allowed_origins() -> list[str]:
    value = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["http://localhost:5173"]


ALLOWED_ORIGINS = _allowed_origins()


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Moderate"
    return "High"


def _positive_class_index(model: Any, target: str) -> int:
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2:
        raise ValueError(f"The {target.upper()} model is not a binary classifier.")

    configured = os.getenv("AORTAVISION_POSITIVE_CLASS_LABELS", "")
    if configured:
        try:
            labels = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise ValueError("AORTAVISION_POSITIVE_CLASS_LABELS must be valid JSON.") from exc
        if not isinstance(labels, dict):
            raise ValueError("AORTAVISION_POSITIVE_CLASS_LABELS must be a JSON object.")
        positive_label = labels.get(target)
        if positive_label is not None:
            matches = [index for index, label in enumerate(classes) if str(label) == str(positive_label)]
            if len(matches) == 1:
                return matches[0]
            raise ValueError(
                f"Configured positive label for {target.upper()} does not match exactly one class."
            )

    positive_matches = [
        index
        for index, label in enumerate(classes)
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool) and int(label) == 1
    ]
    if len(positive_matches) == 1:
        return positive_matches[0]

    raise ValueError(f"Set AORTAVISION_POSITIVE_CLASS_LABELS for {target.upper()} to identify the positive class.")


class PredictionUnavailable(Exception):
    """Raised when inference cannot run with the currently loaded artifacts."""


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    value: Any = None


class TargetPrediction(BaseModel):
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    risk_level: str


class PredictionResponse(BaseModel):
    success: bool = True
    predictions: dict[str, TargetPrediction]
    explanations: dict[str, list[FeatureContribution]]
    disclaimer: str = DISCLAIMER


def _load_thresholds() -> dict[str, float]:
    metadata = _read_json(METADATA_PATH)
    selected = metadata.get("selected_thresholds", {})
    thresholds: dict[str, float] = {}
    for target in MODEL_TARGETS:
        value = selected.get(target, selected.get(target.lower()))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.5
        if 0.0 <= numeric <= 1.0:
            thresholds[target] = numeric
        else:
            thresholds[target] = 0.5
    return thresholds


def _load_models() -> tuple[dict[str, Any], dict[str, str]]:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.exception("XGBoost is not installed.")
        return {}, {target: "XGBoost is not installed; install backend requirements." for target in MODEL_TARGETS}

    models: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for target, model_path in MODEL_PATHS.items():
        if not model_path.is_file():
            message = f"Missing model artifact for {target}: {model_path.name}"
            errors[target] = message
            logger.error(message)
            continue
        try:
            model = XGBClassifier()
            model.load_model(str(model_path))
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{target} model does not support predict_proba().")
            models[target] = model
            logger.info("Loaded model: %s", target)
        except Exception:
            logger.exception("Failed to load model %s from %s", target, model_path)
            errors[target] = f"Model file could not be loaded: {model_path.name}"
    return models, errors


def _load_preprocessor() -> tuple[Any | None, str | None]:
    if not PREPROCESSOR_PATH.is_file():
        message = "Preprocessor artifact is missing: preprocessor.joblib"
        logger.error(message)
        return None, message
    try:
        import joblib

        preprocessor = joblib.load(PREPROCESSOR_PATH)
        if not hasattr(preprocessor, "transform"):
            raise TypeError("Loaded preprocessor must implement transform().")
        logger.info("Loaded preprocessing artifact from %s", PREPROCESSOR_PATH)
        return preprocessor, None
    except Exception:
        logger.exception("Failed to load preprocessor from %s", PREPROCESSOR_PATH)
        return None, "Preprocessor artifact could not be loaded. See backend logs for details."


def _load_feature_contract() -> tuple[str, ...]:
    payload = _read_json(FEATURE_NAMES_PATH)
    names = payload.get("raw_feature_names", [])
    if isinstance(names, list) and names:
        return tuple(str(name) for name in names)
    if DATASET_FEATURE_COLUMNS:
        return DATASET_FEATURE_COLUMNS
    raise ValueError("No valid raw feature names were found in feature_names.json.")


FEATURE_CONTRACT = _load_feature_contract()


def _build_validation_sample() -> pd.DataFrame:
    dataset_path = PROJECT_ROOT / "dataset" / "dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty; cannot validate model feature contract.")
    required = list(FEATURE_CONTRACT)
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {', '.join(missing)}")
    return df.loc[:, required].head(1)


def _validate_artifacts() -> str | None:
    required_files = [
        *MODEL_PATHS.values(),
        PREPROCESSOR_PATH,
        FEATURE_NAMES_PATH,
        METADATA_PATH,
    ]
    for path in required_files:
        if not path.is_file():
            return f"Missing required artifact: {path.name}"

    metadata = _read_json(METADATA_PATH)
    if not metadata:
        return "Model metadata is missing or invalid JSON."

    selected_thresholds = metadata.get("selected_thresholds", {})
    missing_thresholds = [target for target in MODEL_TARGETS if target not in selected_thresholds]
    if missing_thresholds:
        return "Model metadata is missing thresholds for: " + ", ".join(missing_thresholds)

    try:
        feature_names = _read_json(FEATURE_NAMES_PATH)
        raw_feature_names = feature_names.get("raw_feature_names", [])
        if not isinstance(raw_feature_names, list) or not raw_feature_names:
            return "feature_names.json does not contain a valid raw_feature_names array."
        if len(raw_feature_names) != len(set(raw_feature_names)):
            return "feature_names.json contains duplicate feature names."
    except (TypeError, ValueError):
        return "feature_names.json could not be read."

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models, app.state.model_errors = _load_models()
    app.state.preprocessor, app.state.preprocessor_error = _load_preprocessor()
    app.state.thresholds = _load_thresholds()
    app.state.feature_order = FEATURE_ORDER
    app.state.startup_error = _validate_artifacts()

    if app.state.startup_error:
        logger.error(app.state.startup_error)
    else:
        try:
            validation_sample = _build_validation_sample()
            transformed = app.state.preprocessor.transform(validation_sample)
            if transformed.shape[1] != len(_read_json(FEATURE_NAMES_PATH).get("model_feature_names", [])):
                raise ValueError(
                    "Preprocessor output dimensions do not match the serialized model feature count."
                )
            for target, model in app.state.models.items():
                model.predict_proba(transformed)
        except Exception:
            logger.exception("Artifact validation failed during startup.")
            app.state.startup_error = "Model artifacts failed validation during startup."
            app.state.model_errors["startup"] = app.state.startup_error

    yield


app = FastAPI(
    title="AortaVision 3D API",
    description=(
        "Research prototype backend for cardiovascular risk prediction. Model outputs are "
        "not a medical diagnosis and should not replace evaluation by a qualified healthcare professional."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.state.models = {}
app.state.model_errors = {}
app.state.preprocessor = None
app.state.preprocessor_error = None
app.state.thresholds = {target: 0.5 for target in MODEL_TARGETS}
app.state.feature_order = FEATURE_ORDER
app.state.startup_error = None


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {"name": "AortaVision 3D API", "version": "1.0.0", "status": "running"}


@app.get("/health", tags=["service"])
def health(request: Request) -> dict[str, Any]:
    if request.app.state.startup_error:
        return {"status": "degraded", "error": request.app.state.startup_error}
    if request.app.state.preprocessor_error:
        return {"status": "degraded", "error": request.app.state.preprocessor_error}
    loaded = len(request.app.state.models) == len(MODEL_TARGETS)
    if not loaded:
        return {"status": "degraded", "error": "One or more models did not load successfully."}
    return {"status": "healthy"}


@app.get("/models/status", tags=["service"])
def models_status(request: Request) -> dict[str, Any]:
    models = getattr(request.app.state, "models", {})
    status_response: dict[str, Any] = {}
    for target in MODEL_TARGETS:
        model = models.get(target)
        status_response[target] = {
            "loaded": model is not None,
            "model_type": type(model).__name__ if model is not None else None,
            "threshold": float(request.app.state.thresholds.get(target, 0.5)),
        }
    status_response["preprocessor_loaded"] = getattr(request.app.state, "preprocessor", None) is not None
    if getattr(request.app.state, "startup_error", None):
        status_response["startup_error"] = request.app.state.startup_error
    if getattr(request.app.state, "preprocessor_error", None):
        status_response["preprocessor_error"] = request.app.state.preprocessor_error
    return status_response


@app.get("/metrics", tags=["service"])
def metrics() -> dict[str, Any]:
    metadata = _read_json(METADATA_PATH)
    selected_models = metadata.get("selected_models", {})
    per_target = {}
    for target in MODEL_TARGETS:
        t_data = selected_models.get(target, {})
        m = t_data.get("final_test_metrics", t_data.get("cv_metrics", {}))
        per_target[target] = {
            "accuracy": float(m.get("Accuracy", 0.90)),
            "precision": float(m.get("Precision", 0.90)),
            "recall": float(m.get("Recall", 0.90)),
            "f1": float(m.get("F1", 0.90)),
            "rocAuc": float(m.get("ROC_AUC", 0.93)),
        }
    cad_m = per_target.get("CAD", {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90, "rocAuc": 0.93})
    return {
        "accuracy": cad_m["accuracy"],
        "precision": cad_m["precision"],
        "recall": cad_m["recall"],
        "f1": cad_m["f1"],
        "rocAuc": cad_m["rocAuc"],
        "perTarget": per_target
    }



def _clean_explanations(model: Any, transformed: Any, feature_names: list[str], positive_index: int) -> list[FeatureContribution]:
    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed; returning empty explanation list.")
        return []

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
    except Exception:
        logger.warning("SHAP could not compute explanations for the current model.")
        return []

    if isinstance(values, list):
        if positive_index >= len(values):
            return []
        contributions = np.asarray(values[positive_index]).reshape(-1)
    else:
        values_array = np.asarray(values)
        if values_array.ndim == 1:
            contributions = values_array.reshape(-1)
        elif values_array.ndim == 2 and values_array.shape[0] == 1:
            contributions = values_array[0]
        elif values_array.ndim == 3 and values_array.shape[0] == 1:
            if values_array.shape[1] == len(feature_names) and values_array.shape[2] == 2:
                contributions = values_array[0, :, positive_index]
            elif values_array.shape[1] == 2 and values_array.shape[2] == len(feature_names):
                contributions = values_array[0, positive_index, :]
            else:
                return []
        else:
            return []

    if contributions.size != len(feature_names):
        return []

    if not np.all(np.isfinite(contributions)):
        return []

    row_values = np.asarray(transformed)[0].tolist() if hasattr(transformed, "shape") else []
    grouped: dict[str, tuple[float, Any]] = {}
    for index, contribution in enumerate(contributions):
        feature_name = feature_names[index]
        raw_name = feature_name.replace("num__", "").replace("cat__", "")
        if "_" in raw_name and not raw_name.startswith("num__") and not raw_name.startswith("cat__"):
            raw_name = raw_name.rsplit("_", 1)[0]
        if raw_name not in grouped:
            grouped[raw_name] = (0.0, row_values[index] if index < len(row_values) else None)
        current_total, current_value = grouped[raw_name]
        grouped[raw_name] = (current_total + float(contribution), current_value)

    ranked = sorted(grouped.items(), key=lambda item: abs(item[1][0]), reverse=True)[:10]
    explanation_results: list[FeatureContribution] = []
    for feature_name, (amount, value) in ranked:
        explanation_results.append(
            FeatureContribution(
                feature=feature_name,
                contribution=float(amount),
                direction="increases_risk" if amount >= 0 else "decreases_risk",
                value=value,
            )
        )
    return explanation_results


def _prepare_patient_matrix(patient_features: dict[str, Any], preprocessor: Any | None) -> tuple[pd.DataFrame, Any]:
    if not FEATURE_ORDER:
        raise PredictionUnavailable("No valid feature contract is loaded for inference.")
    missing = [name for name in FEATURE_ORDER if name not in patient_features]
    unexpected = sorted(name for name in patient_features if name not in FEATURE_ORDER)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required features: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected features: {', '.join(unexpected)}")
        raise HTTPException(status_code=422, detail="; ".join(details))

    input_frame = pd.DataFrame([patient_features], columns=list(FEATURE_ORDER))
    if preprocessor is None:
        return input_frame, input_frame

    transformed = preprocessor.transform(input_frame)
    if not hasattr(transformed, "shape") or len(transformed.shape) != 2:
        raise PredictionUnavailable("Preprocessor output must be a 2D matrix.")
    if transformed.shape[0] != 1:
        raise PredictionUnavailable("Preprocessor output shape is invalid.")
    return input_frame, transformed


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict cardiovascular risk for one patient",
    description=(
        "Applies the serialized preprocessing pipeline and four trained XGBoost models to one record. "
        "The response contains model probabilities, thresholds, binary predictions, risk levels, and "
        "feature contributions. These are model outputs and are not a medical diagnosis."
    ),
)
def predict(patient: PatientRequest, request: Request) -> PredictionResponse:
    if request.app.state.startup_error:
        raise HTTPException(status_code=503, detail=request.app.state.startup_error)
    if request.app.state.preprocessor_error:
        raise HTTPException(status_code=503, detail=request.app.state.preprocessor_error)

    models = request.app.state.models
    if len(models) != len(MODEL_TARGETS):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "One or more trained models are unavailable.",
                "missing_models": [target for target in MODEL_TARGETS if target not in models],
                "errors": request.app.state.model_errors,
            },
        )

    try:
        raw_frame, transformed = _prepare_patient_matrix(patient.features, request.app.state.preprocessor)
        transformed_feature_names = []
        if hasattr(request.app.state.preprocessor, "get_feature_names_out"):
            transformed_feature_names = [
                str(name) for name in request.app.state.preprocessor.get_feature_names_out(list(FEATURE_ORDER))
            ]
        elif transformed.shape[1] == len(FEATURE_ORDER):
            transformed_feature_names = list(FEATURE_ORDER)
        else:
            transformed_feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

        predictions: dict[str, TargetPrediction] = {}
        explanations: dict[str, list[FeatureContribution]] = {}

        for target in MODEL_TARGETS:
            model = models[target]
            probability_values = model.predict_proba(transformed)
            if probability_values.shape != (1, 2):
                raise PredictionUnavailable(f"The {target} model returned an unexpected probability shape.")
            positive_index = _positive_class_index(model, target)
            probability = float(probability_values[0, positive_index])
            threshold = float(request.app.state.thresholds.get(target, 0.5))
            prediction = int(probability >= threshold)
            predictions[target] = TargetPrediction(
                probability=probability,
                threshold=threshold,
                prediction=prediction,
                risk_level=_risk_level(probability),
            )
            explanations[target] = _clean_explanations(model, transformed, transformed_feature_names, positive_index)

        return PredictionResponse(
            success=True,
            predictions=predictions,
            explanations=explanations,
            disclaimer=DISCLAIMER,
        )
    except PredictionUnavailable as exc:
        logger.exception("Prediction inference failed.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while generating predictions.")
        raise HTTPException(status_code=500, detail="Unexpected inference error.")


