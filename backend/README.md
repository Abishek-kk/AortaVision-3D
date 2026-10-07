# AortaVision 3D Backend

This backend serves the trained XGBoost inference models that were serialized under `model/serialized/`. It loads the saved preprocessing pipeline, validates model artifacts at startup, and exposes a simple FastAPI API for React frontend integration.

## Project purpose

The service supports risk prediction for the four target arteries used in the project:

- CAD
- LAD
- LCX
- RCA

The API is a research prototype for cardiovascular risk visualization and prediction. It does not provide a medical diagnosis.

## Architecture

- `main.py`: FastAPI application, startup validation, model loading, prediction routes, CORS, and logging.
- `schemas/patient_schema.py`: strict patient input validation against the raw model feature contract.
- `model/serialized/`: trained model payloads and preprocessing artifacts.

## Installation

From the project root:

```bash
python -m venv .venv
. .venv/bin/activate  # macOS/Linux
# or .\.venv\Scripts\activate  # Windows PowerShell
pip install -r backend/requirements.txt
```

## Environment variables

Use a local `.env` file if you want to override defaults. A template is provided in `backend/.env.example`.

```bash
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_ORIGIN=http://localhost:5173
```

## Start the backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Alternative when working directly in the backend folder:

```bash
cd backend
uvicorn main:app --reload
```

## API endpoints

- `GET /` — service metadata
- `GET /health` — health status
- `GET /models/status` — model load status and thresholds
- `POST /predict` — generate predictions for CAD, LAD, LCX, and RCA
- `GET /docs` — FastAPI Swagger UI
- `GET /redoc` — ReDoc UI

## Request format

The API expects JSON in the following shape:

```json
{
  "features": {
    "Age": 53,
    "Weight": 90,
    "Length": 175,
    "Sex": "Male",
    "BMI": 29.387755102040817,
    "DM": 0,
    "HTN": 1,
    "Current Smoker": 1,
    "EX-Smoker": 0,
    "FH": 0,
    "Obesity": "Y",
    "CRF": "N",
    "CVA": "N",
    "Airway disease": "N",
    "Thyroid Disease": "N",
    "CHF": "N",
    "DLP": "Y",
    "BP": 110,
    "PR": 80,
    "Edema": 0,
    "Weak Peripheral Pulse": "N",
    "Lung rales": "N",
    "Systolic Murmur": "N",
    "Diastolic Murmur": "N",
    "Typical Chest Pain": 0,
    "Dyspnea": "N",
    "Function Class": 0,
    "Atypical": "N",
    "Nonanginal": "N",
    "LowTH Ang": "N",
    "Q Wave": 0,
    "St Elevation": 0,
    "St Depression": 1,
    "Tinversion": 1,
    "LVH": "N",
    "Poor R Progression": "N",
    "BBB": "N",
    "FBS": 90,
    "CR": 0.7,
    "TG": 250,
    "LDL": 155,
    "HDL": 30,
    "BUN": 8,
    "ESR": 7,
    "HB": 15.6,
    "K": 4.7,
    "Na": 141,
    "WBC": 5700,
    "Lymph": 39,
    "Neut": 52,
    "PLT": 261,
    "EF-TTE": 50,
    "Region RWMA": 0,
    "VHD": "N"
  }
}
```

The request must use only the raw feature names in `model/serialized/feature_names.json`. Target/leakage columns such as `CAD`, `LAD`, `LCX`, `RCA`, and `Cath` are rejected.

## Response format

The prediction response is frontend-friendly and includes probabilities, thresholds, binary predictions, risk levels, and model explanations when SHAP can be safely computed.

```json
{
  "success": true,
  "predictions": {
    "CAD": {
      "probability": 0.90,
      "threshold": 0.35,
      "prediction": 1,
      "risk_level": "High"
    }
  },
  "explanations": {
    "CAD": [
      {
        "feature": "Age",
        "contribution": 0.12,
        "direction": "increases_risk",
        "value": 53
      }
    ]
  },
  "disclaimer": "This system is a research prototype for cardiovascular risk prediction and visualization. Predictions are model outputs and are not a medical diagnosis. They should not be used as a substitute for evaluation by a qualified healthcare professional."
}
```

Risk levels are a UI presentation layer only:

- probability < 0.30 => Low
- 0.30 <= probability < 0.70 => Moderate
- probability >= 0.70 => High

## Model artifact location

Trained models are loaded from:

- `model/serialized/model_cad.json`
- `model/serialized/model_lad.json`
- `model/serialized/model_lcx.json`
- `model/serialized/model_rca.json`
- `model/serialized/preprocessor.joblib`
- `model/serialized/feature_names.json`
- `model/serialized/model_metadata.json`

The backend resolves these paths relative to the project root instead of depending on the current working directory.

## Error responses

The API returns clean JSON errors for invalid requests and unavailable models:

- `400` or `422` for invalid or unsupported patient input
- `500` for unexpected inference errors
- `503` when a model or preprocessing artifact is unavailable

## Frontend connection

The React frontend can call:

```http
POST http://localhost:8000/predict
```

CORS is enabled for the default Vite frontend origins: `http://localhost:5173` and `http://127.0.0.1:5173`. Set `FRONTEND_ORIGIN` to a comma-separated list to override these defaults.

## Medical disclaimer

This system is a research prototype for cardiovascular risk prediction and visualization. Predictions are model outputs and are not a medical diagnosis. They should not be used as a substitute for evaluation by a qualified healthcare professional.
