# AortaVision 3D

**Multimodal AI Hackathon 2026 · Track A: Cardiovascular Risk Visualization & Prediction**

An interactive web app that predicts **coronary artery disease (CAD)** and **stenosis in the LAD, LCX and RCA** from clinical data, then maps the predicted risk onto a rotatable 3D heart. A dashboard explains every prediction with SHAP.

> ⚠️ **Clinical safety disclaimer.** Predictions are for **decision support and educational purposes only**. They are **not** a substitute for formal diagnostic imaging (coronary angiography, CT angiography) or a clinician's judgment. Artery positions in the 3D scene are schematic, not an imaging-based lesion map.

- 🎥 **Demo video:** `<YOUTUBE_LINK>` (3–10 min, unlisted)
- 🏆 **Devpost:** `<DEVPOST_LINK>`

---

## Team (Team Accelerate)

| Name (real full name) | Role |
|---|---|
| `<Full Name>` | `<role>` |
| `<Full Name>` | `<role>` |
| `<Full Name>` | `<role>` |
| `<Full Name>` | `<role>` |

---

## What it does

| Track A requirement | Where it is in this project |
|---|---|
| Predict CAD + LAD/LCX/RCA stenosis | 4 XGBoost classifiers, served by FastAPI (`POST /predict`) |
| No target leakage | `LAD`, `LCX`, `RCA`, `Cath` (and `CAD`) are excluded from inputs; the backend and training code assert this |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC per target (`GET /metrics`, `metrics/`) |
| Interactive 3D heart | React Three Fiber + GLB heart model; rotate, zoom, hover and click vessels |
| Colour-coded arteries | LAD, LCX and RCA colours update from predicted probabilities (green → yellow → red) |
| Vessel-level detail | Selecting a vessel pins its risk card and switches the SHAP view to that vessel |
| Clinical dashboard | CAD status, vessel probabilities, validation metrics |
| Interpretability | SHAP feature contributions for CAD, LAD, LCX and RCA |
| Physiological breakdown | Measurements vs reference ranges, with each one's contribution to the prediction |
| Disclaimer | Always-visible banner, first-visit modal, and a note on the printed report |
| No dedicated GPU needed | Low-poly GLB (~150 KB), no post-processing effects |

---

## Architecture

```
 Patient data (form / sample / CSV / JSON)
                │
                ▼
   React 18 + Vite frontend ──────────────┐
   (R3F 3D heart, SHAP, tables)           │ POST /predict
                ▲                         ▼
                │              FastAPI backend (port 8000)
   probabilities + SHAP       ├─ preprocessing pipeline (joblib)
                └──────────── ├─ 4 XGBoost models (CAD, LAD, LCX, RCA)
                              └─ SHAP TreeExplainer
```

```
.
├── backend/          FastAPI app (main.py), request schema, tests
├── frontend/         React + Vite + Tailwind + React Three Fiber
│   └── public/models/heart.glb
├── model/
│   ├── aortavision_preprocessing.py   preprocessing + feature engineering
│   ├── notebooks/                     training / evaluation scripts
│   └── serialized/                    trained weights (model_*.json, preprocessor.joblib)
├── dataset/          dataset + data dictionary + data-quality report
├── metrics/          evaluation CSVs, ROC curves, confusion matrices, threshold sweeps
├── explainability/   SHAP plots and feature-contribution CSVs
├── docs/             CANONICAL_METRICS.md (how the metrics were produced)
└── scripts/          check_metrics_consistency.py
```

---

## Setup

### Prerequisites
- **Python** 3.10 or newer
- **Node.js** 18 or newer (with npm)
- No GPU required

### 1. Backend (port 8000)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Check it: open http://localhost:8000/health → `{"status":"healthy"}`. API docs are at http://localhost:8000/docs.

### 2. Frontend (port 5173)
```bash
cd frontend
cp .env.example .env               # contains VITE_API_URL=http://localhost:8000
npm install
npm run dev
```
Open **http://localhost:5173**.

### Deploy the frontend to Netlify

Connect the repository to Netlify and deploy using the included `netlify.toml`. It sets the frontend directory as the build base, runs `npm run build`, publishes `dist`, and routes client-side URLs back to the app. If you entered build settings manually, use `frontend` as the base directory, `npm run build` as the build command, and `dist` as the publish directory.

Netlify hosts the static frontend, not the FastAPI prediction backend. To enable predictions, deploy the backend separately, set Netlify's `VITE_API_URL` environment variable to that backend's public URL, and set the backend's `FRONTEND_ORIGIN` variable to your Netlify site URL. Redeploy after changing `VITE_API_URL`.

### 3. Tests and production build (optional)
```bash
pip install pytest httpx
python -m pytest backend/test_backend.py     # run from the repo root
cd frontend && npm run build
```

---

## How to use

1. Open the app and accept the disclaimer.
2. On the home page, choose **Use sample patient data** (High Risk / Low Risk / Borderline), or upload a **CSV** (one row) or **JSON** with the 54 raw feature names.
3. Click **Analyze**.
4. On the analysis page:
   - **Rotate** (drag), **zoom** (scroll / pinch) the 3D heart.
   - **Hover** a vessel to see its probability, **click** it to pin the detail card.
   - Switch the **SHAP tab** (CAD / LAD / LCX / RCA) to see why that risk was predicted. Red bars raise risk, blue bars lower it.
   - Read the **physiological table** for each measurement's value, reference range and contribution.
   - Check the **validation metrics** panel.
   - Use **Export Report** to print or save a PDF.

### API
| Endpoint | Purpose |
|---|---|
| `GET /health` | Model and preprocessor load status |
| `GET /models/status` | Per-target model info and thresholds |
| `GET /metrics` | Repeated-CV and holdout metrics per target |
| `POST /predict` | Body `{"features": {...54 raw features...}}` → probabilities, thresholds, risk levels, SHAP contributions |

Full contract: [`BACKEND_CONTRACT.md`](./BACKEND_CONTRACT.md).

---

## Data and preprocessing

- **Dataset:** Z-Alizadeh Sani coronary artery disease dataset (UCI Machine Learning Repository), `dataset/dataset.csv`.
- **Inputs (54 raw features):** demographics, history, vitals/exam, symptoms, ECG, laboratory values, echocardiography (e.g. Age, BMI, BP, PR, FBS, LDL, HDL, ST elevation/depression, T inversion, LVH, EF-TTE, Region RWMA).
- **Targets:** `CAD` (derived from `Cath == CAD`), `LAD`, `LCX`, `RCA` (Stenotic / Normal).
- **Leakage control:** `CAD`, `LAD`, `LCX`, `RCA` and `Cath` are never model inputs.
- **Cleaning:** category spelling fixes, constant column (`Exertional CP`) removed, robust scaling for numeric features, one-hot encoding for categorical features, derived features (risk-factor count, LDL/HDL ratio, etc.). Details: [`dataset/data_quality_report.md`](./dataset/data_quality_report.md), [`dataset/data_dictionary.md`](./dataset/data_dictionary.md).

> **TODO before submitting:** the task description says the dataset has 303 records, but `dataset/dataset.csv` has **378**. State here exactly where the extra rows come from, or replace the file with the original 303-row version and re-run the metrics.

---

## Models and evaluation

- **Algorithm:** one XGBoost binary classifier per target, with a per-target decision threshold chosen on training data only.
- **Validation:** 5-fold stratified cross-validation repeated 10 times (seed 42), preprocessing fitted inside each fold, plus one fixed multi-label-balanced holdout (n≈76). Repeated CV is the headline estimate; the holdout is reported separately.

| Target | CV accuracy (mean ± SD) | CV ROC-AUC (mean ± SD) | Holdout accuracy | Holdout ROC-AUC |
|---|---:|---:|---:|---:|
| CAD | 0.890 ± 0.008 | 0.935 ± 0.005 | 0.816 | 0.949 |
| LAD | 0.758 ± 0.005 | 0.836 ± 0.009 | 0.776 | 0.906 |
| LCX | 0.678 ± 0.009 | 0.819 ± 0.009 | 0.724 | 0.832 |
| RCA | 0.703 ± 0.010 | 0.802 ± 0.009 | 0.776 | 0.877 |

Precision, recall, F1, PR-AUC, Brier score and 95% intervals are in [`metrics/repeated_cv_summary.csv`](./metrics/repeated_cv_summary.csv) and [`metrics/classification_metrics.csv`](./metrics/classification_metrics.csv). How they were produced: [`docs/CANONICAL_METRICS.md`](./docs/CANONICAL_METRICS.md).

**Verify the README numbers against the CSVs:**
```bash
python scripts/check_metrics_consistency.py
```

**Limitations:** single-centre data, small sample size, thresholds favour recall, LCX and RCA are the weakest targets. Results may not generalise to other hospitals.

---

## 3D pipeline

- **Model:** `frontend/public/models/heart.glb` (open-source heart mesh, optimised for the web). If it fails to load, the app falls back to a procedural heart.
- **Arteries:** LAD, LCX and RCA are tube meshes drawn over the heart. Colour comes from the predicted probability: green (< 50%), yellow (50–70%), red (> 70%).
- **Interaction:** orbit controls (rotate, zoom), hover tooltips, click-to-pin vessel cards, legend toggle, fullscreen.
- **Correspondence:** the same `LAD / LCX / RCA` keys are used by the model outputs, the 3D meshes, the SHAP tabs and the dashboard cards.
- **Extending:** to add a feature, add it to the feature config and reference ranges; to add a vessel, add a mesh entry and a model target. No redesign needed.

---

## Reproducing the models

```bash
pip install -r model/requirements.txt
python model/notebooks/improve_metrics.py --repeats 1 --inner-folds 2 --trials 0   # quick smoke test
```
Trained weights are already included in `model/serialized/`, so training is **not** required to run the app.

> **TODO before submitting:** add the script that trains the shipped models and the repeated-CV evaluation script, and link them here.

---

## Resources

- Dataset: Z-Alizadeh Sani, UCI Machine Learning Repository
- 3D heart mesh: `<name, author and licence of heart.glb>`
- Libraries: XGBoost, scikit-learn, SHAP, FastAPI, React, Vite, Three.js, React Three Fiber, drei, Recharts, Tailwind CSS

## Licence

`<MIT or similar — add text to the LICENSE file>`