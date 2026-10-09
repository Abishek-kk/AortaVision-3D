# CardioVision 3D (AortaVision 3D)

> **Multimodal AI Hackathon 2026 — Track A: Cardiovascular Risk Visualization & Prediction**

CardioVision 3D is a clinical decision support prototype that combines multi-target machine learning predictions (XGBoost models for overall **CAD**, **LAD**, **LCX**, and **RCA** stenosis risk) with an interactive 3D heart visualization and SHAP explainability dashboard.

---

## Architecture Overview

```
 +-----------------------------------------------------------------------------------+
 |                               CardioVision 3D Frontend                             |
 |                              (React 18 + Vite + JSX)                              |
 +----------------------------------------+------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
      +---------v----------+                               +--------v---------+
      | CSV/JSON Patient   |                               | 3D Scene Canvas  |
      | Upload / Sample    |                               | (Three.js / R3F) |
      +---------+----------+                               +--------+---------+
                |                                                   |
                +-------------------------+-------------------------+
                                          |
                                 +--------v--------+
                                 | Patient Data    |
                                 | Context         |
                                 +--------+--------+
                                          |
                                 +--------v--------+
                                 | Fetch API       |
                                 | POST /predict   |
                                 +--------+--------+
                                          |
                                 +--------v--------+
                                 | FastAPI Backend |
                                 | port 8000       |
                                 +-----------------+
```

---

## Key Features

1. **Interactive 3D Coronary Anatomy**: Color-coded 3D heart with animated Left Anterior Descending (**LAD**), Left Circumflex (**LCX**), and Right Coronary Artery (**RCA**). Risk colors dynamically lerp between Green (Low), Yellow (Moderate), Orange (High), and Red (Very High) based on model probabilities.
2. **Clinical Data Input**: Upload a single patient record as CSV or JSON, or use the synthetic sample patient record.
3. **Explainable AI (SHAP) Dashboard**: Top 10 SHAP feature contributions per target variable, selectable across CAD, LAD, LCX, and RCA, with directional tooltips and target-specific summary sentences.
4. **Physiological Reference Table**: Submitted patient measurements compared against general adult reference ranges with status indicators and bidirectional SHAP risk contribution bars for the selected target.
5. **Model Validation Panel**: Displays API-provided repeated-CV and single-holdout metrics alongside selected-threshold precision and recall.
6. **Print & PDF Export**: The Analyze page's **Export Report** button opens the browser print dialog. The print layout includes a 3D canvas snapshot, all four target SHAP charts, the physiological and metrics tables, and the full disclaimer; choose “Save as PDF” in the browser to create a PDF.
7. **Strict Target Leakage Protection**: Enforces runtime assertions preventing targets (`CAD`, `LAD`, `LCX`, `RCA`, `Cath`) from being passed as model input features.
8. **First-Visit Disclaimer Modal**: Requires acknowledgement of the decision-support limitations, schematic 3D anatomy, and small single-centre training dataset before continuing on the first visit.

---

## Model performance

Repeated cross-validation is the headline estimate: 5-fold stratified CV repeated 10 times, with the
preprocessor fitted inside each fold. Values are mean ± SD [95% percentile CI] over the 10
repeat-level means. The single holdout is a separate, one-split result (n≈76); it is not the repeated-CV
estimate and was not used to select the displayed thresholds.

| Target | Repeated-CV accuracy: mean ± SD [95% CI] | Repeated-CV ROC-AUC: mean ± SD [95% CI] | Single holdout accuracy (n≈76) | Single holdout ROC-AUC (n≈76) |
|---|---:|---:|---:|---:|
| CAD | 0.890 ± 0.008 [0.881, 0.902] | 0.935 ± 0.005 [0.926, 0.942] | 0.816 | 0.949 |
| LAD | 0.758 ± 0.005 [0.751, 0.767] | 0.836 ± 0.009 [0.824, 0.848] | 0.776 | 0.906 |
| LCX | 0.678 ± 0.009 [0.664, 0.690] | 0.819 ± 0.009 [0.809, 0.835] | 0.724 | 0.832 |
| RCA | 0.703 ± 0.010 [0.689, 0.718] | 0.802 ± 0.009 [0.788, 0.817] | 0.776 | 0.877 |

The repeated-CV values come from [`metrics/repeated_cv_summary.csv`](./metrics/repeated_cv_summary.csv);
the single-holdout values come from [`metrics/classification_metrics.csv`](./metrics/classification_metrics.csv).
See [`docs/CANONICAL_METRICS.md`](./docs/CANONICAL_METRICS.md) for generation details and provenance.

---

## Quick Start & Setup

### Prerequisites
- **Node.js**: v18.0+
- **Python**: 3.10+ (for backend)

### 1. Start the ML Backend (Port 8000)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend Dev Server (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Environment Variables

Copy [`frontend/.env.example`](./frontend/.env.example) to `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

- `VITE_API_URL`: Base URL of the FastAPI backend service.

The frontend sends one patient record to `POST /predict`. Upload a CSV containing
one data row with the exact raw feature names, or a JSON object containing those
features (either directly or under a `features` property). PDF extraction is not
implemented. The “Use sample patient data” option submits a complete synthetic
record through the same live backend request.

---

## Connecting to the Backend

The frontend communicates with the backend via REST endpoints described in [`BACKEND_CONTRACT.md`](./BACKEND_CONTRACT.md):
- `GET /health`: Health check for model & preprocessor loading.
- `GET /metrics`: Repeated-CV summaries and single-holdout per-target metrics.
- `POST /predict`: Primary inference route accepting 54 raw features and returning predictions for `CAD`, `LAD`, `LCX`, `RCA` along with SHAP explanations.

---

## How to verify reported metrics

From the repository root, verify that the performance table above matches the canonical CSVs
(values are rounded to three decimal places):

```bash
python scripts/check_metrics_consistency.py
```

The check exits non-zero if a README value is missing, stale, or differs from its source CSV by
more than the displayed rounding tolerance.

---

## Customization Guide

### How to Add a New Feature
1. To add a measurement to the physiological reference table, add an entry with its exact raw feature `key`, `label`, `unit`, `low`, and `high` to [`frontend/src/config/referenceRanges.js`](./frontend/src/config/referenceRanges.js).
2. Keep synthetic example records in [`frontend/src/data/samplePatient.js`](./frontend/src/data/samplePatient.js) aligned with any supported feature additions.
3. Adding a reference-range entry alone does not add a model input. Prediction features must be supported by the existing backend feature contract and model artifacts.

### How to Add/Modify Artery Coordinates
1. Open [`frontend/src/config/arteries.ts`](./frontend/src/config/arteries.ts).
2. Edit 3D path coordinates (`main` and `branches`) or `labelAnchor` for `LAD`, `LCX`, or `RCA`.
3. Use debug mode by visiting `http://localhost:5173/?debug=1` to visualize control points and export coordinates.

### How to Swap in a Custom 3D Heart GLB Model
1. Place your `heart.glb` file into `frontend/public/models/heart.glb`.
2. Follow instructions in [`frontend/public/models/README.md`](./frontend/public/models/README.md) regarding polycount and Draco compression.
3. If missing or invalid, the app automatically falls back to the high-performance procedural heart.

---

## Clinical Safety Disclaimers

> **IMPORTANT DISCLAIMER**
> CardioVision 3D is a research prototype for decision support and educational purposes only.
> It is **NOT** a medical diagnostic tool and does **NOT** replace formal diagnostic imaging (e.g., coronary angiography, CT angiography) or clinical judgment by a qualified healthcare professional.
> Coronary artery positions depicted in the 3D scene are schematic representations of ML model probability scores, not imaging-based lesion maps.
> Models were trained on the Z-Alizadeh Sani dataset (303 patient records). Predictions may not generalize to other clinical cohorts.

---

## Demo Video Checklist

- [x] **Input Flow**: Show sample patient selection (High Risk vs Low Risk) and clicking "Analyze Patient".
- [x] **3D Heart Interaction**: Rotate, zoom, hover over LAD/LCX/RCA labels, observe smooth color transition and stenosis narrowing.
- [x] **SHAP Explainability**: Toggle tabs between `CAD`, `LAD`, `LCX`, `RCA` and highlight key risk drivers.
- [x] **Physiological Table & Metrics**: Show reference range indicators and model validation panel.
- [x] **Disclaimers**: Confirm warning banner, first-visit modal, and 3D schematic note are clearly visible.
