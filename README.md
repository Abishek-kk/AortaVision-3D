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
2. **Clinical Data Input Form**: Config-driven form with 54 raw clinical features covering Demographics, Medical History, Vitals & Exam, Symptoms, ECG, Lab Results, and Echocardiogram. Supports pre-set sample patient profiles (High Risk, Low Risk, Borderline).
3. **Explainable AI (SHAP) Dashboard**: Top 10 SHAP feature contributions per target variable with custom directional tooltips and auto-generated summary sentences explaining risk drivers.
4. **Physiological Reference Table**: Patient measurements compared against standard clinical reference ranges with status indicators and bidirectional SHAP risk contribution bars.
5. **Model Validation Panel**: Displays overall and per-target ML performance metrics (Accuracy ~90.4%, ROC-AUC ~0.948) and 2x2 confusion matrices.
6. **Print & PDF Export**: Clinical report view with snapshot of the 3D canvas and patient risk summaries.
7. **Strict Target Leakage Protection**: Enforces runtime assertions preventing targets (`CAD`, `LAD`, `LCX`, `RCA`, `Cath`) from being passed as model input features.

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
- `GET /metrics`: Model test metrics & confusion matrices.
- `POST /predict`: Primary inference route accepting 54 raw features and returning predictions for `CAD`, `LAD`, `LCX`, `RCA` along with SHAP explanations.

---

## Customization Guide

### How to Add a New Feature
1. Open [`frontend/src/config/features.ts`](./frontend/src/config/features.ts).
2. Add a new `FeatureDef` object to `FEATURES` array with `key`, `label`, `section`, `type`, `default`, `unit`, `min`, `max`, `step`.
3. If applicable, add standard reference range to [`frontend/src/config/referenceRanges.ts`](./frontend/src/config/referenceRanges.ts).

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
