# CardioVision 3D - Backend API Contract & Specification

This document details the exact contract between the ML FastAPI backend and the CardioVision 3D frontend.

---

## 1. Endpoints & Server Configuration

- **Base URL**: `http://localhost:8000` (configurable via `VITE_API_URL` env variable)
- **Framework**: FastAPI (Python 3.10+) running with Uvicorn.
- **CORS Configuration**: Enabled via `CORSMiddleware` in `backend/main.py` allowing `http://localhost:5173` (or origins specified in `FRONTEND_ORIGIN`).
- **Frontend upload shape**: One CSV data row or a JSON feature object; PDF extraction and multi-record batch prediction are not supported.

### Routes
| Route | Method | Description |
|---|---|---|
| `/` | `GET` | API Service status |
| `/health` | `GET` | Health check for models & preprocessor |
| `/models/status` | `GET` | Loaded models info and thresholds |
| `/metrics` | `GET` | Overall & per-target model performance metrics |
| `/predict` | `POST` | Primary inference route for 4 targets (`CAD`, `LAD`, `LCX`, `RCA`) & SHAP explanations |

---

## 2. Request Contract (`POST /predict`)

### Body Format
```json
{
  "features": {
    "Age": 65,
    "Sex": "Male",
    "Weight": 78,
    "Length": 172,
    "BMI": 26.37,
    "DM": 1,
    "HTN": 1,
    "Current Smoker": 1,
    "EX-Smoker": 0,
    "FH": 1,
    "Obesity": "Y",
    "CRF": "N",
    "CVA": "N",
    "Airway disease": "N",
    "Thyroid Disease": "N",
    "CHF": "N",
    "DLP": "Y",
    "BP": 135,
    "PR": 78,
    "Edema": 0,
    "Weak Peripheral Pulse": "N",
    "Lung rales": "N",
    "Systolic Murmur": "N",
    "Diastolic Murmur": "N",
    "Typical Chest Pain": 1,
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
    "FBS": 115,
    "CR": 0.9,
    "TG": 180,
    "LDL": 145,
    "HDL": 38,
    "BUN": 14,
    "ESR": 18,
    "HB": 14.2,
    "K": 4.2,
    "Na": 140,
    "WBC": 7200,
    "Lymph": 32,
    "Neut": 60,
    "PLT": 240,
    "EF-TTE": 55,
    "Region RWMA": 0,
    "VHD": "N"
  }
}
```

### Input Field Encodings
- **Categorical (String `Y`/`N`)**: `Obesity`, `CRF`, `CVA`, `Airway disease`, `Thyroid Disease`, `CHF`, `DLP`, `Weak Peripheral Pulse`, `Lung rales`, `Systolic Murmur`, `Diastolic Murmur`, `Dyspnea`, `Atypical`, `Nonanginal`, `LowTH Ang`, `LVH`, `Poor R Progression`.
- **Categorical (Special Strings)**:
  - `Sex`: `"Male"` or `"Female"`
  - `BBB`: `"N"`, `"LBBB"`, or `"RBBB"`
  - `VHD`: `"N"`, `"Mild"`, `"Moderate"`, or `"Severe"`
- **Binary Numeric (0 or 1)**: `DM`, `HTN`, `Current Smoker`, `EX-Smoker`, `FH`, `Edema`, `Typical Chest Pain`, `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`.
- **Numeric Continuous**: `Age`, `Weight`, `Length`, `BMI`, `BP`, `PR`, `FBS`, `CR`, `TG`, `LDL`, `HDL`, `BUN`, `ESR`, `HB`, `K`, `Na`, `WBC`, `Lymph`, `Neut`, `PLT`, `EF-TTE`.
- **Numeric Ordinal**: `Function Class` (0-4), `Region RWMA` (0-4).

### Target Leakage Protection
The following target variables are **strictly forbidden** in request input payloads: `CAD`, `LAD`, `LCX`, `RCA`, `Cath`. Sending any of these will result in an HTTP 422 error.

---

## 3. Response Contract (`POST /predict`)

```json
{
  "success": true,
  "predictions": {
    "CAD": {
      "probability": 0.852,
      "threshold": 0.35,
      "prediction": 1,
      "risk_level": "High"
    },
    "LAD": {
      "probability": 0.781,
      "threshold": 0.30,
      "prediction": 1,
      "risk_level": "High"
    },
    "LCX": {
      "probability": 0.324,
      "threshold": 0.25,
      "prediction": 1,
      "risk_level": "Moderate"
    },
    "RCA": {
      "probability": 0.648,
      "threshold": 0.35,
      "prediction": 1,
      "risk_level": "Moderate"
    }
  },
  "explanations": {
    "CAD": [
      {
        "feature": "Age",
        "contribution": 0.421,
        "direction": "increases_risk",
        "value": 65
      },
      {
        "feature": "EF-TTE",
        "contribution": -0.315,
        "direction": "decreases_risk",
        "value": 55
      }
    ],
    "LAD": [ ... ],
    "LCX": [ ... ],
    "RCA": [ ... ]
  },
  "disclaimer": "This system is a research prototype..."
}
```

---

## 4. Training Order of 56 Input Features

1. `Age`
2. `Weight`
3. `Length`
4. `Sex`
5. `BMI`
6. `DM`
7. `HTN`
8. `Current Smoker`
9. `EX-Smoker`
10. `FH`
11. `Obesity`
12. `CRF`
13. `CVA`
14. `Airway disease`
15. `Thyroid Disease`
16. `CHF`
17. `DLP`
18. `BP`
19. `PR`
20. `Edema`
21. `Weak Peripheral Pulse`
22. `Lung rales`
23. `Systolic Murmur`
24. `Diastolic Murmur`
25. `Typical Chest Pain`
26. `Dyspnea`
27. `Function Class`
28. `Atypical`
29. `Nonanginal`
30. `LowTH Ang`
31. `Q Wave`
32. `St Elevation`
33. `St Depression`
34. `Tinversion`
35. `LVH`
36. `Poor R Progression`
37. `BBB`
38. `FBS`
39. `CR`
40. `TG`
41. `LDL`
42. `HDL`
43. `BUN`
44. `ESR`
45. `HB`
46. `K`
47. `Na`
48. `WBC`
49. `Lymph`
50. `Neut`
51. `PLT`
52. `EF-TTE`
53. `Region RWMA`
54. `VHD`

*(Note: `Exertional CP` was removed during data cleaning as it was constant in training data).*
