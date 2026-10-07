# Dataset inspection and data dictionary

Source file audited: `dataset/dataset.csv`.

## Dataset summary

- Number of records: 378
- Number of columns: 59
- Numerical columns: 34
- Categorical/text columns: 25
- Missing values: 0 total
- Blank string values: 0 total
- Duplicate rows: 0 exact duplicate data rows
- Patient identifier column: none present; duplicate-patient checks beyond exact-row matching are not possible.
- Source `CAD` column: not present. The training pipeline derives CAD from `Cath == CAD`.
- Present label columns: `LAD`, `LCX`, `RCA`, `Cath`.
- Target/leakage columns excluded from model inputs: `CAD`, `LAD`, `LCX`, `RCA`, `Cath`.

## Target class counts

- `CAD` derived from `Cath`: `CAD` = 291, `Normal` = 87
- `LAD`: `Stenotic` = 238, `Normal` = 140
- `LCX`: `Normal` = 189, `Stenotic` = 189
- `RCA`: `Stenotic` = 189, `Normal` = 189
- `Cath`: `CAD` = 291, `Normal` = 87

## Column-level data dictionary

Descriptions are limited to the provided column names because no official codebook with units or clinical definitions was supplied. Numeric summaries describe the current CSV values.

| Column | Type | Observed values / summary |
|--------|------|---------------------------|
| `Age` | Numerical | min 30; max 86 |
| `Weight` | Numerical | min 48; max 120 |
| `Length` | Numerical | min 140; max 188 |
| `Sex` | Categorical | `Male`, `Female` |
| `BMI` | Numerical | min 18.1154; max 40.9007 |
| `DM` | Binary numeric | `0`, `1` |
| `HTN` | Binary numeric | `0`, `1` |
| `Current Smoker` | Binary numeric | `0`, `1` |
| `EX-Smoker` | Binary numeric | `0`, `1` |
| `FH` | Binary numeric | `0`, `1` |
| `Obesity` | Categorical | `Y`, `N` |
| `CRF` | Categorical | `Y`, `N` |
| `CVA` | Categorical | `Y`, `N` |
| `Airway disease` | Categorical | `Y`, `N` |
| `Thyroid Disease` | Categorical | `Y`, `N` |
| `CHF` | Categorical | `Y`, `N` |
| `DLP` | Categorical | `Y`, `N` |
| `BP` | Numerical | min 90; max 190 |
| `PR` | Numerical | min 50; max 110 |
| `Edema` | Binary numeric | `0`, `1` |
| `Weak Peripheral Pulse` | Categorical | `Y`, `N` |
| `Lung rales` | Categorical | `Y`, `N` |
| `Systolic Murmur` | Categorical | `Y`, `N` |
| `Diastolic Murmur` | Categorical | `Y`, `N` |
| `Typical Chest Pain` | Binary numeric | `0`, `1` |
| `Dyspnea` | Categorical | `Y`, `N` |
| `Function Class` | Numeric code | `0`, `1`, `2`, `3` |
| `Atypical` | Categorical | `Y`, `N` |
| `Nonanginal` | Categorical | `Y`, `N` |
| `Exertional CP` | Categorical | `N` only; removed from model inputs as a constant feature |
| `LowTH Ang` | Categorical | `Y`, `N` |
| `Q Wave` | Binary numeric | `0`, `1` |
| `St Elevation` | Binary numeric | `0`, `1` |
| `St Depression` | Binary numeric | `0`, `1` |
| `Tinversion` | Binary numeric | `0`, `1` |
| `LVH` | Categorical | `Y`, `N` |
| `Poor R Progression` | Categorical | `Y`, `N` |
| `BBB` | Categorical | `N`, `LBBB`, `RBBB` |
| `FBS` | Numerical | min 62; max 400 |
| `CR` | Numerical | min 0.5; max 2.2 |
| `TG` | Numerical | min 37; max 1050 |
| `LDL` | Numerical | min 18; max 232 |
| `HDL` | Numerical | min 15.9; max 111 |
| `BUN` | Numerical | min 6; max 52 |
| `ESR` | Numerical | min 1; max 90 |
| `HB` | Numerical | min 8.9; max 17.6 |
| `K` | Numerical | min 3.0; max 6.6 |
| `Na` | Numerical | min 128; max 156 |
| `WBC` | Numerical | min 3700; max 18000 |
| `Lymph` | Numerical | min 7; max 60 |
| `Neut` | Numerical | min 32; max 89 |
| `PLT` | Numerical | min 25; max 742 |
| `EF-TTE` | Numerical | min 15; max 60 |
| `Region RWMA` | Numeric code | `0`, `1`, `2`, `3`, `4` |
| `VHD` | Categorical | `N`, `Mild`, `Moderate`, `Severe` |
| `LAD` | Target label | `Normal`, `Stenotic` |
| `LCX` | Target label | `Normal`, `Stenotic` |
| `RCA` | Target label | `Normal`, `Stenotic` |
| `Cath` | Target/source label | `Normal`, `CAD` |

## Data-quality observations and handling notes

- No missing values, blank strings, or exact duplicate rows are present.
- `Exertional CP` is constant (`N`) in the current CSV and is removed from final model inputs.
- Several columns are near-constant, including `CHF`, `LowTH Ang`, `CVA`, `CRF`, `Weak Peripheral Pulse`, `Thyroid Disease`, `Edema`, and `EX-Smoker`.
- `Sex` was normalized from `Fmale` to `Female`.
- `VHD` was normalized from lowercase `mild` to `Mild`.
- One row has `Cath = Normal` while at least one vessel label is `Stenotic`; this should be manually verified against the source data before publication or clinical use.
- `TG` contains one value above 1000; this should be clinically/source verified before publication or clinical use.
- BMI values match `Weight / (Length/100)^2` within audit tolerance.

## Dataset usage

- This CSV is the current working dataset used by the saved model artifacts.
- Model training derives `CAD` from `Cath == CAD`; `Cath` itself is excluded from input features.
- `LAD`, `LCX`, `RCA`, and `Cath` are labels/leakage columns and must not be used as model input features.
- Preprocessing, imputation, one-hot encoding, thresholding, and model serialization are handled in `model/notebooks/final_xgb_optimization.py` and saved under `model/serialized/`.
