"""Reusable preprocessing helpers for AortaVision 3D training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic, leakage-free clinical aggregate features."""

    engineered_features = (
        "Risk Factor Count",
        "Symptom Burden",
        "ECG Burden",
        "LDL HDL Ratio",
        "TG HDL Ratio",
        "Neut Lymph Ratio",
        "Platelet Lymph Ratio",
        "Age BMI Product",
        "BP PR Product",
        "Renal Load",
        "Inflammation Load",
        "EF RWMA Interaction",
    )

    def fit(self, X: pd.DataFrame, y: object = None) -> "ClinicalFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        def numeric(column: str) -> pd.Series:
            if column not in frame:
                return pd.Series(0.0, index=frame.index)
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        def yes(column: str) -> pd.Series:
            if column not in frame:
                return pd.Series(0.0, index=frame.index)
            values = frame[column].astype(str).str.strip().str.casefold()
            return values.isin({"y", "yes", "1", "true", "moderate", "severe", "mild", "lbbb", "rbbb"}).astype(float)

        def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            denominator = denominator.replace(0, np.nan)
            return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        risk_columns = ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH"]
        risk_count = sum(numeric(column) for column in risk_columns)
        risk_count = risk_count + yes("Obesity") + yes("CRF") + yes("DLP")

        symptom_count = (
            numeric("Typical Chest Pain")
            + yes("Dyspnea")
            + yes("Atypical")
            + yes("Exertional CP")
            + yes("LowTH Ang")
        )
        ecg_count = (
            numeric("Q Wave")
            + numeric("St Elevation")
            + numeric("St Depression")
            + numeric("Tinversion")
            + yes("LVH")
            + yes("Poor R Progression")
            + yes("BBB")
        )

        frame["Risk Factor Count"] = risk_count
        frame["Symptom Burden"] = symptom_count
        frame["ECG Burden"] = ecg_count
        frame["LDL HDL Ratio"] = safe_ratio(numeric("LDL"), numeric("HDL"))
        frame["TG HDL Ratio"] = safe_ratio(numeric("TG"), numeric("HDL"))
        frame["Neut Lymph Ratio"] = safe_ratio(numeric("Neut"), numeric("Lymph"))
        frame["Platelet Lymph Ratio"] = safe_ratio(numeric("PLT"), numeric("Lymph"))
        frame["Age BMI Product"] = numeric("Age") * numeric("BMI")
        frame["BP PR Product"] = numeric("BP") * numeric("PR")
        frame["Renal Load"] = numeric("CR") * numeric("BUN")
        frame["Inflammation Load"] = numeric("ESR") * numeric("WBC")
        frame["EF RWMA Interaction"] = numeric("EF-TTE") * numeric("Region RWMA")
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(self.engineered_features, dtype=object)
        names = [str(feature) for feature in input_features]
        for feature in self.engineered_features:
            if feature not in names:
                names.append(feature)
        return np.asarray(names, dtype=object)
