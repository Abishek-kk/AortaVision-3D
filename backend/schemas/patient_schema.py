from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

FeatureValue: TypeAlias = bool | int | float | str
TARGET_COLUMNS = frozenset({"cad", "lad", "lcx", "rca", "cath"})


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_raw_feature_names() -> tuple[str, ...]:
    feature_names_path = _project_root() / "model" / "serialized" / "feature_names.json"
    if feature_names_path.is_file():
        try:
            payload = json.loads(feature_names_path.read_text(encoding="utf-8"))
            raw_names = payload.get("raw_feature_names", [])
            if isinstance(raw_names, list) and raw_names:
                return tuple(str(name) for name in raw_names)
        except (OSError, TypeError, ValueError):
            pass
    dataset_path = _project_root() / "dataset" / "dataset.csv"
    if dataset_path.is_file():
        try:
            with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
                header = csv_file.readline().strip()
            if header:
                names = [name.strip() for name in header.split(",") if name.strip()]
                return tuple(
                    name for name in names if name.casefold() not in TARGET_COLUMNS
                )
        except OSError:
            pass
    return tuple()


DATASET_FEATURE_COLUMNS = _load_raw_feature_names()


class PatientRequest(BaseModel):
    """One patient represented by the raw input features used during model training."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        min_length=1,
        description=(
            "Map of raw model input features to scalar values. CAD, LAD, LCX, RCA, and Cath are "
            "not valid prediction inputs."
        ),
    )

    @field_validator("features")
    @classmethod
    def validate_feature_keys(cls, features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        blocked = [name for name in features if name.strip().casefold() in TARGET_COLUMNS]
        if blocked:
            raise ValueError(
                "Target/leakage columns cannot be prediction inputs: " + ", ".join(blocked)
            )

        if not DATASET_FEATURE_COLUMNS:
            raise ValueError("Feature contract is unavailable; model artifacts could not be read.")

        unknown = sorted(name for name in features if name not in DATASET_FEATURE_COLUMNS)
        if unknown:
            raise ValueError(
                "Unsupported feature names: " + ", ".join(unknown) + ". Use only the raw model features."
            )
        return features
