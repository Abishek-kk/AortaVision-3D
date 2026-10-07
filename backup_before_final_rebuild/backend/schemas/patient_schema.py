"""Request schema for patient prediction input features.

This schema intentionally follows the dataset-driven feature list in
``dataset/dataset.csv`` and explicitly excludes the leakage/target columns:
CAD, LAD, LCX, RCA, and Cath.

If the training notebook later defines a different final feature order, update
``DATASET_FEATURE_COLUMNS`` below or set AORTAVISION_FEATURE_ORDER before
launching the backend. Do not add target columns back into the model input.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

FeatureValue: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr
_TARGET_COLUMNS = frozenset({"cad", "lad", "lcx", "rca", "cath"})


def _load_dataset_feature_columns() -> tuple[str, ...]:
    """Read the dataset header and exclude all leakage/target columns."""
    dataset_path = Path(__file__).resolve().parents[2] / "dataset" / "dataset.csv"
    if not dataset_path.is_file():
        return tuple()

    with dataset_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration:
            return tuple()

    columns: list[str] = []
    for column_name in header:
        cleaned = column_name.strip()
        if not cleaned:
            continue
        if cleaned.casefold() in _TARGET_COLUMNS:
            continue
        columns.append(cleaned)
    return tuple(columns)


def _resolve_feature_columns() -> tuple[str, ...]:
    env_order = tuple(
        feature.strip()
        for feature in os.getenv("AORTAVISION_FEATURE_ORDER", "").split(",")
        if feature.strip()
    )
    if env_order:
        return env_order
    return _load_dataset_feature_columns()


DATASET_FEATURE_COLUMNS = _resolve_feature_columns()


class PatientRequest(BaseModel):
    """One patient's raw features before model-specific preprocessing."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        min_length=1,
        description=(
            "Map of exact model input feature names to scalar values. Target/leakage "
            "columns CAD, LAD, LCX, RCA, and Cath are prohibited."
        ),
    )

    @field_validator("features")
    @classmethod
    def reject_target_columns(
        cls, features: dict[str, FeatureValue]
    ) -> dict[str, FeatureValue]:
        blocked = [
            name
            for name in features
            if name.strip().casefold() in _TARGET_COLUMNS
        ]
        if blocked:
            raise ValueError(
                "Target/leakage columns cannot be prediction inputs: "
                + ", ".join(blocked)
            )

        if DATASET_FEATURE_COLUMNS:
            unknown = sorted(
                name for name in features if name not in DATASET_FEATURE_COLUMNS
            )
            if unknown:
                raise ValueError(
                    "Unsupported feature names: "
                    + ", ".join(unknown)
                    + ". Allowed features are the dataset-defined model inputs."
                )
        return features