"""Check that rounded performance figures in README.md match canonical CSVs."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
HOLDOUT_PATH = ROOT / "metrics" / "classification_metrics.csv"
REPEATED_CV_PATH = ROOT / "metrics" / "repeated_cv_summary.csv"
TARGETS = ("CAD", "LAD", "LCX", "RCA")
ESTIMATE_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*±\s*(-?\d+(?:\.\d+)?)\s*"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*$"
)
ROUNDING_TOLERANCE = 0.000500001


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_performance_table() -> dict[str, list[str]]:
    lines = README_PATH.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().startswith("| Target |")
            and "Repeated-CV accuracy" in line
            and "Single holdout accuracy" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("README performance table header was not found.")

    rows: dict[str, list[str]] = {}
    for line in lines[header_index + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) == 5 and cells[0] in TARGETS:
            rows[cells[0]] = cells[1:]
    return rows


def close_enough(reported: float, source: float) -> bool:
    return abs(reported - source) <= ROUNDING_TOLERANCE


def check_metrics() -> list[str]:
    holdout_rows = {row["Target"]: row for row in read_rows(HOLDOUT_PATH)}
    cv_rows = {
        (row["Target"], row["Metric"]): row
        for row in read_rows(REPEATED_CV_PATH)
    }
    readme_rows = read_performance_table()
    errors: list[str] = []

    missing = set(TARGETS).difference(readme_rows)
    if missing:
        errors.append(f"README table is missing targets: {', '.join(sorted(missing))}.")

    for target in TARGETS:
        if target not in readme_rows:
            continue
        if target not in holdout_rows:
            errors.append(f"Holdout CSV is missing {target}.")
            continue
        accuracy_text, roc_auc_text, holdout_accuracy_text, holdout_roc_auc_text = readme_rows[target]
        for metric, cell_text in (("Accuracy", accuracy_text), ("ROC_AUC", roc_auc_text)):
            row = cv_rows.get((target, metric))
            if row is None:
                errors.append(f"Repeated-CV CSV is missing {target} {metric}.")
                continue
            match = ESTIMATE_PATTERN.fullmatch(cell_text)
            if match is None:
                errors.append(f"README {target} {metric} cell is not a mean ± SD [CI] value: {cell_text!r}.")
                continue
            fields = ("Mean", "SD", "CI_95_Lower", "CI_95_Upper")
            for field, reported_text in zip(fields, match.groups()):
                if not close_enough(float(reported_text), float(row[field])):
                    errors.append(
                        f"README {target} {metric} {field}={reported_text} "
                        f"does not match CSV value {row[field]} within rounding."
                    )

        for metric, cell_text, csv_column in (
            ("Accuracy", holdout_accuracy_text, "Accuracy"),
            ("ROC_AUC", holdout_roc_auc_text, "ROC_AUC"),
        ):
            try:
                reported_value = float(cell_text)
            except ValueError:
                errors.append(f"README {target} holdout {metric} is not numeric: {cell_text!r}.")
                continue
            source_value = float(holdout_rows[target][csv_column])
            if not close_enough(reported_value, source_value):
                errors.append(
                    f"README {target} holdout {metric}={reported_value} "
                    f"does not match CSV value {source_value} within rounding."
                )
    return errors


def main() -> int:
    try:
        errors = check_metrics()
    except (OSError, KeyError, ValueError, csv.Error) as exc:
        print(f"Metric consistency check failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("Metric consistency check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("README performance table matches both canonical metric CSVs to displayed rounding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
