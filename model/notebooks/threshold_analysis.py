"""Analyze selected production thresholds using training OOF predictions only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

import final_rebuild_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "dataset" / "dataset.csv"
METADATA_PATH = ROOT / "model" / "serialized" / "model_metadata.json"
PREPROCESSOR_PATH = ROOT / "model" / "serialized" / "preprocessor.joblib"
MODEL_DIR = ROOT / "model" / "serialized"
METRICS_DIR = ROOT / "metrics"
PLOT_DIR = METRICS_DIR / "thresholds"
SWEEP_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def read_metadata() -> dict[str, Any]:
    with METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    total = len(truth)
    return {
        "Precision": float(precision_score(truth, prediction, zero_division=0)),
        "Recall": float(recall_score(truth, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1": float(f1_score(truth, prediction, zero_division=0)),
        "F2": float(fbeta_score(truth, prediction, beta=2, zero_division=0)),
        "Accuracy": float(accuracy_score(truth, prediction)),
        "False_Negatives_per_100": float(fn / total * 100) if total else 0.0,
        "False_Positives_per_100": float(fp / total * 100) if total else 0.0,
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "True_Negatives": int(tn),
        "False_Negatives": int(fn),
    }


def youden_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    valid = np.isfinite(thresholds)
    if not valid.any():
        raise RuntimeError("ROC curve did not yield any finite candidate thresholds.")
    youden_j = true_positive_rate[valid] - false_positive_rate[valid]
    valid_thresholds = thresholds[valid]
    best_j = np.max(youden_j)
    # roc_curve returns thresholds descending, so the first tie is the more specific cutoff.
    return float(valid_thresholds[np.flatnonzero(youden_j == best_j)[0]])


def build_training_predictions(
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    cleaned = df.copy()
    if "Sex" in cleaned and "Fmale" in set(cleaned["Sex"].astype(str)):
        cleaned["Sex"] = cleaned["Sex"].replace({"Fmale": "Female"})
    if "VHD" in cleaned and "mild" in set(cleaned["VHD"].astype(str)):
        cleaned["VHD"] = cleaned["VHD"].replace({"mild": "Mild"})

    y_all = pipeline.derive_targets(cleaned)
    feature_names = metadata["raw_feature_names"]
    X_all = cleaned[feature_names].copy()
    train_idx, test_idx, _ = pipeline.deterministic_multilabel_holdout(
        y_all, pipeline.TEST_SIZE, pipeline.RANDOM_STATE
    )
    X_train = X_all.loc[train_idx].copy()
    X_test = X_all.loc[test_idx].copy()
    y_train = y_all.loc[train_idx].copy()
    y_test = y_all.loc[test_idx].copy()
    cv = StratifiedKFold(
        n_splits=pipeline.N_SPLITS,
        shuffle=True,
        random_state=pipeline.RANDOM_STATE,
    )
    out_of_fold: dict[str, np.ndarray] = {}

    for target in pipeline.TARGETS:
        selected = metadata["selected_models"][target]
        candidate = next(
            (
                item
                for item in pipeline.candidate_pipelines(
                    metadata["numeric_features"],
                    metadata["categorical_features"],
                    y_train[target],
                )
                if item["algorithm"] == selected["algorithm"]
                and item["strategy"] == selected["strategy"]
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(f"No matching selected estimator found for {target}.")
        out_of_fold[target] = cross_val_predict(
            candidate["estimator"],
            X_train,
            y_train[target],
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    return X_test, y_train, y_test, out_of_fold


def summarize_target(
    target: str,
    y_train: pd.Series,
    oof_probability: np.ndarray,
    y_test: pd.Series,
    holdout_probability: np.ndarray,
    production_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweep_rows = [
        {"Threshold": float(threshold), **threshold_metrics(y_train, oof_probability, float(threshold))}
        for threshold in SWEEP_THRESHOLDS
    ]
    sweep = pd.DataFrame(sweep_rows)

    selected_threshold, _, _ = pipeline.choose_threshold(y_train, oof_probability)
    youden = youden_threshold(y_train, oof_probability)
    operating_points = {
        "production": threshold_metrics(y_train, oof_probability, production_threshold),
        "default_0_5": threshold_metrics(y_train, oof_probability, 0.5),
        "youden_j_optimum": threshold_metrics(y_train, oof_probability, youden),
    }
    holdout_point = threshold_metrics(y_test, holdout_probability, production_threshold)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for column, label in (("Precision", "Precision"), ("Recall", "Recall"), ("Specificity", "Specificity")):
        axis.plot(sweep["Threshold"], sweep[column], marker="o", markersize=3, label=label)
    axis.axvline(production_threshold, color="#dc2626", linestyle="--", linewidth=1.5,
                 label=f"Production {production_threshold:.2f}")
    axis.axvline(0.5, color="#475569", linestyle=":", linewidth=1.5, label="Default 0.50")
    axis.set(title=f"{target} threshold trade-offs (training OOF)", xlabel="Decision threshold", ylabel="Rate")
    axis.set_xlim(0.05, 0.95)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target.lower()}_tradeoff.png", dpi=180)
    plt.close(fig)

    summary = {
        "production_threshold": float(production_threshold),
        "production_threshold_matches_current_rule": bool(np.isclose(production_threshold, selected_threshold)),
        "threshold_from_current_rule": float(selected_threshold),
        "current_rule_differs_from_production_threshold": not bool(np.isclose(production_threshold, selected_threshold)),
        "objective": pipeline.THRESHOLD_OBJECTIVE,
        "tie_breakers": list(pipeline.THRESHOLD_TIE_BREAKERS[1:]),
        "recall_floor": pipeline.THRESHOLD_RECALL_FLOOR,
        "production_operating_point": operating_points["production"],
        "default_0_5_operating_point": operating_points["default_0_5"],
        "youden_j_optimum": {
            "threshold": float(youden),
            **operating_points["youden_j_optimum"],
        },
        "holdout_report_only": {
            "threshold": float(production_threshold),
            **holdout_point,
        },
    }
    return sweep, summary


def markdown_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def make_report(rationales: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Decision-threshold analysis",
        "",
        "## Clinical rationale",
        "",
        "These thresholds are intended for screening and decision support, not diagnosis. A missed "
        "stenosis can delay further evaluation, while a false alarm generally prompts review or "
        "confirmatory testing. The operating point therefore favors sensitivity (recall) over precision, "
        "accepting additional false positives to reduce missed disease. Clinical evaluation and formal "
        "diagnostic testing remain necessary.",
        "",
        "## Selection rule and training-only comparison",
        "",
        "The production rule maximizes F1 on five-fold out-of-fold predictions from the training split "
        "over the rebuild grid 0.20–0.70 in 0.05 increments. Ties prefer higher recall, then precision, "
        "then accuracy. There is no explicit recall floor. The broader 0.05–0.95 sweep below is descriptive "
        "and does not alter the production thresholds. Youden-J maximizes sensitivity + specificity − 1 "
        "on the same training OOF predictions. The holdout is never used to select a threshold.",
        "",
        "| Target | Production threshold | Production OOF P / R / specificity / F1 | Default 0.50 P / R / specificity / F1 | Youden threshold | Youden OOF P / R / specificity / F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in rationales.items():
        production = item["production_operating_point"]
        default = item["default_0_5_operating_point"]
        youden = item["youden_j_optimum"]
        lines.append(
            f"| {target} | {item['production_threshold']:.2f} | "
            f"{markdown_pct(production['Precision'])} / {markdown_pct(production['Recall'])} / "
            f"{markdown_pct(production['Specificity'])} / {markdown_pct(production['F1'])} | "
            f"{markdown_pct(default['Precision'])} / {markdown_pct(default['Recall'])} / "
            f"{markdown_pct(default['Specificity'])} / {markdown_pct(default['F1'])} | "
            f"{youden['threshold']:.3f} | "
            f"{markdown_pct(youden['Precision'])} / {markdown_pct(youden['Recall'])} / "
            f"{markdown_pct(youden['Specificity'])} / {markdown_pct(youden['F1'])} |"
        )

    lines.extend(
        [
            "",
            "## Holdout report only",
            "",
            "The following values are descriptive performance at the saved production threshold; "
            "they were not used to choose or revise thresholds.",
            "",
            "| Target | Threshold | Precision | Recall | Specificity | False negatives / 100 | False positives / 100 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, item in rationales.items():
        report = item["holdout_report_only"]
        lines.append(
            f"| {target} | {report['threshold']:.2f} | {markdown_pct(report['Precision'])} | "
            f"{markdown_pct(report['Recall'])} | {markdown_pct(report['Specificity'])} | "
            f"{report['False_Negatives_per_100']:.1f} | {report['False_Positives_per_100']:.1f} |"
        )

    differences = [
        f"{target} (saved {item['production_threshold']:.2f}, current rule {item['threshold_from_current_rule']:.2f})"
        for target, item in rationales.items()
        if item["current_rule_differs_from_production_threshold"]
    ]
    lines.extend(
        [
            "",
            "## Consistency check",
            "",
            (
                f"The documented F1 rule selects a different threshold for: {', '.join(differences)}. "
                "Saved production thresholds were left unchanged; this difference should be reconciled before "
                "changing deployed behavior."
                if differences
                else "The documented F1 rule reproduces all saved production thresholds; no thresholds were changed."
            ),
            "",
            "Complete 0.05–0.95 sweep metrics and trade-off figures are in the matching "
            "`threshold_sweep_*.csv` files and `thresholds/*_tradeoff.png` figures.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata()
    actual_hash = sha256(DATASET_PATH)
    expected_hash = metadata.get("dataset_sha256")
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError("Dataset hash does not match model_metadata.json; refusing to analyze a different dataset.")

    df = pd.read_csv(DATASET_PATH)
    X_test, y_train, y_test, oof_predictions = build_training_predictions(df, metadata)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_processed = preprocessor.transform(X_test)
    rationale_by_target: dict[str, dict[str, Any]] = {}

    for target in pipeline.TARGETS:
        production_threshold = float(metadata["selected_thresholds"][target])
        model = XGBClassifier()
        model.load_model(str(MODEL_DIR / f"model_{target.lower()}.json"))
        holdout_probability = model.predict_proba(X_test_processed)[:, 1]
        sweep, rationale = summarize_target(
            target,
            y_train[target],
            oof_predictions[target],
            y_test[target],
            holdout_probability,
            production_threshold,
        )
        sweep.to_csv(METRICS_DIR / f"threshold_sweep_{target.lower()}.csv", index=False)
        rationale_by_target[target] = rationale

    metadata["threshold_rationale"] = rationale_by_target
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (METRICS_DIR / "threshold_report.md").write_text(make_report(rationale_by_target), encoding="utf-8")


if __name__ == "__main__":
    main()
