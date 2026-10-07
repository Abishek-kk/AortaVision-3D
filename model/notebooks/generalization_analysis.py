"""Compare the final holdout with repeated CV and evaluate holdout stability.

Run from the repository root with:
    python model/notebooks/generalization_analysis.py

The holdout is used only for the descriptive comparison and seed-stability
analysis. Regularization candidates are evaluated only on the seed-42 training
partition, with fixed repeated-CV splits and no holdout-based selection.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "dataset" / "dataset.csv"
METADATA_PATH = ROOT / "model" / "serialized" / "model_metadata.json"
SUMMARY_PATH = ROOT / "metrics" / "repeated_cv_summary.csv"
FOLDS_PATH = ROOT / "metrics" / "repeated_cv_folds.csv"
OVERFITTING_PATH = ROOT / "metrics" / "final_rebuild_overfitting.csv"
METRICS_DIR = ROOT / "metrics"

TEST_SIZE = 0.20
HOLDOUT_SEEDS = tuple(range(42, 242, 10))
REGULARIZATION_FOLDS = 5
REGULARIZATION_REPEATS = 5
REGULARIZATION_SEED = 42
TARGETS = ("CAD", "LAD", "LCX", "RCA")
LEAKAGE_COLUMNS = {"CAD", "LAD", "LCX", "RCA", "Cath"}

REGULARIZATION_GRID: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "shallow_high_child_weight",
        {
            "max_depth": 1,
            "min_child_weight": 8,
            "n_estimators": 300,
            "learning_rate": 0.02,
        },
    ),
    (
        "stronger_sampling_and_l2",
        {
            "max_depth": 2,
            "min_child_weight": 8,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 12.0,
            "n_estimators": 300,
            "learning_rate": 0.02,
        },
    ),
    (
        "stronger_l2",
        {
            "max_depth": 2,
            "min_child_weight": 5,
            "reg_lambda": 15.0,
            "n_estimators": 300,
            "learning_rate": 0.02,
        },
    ),
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required_paths = (DATASET_PATH, METADATA_PATH, SUMMARY_PATH, FOLDS_PATH, OVERFITTING_PATH)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required analysis inputs are missing: {missing_paths}")

    with METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    frame = pd.read_csv(DATASET_PATH)
    if "Sex" in frame.columns:
        frame["Sex"] = frame["Sex"].replace({"Fmale": "Female"})
    if "VHD" in frame.columns:
        frame["VHD"] = frame["VHD"].replace({"mild": "Mild"})
    summary = pd.read_csv(SUMMARY_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    overfitting = pd.read_csv(OVERFITTING_PATH)

    targets = list(metadata.get("targets", []))
    if targets != list(TARGETS):
        raise ValueError(f"Expected metadata targets {list(TARGETS)}, got {targets}.")
    missing_features = set(metadata["raw_feature_names"]).difference(frame.columns)
    forbidden = LEAKAGE_COLUMNS.intersection(metadata["raw_feature_names"])
    if missing_features:
        raise ValueError(f"Dataset is missing model features: {sorted(missing_features)}")
    if forbidden:
        raise ValueError(f"Target/leakage columns found among model inputs: {sorted(forbidden)}")

    for target in TARGETS:
        for source, table in (("repeated-CV summary", summary), ("repeated-CV folds", folds), ("holdout metrics", overfitting)):
            if target not in set(table["Target"]):
                raise ValueError(f"{source} is missing target {target}.")
    return frame, summary, metadata, folds, overfitting


def derive_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if "Cath" not in frame.columns:
        raise ValueError("Dataset is missing the Cath column required to derive CAD.")
    labels = pd.DataFrame(index=frame.index)
    labels["CAD"] = frame["Cath"].astype(str).str.strip().eq("CAD").astype(int)
    for target in ("LAD", "LCX", "RCA"):
        if target not in frame.columns:
            raise ValueError(f"Dataset is missing the {target} target column.")
        labels[target] = frame[target].astype(str).str.strip().eq("Stenotic").astype(int)
    return labels.loc[:, list(TARGETS)]


def multilabel_balanced_holdout(
    labels: pd.DataFrame, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror deterministic_multilabel_holdout in final_rebuild_pipeline.py."""
    n_test = int(round(len(labels) * test_size))
    label_indices = labels.index.to_numpy()
    label_matrix = labels.loc[:, list(TARGETS)].to_numpy(dtype=float)
    row_positions = {int(index): position for position, index in enumerate(label_indices)}
    desired = label_matrix.sum(axis=0) * (n_test / len(labels))
    rng = np.random.default_rng(random_state)
    order = rng.permutation(label_indices)
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_counts = np.zeros(len(TARGETS), dtype=float)

    while len(selected) < n_test:
        best_index: int | None = None
        remaining_indices = [int(index) for index in order if int(index) not in selected_set]
        remaining_labels = label_matrix[[row_positions[index] for index in remaining_indices]]
        next_counts = selected_counts + remaining_labels
        overshoots = np.maximum(0, next_counts - desired).sum(axis=1)
        distances = np.abs(desired - next_counts).sum(axis=1)
        remaining_slots = n_test - len(selected) - 1
        rarity_bonuses = -remaining_labels.sum(axis=1) if remaining_slots > 0 else np.zeros(len(remaining_indices))
        best_overshoot = overshoots.min()
        candidates = np.flatnonzero(overshoots == best_overshoot)
        best_distance = distances[candidates].min()
        candidates = candidates[distances[candidates] == best_distance]
        best_rarity = rarity_bonuses[candidates].min()
        best_position = int(candidates[np.flatnonzero(rarity_bonuses[candidates] == best_rarity)[0]])
        best_index = remaining_indices[best_position]
        if best_index is None:
            raise RuntimeError("Could not complete the multi-label-balanced holdout.")
        selected.append(best_index)
        selected_set.add(best_index)
        selected_counts += labels.loc[best_index, list(TARGETS)].to_numpy(dtype=float)

    test_indices = np.array(sorted(selected), dtype=int)
    train_indices = np.array(sorted(set(labels.index) - selected_set), dtype=int)
    return train_indices, test_indices


def make_preprocessor(metadata: dict[str, Any]) -> ColumnTransformer:
    numeric = list(metadata["numeric_features"])
    categorical = list(metadata["categorical_features"])
    if set(numeric).union(categorical) != set(metadata["raw_feature_names"]):
        raise ValueError("Metadata numeric/categorical features do not match raw_feature_names.")
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model_pipeline(
    metadata: dict[str, Any],
    target: str,
    random_state: int,
    parameter_overrides: dict[str, Any] | None = None,
) -> Pipeline:
    config = metadata["selected_models"][target]
    if config.get("algorithm") != "XGBoost":
        raise ValueError(f"Unsupported production algorithm for {target}: {config.get('algorithm')!r}")
    parameters = {
        name: value
        for name, value in config["hyperparameters"].items()
        if value is not None
    }
    parameters["random_state"] = random_state
    if parameter_overrides:
        parameters.update(parameter_overrides)
    steps: list[tuple[str, Any]] = [("preprocessor", make_preprocessor(metadata))]
    if config.get("strategy") == "smote_regularized":
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
        pipeline_type: Any = ImbalancedPipeline
    else:
        pipeline_type = Pipeline
    steps.append(("model", XGBClassifier(**parameters)))
    return pipeline_type(steps)


def compute_metrics(labels: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "F1": float(f1_score(labels, probabilities >= threshold, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
    }


def run_holdout_seed_stability(
    frame: pd.DataFrame, metadata: dict[str, Any], labels: pd.DataFrame
) -> pd.DataFrame:
    features = frame.loc[:, metadata["raw_feature_names"]]
    rows: list[dict[str, Any]] = []
    for seed in HOLDOUT_SEEDS:
        train_indices, test_indices = multilabel_balanced_holdout(labels, TEST_SIZE, seed)
        for target in TARGETS:
            config = metadata["selected_models"][target]
            model = make_model_pipeline(metadata, target, random_state=42)
            model.fit(features.loc[train_indices], labels.loc[train_indices, target])
            probabilities = model.predict_proba(features.loc[test_indices])[:, 1]
            rows.append(
                {
                    "Target": target,
                    "Seed": seed,
                    "Train_Samples": len(train_indices),
                    "Test_Samples": len(test_indices),
                    **compute_metrics(
                        labels.loc[test_indices, target],
                        probabilities,
                        float(config["threshold"]),
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_scores = scores.loc[scores["Target"] == target]
        for metric in ("ROC_AUC", "F1"):
            values = target_scores[metric]
            summary_rows.append(
                {
                    "Target": target,
                    "Metric": metric,
                    "Mean": float(values.mean()),
                    "SD": float(values.std(ddof=1)),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                    "N_Seeds": len(HOLDOUT_SEEDS),
                    "Seeds": ";".join(map(str, HOLDOUT_SEEDS)),
                }
            )
    scores.to_csv(METRICS_DIR / "holdout_seed_stability_runs.csv", index=False)
    return pd.DataFrame(summary_rows)


def run_regularization_cv(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    labels: pd.DataFrame,
    overfitting: pd.DataFrame,
) -> pd.DataFrame:
    train_indices, _ = multilabel_balanced_holdout(
        labels, TEST_SIZE, int(metadata["random_state"])
    )
    features = frame.loc[train_indices, metadata["raw_feature_names"]]
    training_labels = labels.loc[train_indices]
    gaps = overfitting.set_index("Target")["Train_CV_AUC_Gap"]
    affected_targets = [target for target in TARGETS if float(gaps[target]) > 0.10]
    if not affected_targets:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for target in affected_targets:
        splitter = RepeatedStratifiedKFold(
            n_splits=REGULARIZATION_FOLDS,
            n_repeats=REGULARIZATION_REPEATS,
            random_state=REGULARIZATION_SEED,
        )
        splits = list(splitter.split(features, training_labels[target]))
        settings = [("production", {})] + list(REGULARIZATION_GRID)
        for setting_name, overrides in settings:
            cv_by_repeat: dict[int, list[float]] = {}
            train_by_repeat: dict[int, list[float]] = {}
            effective_parameters = {
                name: value
                for name, value in metadata["selected_models"][target]["hyperparameters"].items()
                if value is not None
            }
            effective_parameters.update(overrides)
            for fold_index, (fit_indices, validation_indices) in enumerate(splits):
                repeat = fold_index // REGULARIZATION_FOLDS + 1
                model = make_model_pipeline(
                    metadata,
                    target,
                    random_state=REGULARIZATION_SEED,
                    parameter_overrides=overrides,
                )
                x_fit = features.iloc[fit_indices]
                y_fit = training_labels[target].iloc[fit_indices]
                x_validation = features.iloc[validation_indices]
                y_validation = training_labels[target].iloc[validation_indices]
                model.fit(x_fit, y_fit)
                train_probabilities = model.predict_proba(x_fit)[:, 1]
                validation_probabilities = model.predict_proba(x_validation)[:, 1]
                train_by_repeat.setdefault(repeat, []).append(
                    float(roc_auc_score(y_fit, train_probabilities))
                )
                cv_by_repeat.setdefault(repeat, []).append(
                    float(roc_auc_score(y_validation, validation_probabilities))
                )

            repeat_cv_means = pd.Series(
                {repeat: float(np.mean(values)) for repeat, values in cv_by_repeat.items()}
            )
            repeat_train_means = pd.Series(
                {repeat: float(np.mean(values)) for repeat, values in train_by_repeat.items()}
            )
            rows.append(
                {
                    "Target": target,
                    "Setting": setting_name,
                    "Parameters": json.dumps(
                        {
                            key: effective_parameters.get(key)
                            for key in (
                                "max_depth",
                                "min_child_weight",
                                "subsample",
                                "colsample_bytree",
                                "reg_lambda",
                                "n_estimators",
                                "learning_rate",
                            )
                        },
                        sort_keys=True,
                    ),
                    "Train_ROC_AUC_Mean": float(repeat_train_means.mean()),
                    "CV_ROC_AUC_Mean": float(repeat_cv_means.mean()),
                    "CV_ROC_AUC_SD": float(repeat_cv_means.std(ddof=1)),
                    "Train_CV_AUC_Gap": float(repeat_train_means.mean() - repeat_cv_means.mean()),
                    "CV_AUC_Gap_Shrinkage_vs_Production": np.nan,
                    "CV_AUC_Improvement_vs_Production": np.nan,
                    "Improves_by_more_than_1_SD": False,
                    "N_Folds": REGULARIZATION_FOLDS,
                    "N_Repeats": REGULARIZATION_REPEATS,
                    "Random_State": REGULARIZATION_SEED,
                }
            )

    results = pd.DataFrame(rows)
    for target in affected_targets:
        target_indices = results.index[results["Target"] == target]
        baseline_index = results.index[
            (results["Target"] == target) & (results["Setting"] == "production")
        ][0]
        baseline_mean = float(results.loc[baseline_index, "CV_ROC_AUC_Mean"])
        baseline_sd = float(results.loc[baseline_index, "CV_ROC_AUC_SD"])
        baseline_gap = float(results.loc[baseline_index, "Train_CV_AUC_Gap"])
        for index in target_indices:
            results.loc[index, "CV_AUC_Improvement_vs_Production"] = (
                float(results.loc[index, "CV_ROC_AUC_Mean"]) - baseline_mean
            )
            results.loc[index, "CV_AUC_Gap_Shrinkage_vs_Production"] = (
                baseline_gap - float(results.loc[index, "Train_CV_AUC_Gap"])
            )
            results.loc[index, "Improves_by_more_than_1_SD"] = (
                results.loc[index, "Setting"] != "production"
                and float(results.loc[index, "CV_ROC_AUC_Mean"]) > baseline_mean + baseline_sd
            )
    return results


def percentile_within_distribution(value: float, distribution: pd.Series) -> float:
    values = distribution.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Repeated-CV distribution contains no usable AUC values.")
    return float(100.0 * np.mean(values <= value))


def make_holdout_comparison(
    summary: pd.DataFrame, folds: pd.DataFrame, overfitting: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        summary_auc = summary.loc[
            (summary["Target"] == target) & (summary["Metric"] == "ROC_AUC")
        ]
        holdout = overfitting.loc[overfitting["Target"] == target]
        target_folds = folds.loc[folds["Target"] == target, "ROC_AUC"]
        if len(summary_auc) != 1 or len(holdout) != 1 or target_folds.empty:
            raise ValueError(f"Expected one CV summary and holdout record plus fold values for {target}.")
        cv_row = summary_auc.iloc[0]
        holdout_auc = float(holdout.iloc[0]["Test_ROC_AUC"])
        percentile = percentile_within_distribution(holdout_auc, target_folds)
        rows.append(
            {
                "Target": target,
                "Repeated_CV_ROC_AUC_Mean": float(cv_row["Mean"]),
                "Repeated_CV_ROC_AUC_SD": float(cv_row["SD"]),
                "Repeated_CV_ROC_AUC_CI_95_Lower": float(cv_row["CI_95_Lower"]),
                "Repeated_CV_ROC_AUC_CI_95_Upper": float(cv_row["CI_95_Upper"]),
                "Holdout_ROC_AUC": holdout_auc,
                "Holdout_F1": float(holdout.iloc[0]["Test_F1"]),
                "Holdout_Percentile_in_CV_Fold_AUCs": percentile,
                "CV_Fold_AUC_90th_Percentile": float(target_folds.quantile(0.90)),
                "Above_90th_Percentile": percentile > 90.0,
                "N_CV_Fold_AUCs": int(target_folds.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def archive_outdated_cv_stability() -> Path | None:
    outdated_path = METRICS_DIR / "cv_seed_stability.csv"
    if not outdated_path.exists():
        archives = sorted(METRICS_DIR.glob("cv_seed_stability_archived*.csv"))
        return archives[0] if archives else None
    archive_path = METRICS_DIR / "cv_seed_stability_archived.csv"
    counter = 1
    while archive_path.exists():
        archive_path = METRICS_DIR / f"cv_seed_stability_archived_{counter}.csv"
        counter += 1
    shutil.move(str(outdated_path), str(archive_path))
    return archive_path


def write_report(
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    regularization: pd.DataFrame,
    metadata: dict[str, Any],
    archived_path: Path | None,
) -> None:
    lines = [
        "# Generalization analysis",
        "",
        "## Headline to quote",
        "",
        "Use the repeated cross-validation ROC-AUC mean and its 95% percentile interval as the primary performance estimate. "
        "Show the final holdout as a secondary, single-split result rather than choosing whichever estimate is higher. "
        "The intervals come from ten repeat-level means of five-fold CV; they describe resampling variability in this dataset, "
        "not uncertainty across hospitals or patient populations.",
        "",
        "All repeated-CV values below use the fixed production model settings and thresholds in "
        "`model/serialized/model_metadata.json`. The repeated CV uses 5 folds × 10 repeats (seed 42). "
        "Holdout percentiles compare the seed-42 holdout AUC with the 50 individual fold AUCs and are empirical diagnostics, "
        "not calibrated probabilities or independent-test significance tests.",
        "",
        "## Holdout compared with repeated CV",
        "",
        "| Target | Repeated-CV ROC-AUC (mean; 95% interval) | Holdout ROC-AUC | Holdout F1 | Percentile among 50 CV fold AUCs | Above 90th percentile? |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Target} | {row.Repeated_CV_ROC_AUC_Mean:.3f} "
            f"[{row.Repeated_CV_ROC_AUC_CI_95_Lower:.3f}, {row.Repeated_CV_ROC_AUC_CI_95_Upper:.3f}] "
            f"| {row.Holdout_ROC_AUC:.3f} | {row.Holdout_F1:.3f} "
            f"| {row.Holdout_Percentile_in_CV_Fold_AUCs:.1f}th | "
            f"{'Yes' if row.Above_90th_Percentile else 'No'} |"
        )

    lines.extend(
        [
            "",
            "A holdout above the 90th percentile is unusually favorable relative to the CV folds, but does not establish why that split scored well. "
            "With only 76 holdout rows, a few cases can materially affect AUC and F1.",
            "",
            "## Holdout split-seed stability",
            "",
            f"The fixed production models and thresholds were evaluated on 20 multi-label-balanced 80/20 splits "
            f"(seeds {HOLDOUT_SEEDS[0]}, {HOLDOUT_SEEDS[1]}, …, {HOLDOUT_SEEDS[-1]}). "
            "These are sensitivity results, not 20 independent external test sets.",
            "",
            "| Target | Metric | Mean | SD | Min | Max |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in stability.itertuples(index=False):
        lines.append(
            f"| {row.Target} | {row.Metric} | {row.Mean:.3f} | {row.SD:.3f} | {row.Min:.3f} | {row.Max:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Training-only regularization experiment",
            "",
            "Targets with a production train-to-CV ROC-AUC gap greater than 0.10 were tested: "
            + (
                ", ".join(regularization["Target"].drop_duplicates().tolist())
                if not regularization.empty
                else "none"
            )
            + ". The 76-row holdout was not used. Each configuration used identical 5-fold × 5-repeat stratified splits "
            "within the seed-42 training partition. The gap is the fold-training ROC-AUC mean minus the validation ROC-AUC mean. "
            "The 1-SD promotion gate compares each candidate's repeated-CV mean against the production baseline mean plus "
            "the production baseline SD across repeat means.",
            "",
            "| Target | Setting | CV ROC-AUC mean ± SD | Train-CV gap | Gap shrinkage vs production | CV-AUC improvement vs production | >1 baseline SD? |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    if regularization.empty:
        lines.append("| — | No targets met the gap threshold | — | — | — | — | — |")
    else:
        for row in regularization.itertuples(index=False):
            lines.append(
                f"| {row.Target} | {row.Setting} | {row.CV_ROC_AUC_Mean:.3f} ± {row.CV_ROC_AUC_SD:.3f} "
                f"| {row.Train_CV_AUC_Gap:.3f} | {row.CV_AUC_Gap_Shrinkage_vs_Production:+.3f} "
                f"| {row.CV_AUC_Improvement_vs_Production:+.3f} "
                f"| {'Yes' if row.Improves_by_more_than_1_SD else 'No'} |"
            )
    qualified = regularization.loc[regularization["Improves_by_more_than_1_SD"]] if not regularization.empty else regularization
    if qualified.empty:
        lines.extend(
            [
                "",
                "No regularization candidate improved repeated-CV ROC-AUC by more than one production-baseline SD. "
                "The production models were therefore left unchanged; the configurations above are documented experiments.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "At least one candidate cleared the predeclared >1-SD repeated-CV gate. This analysis does not promote it automatically; "
                "any production change should be separately reviewed and retrained using training data only, without consulting holdout scores.",
            ]
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The dataset contains 378 observations from a single center. It is too small to support strong claims about transportability.",
            "- The holdout contains 76 observations and is only one split; its metrics can be optimistic or pessimistic by chance.",
            "- Repeated cross-validation reuses the same patients across folds/repeats. Its intervals are resampling summaries, not external-validation confidence intervals.",
            "- No independent center, time period, or prospective cohort was available for external validation.",
            "- Results reflect the documented preprocessing, label definitions, selected production configurations, and thresholds; changes to these invalidate direct comparisons.",
            "",
            "## Reproducibility",
            "",
            f"Holdout split seed used by the final rebuild: {int(metadata['random_state'])}. "
            f"Holdout stability seeds: {', '.join(map(str, HOLDOUT_SEEDS))}. "
            f"Regularization CV seed: {REGULARIZATION_SEED}.",
        ]
    )
    if archived_path:
        lines.extend(["", f"The outdated seed-stability file was preserved as `{archived_path.relative_to(ROOT).as_posix()}`."])
    (METRICS_DIR / "generalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frame, summary, metadata, folds, overfitting = load_inputs()
    labels = derive_labels(frame)
    if len(frame) != len(labels):
        raise ValueError("Feature and target row counts do not match.")

    comparison = make_holdout_comparison(summary, folds, overfitting)
    stability = run_holdout_seed_stability(frame, metadata, labels)
    regularization = run_regularization_cv(frame, metadata, labels, overfitting)
    archived_path = archive_outdated_cv_stability()

    comparison.to_csv(METRICS_DIR / "generalization_holdout_comparison.csv", index=False)
    stability.to_csv(METRICS_DIR / "holdout_seed_stability.csv", index=False)
    regularization.to_csv(METRICS_DIR / "regularization_cv_results.csv", index=False)
    write_report(comparison, stability, regularization, metadata, archived_path)

    print(f"Analyzed {len(frame)} rows; holdout seed stability evaluated {len(HOLDOUT_SEEDS)} splits.")
    print("Holdout percentile within repeated-CV fold AUCs:")
    for row in comparison.itertuples(index=False):
        print(
            f"  {row.Target}: {row.Holdout_Percentile_in_CV_Fold_AUCs:.1f}th percentile "
            f"({'above' if row.Above_90th_Percentile else 'not above'} 90th)"
        )
    print(f"Outputs: {METRICS_DIR / 'generalization_report.md'}, "
          f"{METRICS_DIR / 'generalization_holdout_comparison.csv'}, "
          f"{METRICS_DIR / 'holdout_seed_stability.csv'}, "
          f"{METRICS_DIR / 'regularization_cv_results.csv'}")


if __name__ == "__main__":
    main()
