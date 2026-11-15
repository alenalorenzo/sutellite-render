
"""Experimental MODIS Aqua / VIIRS SNPP count harmonisation model.

This experiment predicts VIIRS SNPP detection counts from MODIS Aqua
detection counts, conditional on both sensors recording detections in
the same monthly 0.1-degree cell and day/night group.

It does not estimate satellite observation coverage, reconstruct unique
fires, infer confirmed zero-detection cases, or produce 375 m predictions.

Chronological split:
    Training:   2013-2022
    Validation: 2023-2024
    Test:       2025

The 2025 test data is not used for model selection or fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "1.0"

DEFAULT_DATASET_FILE = Path(
    "processed/harmonization_spain/spain_monthly_cell_counts.parquet"
)
DEFAULT_OUTPUT_DIR = Path("processed/harmonization_spain")

TRAIN_START_YEAR = 2013
TRAIN_END_YEAR = 2022

VALIDATION_START_YEAR = 2023
VALIDATION_END_YEAR = 2024

TEST_YEAR = 2025

RANDOM_STATE = 42

KEY_COLUMNS = (
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "daynight",
)

REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    "modis_count",
    "viirs_count",
    "modis_present",
    "viirs_present",
)

FEATURE_COLUMNS = (
    "modis_log_count",
    "month_sin",
    "month_cos",
    "is_night",
)

ML_MODEL_NAMES = ("ridge", "random_forest")

PREDICTION_COLUMNS = {
    "naive_modis": "prediction_naive_modis",
    "month_daynight_mean": "prediction_month_daynight_mean",
    "ridge": "prediction_ridge",
    "random_forest": "prediction_random_forest",
}


def validate_model_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate the harmonised dataset before model preparation."""

    if dataset.empty:
        raise ValueError("The harmonisation dataset is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            "The harmonisation dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    data = dataset.copy()

    numeric_columns = (
        "year",
        "month",
        "grid_lat_index",
        "grid_lon_index",
        "modis_count",
        "viirs_count",
    )

    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"Column '{column}' contains missing or non-numeric values."
            )

        array = values.to_numpy(dtype=np.float64)

        if not np.isfinite(array).all():
            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )

        if not np.equal(array, np.floor(array)).all():
            raise ValueError(
                f"Column '{column}' must contain integer values."
            )

        data[column] = values.astype("int64")

    if not data["year"].between(
        TRAIN_START_YEAR,
        TEST_YEAR,
    ).all():
        raise ValueError(
            f"Dataset years must be between {TRAIN_START_YEAR} "
            f"and {TEST_YEAR}."
        )

    if not data["month"].between(1, 12).all():
        raise ValueError("Month values must be between 1 and 12.")

    for column in ("modis_count", "viirs_count"):
        if (data[column] < 0).any():
            raise ValueError(
                f"Column '{column}' cannot contain negative counts."
            )

    for column in ("modis_present", "viirs_present"):
        if data[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )

        if not data[column].isin((True, False)).all():
            raise ValueError(
                f"Column '{column}' must contain boolean values."
            )

        data[column] = data[column].astype(bool)

    if not data["daynight"].isin(("D", "N")).all():
        raise ValueError(
            "Day/night values must contain only 'D' or 'N'."
        )

    for sensor in ("modis", "viirs"):
        expected_presence = data[f"{sensor}_count"].gt(0)

        if not data[f"{sensor}_present"].equals(expected_presence):
            raise ValueError(
                f"'{sensor}_present' is inconsistent with "
                f"'{sensor}_count'."
            )

    duplicated = data.duplicated(
        subset=list(KEY_COLUMNS),
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "The dataset contains duplicated spatial-temporal groups."
        )

    return data


def prepare_paired_splits(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter paired detection groups and create chronological splits.

    One-sided groups are deliberately excluded because a missing
    detection record cannot be interpreted as a confirmed zero.
    """

    data = validate_model_dataset(dataset)

    paired_mask = data["modis_present"] & data["viirs_present"]
    paired = data.loc[paired_mask].copy()

    if paired.empty:
        raise ValueError(
            "No paired MODIS/VIIRS detection groups are available."
        )

    if (
        paired["modis_count"].le(0).any()
        or paired["viirs_count"].le(0).any()
    ):
        raise ValueError(
            "Paired groups must have positive counts for both sensors."
        )

    training = paired.loc[
        paired["year"].between(
            TRAIN_START_YEAR,
            TRAIN_END_YEAR,
        )
    ].copy()

    validation = paired.loc[
        paired["year"].between(
            VALIDATION_START_YEAR,
            VALIDATION_END_YEAR,
        )
    ].copy()

    test = paired.loc[paired["year"].eq(TEST_YEAR)].copy()

    if training.empty:
        raise ValueError("The training split is empty.")

    if validation.empty:
        raise ValueError("The validation split is empty.")

    if test.empty:
        raise ValueError("The 2025 test split is empty.")

    return (
        training.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build input features without using VIIRS target information."""

    required = {
        "modis_count",
        "month",
        "daynight",
    }

    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(
            f"Cannot build model features. Missing columns: {missing}"
        )

    month_zero_based = dataset["month"].astype(np.float64) - 1.0

    month_angle = 2.0 * np.pi * month_zero_based / 12.0

    features = pd.DataFrame(index=dataset.index)

    features["modis_log_count"] = np.log1p(
        dataset["modis_count"].astype(np.float64)
    )

    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    features["is_night"] = (
        dataset["daynight"].eq("N").astype(np.float64)
    )

    return features.loc[:, list(FEATURE_COLUMNS)]


def build_estimators() -> dict[str, Any]:
    """Create the fixed candidate ML models."""

    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    random_forest_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "ridge": ridge_model,
        "random_forest": random_forest_model,
    }


def predict_month_daynight_mean(
    training_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> np.ndarray:
    """Predict using training-only monthly and day/night mean counts."""

    if training_data.empty:
        raise ValueError(
            "Cannot calculate the monthly baseline from empty training data."
        )

    means = (
        training_data.groupby(
            ["month", "daynight"],
            observed=True,
        )["viirs_count"]
        .mean()
        .to_dict()
    )

    global_mean = float(training_data["viirs_count"].mean())

    predictions = [
        means.get(
            (int(month), str(daynight)),
            global_mean,
        )
        for month, daynight in zip(
            prediction_data["month"],
            prediction_data["daynight"],
        )
    ]

    return np.asarray(predictions, dtype=np.float64)


def _predict_model(
    estimator: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Convert log-scale predictions to non-negative detection counts."""

    log_predictions = np.asarray(
        estimator.predict(features),
        dtype=np.float64,
    )

    predictions = np.expm1(log_predictions)

    return np.maximum(predictions, 0.0)


def create_prediction_frame(
    prediction_data: pd.DataFrame,
    *,
    baseline_training_data: pd.DataFrame,
    estimators: dict[str, Any],
) -> pd.DataFrame:
    """Create predictions for the baselines and supplied ML estimators."""

    output_columns = [
        *KEY_COLUMNS,
        "modis_count",
        "viirs_count",
    ]

    predictions = prediction_data.loc[
        :,
        output_columns,
    ].copy()

    predictions["prediction_naive_modis"] = (
        prediction_data["modis_count"].to_numpy(dtype=np.float64)
    )

    predictions["prediction_month_daynight_mean"] = (
        predict_month_daynight_mean(
            baseline_training_data,
            prediction_data,
        )
    )

    features = build_model_features(prediction_data)

    for model_name, estimator in estimators.items():
        predictions[f"prediction_{model_name}"] = _predict_model(
            estimator,
            features,
        )

    return predictions


def assign_seasons(months: pd.Series) -> pd.Series:
    """Map calendar months to meteorological seasons."""

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    return months.map(season_map)


def assign_activity_levels(counts: pd.Series) -> pd.Series:
    """Create fixed target-count strata for descriptive evaluation."""

    return pd.Series(
        np.select(
            [
                counts.eq(1),
                counts.between(2, 5),
                counts.ge(6),
            ],
            [
                "Low (1)",
                "Medium (2-5)",
                "High (6+)",
            ],
            default="Unknown",
        ),
        index=counts.index,
        dtype="string",
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    split_name: str,
    prediction_columns: list[str],
) -> pd.DataFrame:
    """Calculate overall and stratified metrics on the original count scale.

    Signed bias is mean(prediction - actual). Positive values indicate
    overprediction and negative values indicate underprediction.
    """

    if predictions.empty:
        raise ValueError("Cannot evaluate an empty prediction dataset.")

    required = {"viirs_count", *prediction_columns}

    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Cannot evaluate predictions. Missing columns: {missing}"
        )

    work = predictions.copy()

    work["season"] = assign_seasons(work["month"])
    work["activity_level"] = assign_activity_levels(
        work["viirs_count"]
    )

    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "all", work),
    ]

    for dimension in (
        "year",
        "season",
        "daynight",
        "activity_level",
    ):
        for group_value, group_data in work.groupby(
            dimension,
            sort=True,
            observed=True,
        ):
            scopes.append(
                (
                    dimension,
                    str(group_value),
                    group_data,
                )
            )

    metric_rows: list[dict[str, Any]] = []

    for prediction_column in prediction_columns:
        model_name = prediction_column.removeprefix("prediction_")

        for scope_name, group_name, group_data in scopes:
            actual = group_data["viirs_count"].to_numpy(
                dtype=np.float64
            )

            predicted = group_data[prediction_column].to_numpy(
                dtype=np.float64
            )

            errors = predicted - actual

            metric_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "scope": scope_name,
                    "group": group_name,
                    "n": int(len(group_data)),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "signed_bias": float(np.mean(errors)),
                }
            )

    return pd.DataFrame(metric_rows)


def _overall_metrics_by_model(
    metrics: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Convert overall metric rows into a JSON-compatible dictionary."""

    overall = metrics.loc[
        metrics["scope"].eq("overall")
    ]

    results: dict[str, dict[str, float]] = {}

    for _, row in overall.iterrows():
        results[str(row["model"])] = {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "signed_bias": float(row["signed_bias"]),
        }

    return results


def _split_summary(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise paired groups and day/night distributions."""

    result: dict[str, Any] = {}

    for split_name, data in (
        ("training", training),
        ("validation", validation),
        ("test", test),
    ):
        result[split_name] = {
            "rows": int(len(data)),
            "years": sorted(
                int(year)
                for year in data["year"].unique()
            ),
            "daynight_groups": {
                "D": int(data["daynight"].eq("D").sum()),
                "N": int(data["daynight"].eq("N").sum()),
            },
        }

    return result


def run_experiment(
    dataset_file: Path = DEFAULT_DATASET_FILE,
    output_directory: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run chronological validation, select a model and evaluate 2025."""

    dataset_file = Path(dataset_file)
    output_directory = Path(output_directory)

    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Prepared feature dataset not found: {dataset_file}"
        )

    dataset = pd.read_parquet(dataset_file)

    training, validation, test = prepare_paired_splits(dataset)

    # Stage 1: fit candidate models using training years only.
    candidate_estimators = build_estimators()

    training_features = build_model_features(training)
    training_target = np.log1p(
        training["viirs_count"].to_numpy(dtype=np.float64)
    )

    for estimator in candidate_estimators.values():
        estimator.fit(
            training_features,
            training_target,
        )

    # Stage 2: compare candidates on the validation period.
    validation_predictions = create_prediction_frame(
        validation,
        baseline_training_data=training,
        estimators=candidate_estimators,
    )

    validation_prediction_columns = [
        PREDICTION_COLUMNS["naive_modis"],
        PREDICTION_COLUMNS["month_daynight_mean"],
        PREDICTION_COLUMNS["ridge"],
        PREDICTION_COLUMNS["random_forest"],
    ]

    validation_metrics = evaluate_predictions(
        validation_predictions,
        split_name="validation",
        prediction_columns=validation_prediction_columns,
    )

    validation_overall = validation_metrics.loc[
        validation_metrics["scope"].eq("overall")
        & validation_metrics["model"].isin(ML_MODEL_NAMES)
    ]

    if validation_overall.empty:
        raise RuntimeError(
            "No ML candidate produced validation metrics."
        )

    # Select the candidate with the lowest validation RMSE.
    # Using a NumPy array and positional indexing avoids ambiguous
    # two-dimensional DataFrame .loc typing in Pylance.
    validation_rmse = validation_overall["rmse"].to_numpy(
        dtype=np.float64
    )
    best_model_position = int(np.argmin(validation_rmse))

    selected_model_name = str(
        validation_overall["model"].to_numpy()[best_model_position]
    )

    # Stage 3: refit only the selected ML model on training + validation.
    development_data = pd.concat(
        [training, validation],
        ignore_index=True,
    )

    selected_estimator = build_estimators()[selected_model_name]

    development_features = build_model_features(development_data)
    development_target = np.log1p(
        development_data["viirs_count"].to_numpy(dtype=np.float64)
    )

    selected_estimator.fit(
        development_features,
        development_target,
    )

    # Stage 4: evaluate the untouched 2025 test period.
    # No model or hyperparameter selection is performed using these results.
    test_predictions = create_prediction_frame(
        test,
        baseline_training_data=development_data,
        estimators={
            selected_model_name: selected_estimator,
        },
    )

    test_prediction_columns = [
        PREDICTION_COLUMNS["naive_modis"],
        PREDICTION_COLUMNS["month_daynight_mean"],
        PREDICTION_COLUMNS[selected_model_name],
    ]

    test_metrics = evaluate_predictions(
        test_predictions,
        split_name="test",
        prediction_columns=test_prediction_columns,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_metrics_path = (
        output_directory / "validation_metrics.csv"
    )
    test_metrics_path = output_directory / "test_metrics.csv"
    validation_predictions_path = (
        output_directory / "validation_predictions.csv"
    )
    test_predictions_path = (
        output_directory / "test_predictions.csv"
    )
    model_path = output_directory / "selected_model.joblib"
    summary_path = output_directory / "model_summary.json"

    validation_metrics.to_csv(
        validation_metrics_path,
        index=False,
    )
    test_metrics.to_csv(
        test_metrics_path,
        index=False,
    )
    validation_predictions.to_csv(
        validation_predictions_path,
        index=False,
    )
    test_predictions.to_csv(
        test_predictions_path,
        index=False,
    )

    model_artifact = {
        "model_version": MODEL_VERSION,
        "model_name": selected_model_name,
        "estimator": selected_estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "log1p(viirs_count)",
        "prediction_inverse_transform": "expm1, clipped at zero",
        "training_years": [
            TRAIN_START_YEAR,
            TRAIN_END_YEAR,
        ],
        "validation_years": [
            VALIDATION_START_YEAR,
            VALIDATION_END_YEAR,
        ],
        "final_fit_years": [
            TRAIN_START_YEAR,
            VALIDATION_END_YEAR,
        ],
        "test_year": TEST_YEAR,
        "population": (
            "Groups with both MODIS Aqua and VIIRS SNPP detections."
        ),
    }

    joblib.dump(
        model_artifact,
        model_path,
    )

    summary = {
        "model_version": MODEL_VERSION,
        "experiment": (
            "Spain MODIS Aqua / VIIRS SNPP conditional count comparison"
        ),
        "dataset_file": dataset_file.name,
        "population": {
            "selection": (
                "Only groups where both modis_present and viirs_present "
                "are true."
            ),
            "paired_groups": int(
                len(training) + len(validation) + len(test)
            ),
            "excluded_one_sided_groups": int(
                len(dataset)
                - (
                    len(training)
                    + len(validation)
                    + len(test)
                )
            ),
            "interpretation": (
                "Predictions are conditional on both sensors recording "
                "detections in the same spatial-temporal group."
            ),
        },
        "split": _split_summary(
            training,
            validation,
            test,
        ),
        "model_selection": {
            "criterion": (
                "Lowest validation RMSE among Ridge and Random Forest."
            ),
            "candidate_models": list(ML_MODEL_NAMES),
            "selected_model": selected_model_name,
            "validation_overall_metrics": (
                _overall_metrics_by_model(validation_metrics)
            ),
        },
        "test_overall_metrics": (
            _overall_metrics_by_model(test_metrics)
        ),
        "metric_definition": {
            "mae": "Mean absolute error in VIIRS detection counts.",
            "rmse": "Root mean squared error in VIIRS detection counts.",
            "signed_bias": (
                "Mean(prediction - actual). Positive means overprediction."
            ),
        },
        "limitations": [
            "The input dataset contains detections, not satellite observation opportunity.",
            "Missing detections cannot be interpreted as confirmed zero activity.",
            "The paired population excludes one-sided sensor groups.",
            "The model cannot estimate VIIRS detections in groups without MODIS detections.",
            "The model does not predict unique fires, burned area, energy or 375 m pixels.",
            "The 0.1-degree grid is angular and not equal-area.",
            "Counts are source observations, not unique fire events.",
            "The 2025 test period is a single-year temporal holdout.",
            "Night and some year/season/activity strata have smaller sample sizes.",
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    print("SPAIN SENSOR HARMONISATION MODEL")
    print(f"Paired groups: {summary['population']['paired_groups']:,}")
    print(f"Selected model: {selected_model_name}")
    print(
        f"Training groups: {len(training):,} "
        f"({TRAIN_START_YEAR}-{TRAIN_END_YEAR})"
    )
    print(
        f"Validation groups: {len(validation):,} "
        f"({VALIDATION_START_YEAR}-{VALIDATION_END_YEAR})"
    )
    print(
        f"Test groups: {len(test):,} ({TEST_YEAR})"
    )

    print("\nVALIDATION OVERALL METRICS")
    print(
        validation_metrics.loc[
            validation_metrics["scope"].eq("overall"),
            ["model", "n", "mae", "rmse", "signed_bias"],
        ].to_string(index=False)
    )

    print("\nTEST OVERALL METRICS")
    print(
        test_metrics.loc[
            test_metrics["scope"].eq("overall"),
            ["model", "n", "mae", "rmse", "signed_bias"],
        ].to_string(index=False)
    )

    print("\nOUTPUT FILES")
    for path in (
        validation_metrics_path,
        test_metrics_path,
        validation_predictions_path,
        test_predictions_path,
        model_path,
        summary_path,
    ):
        print(path)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the experimental Spain MODIS/VIIRS "
            "conditional count harmonisation model."
        )
    )

    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Prepared harmonisation Parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions and model artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    run_experiment(
        dataset_file=args.dataset_file,
        output_directory=args.output_dir,
    )


if __name__ == "__main__":
    main()