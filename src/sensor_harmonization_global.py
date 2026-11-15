"""Full-dataset MODIS/VIIRS count-mapping experiment for Sutellite.

This module scans all canonical Parquet files below an input directory,
aggregates ARCHIVE observations across every region and platform, and trains
models to estimate VIIRS sensor-family detection counts from MODIS counts.

IMPORTANT SCIENTIFIC SCOPE
--------------------------
The supervised model is trained/evaluated only on monthly grid groups where
both sensor families have at least one detection. A missing detection record
is NOT a confirmed zero and this module does not infer satellite observation
opportunity. Therefore predictions for MODIS-only groups are outside the
validated training population and must be labelled as exploratory.

Counts are source detection records, not unique fires, burned area, FRP, or
375 m pixels. ARCHIVE is used by default; NRT is excluded because it can
overlap ARCHIVE records.

Commands
--------
Prepare all canonical Parquets:
    python src/sensor_harmonization_global.py prepare \
        --input-dir processed/canonical_v1.1 \
        --output-dir processed/harmonization_global

Train / select / evaluate with temporal holdout:
    python src/sensor_harmonization_global.py train \
        --dataset-file processed/harmonization_global/global_monthly_cell_counts.parquet \
        --output-dir processed/harmonization_global/model

Run historical inference on paired groups:
    python src/sensor_harmonization_global.py infer \
        --model-file processed/harmonization_global/model/global_model.joblib \
        --input-file processed/harmonization_global/global_monthly_cell_counts.parquet \
        --output-file processed/harmonization_global/model/historical_predictions.parquet

For exploratory MODIS-only inference, explicitly pass --include-modis-only.
Those rows are flagged OUTSIDE_VALIDATED_POPULATION in the output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_VERSION = "2.0.0"
FEATURES_VERSION = "2.0.0"
RANDOM_STATE = 42

DEFAULT_INPUT_DIR = Path("processed/canonical_v1.1")
DEFAULT_OUTPUT_DIR = Path("processed/harmonization_global")
DEFAULT_DATASET_FILE = DEFAULT_OUTPUT_DIR / "global_monthly_cell_counts.parquet"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "model"
DEFAULT_GRID_SIZE_DEGREES = 0.1
DEFAULT_TRAIN_END_YEAR = 2022
DEFAULT_VALIDATION_START_YEAR = 2023
DEFAULT_VALIDATION_END_YEAR = 2024
DEFAULT_TEST_START_YEAR = 2025
DEFAULT_TEST_END_YEAR = 2025

PLATFORM_COLUMNS = (
    "modis_aqua_count",
    "modis_terra_count",
    "viirs_snpp_count",
    "viirs_noaa20_count",
    "viirs_noaa21_count",
)

KEY_COLUMNS = (
    "region_id",
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "daynight",
)

DATASET_COLUMNS = (
    *KEY_COLUMNS,
    "grid_lat_center_deg",
    "grid_lon_center_deg",
    *PLATFORM_COLUMNS,
    "modis_count",
    "viirs_count",
    "modis_present",
    "viirs_present",
)

NUMERIC_FEATURES = (
    "log_modis_aqua_count",
    "log_modis_terra_count",
    "log_modis_total_count",
    "n_modis_platforms_with_detections",
    "year",
    "latitude_deg",
    "longitude_deg",
    "month_sin",
    "month_cos",
    "is_night",
)
CATEGORICAL_FEATURES = ("region_id",)
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

PREDICTION_COLUMNS = {
    "naive_modis": "prediction_naive_modis",
    "region_month_daynight_mean": "prediction_region_month_daynight_mean",
    "ridge_smearing": "prediction_ridge_smearing",
    "hist_gradient_boosting_poisson": "prediction_hist_gradient_boosting_poisson",
}


def _require_duckdb() -> Any:
    """Import DuckDB only for Parquet preparation/loading commands."""

    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised on user machine
        raise RuntimeError(
            "DuckDB is required for full-dataset Parquet scanning. "
            "Install the project requirements with: pip install -r requirements.txt"
        ) from exc
    return duckdb


def _sql_string(value: str) -> str:
    """Quote a string literal for DuckDB SQL."""

    return "'" + str(value).replace("'", "''") + "'"


def _discover_parquet_files(
    input_dir: Path,
    *,
    exclude_dir: Path | None = None,
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Canonical Parquet directory not found: {input_dir}")

    excluded = Path(exclude_dir).resolve() if exclude_dir is not None else None
    files = []
    for path in input_dir.rglob("*.parquet"):
        if not path.is_file() or path.name.startswith("."):
            continue
        resolved = path.resolve()
        if excluded is not None:
            try:
                resolved.relative_to(excluded)
                continue
            except ValueError:
                pass
        files.append(resolved)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No .parquet files were found recursively under: {input_dir}"
        )
    return files


def _global_aggregation_sql(
    files: list[Path],
    *,
    grid_size_degrees: float,
) -> str:
    """Build SQL that aggregates all input Parquets without loading raw rows."""

    path_list = "[" + ",\n".join(_sql_string(str(path)) for path in files) + "]"
    grid = repr(float(grid_size_degrees))

    return f"""
        WITH raw AS (
            SELECT
                NULLIF(TRIM(CAST(region_id AS VARCHAR)), '') AS region_id,
                UPPER(TRIM(CAST(sensor AS VARCHAR))) AS sensor,
                UPPER(TRIM(CAST(platform AS VARCHAR))) AS platform,
                UPPER(TRIM(CAST(processing_stream AS VARCHAR))) AS processing_stream,
                TRY_CAST(acq_date AS DATE) AS obs_date,
                TRY_CAST(latitude_deg AS DOUBLE) AS latitude_deg,
                TRY_CAST(longitude_deg AS DOUBLE) AS longitude_deg,
                UPPER(TRIM(CAST(daynight AS VARCHAR))) AS daynight
            FROM read_parquet({path_list}, union_by_name = true, hive_partitioning = false)
        ),
        eligible AS (
            SELECT
                region_id,
                YEAR(obs_date)::INTEGER AS year,
                MONTH(obs_date)::INTEGER AS month,
                FLOOR(latitude_deg / {grid})::INTEGER AS grid_lat_index,
                FLOOR(longitude_deg / {grid})::INTEGER AS grid_lon_index,
                daynight,
                sensor,
                platform
            FROM raw
            WHERE processing_stream = 'ARCHIVE'
              AND sensor IN ('MODIS', 'VIIRS')
              AND platform IN ('AQUA', 'TERRA', 'SNPP', 'NOAA-20', 'NOAA-21')
              AND region_id IS NOT NULL
              AND obs_date IS NOT NULL
              AND latitude_deg BETWEEN -90.0 AND 90.0
              AND longitude_deg BETWEEN -180.0 AND 180.0
              AND daynight IN ('D', 'N')
        )
        SELECT
            region_id,
            year,
            month,
            grid_lat_index,
            grid_lon_index,
            daynight,
            LEAST(90.0, ((grid_lat_index + 0.5) * {grid}))::DOUBLE AS grid_lat_center_deg,
            LEAST(180.0, ((grid_lon_index + 0.5) * {grid}))::DOUBLE AS grid_lon_center_deg,
            COUNT(*) FILTER (WHERE sensor = 'MODIS' AND platform = 'AQUA')::BIGINT AS modis_aqua_count,
            COUNT(*) FILTER (WHERE sensor = 'MODIS' AND platform = 'TERRA')::BIGINT AS modis_terra_count,
            COUNT(*) FILTER (WHERE sensor = 'VIIRS' AND platform = 'SNPP')::BIGINT AS viirs_snpp_count,
            COUNT(*) FILTER (WHERE sensor = 'VIIRS' AND platform = 'NOAA-20')::BIGINT AS viirs_noaa20_count,
            COUNT(*) FILTER (WHERE sensor = 'VIIRS' AND platform = 'NOAA-21')::BIGINT AS viirs_noaa21_count,
            (
                COUNT(*) FILTER (WHERE sensor = 'MODIS')
            )::BIGINT AS modis_count,
            (
                COUNT(*) FILTER (WHERE sensor = 'VIIRS')
            )::BIGINT AS viirs_count,
            (COUNT(*) FILTER (WHERE sensor = 'MODIS') > 0) AS modis_present,
            (COUNT(*) FILTER (WHERE sensor = 'VIIRS') > 0) AS viirs_present
        FROM eligible
        GROUP BY
            region_id, year, month, grid_lat_index, grid_lon_index, daynight
    """


def prepare_global_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    memory_limit: str = "8GB",
    threads: int | None = None,
) -> dict[str, Path]:
    """Aggregate every canonical Parquet file using DuckDB's out-of-core scan.

    ARCHIVE is used and NRT is excluded to avoid mixing potentially overlapping
    archive and near-real-time source records. MODIS and VIIRS platform counts
    are preserved separately before calculating family-level totals.
    """

    if not np.isfinite(grid_size_degrees) or not 0 < grid_size_degrees <= 180:
        raise ValueError("grid_size_degrees must be finite and in the interval (0, 180].")
    if threads is not None and threads < 1:
        raise ValueError("threads must be a positive integer.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _discover_parquet_files(input_dir, exclude_dir=output_dir)
    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    duckdb = _require_duckdb()
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
        connection.execute(f"SET temp_directory = {_sql_string(str(temp_dir.resolve()))}")
        if threads is not None:
            connection.execute(f"SET threads = {int(threads)}")

        query = _global_aggregation_sql(
            files,
            grid_size_degrees=grid_size_degrees,
        )
        parquet_path = (output_dir / "global_monthly_cell_counts.parquet").resolve()
        parquet_literal = _sql_string(str(parquet_path))
        connection.execute(
            f"COPY ({query}) TO {parquet_literal} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        # The aggregate is many orders of magnitude smaller than the raw
        # observations. Load only this grouped table to build metadata.
        dataset = connection.execute(
            f"SELECT * FROM read_parquet({parquet_literal})"
        ).fetchdf()
    finally:
        connection.close()

    if dataset.empty:
        raise ValueError(
            "No eligible ARCHIVE MODIS/VIIRS observations were found. "
            "Check that the input directory contains canonical schema Parquets."
        )

    dataset = dataset.loc[:, list(DATASET_COLUMNS)]
    dataset = validate_global_dataset(dataset)

    paired = dataset["modis_present"] & dataset["viirs_present"]
    modis_only = dataset["modis_present"] & ~dataset["viirs_present"]
    viirs_only = ~dataset["modis_present"] & dataset["viirs_present"]

    summary = {
        "features_version": FEATURES_VERSION,
        "input_directory": str(Path(input_dir)),
        "files_discovered": len(files),
        "files": [str(path) for path in files],
        "processing_stream": "ARCHIVE",
        "excluded_streams": ["NRT"],
        "grid_size_degrees": float(grid_size_degrees),
        "grid_definition": (
            "floor(latitude_deg / grid_size_degrees), "
            "floor(longitude_deg / grid_size_degrees); global angular grid"
        ),
        "temporal_unit": "calendar_month",
        "grouping_keys": list(KEY_COLUMNS),
        "platforms": {
            "modis": ["AQUA", "TERRA"],
            "viirs": ["SNPP", "NOAA-20", "NOAA-21"],
        },
        "dataset": {
            "rows": int(len(dataset)),
            "regions": sorted(dataset["region_id"].astype(str).unique().tolist()),
            "year_min": int(dataset["year"].min()),
            "year_max": int(dataset["year"].max()),
            "modis_aqua_observations": int(dataset["modis_aqua_count"].sum()),
            "modis_terra_observations": int(dataset["modis_terra_count"].sum()),
            "viirs_snpp_observations": int(dataset["viirs_snpp_count"].sum()),
            "viirs_noaa20_observations": int(dataset["viirs_noaa20_count"].sum()),
            "viirs_noaa21_observations": int(dataset["viirs_noaa21_count"].sum()),
            "modis_observations_total": int(dataset["modis_count"].sum()),
            "viirs_observations_total": int(dataset["viirs_count"].sum()),
            "groups_with_both_sensors": int(paired.sum()),
            "groups_with_modis_only": int(modis_only.sum()),
            "groups_with_viirs_only": int(viirs_only.sum()),
        },
        "interpretation": {
            "count_unit": "source observation record",
            "target": "total VIIRS detection records across SNPP, NOAA-20 and NOAA-21",
            "predictors": "MODIS Aqua and Terra detection records, month, day/night, year and location",
            "paired_population": (
                "Only region-month-grid-daynight groups with at least one MODIS "
                "and one VIIRS detection are supervised examples."
            ),
            "missingness": (
                "A missing detection record is not evidence of zero fire activity "
                "or confirmed satellite observation coverage."
            ),
            "limitations": [
                "Detection counts are not unique fire events.",
                "The grid is angular, not equal-area.",
                "The model does not estimate observation opportunity or coverage.",
                "Training on paired detection groups introduces selection bias.",
                "Predictions for MODIS-only groups are outside the validated population.",
                "Platform availability changes over time and can affect family totals.",
                "No cross-sensor event deduplication is performed.",
                "ARCHIVE only is used; NRT records are excluded to avoid overlap.",
            ],
        },
    }

    summary_path = output_dir / "global_feature_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"dataset": parquet_path, "summary": summary_path}


def validate_global_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate the aggregated multi-region dataset."""

    if dataset.empty:
        raise ValueError("The global harmonisation dataset is empty.")

    missing = sorted(set(DATASET_COLUMNS) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = dataset.loc[:, list(DATASET_COLUMNS)].copy()
    data["region_id"] = data["region_id"].astype("string").str.strip()
    if data["region_id"].isna().any() or data["region_id"].eq("").any():
        raise ValueError("region_id contains missing or empty values.")

    integer_columns = (
        "year", "month", "grid_lat_index", "grid_lon_index", *PLATFORM_COLUMNS,
        "modis_count", "viirs_count",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"'{column}' must contain finite integer values.")
        data[column] = numeric.astype("int64")

    if not data["month"].between(1, 12).all():
        raise ValueError("month must be between 1 and 12.")
    if not data["daynight"].isin(("D", "N")).all():
        raise ValueError("daynight must contain only 'D' or 'N'.")

    count_columns = (*PLATFORM_COLUMNS, "modis_count", "viirs_count")
    for column in count_columns:
        if (data[column] < 0).any():
            raise ValueError(f"'{column}' cannot contain negative counts.")

    expected_modis = data["modis_aqua_count"] + data["modis_terra_count"]
    expected_viirs = (
        data["viirs_snpp_count"]
        + data["viirs_noaa20_count"]
        + data["viirs_noaa21_count"]
    )
    if not expected_modis.equals(data["modis_count"]):
        raise ValueError("modis_count is inconsistent with platform-specific counts.")
    if not expected_viirs.equals(data["viirs_count"]):
        raise ValueError("viirs_count is inconsistent with platform-specific counts.")

    for column, count_column in (
        ("modis_present", "modis_count"),
        ("viirs_present", "viirs_count"),
    ):
        if data[column].isna().any() or not data[column].isin((True, False, 0, 1)).all():
            raise ValueError(f"'{column}' must contain boolean values.")
        data[column] = data[column].astype(bool)
        if not data[column].equals(data[count_column].gt(0)):
            raise ValueError(f"'{column}' is inconsistent with '{count_column}'.")

    for column, low, high in (
        ("grid_lat_center_deg", -90.0, 90.0),
        ("grid_lon_center_deg", -180.0, 180.0),
    ):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < low - 1e-8).any() or (values > high + 1e-8).any():
            raise ValueError(f"'{column}' contains invalid coordinates.")
        data[column] = values

    duplicates = data.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError("Dataset contains duplicate region/grid/month/daynight groups.")

    return data.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def prepare_paired_splits(
    dataset: pd.DataFrame,
    *,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    validation_start_year: int = DEFAULT_VALIDATION_START_YEAR,
    validation_end_year: int = DEFAULT_VALIDATION_END_YEAR,
    test_start_year: int = DEFAULT_TEST_START_YEAR,
    test_end_year: int = DEFAULT_TEST_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Create chronological splits using paired sensor-presence groups only.

    Data after test_end_year is deliberately not folded into training or test.
    This prevents an incomplete current year from being compared with a full
    historical year by accident.
    """

    if not (train_end_year < validation_start_year <= validation_end_year < test_start_year <= test_end_year):
        raise ValueError("Year ranges must be strictly chronological and non-overlapping.")

    data = validate_global_dataset(dataset)
    paired_mask = data["modis_present"] & data["viirs_present"]
    paired = data.loc[paired_mask].copy()

    training = paired.loc[paired["year"] <= train_end_year].copy()
    validation = paired.loc[paired["year"].between(validation_start_year, validation_end_year)].copy()
    test = paired.loc[paired["year"].between(test_start_year, test_end_year)].copy()

    for name, frame in (("training", training), ("validation", validation), ("test", test)):
        if frame.empty:
            raise ValueError(
                f"{name} split is empty. Check the available years or adjust the CLI year ranges."
            )

    ignored_future = int((paired["year"] > test_end_year).sum())
    ignored_between = int(
        paired["year"].between(train_end_year + 1, validation_start_year - 1).sum()
        + paired["year"].between(validation_end_year + 1, test_start_year - 1).sum()
    )
    split_counts = {
        "all_groups": int(len(data)),
        "paired_groups": int(len(paired)),
        "one_sided_groups_excluded": int(len(data) - len(paired)),
        "training_groups": int(len(training)),
        "validation_groups": int(len(validation)),
        "test_groups": int(len(test)),
        "paired_groups_after_test_window_not_used": ignored_future,
        "paired_groups_in_unassigned_year_gaps": ignored_between,
    }
    return (
        training.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
        split_counts,
    )


def build_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build predictor-only features; VIIRS counts never enter this function."""

    required = {
        "region_id", "year", "month", "daynight", "grid_lat_center_deg",
        "grid_lon_center_deg", "modis_aqua_count", "modis_terra_count", "modis_count",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Cannot build model features; missing columns: {missing}")

    month = pd.to_numeric(dataset["month"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(month).all() or not ((month >= 1) & (month <= 12)).all():
        raise ValueError("month must contain finite values from 1 to 12.")

    features = pd.DataFrame(index=dataset.index)
    features["log_modis_aqua_count"] = np.log1p(
        pd.to_numeric(dataset["modis_aqua_count"], errors="raise").astype(np.float64)
    )
    features["log_modis_terra_count"] = np.log1p(
        pd.to_numeric(dataset["modis_terra_count"], errors="raise").astype(np.float64)
    )
    features["log_modis_total_count"] = np.log1p(
        pd.to_numeric(dataset["modis_count"], errors="raise").astype(np.float64)
    )
    features["n_modis_platforms_with_detections"] = (
        dataset["modis_aqua_count"].gt(0).astype(np.float64)
        + dataset["modis_terra_count"].gt(0).astype(np.float64)
    )
    features["year"] = pd.to_numeric(dataset["year"], errors="raise").astype(np.float64)
    features["latitude_deg"] = pd.to_numeric(
        dataset["grid_lat_center_deg"], errors="raise"
    ).astype(np.float64)
    features["longitude_deg"] = pd.to_numeric(
        dataset["grid_lon_center_deg"], errors="raise"
    ).astype(np.float64)

    angle = 2.0 * np.pi * (month - 1.0) / 12.0
    features["month_sin"] = np.sin(angle)
    features["month_cos"] = np.cos(angle)
    features["is_night"] = dataset["daynight"].eq("N").astype(np.float64)
    features["region_id"] = dataset["region_id"].astype(str)

    features = features.loc[:, list(FEATURE_COLUMNS)]
    numeric = features.loc[:, list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Model features contain non-finite values.")
    return features


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "region",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_ridge_smearing(
    training_data: pd.DataFrame,
    *,
    cv_folds: int = 3,
) -> tuple[Pipeline, float]:
    """Fit log-count Ridge and estimate Duan smearing from training-only OOF residuals."""

    if training_data.empty:
        raise ValueError("Cannot fit Ridge on an empty training dataset.")
    if cv_folds < 2 or len(training_data) < cv_folds:
        raise ValueError("Ridge smearing cross-validation needs at least cv_folds rows.")

    features = build_model_features(training_data)
    target_log = np.log1p(training_data["viirs_count"].to_numpy(dtype=np.float64))
    estimator = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            ("regressor", Ridge(alpha=2.0)),
        ]
    )

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_log_predictions = cross_val_predict(
        estimator,
        features,
        target_log,
        cv=splitter,
        n_jobs=1,
        method="predict",
    )
    residuals = target_log - np.asarray(oof_log_predictions, dtype=np.float64)
    smearing_factor = float(np.mean(np.exp(np.clip(residuals, -20.0, 20.0))))
    if not np.isfinite(smearing_factor) or smearing_factor <= 0:
        raise RuntimeError("Could not calculate a finite positive smearing factor.")

    estimator.fit(features, target_log)
    return estimator, smearing_factor


def fit_poisson_gradient_boosting(training_data: pd.DataFrame) -> Pipeline:
    """Fit a nonlinear count-mean model using Poisson loss on raw positive counts."""

    if training_data.empty:
        raise ValueError("Cannot fit Poisson boosting on an empty training dataset.")
    estimator = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_iter=140,
                    max_leaf_nodes=31,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    estimator.fit(
        build_model_features(training_data),
        training_data["viirs_count"].to_numpy(dtype=np.float64),
    )
    return estimator


def predict_ridge(estimator: Pipeline, features: pd.DataFrame, smearing_factor: float) -> np.ndarray:
    log_predictions = np.asarray(estimator.predict(features), dtype=np.float64)
    # E[y + 1] ~= exp(E[log(y + 1)]) * E[exp(residual)].
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = np.exp(np.clip(log_predictions, -20.0, 20.0)) * smearing_factor - 1.0
    if not np.isfinite(predictions).all():
        raise ValueError("Ridge produced non-finite predictions after inverse transformation.")
    return np.maximum(predictions, 0.0)


def predict_region_month_daynight_mean(
    training_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> np.ndarray:
    """Training-only regional seasonal mean with progressively broader fallbacks."""

    if training_data.empty:
        raise ValueError("Cannot calculate baseline from empty training data.")
    exact = training_data.groupby(
        ["region_id", "month", "daynight"], observed=True
    )["viirs_count"].mean().to_dict()
    month_dn = training_data.groupby(
        ["month", "daynight"], observed=True
    )["viirs_count"].mean().to_dict()
    region_mean = training_data.groupby("region_id", observed=True)["viirs_count"].mean().to_dict()
    global_mean = float(training_data["viirs_count"].mean())

    values: list[float] = []
    for region, month, daynight in zip(
        prediction_data["region_id"].astype(str),
        prediction_data["month"].astype(int),
        prediction_data["daynight"].astype(str),
    ):
        value = exact.get((region, month, daynight))
        if value is None:
            value = month_dn.get((month, daynight))
        if value is None:
            value = region_mean.get(region, global_mean)
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _prediction_frame(
    prediction_data: pd.DataFrame,
    *,
    baseline_training_data: pd.DataFrame,
    fitted_models: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    output = prediction_data.loc[
        :, [*KEY_COLUMNS, "modis_count", "viirs_count"]
    ].copy()
    output["prediction_naive_modis"] = prediction_data["modis_count"].to_numpy(dtype=np.float64)
    output["prediction_region_month_daynight_mean"] = predict_region_month_daynight_mean(
        baseline_training_data, prediction_data
    )
    features = build_model_features(prediction_data)

    for name, details in fitted_models.items():
        estimator = details["estimator"]
        if name == "ridge_smearing":
            predicted = predict_ridge(
                estimator,
                features,
                float(details["smearing_factor"]),
            )
        elif name == "hist_gradient_boosting_poisson":
            predicted = np.asarray(estimator.predict(features), dtype=np.float64)
            predicted = np.maximum(predicted, 0.0)
        else:
            raise ValueError(f"Unsupported fitted model: {name}")
        if len(predicted) != len(output) or not np.isfinite(predicted).all():
            raise ValueError(f"Model '{name}' returned invalid predictions.")
        output[PREDICTION_COLUMNS[name]] = predicted
    return output


def _season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    split_name: str,
    prediction_columns: Iterable[str],
) -> pd.DataFrame:
    """Report overall and stratified original-scale count metrics."""

    if predictions.empty:
        raise ValueError("Cannot evaluate an empty prediction frame.")
    prediction_columns = list(prediction_columns)
    required = {"viirs_count", "region_id", "year", "month", "daynight", *prediction_columns}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing evaluation columns: {missing}")

    work = predictions.copy()
    work["season"] = work["month"].map(_season_name)
    work["activity_level"] = pd.cut(
        work["viirs_count"],
        bins=[0, 1, 5, 20, 100, np.inf],
        labels=["1", "2-5", "6-20", "21-100", ">100"],
        include_lowest=True,
    ).astype("string")

    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", work)]
    for dimension in ("region_id", "year", "season", "daynight", "activity_level"):
        for group_value, group in work.groupby(dimension, sort=True, observed=True, dropna=False):
            scopes.append((dimension, str(group_value), group))

    rows: list[dict[str, Any]] = []
    for prediction_column in prediction_columns:
        model_name = prediction_column.removeprefix("prediction_")
        for scope, group_name, subset in scopes:
            actual = subset["viirs_count"].to_numpy(dtype=np.float64)
            predicted = subset[prediction_column].to_numpy(dtype=np.float64)
            errors = predicted - actual
            denominator = float(np.sum(np.abs(actual)))
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "scope": scope,
                    "group": group_name,
                    "n": int(len(subset)),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "signed_bias": float(np.mean(errors)),
                    "wape": float(np.sum(np.abs(errors)) / denominator) if denominator > 0 else math.nan,
                    "mean_actual": float(np.mean(actual)),
                    "mean_prediction": float(np.mean(predicted)),
                }
            )
    return pd.DataFrame(rows)


def _overall_metrics(metrics: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    overall = metrics.loc[metrics["scope"].eq("overall")]
    result: dict[str, dict[str, float | int]] = {}
    for _, row in overall.iterrows():
        result[str(row["model"])] = {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "signed_bias": float(row["signed_bias"]),
            "wape": float(row["wape"]),
        }
    return result


def _fit_models(
    training_data: pd.DataFrame,
    *,
    ridge_cv_folds: int = 3,
    include_poisson_boosting: bool = True,
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    ridge, smear = fit_ridge_smearing(training_data, cv_folds=ridge_cv_folds)
    models["ridge_smearing"] = {
        "estimator": ridge,
        "smearing_factor": smear,
    }
    if include_poisson_boosting:
        models["hist_gradient_boosting_poisson"] = {
            "estimator": fit_poisson_gradient_boosting(training_data),
            "smearing_factor": 1.0,
        }
    return models


def run_experiment(
    dataset_file: Path = DEFAULT_DATASET_FILE,
    output_directory: Path = DEFAULT_MODEL_DIR,
    *,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    validation_start_year: int = DEFAULT_VALIDATION_START_YEAR,
    validation_end_year: int = DEFAULT_VALIDATION_END_YEAR,
    test_start_year: int = DEFAULT_TEST_START_YEAR,
    test_end_year: int = DEFAULT_TEST_END_YEAR,
    ridge_cv_folds: int = 3,
    include_poisson_boosting: bool = True,
) -> dict[str, Path]:
    """Select a model on validation years and evaluate once on held-out test years."""

    dataset_file = Path(dataset_file)
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Prepared global feature Parquet not found: {dataset_file}")
    duckdb = _require_duckdb()
    connection = duckdb.connect(database=":memory:")
    try:
        dataset_literal = _sql_string(str(dataset_file.resolve()))
        data = connection.execute(
            f"SELECT * FROM read_parquet({dataset_literal})"
        ).fetchdf()
    finally:
        connection.close()

    data = validate_global_dataset(data)
    training, validation, test, split_counts = prepare_paired_splits(
        data,
        train_end_year=train_end_year,
        validation_start_year=validation_start_year,
        validation_end_year=validation_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    )

    candidate_models = _fit_models(
        training,
        ridge_cv_folds=ridge_cv_folds,
        include_poisson_boosting=include_poisson_boosting,
    )
    validation_predictions = _prediction_frame(
        validation,
        baseline_training_data=training,
        fitted_models=candidate_models,
    )
    validation_columns = [
        PREDICTION_COLUMNS["naive_modis"],
        PREDICTION_COLUMNS["region_month_daynight_mean"],
        *[PREDICTION_COLUMNS[name] for name in candidate_models],
    ]
    validation_metrics = evaluate_predictions(
        validation_predictions,
        split_name="validation",
        prediction_columns=validation_columns,
    )

    candidate_overall = validation_metrics.loc[
        validation_metrics["scope"].eq("overall")
        & validation_metrics["model"].isin(candidate_models.keys())
    ].sort_values(["mae", "rmse"], ascending=[True, True], kind="stable")
    if candidate_overall.empty:
        raise RuntimeError("No candidate model produced validation metrics.")
    selected_model_name = str(candidate_overall.iloc[0]["model"])

    development = pd.concat([training, validation], ignore_index=True)
    final_models = _fit_models(
        development,
        ridge_cv_folds=ridge_cv_folds,
        include_poisson_boosting=False,
    ) if selected_model_name == "ridge_smearing" else {
        "hist_gradient_boosting_poisson": {
            "estimator": fit_poisson_gradient_boosting(development),
            "smearing_factor": 1.0,
        }
    }

    # Do not include validation metrics or test metrics in fitting. The test
    # period is opened only after candidate selection and final development fit.
    test_predictions = _prediction_frame(
        test,
        baseline_training_data=development,
        fitted_models=final_models,
    )
    test_columns = [
        PREDICTION_COLUMNS["naive_modis"],
        PREDICTION_COLUMNS["region_month_daynight_mean"],
        PREDICTION_COLUMNS[selected_model_name],
    ]
    test_metrics = evaluate_predictions(
        test_predictions,
        split_name="test",
        prediction_columns=test_columns,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    validation_metrics_path = output_directory / "validation_metrics.csv"
    test_metrics_path = output_directory / "test_metrics.csv"
    validation_predictions_path = output_directory / "validation_predictions.parquet"
    test_predictions_path = output_directory / "test_predictions.parquet"
    model_path = output_directory / "global_model.joblib"
    summary_path = output_directory / "global_model_summary.json"

    validation_metrics.to_csv(validation_metrics_path, index=False)
    test_metrics.to_csv(test_metrics_path, index=False)
    validation_predictions.to_parquet(validation_predictions_path, index=False)
    test_predictions.to_parquet(test_predictions_path, index=False)

    selected_details = final_models[selected_model_name]
    artifact = {
        "model_version": MODEL_VERSION,
        "model_name": selected_model_name,
        "estimator": selected_details["estimator"],
        "smearing_factor": float(selected_details["smearing_factor"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": "viirs_count",
        "target_unit": "VIIRS source detection records per monthly grid/daynight group",
        "training_population": "paired MODIS/VIIRS detection groups only",
        "supported_modis_platforms": ["AQUA", "TERRA"],
        "target_viirs_platforms": ["SNPP", "NOAA-20", "NOAA-21"],
        "training_end_year": int(train_end_year),
        "validation_years": [int(validation_start_year), int(validation_end_year)],
        "final_fit_years": [int(training["year"].min()), int(validation_end_year)],
        "test_years": [int(test_start_year), int(test_end_year)],
        "random_state": RANDOM_STATE,
        "limitations": [
            "The model is conditional on at least one detection from both sensor families in a group.",
            "Predictions for MODIS-only groups are outside the validated population.",
            "A missing detection is not a confirmed zero or proof of no observation opportunity.",
            "The model estimates aggregate VIIRS detection records, not unique fires or physical fire intensity.",
            "Platform availability changes over time and may affect system-level counts.",
            "No event matching or cross-sensor deduplication is performed.",
            "The model has not been validated for unseen regions or years beyond the held-out period.",
        ],
    }
    joblib.dump(artifact, model_path)

    summary = {
        "model_version": MODEL_VERSION,
        "dataset_file": str(dataset_file),
        "split_counts": split_counts,
        "splits": {
            "training_years": [int(training["year"].min()), int(training["year"].max())],
            "validation_years": [int(validation_start_year), int(validation_end_year)],
            "test_years": [int(test_start_year), int(test_end_year)],
        },
        "model_selection": {
            "criterion": "lowest validation MAE, with RMSE as tie-breaker",
            "selected_model": selected_model_name,
            "validation_metrics": _overall_metrics(validation_metrics),
        },
        "test_metrics": _overall_metrics(test_metrics),
        "test_metrics_by_region": test_metrics.loc[
            test_metrics["scope"].eq("region_id")
        ].to_dict(orient="records"),
        "smearing_factor": float(selected_details["smearing_factor"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "scientific_scope": (
            "Paired-detection conditional count mapping. Not a correction for all MODIS-only/VIIRS-only groups."
        ),
        "outputs": {
            "validation_metrics": validation_metrics_path.name,
            "test_metrics": test_metrics_path.name,
            "validation_predictions": validation_predictions_path.name,
            "test_predictions": test_predictions_path.name,
            "model": model_path.name,
        },
        "limitations": artifact["limitations"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "model": model_path,
        "summary": summary_path,
        "validation_metrics": validation_metrics_path,
        "test_metrics": test_metrics_path,
        "validation_predictions": validation_predictions_path,
        "test_predictions": test_predictions_path,
    }


def load_model_artifact(model_file: Path) -> dict[str, Any]:
    model_file = Path(model_file)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    artifact = joblib.load(model_file)
    required = {
        "model_version", "model_name", "estimator", "smearing_factor",
        "feature_columns", "target", "training_population",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Model artifact is missing required fields: {missing}")
    if artifact["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"Incompatible model version {artifact['model_version']}; expected {MODEL_VERSION}."
        )
    if artifact["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Artifact feature columns do not match this inference implementation.")
    if artifact["target"] != "viirs_count":
        raise ValueError("Artifact target is not the supported VIIRS count target.")
    if artifact["model_name"] not in ("ridge_smearing", "hist_gradient_boosting_poisson"):
        raise ValueError(f"Unsupported model name: {artifact['model_name']}")
    if not callable(getattr(artifact["estimator"], "predict", None)):
        raise ValueError("Artifact estimator does not provide predict().")
    return artifact


def validate_inference_input(input_data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Validate MODIS predictor rows; VIIRS target columns are optional.

    If VIIRS truth is present, it is used only to identify historical paired
    groups and to report the observed count. It is never passed to the model.
    Feature-only input is accepted for prospective use, but such predictions
    are necessarily outside the paired-detection validation population.
    """

    if input_data.empty:
        raise ValueError("Inference input is empty.")

    required = {
        *KEY_COLUMNS,
        "grid_lat_center_deg",
        "grid_lon_center_deg",
        "modis_aqua_count",
        "modis_terra_count",
        "modis_count",
    }
    missing = sorted(required - set(input_data.columns))
    if missing:
        raise ValueError(f"Inference input is missing required columns: {missing}")

    data = input_data.copy()
    data["region_id"] = data["region_id"].astype("string").str.strip()
    if data["region_id"].isna().any() or data["region_id"].eq("").any():
        raise ValueError("Inference input region_id contains missing or empty values.")

    for column in (
        "year", "month", "grid_lat_index", "grid_lon_index",
        "modis_aqua_count", "modis_terra_count", "modis_count",
    ):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Inference column '{column}' must contain finite integers.")
        data[column] = values.astype("int64")

    if not data["month"].between(1, 12).all():
        raise ValueError("Inference month must be between 1 and 12.")
    if not data["daynight"].isin(("D", "N")).all():
        raise ValueError("Inference daynight must contain only 'D' or 'N'.")
    if (data[["modis_aqua_count", "modis_terra_count", "modis_count"]] < 0).any().any():
        raise ValueError("Inference MODIS counts cannot be negative.")
    if not (
        data["modis_aqua_count"] + data["modis_terra_count"]
    ).equals(data["modis_count"]):
        raise ValueError("Inference modis_count is inconsistent with platform counts.")

    for column, low, high in (
        ("grid_lat_center_deg", -90.1, 90.1),
        ("grid_lon_center_deg", -180.1, 180.1),
    ):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < low).any() or (values > high).any():
            raise ValueError(f"Inference column '{column}' contains invalid coordinates.")
        data[column] = values

    duplicates = data.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError("Inference input contains duplicated region/grid/month/daynight groups.")

    truth_columns = {"viirs_count", "viirs_present"}
    present_truth_columns = truth_columns.intersection(data.columns)
    if present_truth_columns and present_truth_columns != truth_columns:
        raise ValueError("Provide both viirs_count and viirs_present, or neither.")

    has_viirs_truth = present_truth_columns == truth_columns
    if has_viirs_truth:
        values = pd.to_numeric(data["viirs_count"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError("Inference viirs_count must contain finite non-negative integers.")
        data["viirs_count"] = values.astype("int64")
        if data["viirs_present"].isna().any() or not data["viirs_present"].isin((True, False, 0, 1)).all():
            raise ValueError("Inference viirs_present must contain boolean values.")
        data["viirs_present"] = data["viirs_present"].astype(bool)
        if not data["viirs_present"].equals(data["viirs_count"].gt(0)):
            raise ValueError("Inference viirs_present is inconsistent with viirs_count.")

    return data, has_viirs_truth


def predict_global_counts(
    input_data: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    include_modis_only: bool = False,
) -> pd.DataFrame:
    """Predict VIIRS counts and explicitly label out-of-population rows.

    By default, only historical groups with VIIRS truth available and both
    sensors present are predicted. For a feature-only / prospective MODIS
    input, explicitly pass include_modis_only=True; every row is then marked
    OUTSIDE_VALIDATED_POPULATION because VIIRS presence is unknown.
    """

    data, has_viirs_truth = validate_inference_input(input_data)
    if not callable(getattr(artifact.get("estimator"), "predict", None)):
        raise ValueError("Invalid model artifact: no estimator.predict method.")
    if artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Model artifact feature columns do not match.")

    modis_mask = data["modis_count"].gt(0)
    if has_viirs_truth:
        paired_mask = modis_mask & data["viirs_present"]
        selected = data.loc[modis_mask if include_modis_only else paired_mask].copy()
        observed_counts = selected["viirs_count"].to_numpy(dtype=np.float64)
        in_validated_population = selected["viirs_present"].to_numpy(dtype=bool)
    else:
        if not include_modis_only:
            raise ValueError(
                "VIIRS truth is absent, so paired-detection membership cannot be checked. "
                "Pass include_modis_only=True only for explicitly exploratory inference."
            )
        selected = data.loc[modis_mask].copy()
        observed_counts = np.full(len(selected), np.nan, dtype=np.float64)
        in_validated_population = np.zeros(len(selected), dtype=bool)

    if selected.empty:
        raise ValueError("No eligible MODIS groups are available for inference.")

    features = build_model_features(selected)
    model_name = str(artifact["model_name"])
    if model_name == "ridge_smearing":
        predicted = predict_ridge(
            artifact["estimator"],
            features,
            float(artifact["smearing_factor"]),
        )
    elif model_name == "hist_gradient_boosting_poisson":
        predicted = np.maximum(
            np.asarray(artifact["estimator"].predict(features), dtype=np.float64),
            0.0,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    result = selected.loc[:, list(KEY_COLUMNS)].copy()
    result["modis_count"] = selected["modis_count"].to_numpy(dtype=np.int64)
    result["viirs_count_observed"] = observed_counts
    result["viirs_count_estimated"] = predicted
    result["model_name"] = model_name
    result["model_population_status"] = np.where(
        in_validated_population,
        "WITHIN_PAIRED_DETECTION_POPULATION",
        "OUTSIDE_VALIDATED_POPULATION",
    )
    result["is_within_paired_detection_population"] = in_validated_population
    return result


def run_inference(
    model_file: Path,
    input_file: Path,
    output_file: Path,
    *,
    include_modis_only: bool = False,
) -> dict[str, Path]:
    duckdb = _require_duckdb()
    artifact = load_model_artifact(model_file)
    connection = duckdb.connect(database=":memory:")
    try:
        input_literal = _sql_string(str(Path(input_file).resolve()))
        input_data = connection.execute(
            f"SELECT * FROM read_parquet({input_literal})"
        ).fetchdf()
    finally:
        connection.close()

    predictions = predict_global_counts(
        input_data,
        artifact,
        include_modis_only=include_modis_only,
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_file, index=False)
    summary_path = output_file.with_suffix(".json")
    summary = {
        "model_version": artifact["model_version"],
        "model_name": artifact["model_name"],
        "rows_predicted": int(len(predictions)),
        "within_paired_detection_population": int(
            predictions["is_within_paired_detection_population"].sum()
        ),
        "outside_validated_population": int(
            (~predictions["is_within_paired_detection_population"]).sum()
        ),
        "include_modis_only": bool(include_modis_only),
        "warning": (
            "OUTSIDE_VALIDATED_POPULATION estimates are exploratory only. "
            "The model was trained only where both sensors had detections."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"predictions": output_file, "summary": summary_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-dataset MODIS/VIIRS sensor-family count mapping experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Aggregate all canonical Parquet files into a global monthly grid."
    )
    prepare_parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--grid-size-degrees", type=float, default=DEFAULT_GRID_SIZE_DEGREES)
    prepare_parser.add_argument("--memory-limit", default="8GB")
    prepare_parser.add_argument("--threads", type=int, default=None)

    train_parser = subparsers.add_parser(
        "train", help="Select a model on validation years and evaluate a held-out test period."
    )
    train_parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    train_parser.add_argument("--validation-start-year", type=int, default=DEFAULT_VALIDATION_START_YEAR)
    train_parser.add_argument("--validation-end-year", type=int, default=DEFAULT_VALIDATION_END_YEAR)
    train_parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    train_parser.add_argument("--test-end-year", type=int, default=DEFAULT_TEST_END_YEAR)
    train_parser.add_argument("--ridge-cv-folds", type=int, default=3)
    train_parser.add_argument(
        "--skip-poisson-boosting",
        action="store_true",
        help="Train Ridge only (faster, lower memory use).",
    )

    infer_parser = subparsers.add_parser(
        "infer", help="Generate predictions from a prepared global dataset."
    )
    infer_parser.add_argument("--model-file", type=Path, required=True)
    infer_parser.add_argument("--input-file", type=Path, required=True)
    infer_parser.add_argument("--output-file", type=Path, required=True)
    infer_parser.add_argument(
        "--include-modis-only",
        action="store_true",
        help="Also predict MODIS-only groups; these are flagged OUTSIDE_VALIDATED_POPULATION.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        paths = prepare_global_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            grid_size_degrees=args.grid_size_degrees,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print("GLOBAL SENSOR HARMONISATION DATASET PREPARED")
        print(f"Dataset: {paths['dataset']}")
        print(f"Summary: {paths['summary']}")
    elif args.command == "train":
        paths = run_experiment(
            dataset_file=args.dataset_file,
            output_directory=args.output_dir,
            train_end_year=args.train_end_year,
            validation_start_year=args.validation_start_year,
            validation_end_year=args.validation_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            ridge_cv_folds=args.ridge_cv_folds,
            include_poisson_boosting=not args.skip_poisson_boosting,
        )
        print("GLOBAL SENSOR HARMONISATION MODEL TRAINED")
        for label, path in paths.items():
            print(f"{label}: {path}")
    elif args.command == "infer":
        paths = run_inference(
            model_file=args.model_file,
            input_file=args.input_file,
            output_file=args.output_file,
            include_modis_only=args.include_modis_only,
        )
        print("GLOBAL SENSOR HARMONISATION INFERENCE COMPLETE")
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
