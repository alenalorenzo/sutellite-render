
"""Independent inference interface for the IA2 sensor model.

Loads an existing trained model and estimates VIIRS SNPP detection counts
from prepared MODIS Aqua input features.

This interface does not train or tune models.

IMPORTANT:
The trained model was evaluated only on groups where both sensors recorded
detections. Predictions outside that paired-detection population are not
validated. The caller must explicitly acknowledge this limitation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.sensor_harmonization_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    build_model_features,
)


DEFAULT_MODEL_FILE = Path(
    "processed/harmonization_spain/selected_model.joblib"
)

DEFAULT_OUTPUT_FILE = Path(
    "processed/harmonization_spain/inference_predictions.csv"
)

REQUIRED_INPUT_COLUMNS = (
    "month",
    "modis_count",
    "daynight",
)

EXPECTED_TARGET = "log1p(viirs_count)"

EXPECTED_POPULATION = (
    "Groups with both MODIS Aqua and VIIRS SNPP detections."
)

PREDICTION_COLUMN = "predicted_viirs_count"


def load_model_artifact(model_file: Path) -> dict[str, Any]:
    """Load and validate a previously trained IA2 model artifact."""

    model_file = Path(model_file)

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_file}"
        )

    artifact = joblib.load(model_file)

    if not isinstance(artifact, dict):
        raise ValueError(
            "The model artifact must contain a dictionary."
        )

    required_keys = {
        "model_version",
        "model_name",
        "estimator",
        "feature_columns",
        "target",
        "population",
    }

    missing_keys = sorted(required_keys - set(artifact))
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing keys: {missing_keys}"
        )

    if artifact["model_version"] != MODEL_VERSION:
        raise ValueError(
            "Incompatible model version: "
            f"expected {MODEL_VERSION}, "
            f"received {artifact['model_version']}."
        )

    if artifact["model_name"] not in ("ridge", "random_forest"):
        raise ValueError(
            f"Unsupported model name: {artifact['model_name']}"
        )

    if artifact["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError(
            "Model feature columns do not match the inference pipeline."
        )

    if artifact["target"] != EXPECTED_TARGET:
        raise ValueError(
            f"Unsupported model target: {artifact['target']}"
        )

    if artifact["population"] != EXPECTED_POPULATION:
        raise ValueError(
            "The model artifact does not declare the expected "
            "paired-detection population."
        )

    predict_method = getattr(artifact["estimator"], "predict", None)

    if not callable(predict_method):
        raise ValueError(
            "The saved estimator does not provide a predict() method."
        )

    return artifact


def validate_inference_input(
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate new observations before constructing model features.

    Input rows must contain a positive MODIS count, a valid month and
    a valid day/night category.

    When VIIRS presence information is supplied, all rows must belong
    to the paired-detection population.
    """

    if input_data.empty:
        raise ValueError("Inference input is empty.")

    data = input_data.copy()

    missing_columns = sorted(
        set(REQUIRED_INPUT_COLUMNS) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Inference input is missing required columns: "
            f"{missing_columns}"
        )

    for column in ("month", "modis_count"):
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Input column '{column}' contains missing or "
                "non-numeric values."
            )

        array = values.to_numpy(dtype=np.float64)

        if not np.isfinite(array).all():
            raise ValueError(
                f"Input column '{column}' contains non-finite values."
            )

        if not np.equal(array, np.floor(array)).all():
            raise ValueError(
                f"Input column '{column}' must contain integer values."
            )

        data[column] = values.astype("int64")

    if not data["month"].between(1, 12).all():
        raise ValueError("Month values must be between 1 and 12.")

    if (data["modis_count"] <= 0).any():
        raise ValueError(
            "MODIS counts must be positive for this paired-detection model."
        )

    if not data["daynight"].isin(("D", "N")).all():
        raise ValueError(
            "Day/night values must contain only 'D' or 'N'."
        )

    # If the caller supplies VIIRS presence information, reject groups
    # that do not match the population used for model training.
    if "viirs_present" in data.columns:
        if data["viirs_present"].isna().any():
            raise ValueError(
                "Input column 'viirs_present' contains missing values."
            )

        if not data["viirs_present"].isin((True, False)).all():
            raise ValueError(
                "Input column 'viirs_present' must contain booleans."
            )

        if not data["viirs_present"].astype(bool).all():
            raise ValueError(
                "Inference input contains groups without VIIRS detections. "
                "These groups are outside the validated model population."
            )

    if "viirs_count" in data.columns:
        viirs_values = pd.to_numeric(
            data["viirs_count"],
            errors="coerce",
        )

        if viirs_values.isna().any():
            raise ValueError(
                "Input column 'viirs_count' contains missing or "
                "non-numeric values."
            )

        viirs_array = viirs_values.to_numpy(dtype=np.float64)

        if (
            not np.isfinite(viirs_array).all()
            or (viirs_array <= 0).any()
        ):
            raise ValueError(
                "When supplied, VIIRS counts must be finite and positive "
                "for the paired-detection population."
            )

    group_columns = (
        "year",
        "month",
        "grid_lat_index",
        "grid_lon_index",
        "daynight",
    )

    if all(column in data.columns for column in group_columns):
        duplicate_mask = data.duplicated(
            subset=list(group_columns),
            keep=False,
        )

        if duplicate_mask.any():
            raise ValueError(
                "Inference input contains duplicated spatial-temporal groups."
            )

    return data


def predict_viirs_counts(
    input_data: pd.DataFrame,
    model_artifact: dict[str, Any],
    *,
    assume_paired_population: bool = False,
) -> pd.DataFrame:
    """Generate VIIRS count estimates using an already trained model.

    The caller must explicitly confirm that the input belongs to the
    paired-detection population. This assertion does not independently
    establish satellite coverage or prove that future groups are paired.
    """

    if not assume_paired_population:
        raise ValueError(
            "Prediction requires explicit acknowledgement of the "
            "paired-detection population. Set "
            "assume_paired_population=True only when this assumption "
            "is appropriate for the input data."
        )

    data = validate_inference_input(input_data)

    # Check the artifact even when this function is called directly
    # rather than through load_model_artifact().
    if model_artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError(
            "Model feature columns do not match the inference pipeline."
        )

    if model_artifact.get("target") != EXPECTED_TARGET:
        raise ValueError("Unsupported model target.")

    estimator = model_artifact.get("estimator")
    predict_method = getattr(estimator, "predict", None)

    if not callable(predict_method):
        raise ValueError(
            "Model artifact does not contain a valid estimator."
        )

    features = build_model_features(data)

    log_predictions = np.asarray(
        predict_method(features),
        dtype=np.float64,
    )

    if log_predictions.ndim != 1 or len(log_predictions) != len(data):
        raise ValueError(
            "The model returned an unexpected prediction shape."
        )

    if not np.isfinite(log_predictions).all():
        raise ValueError(
            "The model returned non-finite log-scale predictions."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        predictions = np.expm1(log_predictions)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Inverse-transformed predictions contain non-finite values."
        )

    predictions = np.maximum(predictions, 0.0)

    result = data.copy()
    result[PREDICTION_COLUMN] = predictions
    result["model_name"] = str(model_artifact["model_name"])

    return result


def read_inference_input(input_file: Path) -> pd.DataFrame:
    """Read CSV or Parquet inference input."""

    input_file = Path(input_file)

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Inference input file not found: {input_file}"
        )

    suffix = input_file.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_file)

    if suffix == ".csv":
        return pd.read_csv(input_file)

    raise ValueError(
        "Unsupported inference input format. Use .parquet or .csv."
    )


def export_predictions(
    predictions: pd.DataFrame,
    output_file: Path,
    model_artifact: dict[str, Any],
) -> dict[str, Path]:
    """Export predictions and an accompanying metadata JSON file."""

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_file.suffix.lower()

    if suffix == ".csv":
        predictions.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )
    elif suffix == ".parquet":
        predictions.to_parquet(
            output_file,
            index=False,
        )
    else:
        raise ValueError(
            "Unsupported output format. Use .parquet or .csv."
        )

    summary_file = output_file.with_suffix(".json")

    summary = {
        "model_version": model_artifact["model_version"],
        "model_name": model_artifact["model_name"],
        "input_population": EXPECTED_POPULATION,
        "prediction_column": PREDICTION_COLUMN,
        "prediction_scale": "original VIIRS detection count scale",
        "inverse_transform": "expm1, clipped at zero",
        "rows_predicted": int(len(predictions)),
        "limitations": [
            "The model was evaluated only on paired MODIS/VIIRS detection groups.",
            "Predictions for MODIS-only groups are not validated.",
            "The paired-population assumption must be assessed by the caller.",
            "The model does not estimate observation coverage or confirmed zero detections.",
        ],
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "predictions": output_file,
        "summary": summary_file,
    }


def run_inference(
    model_file: Path,
    input_file: Path,
    output_file: Path,
    *,
    assume_paired_population: bool,
) -> dict[str, Path]:
    """Load a model, predict input observations and export the results."""

    model_artifact = load_model_artifact(model_file)
    input_data = read_inference_input(input_file)

    predictions = predict_viirs_counts(
        input_data,
        model_artifact,
        assume_paired_population=assume_paired_population,
    )

    return export_predictions(
        predictions,
        output_file,
        model_artifact,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent inference using a trained IA2 "
            "MODIS Aqua / VIIRS SNPP model."
        )
    )

    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="Previously trained joblib model artifact.",
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input CSV or Parquet file containing MODIS features.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV or Parquet prediction file.",
    )

    parser.add_argument(
        "--assume-paired-population",
        action="store_true",
        required=True,
        help=(
            "Explicitly acknowledge that input groups belong to the "
            "paired-detection population used to train and evaluate IA2."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_paths = run_inference(
        model_file=args.model_file,
        input_file=args.input_file,
        output_file=args.output_file,
        assume_paired_population=args.assume_paired_population,
    )

    print("SUTELLITE - IA2 INFERENCE")
    print("=" * 55)

    print(f"Predictions: {output_paths['predictions']}")
    print(f"Summary: {output_paths['summary']}")

    print(
        "\nWARNING: Predictions are conditional on the paired-detection "
        "population. They are not validated for MODIS-only groups."
    )


if __name__ == "__main__":
    main()
