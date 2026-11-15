
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.sensor_harmonization_inference import (
    EXPECTED_POPULATION,
    EXPECTED_TARGET,
    PREDICTION_COLUMN,
    export_predictions,
    load_model_artifact,
    predict_viirs_counts,
    run_inference,
    validate_inference_input,
)
from src.sensor_harmonization_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    build_estimators,
    build_model_features,
)


def make_model_artifact() -> dict:
    """Fit a small synthetic Ridge estimator for inference tests."""

    rows = []

    for month in range(1, 13):
        for daynight in ("D", "N"):
            rows.append(
                {
                    "month": month,
                    "modis_count": 1 + month,
                    "daynight": daynight,
                }
            )

    training_data = pd.DataFrame(rows)

    features = build_model_features(training_data)

    target = np.log1p(
        np.asarray(
            [
                2 + month * 2 + (3 if daynight == "N" else 0)
                for month in range(1, 13)
                for daynight in ("D", "N")
            ],
            dtype=np.float64,
        )
    )

    estimator = build_estimators()["ridge"]
    estimator.fit(features, target)

    return {
        "model_version": MODEL_VERSION,
        "model_name": "ridge",
        "estimator": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": EXPECTED_TARGET,
        "population": EXPECTED_POPULATION,
    }


def make_inference_input() -> pd.DataFrame:
    """Create valid input rows without VIIRS target values."""

    return pd.DataFrame(
        {
            "year": [2026, 2026],
            "month": [1, 7],
            "grid_lat_index": [-61, -62],
            "grid_lon_index": [130, 131],
            "daynight": ["D", "N"],
            "modis_count": [3, 8],
        }
    )


class TestSensorHarmonizationInference(unittest.TestCase):

    def test_loads_valid_model_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_file = Path(temporary_directory) / "model.joblib"

            joblib.dump(
                make_model_artifact(),
                model_file,
            )

            artifact = load_model_artifact(model_file)

            self.assertEqual(artifact["model_name"], "ridge")
            self.assertEqual(
                artifact["feature_columns"],
                list(FEATURE_COLUMNS),
            )

    def test_rejects_missing_model_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_file = Path(temporary_directory) / "missing.joblib"

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Model artifact not found",
            ):
                load_model_artifact(model_file)

    def test_rejects_incompatible_model_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_file = Path(temporary_directory) / "model.joblib"

            invalid_artifact = make_model_artifact()
            invalid_artifact["feature_columns"] = [
                "wrong_feature"
            ]

            joblib.dump(
                invalid_artifact,
                model_file,
            )

            with self.assertRaisesRegex(
                ValueError,
                "feature columns",
            ):
                load_model_artifact(model_file)

    def test_validates_input_without_viirs_target_columns(self):
        input_data = make_inference_input()

        validated = validate_inference_input(input_data)

        self.assertEqual(len(validated), 2)
        self.assertNotIn("viirs_count", validated.columns)

    def test_rejects_missing_required_input_columns(self):
        input_data = pd.DataFrame(
            {
                "month": [1],
                "modis_count": [3],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing required columns",
        ):
            validate_inference_input(input_data)

    def test_rejects_zero_modis_count(self):
        input_data = make_inference_input()
        input_data["modis_count"] = [3, 0]

        with self.assertRaisesRegex(
            ValueError,
            "must be positive",
        ):
            validate_inference_input(input_data)

    def test_rejects_invalid_month(self):
        input_data = make_inference_input()
        input_data["month"] = [1, 13]

        with self.assertRaisesRegex(
            ValueError,
            "between 1 and 12",
        ):
            validate_inference_input(input_data)

    def test_rejects_invalid_daynight(self):
        input_data = make_inference_input()
        input_data["daynight"] = ["D", "X"]

        with self.assertRaisesRegex(
            ValueError,
            "only 'D' or 'N'",
        ):
            validate_inference_input(input_data)

    def test_rejects_unpaired_groups_when_flag_is_supplied(self):
        input_data = make_inference_input()
        input_data["viirs_present"] = [True, False]

        with self.assertRaisesRegex(
            ValueError,
            "outside the validated model population",
        ):
            validate_inference_input(input_data)

    def test_requires_explicit_paired_population_acknowledgement(self):
        input_data = make_inference_input()
        artifact = make_model_artifact()

        with self.assertRaisesRegex(
            ValueError,
            "explicit acknowledgement",
        ):
            predict_viirs_counts(
                input_data,
                artifact,
            )

    def test_predicts_non_negative_counts_without_retraining(self):
        input_data = make_inference_input()
        artifact = make_model_artifact()

        predictions = predict_viirs_counts(
            input_data,
            artifact,
            assume_paired_population=True,
        )

        self.assertEqual(len(predictions), 2)
        self.assertIn(
            PREDICTION_COLUMN,
            predictions.columns,
        )

        predicted_values = predictions[
            PREDICTION_COLUMN
        ].to_numpy(dtype=np.float64)

        self.assertTrue(np.isfinite(predicted_values).all())
        self.assertTrue((predicted_values >= 0).all())

        self.assertTrue(
            predictions["model_name"].eq("ridge").all()
        )

    def test_rejects_duplicate_spatial_temporal_groups(self):
        input_data = make_inference_input()
        duplicate = input_data.iloc[[0]].copy()

        input_data = pd.concat(
            [input_data, duplicate],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicated spatial-temporal groups",
        ):
            validate_inference_input(input_data)

    def test_exports_predictions_and_metadata(self):
        input_data = make_inference_input()
        artifact = make_model_artifact()

        predictions = predict_viirs_counts(
            input_data,
            artifact,
            assume_paired_population=True,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = (
                Path(temporary_directory) / "predictions.csv"
            )

            paths = export_predictions(
                predictions,
                output_file,
                artifact,
            )

            self.assertTrue(paths["predictions"].is_file())
            self.assertTrue(paths["summary"].is_file())

            exported = pd.read_csv(paths["predictions"])

            self.assertEqual(len(exported), 2)
            self.assertIn(
                PREDICTION_COLUMN,
                exported.columns,
            )

    def test_runs_complete_independent_inference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            model_file = root / "model.joblib"
            input_file = root / "input.parquet"
            output_file = root / "predictions.parquet"

            joblib.dump(
                make_model_artifact(),
                model_file,
            )

            make_inference_input().to_parquet(
                input_file,
                index=False,
            )

            output_paths = run_inference(
                model_file=model_file,
                input_file=input_file,
                output_file=output_file,
                assume_paired_population=True,
            )

            self.assertTrue(
                output_paths["predictions"].is_file()
            )
            self.assertTrue(
                output_paths["summary"].is_file()
            )

            exported = pd.read_parquet(
                output_paths["predictions"]
            )

            self.assertEqual(len(exported), 2)
            self.assertIn(
                PREDICTION_COLUMN,
                exported.columns,
            )


if __name__ == "__main__":
    unittest.main()
