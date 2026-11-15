
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.sensor_harmonization_model import (
    build_model_features,
    evaluate_predictions,
    prepare_paired_splits,
    run_experiment,
    validate_model_dataset,
)


def make_synthetic_dataset() -> pd.DataFrame:
    """Generate paired and one-sided groups across all experiment years."""

    rows = []

    for year in range(2013, 2026):
        for month in range(1, 13):
            for daynight in ("D", "N"):
                modis_count = 1 + (month % 4)
                viirs_count = 1 + ((month + year) % 7)

                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "grid_lat_index": -61,
                        "grid_lon_index": 130,
                        "daynight": daynight,
                        "modis_count": modis_count,
                        "viirs_count": viirs_count,
                        "modis_present": True,
                        "viirs_present": True,
                    }
                )

        # Add one MODIS-only group per year.
        rows.append(
            {
                "year": year,
                "month": 1,
                "grid_lat_index": -70,
                "grid_lon_index": 130,
                "daynight": "D",
                "modis_count": 1,
                "viirs_count": 0,
                "modis_present": True,
                "viirs_present": False,
            }
        )

    return pd.DataFrame(rows)


class TestSensorHarmonizationModel(unittest.TestCase):

    def test_validation_accepts_valid_dataset(self):
        dataset = make_synthetic_dataset()

        validated = validate_model_dataset(dataset)

        self.assertEqual(len(validated), len(dataset))

    def test_validation_rejects_missing_columns(self):
        dataset = pd.DataFrame(
            {
                "year": [2013],
                "month": [1],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing required columns",
        ):
            validate_model_dataset(dataset)

    def test_validation_rejects_inconsistent_presence_flags(self):
        dataset = make_synthetic_dataset()
        dataset.loc[0, "modis_present"] = False

        with self.assertRaisesRegex(
            ValueError,
            "inconsistent",
        ):
            validate_model_dataset(dataset)

    def test_validation_rejects_duplicate_groups(self):
        dataset = make_synthetic_dataset()
        duplicate = dataset.iloc[[0]].copy()

        dataset = pd.concat(
            [dataset, duplicate],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicated spatial-temporal groups",
        ):
            validate_model_dataset(dataset)

    def test_paired_split_excludes_one_sided_groups(self):
        dataset = make_synthetic_dataset()

        training, validation, test = prepare_paired_splits(dataset)

        self.assertEqual(len(training), 240)
        self.assertEqual(len(validation), 48)
        self.assertEqual(len(test), 24)

        for split in (training, validation, test):
            self.assertTrue(split["modis_present"].all())
            self.assertTrue(split["viirs_present"].all())

        self.assertEqual(
            set(training["year"].unique()),
            set(range(2013, 2023)),
        )
        self.assertEqual(
            set(validation["year"].unique()),
            {2023, 2024},
        )
        self.assertEqual(
            set(test["year"].unique()),
            {2025},
        )

    def test_paired_split_rejects_empty_intersection(self):
        dataset = make_synthetic_dataset()

        dataset["viirs_present"] = False
        dataset["viirs_count"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "No paired",
        ):
            prepare_paired_splits(dataset)

    def test_feature_builder_uses_expected_transformations(self):
        dataset = pd.DataFrame(
            {
                "month": [1, 12],
                "modis_count": [3, 7],
                "daynight": ["D", "N"],
            }
        )

        features = build_model_features(dataset)

        self.assertEqual(
            list(features.columns),
            [
                "modis_log_count",
                "month_sin",
                "month_cos",
                "is_night",
            ],
        )

        modis_log_values = features[
            "modis_log_count"
        ].to_numpy(dtype=np.float64)

        month_sin_values = features[
            "month_sin"
        ].to_numpy(dtype=np.float64)

        month_cos_values = features[
            "month_cos"
        ].to_numpy(dtype=np.float64)

        night_values = features[
            "is_night"
        ].to_numpy(dtype=np.float64)

        np.testing.assert_allclose(
            modis_log_values,
            np.array(
                [
                    np.log1p(3),
                    np.log1p(7),
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1e-12,
        )

        # January must have sin(0) = 0.
        np.testing.assert_allclose(
            month_sin_values[0:1],
            np.array([0.0], dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )

        # January must have cos(0) = 1.
        np.testing.assert_allclose(
            month_cos_values[0:1],
            np.array([1.0], dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )

        # Day = 0, Night = 1.
        np.testing.assert_array_equal(
            night_values,
            np.array([0.0, 1.0], dtype=np.float64),
        )

        # December is adjacent to January in the circular encoding.
        np.testing.assert_allclose(
            month_sin_values[1:2],
            np.array([-0.5], dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )

    def test_evaluation_reports_expected_metrics(self):
        predictions = pd.DataFrame(
            {
                "year": [2025, 2025],
                "month": [1, 7],
                "grid_lat_index": [-61, -62],
                "grid_lon_index": [130, 131],
                "daynight": ["D", "N"],
                "modis_count": [2, 4],
                "viirs_count": [3, 5],
                "prediction_test": [2.0, 7.0],
            }
        )

        metrics = evaluate_predictions(
            predictions,
            split_name="test",
            prediction_columns=["prediction_test"],
        )

        overall = metrics.loc[
            metrics["scope"].eq("overall")
        ]

        self.assertEqual(len(overall), 1)

        actual_metrics = overall[
            [
                "n",
                "mae",
                "rmse",
                "signed_bias",
            ]
        ].to_numpy(dtype=np.float64)

        expected_metrics = np.array(
            [
                [
                    2.0,
                    1.5,
                    np.sqrt(2.5),
                    0.5,
                ]
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            actual_metrics,
            expected_metrics,
            rtol=0.0,
            atol=1e-12,
        )

        self.assertIn(
            "daynight",
            set(metrics["scope"]),
        )
        self.assertIn(
            "season",
            set(metrics["scope"]),
        )
        self.assertIn(
            "activity_level",
            set(metrics["scope"]),
        )

    def test_complete_experiment_exports_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            dataset_file = root / "synthetic_dataset.parquet"
            output_directory = root / "output"

            dataset = make_synthetic_dataset()
            dataset.to_parquet(
                dataset_file,
                index=False,
            )

            summary = run_experiment(
                dataset_file=dataset_file,
                output_directory=output_directory,
            )

            expected_files = [
                "validation_metrics.csv",
                "test_metrics.csv",
                "validation_predictions.csv",
                "test_predictions.csv",
                "selected_model.joblib",
                "model_summary.json",
            ]

            for filename in expected_files:
                self.assertTrue(
                    (output_directory / filename).is_file(),
                    msg=f"Missing output file: {filename}",
                )

            self.assertEqual(
                summary["split"]["training"]["rows"],
                240,
            )
            self.assertEqual(
                summary["split"]["validation"]["rows"],
                48,
            )
            self.assertEqual(
                summary["split"]["test"]["rows"],
                24,
            )

            validation_metrics = pd.read_csv(
                output_directory / "validation_metrics.csv"
            )
            test_metrics = pd.read_csv(
                output_directory / "test_metrics.csv"
            )

            self.assertIn(
                "random_forest",
                set(validation_metrics["model"]),
            )
            self.assertIn(
                "ridge",
                set(validation_metrics["model"]),
            )

            self.assertEqual(
                set(test_metrics["model"]),
                {
                    "naive_modis",
                    "month_daynight_mean",
                    summary["model_selection"]["selected_model"],
                },
            )


if __name__ == "__main__":
    unittest.main()
