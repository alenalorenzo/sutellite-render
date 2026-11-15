
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.sensor_harmonization_features import (
    export_harmonization_dataset,
    prepare_harmonization_dataset,
)


INPUT_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string()),
        pa.field("region_id", pa.string()),
        pa.field("sensor", pa.string()),
        pa.field("platform", pa.string()),
        pa.field("processing_stream", pa.string()),
        pa.field("latitude_deg", pa.float64()),
        pa.field("longitude_deg", pa.float64()),
        pa.field("acq_date", pa.date32()),
        pa.field("daynight", pa.string()),
    ]
)


def make_row(
    observation_id: str,
    platform: str,
    acquisition_date: date,
    *,
    region_id: str = "espana",
    processing_stream: str = "ARCHIVE",
    daynight: str = "D",
    latitude: float = -6.05,
    longitude: float = 13.05,
) -> dict:
    sensor = "MODIS" if platform in {"AQUA", "TERRA"} else "VIIRS"

    return {
        "observation_id": observation_id,
        "region_id": region_id,
        "sensor": sensor,
        "platform": platform,
        "processing_stream": processing_stream,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "acq_date": acquisition_date,
        "daynight": daynight,
    }


def write_fixture(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=INPUT_SCHEMA)
    pq.write_table(table, path)


def create_standard_inputs(root: Path) -> tuple[Path, Path]:
    """Create fixtures with common, MODIS-only and VIIRS-only groups."""

    modis_file = root / "modis.parquet"
    viirs_file = root / "viirs.parquet"

    modis_rows = [
        # Two detections in the same cell-month and day group.
        make_row("m1", "AQUA", date(2013, 1, 5)),
        make_row(
            "m2",
            "AQUA",
            date(2013, 1, 6),
            latitude=-6.04,
            longitude=13.04,
        ),

        # A separate night group in the same cell.
        make_row("m3", "AQUA", date(2013, 1, 5), daynight="N"),

        # A MODIS-only spatial cell.
        make_row(
            "m4",
            "AQUA",
            date(2013, 1, 5),
            latitude=-6.15,
        ),

        # These records must be excluded.
        make_row("m5", "TERRA", date(2013, 1, 5)),
        make_row(
            "m6",
            "AQUA",
            date(2013, 1, 5),
            processing_stream="NRT",
        ),
        make_row("m7", "AQUA", date(2012, 12, 31)),
        make_row("m8", "AQUA", date(2026, 1, 1)),
        make_row(
            "m9",
            "AQUA",
            date(2013, 1, 5),
            region_id="portugal",
        ),
    ]

    viirs_rows = [
        # Common day and night groups.
        make_row("v1", "SNPP", date(2013, 1, 5)),
        make_row("v2", "SNPP", date(2013, 1, 5), daynight="N"),

        # A VIIRS-only spatial cell.
        make_row(
            "v3",
            "SNPP",
            date(2013, 1, 5),
            latitude=-6.25,
        ),

        # These records must be excluded.
        make_row("v4", "NOAA-20", date(2013, 1, 5)),
        make_row(
            "v5",
            "SNPP",
            date(2013, 1, 5),
            processing_stream="NRT",
        ),
        make_row("v6", "SNPP", date(2026, 1, 1)),
    ]

    write_fixture(modis_file, modis_rows)
    write_fixture(viirs_file, viirs_rows)

    return modis_file, viirs_file


def get_group(
    dataset: pd.DataFrame,
    year: int,
    month: int,
    grid_lat_index: int,
    grid_lon_index: int,
    daynight: str,
) -> pd.Series:
    """Retrieve exactly one dataset row using a boolean mask."""

    mask = (
        dataset["year"].eq(year)
        & dataset["month"].eq(month)
        & dataset["grid_lat_index"].eq(grid_lat_index)
        & dataset["grid_lon_index"].eq(grid_lon_index)
        & dataset["daynight"].eq(daynight)
    )

    matches = dataset.loc[mask]

    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one matching group, found {len(matches)}."
        )

    return matches.iloc[0]


class TestPrepareHarmonizationDataset(unittest.TestCase):

    def test_filters_and_aggregates_sensor_observations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            dataset, summary = prepare_harmonization_dataset(
                modis_file,
                viirs_file,
                start_year=2013,
                end_year=2025,
                grid_size_degrees=0.1,
            )

            self.assertEqual(len(dataset), 4)

            self.assertEqual(
                list(dataset.columns),
                [
                    "year",
                    "month",
                    "grid_lat_index",
                    "grid_lon_index",
                    "daynight",
                    "modis_count",
                    "viirs_count",
                    "modis_present",
                    "viirs_present",
                ],
            )

            self.assertEqual(
                summary["input"]["modis"]["rows_read"],
                9,
            )
            self.assertEqual(
                summary["input"]["modis"]["rows_region"],
                8,
            )
            self.assertEqual(
                summary["input"]["modis"]["rows_archive"],
                7,
            )
            self.assertEqual(
                summary["input"]["modis"]["rows_target_platform"],
                6,
            )
            self.assertEqual(
                summary["input"]["modis"]["rows_in_year_range"],
                4,
            )
            self.assertEqual(
                summary["input"]["modis"]["eligible_observations"],
                4,
            )

            self.assertEqual(
                summary["input"]["viirs"]["eligible_observations"],
                3,
            )

            self.assertEqual(
                summary["dataset"]["modis_detection_observations"],
                4,
            )
            self.assertEqual(
                summary["dataset"]["viirs_detection_observations"],
                3,
            )

    def test_outer_join_preserves_both_common_and_one_sided_groups(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            dataset, _ = prepare_harmonization_dataset(
                modis_file,
                viirs_file,
            )

            common_day = get_group(
                dataset,
                2013,
                1,
                -61,
                130,
                "D",
            )
            self.assertEqual(common_day["modis_count"], 2)
            self.assertEqual(common_day["viirs_count"], 1)
            self.assertTrue(common_day["modis_present"])
            self.assertTrue(common_day["viirs_present"])

            common_night = get_group(
                dataset,
                2013,
                1,
                -61,
                130,
                "N",
            )
            self.assertEqual(common_night["modis_count"], 1)
            self.assertEqual(common_night["viirs_count"], 1)
            self.assertTrue(common_night["modis_present"])
            self.assertTrue(common_night["viirs_present"])

            modis_only = get_group(
                dataset,
                2013,
                1,
                -62,
                130,
                "D",
            )
            self.assertEqual(modis_only["modis_count"], 1)
            self.assertEqual(modis_only["viirs_count"], 0)
            self.assertTrue(modis_only["modis_present"])
            self.assertFalse(modis_only["viirs_present"])

            viirs_only = get_group(
                dataset,
                2013,
                1,
                -63,
                130,
                "D",
            )
            self.assertEqual(viirs_only["modis_count"], 0)
            self.assertEqual(viirs_only["viirs_count"], 1)
            self.assertFalse(viirs_only["modis_present"])
            self.assertTrue(viirs_only["viirs_present"])

    def test_negative_coordinates_use_floor_grid_indices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            modis_file = root / "modis.parquet"
            viirs_file = root / "viirs.parquet"

            write_fixture(
                modis_file,
                [
                    make_row(
                        "m1",
                        "AQUA",
                        date(2013, 1, 5),
                        latitude=-6.05,
                        longitude=13.05,
                    )
                ],
            )
            write_fixture(
                viirs_file,
                [
                    make_row(
                        "v1",
                        "SNPP",
                        date(2013, 1, 5),
                        latitude=-6.05,
                        longitude=13.05,
                    )
                ],
            )

            dataset, _ = prepare_harmonization_dataset(
                modis_file,
                viirs_file,
            )

            self.assertEqual(dataset.loc[0, "grid_lat_index"], -61)
            self.assertEqual(dataset.loc[0, "grid_lon_index"], 130)

    def test_summary_reports_detection_coverage_limitations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            _, summary = prepare_harmonization_dataset(
                modis_file,
                viirs_file,
            )

            dataset_summary = summary["dataset"]

            self.assertEqual(dataset_summary["rows"], 4)
            self.assertEqual(
                dataset_summary["groups_with_both_sensors"],
                2,
            )
            self.assertEqual(
                dataset_summary["groups_with_modis_only"],
                1,
            )
            self.assertEqual(
                dataset_summary["groups_with_viirs_only"],
                1,
            )

            self.assertIn(
                "coverage",
                summary["interpretation"]["missing_count"].lower(),
            )

    def test_export_writes_parquet_csv_and_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            dataset, summary = prepare_harmonization_dataset(
                modis_file,
                viirs_file,
            )

            output_directory = root / "output"
            paths = export_harmonization_dataset(
                dataset,
                summary,
                output_directory,
            )

            self.assertTrue(paths["parquet"].is_file())
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["summary"].is_file())

            exported_parquet = pd.read_parquet(paths["parquet"])
            self.assertEqual(len(exported_parquet), len(dataset))

            exported_csv = pd.read_csv(paths["csv"])
            self.assertEqual(len(exported_csv), len(dataset))

            with paths["summary"].open(
                "r",
                encoding="utf-8",
            ) as json_file:
                exported_summary = json.load(json_file)

            self.assertEqual(
                exported_summary["dataset"]["rows"],
                len(dataset),
            )

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file = root / "modis.parquet"
            viirs_file = root / "viirs.parquet"

            pq.write_table(
                pa.table({"observation_id": ["m1"]}),
                modis_file,
            )
            write_fixture(
                viirs_file,
                [
                    make_row(
                        "v1",
                        "SNPP",
                        date(2013, 1, 5),
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required columns",
            ):
                prepare_harmonization_dataset(
                    modis_file,
                    viirs_file,
                )

    def test_rejects_invalid_daynight_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file = root / "modis.parquet"
            viirs_file = root / "viirs.parquet"

            write_fixture(
                modis_file,
                [
                    make_row(
                        "m1",
                        "AQUA",
                        date(2013, 1, 5),
                        daynight="X",
                    )
                ],
            )
            write_fixture(
                viirs_file,
                [
                    make_row(
                        "v1",
                        "SNPP",
                        date(2013, 1, 5),
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "invalid daynight values",
            ):
                prepare_harmonization_dataset(
                    modis_file,
                    viirs_file,
                )

    def test_rejects_duplicate_eligible_observation_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file = root / "modis.parquet"
            viirs_file = root / "viirs.parquet"

            duplicate_rows = [
                make_row(
                    "duplicate",
                    "AQUA",
                    date(2013, 1, 5),
                ),
                make_row(
                    "duplicate",
                    "AQUA",
                    date(2013, 1, 6),
                ),
            ]

            write_fixture(modis_file, duplicate_rows)
            write_fixture(
                viirs_file,
                [
                    make_row(
                        "v1",
                        "SNPP",
                        date(2013, 1, 5),
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicated observation IDs",
            ):
                prepare_harmonization_dataset(
                    modis_file,
                    viirs_file,
                )

    def test_rejects_invalid_grid_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            with self.assertRaisesRegex(
                ValueError,
                "finite positive number",
            ):
                prepare_harmonization_dataset(
                    modis_file,
                    viirs_file,
                    grid_size_degrees=0,
                )

    def test_rejects_reversed_year_range(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modis_file, viirs_file = create_standard_inputs(root)

            with self.assertRaisesRegex(
                ValueError,
                "cannot exceed",
            ):
                prepare_harmonization_dataset(
                    modis_file,
                    viirs_file,
                    start_year=2025,
                    end_year=2013,
                )


if __name__ == "__main__":
    unittest.main()
