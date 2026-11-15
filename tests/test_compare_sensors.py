import csv
import json
import tempfile
import unittest
from datetime import date, datetime, time, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA
from src.compare_sensors import compare_dataset


PLATFORM_DETAILS = {
    "TERRA": ("MODIS", "Terra", 1000, "M-C61", "MODIS_C61"),
    "AQUA": ("MODIS", "Aqua", 1000, "M-C61", "MODIS_C61"),
    "SNPP": ("VIIRS", "NPP", 375, "SV-C2", "VIIRS_C2"),
    "NOAA-20": ("VIIRS", "NOAA-20", 375, "J1V-C2", "VIIRS_C2"),
}


def make_row(
    observation_id: str,
    platform: str,
    acquisition_date: date,
    *,
    daynight: str = "D",
    frp: float | None = 10.0,
    processing_stream: str = "ARCHIVE",
    latitude: float = -6.05,
    longitude: float = 13.05,
) -> dict:
    sensor, satellite, resolution, token, collection = PLATFORM_DETAILS[platform]
    is_modis = sensor == "MODIS"
    time_value = time(10, 6) if daynight == "D" else time(22, 6)
    acquired_at = datetime.combine(
        acquisition_date,
        time_value,
        tzinfo=timezone.utc,
    )
    return {
        "observation_id": observation_id * 64,
        "region_id": "angola",
        "sensor": sensor,
        "platform": platform,
        "satellite_raw": satellite,
        "instrument_raw": "MODIS" if is_modis else "VIIRS",
        "processing_stream": processing_stream,
        "distribution_service": "NASA_FIRMS",
        "processing_provider": (
            "UNIVERSITY_OF_MARYLAND"
            if processing_stream == "ARCHIVE"
            else "NASA_LANCE_FIRMS"
        ),
        "source_collection": collection,
        "source_product_token_raw": token,
        "source_file": f"angola_{platform.lower()}/fixture.csv",
        "source_record_index": int(observation_id, 16),
        "firms_version_raw": "6.03" if is_modis else "2",
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "acq_date": acquisition_date,
        "acq_time_raw": acquired_at.strftime("%H%M"),
        "acq_time_hhmm_utc": acquired_at.strftime("%H%M"),
        "acq_datetime_utc": acquired_at,
        "modis_bt21_22_k": 315.0 if is_modis else None,
        "modis_bt31_k": 292.0 if is_modis else None,
        "viirs_bt_i4_k": None if is_modis else 330.0,
        "viirs_bt_i5_k": None if is_modis else 295.0,
        "frp_mw": frp,
        "scan_km": 1.0 if is_modis else 0.4,
        "track_km": 1.0 if is_modis else 0.4,
        "confidence_raw": "80" if is_modis else "n",
        "confidence_system": (
            "MODIS_PERCENT_0_100" if is_modis else "VIIRS_LNH"
        ),
        "confidence_modis_pct": 80.0 if is_modis else None,
        "confidence_viirs_class": None if is_modis else "n",
        "detection_type_code": None if processing_stream == "NRT" else 0,
        "detection_type_label": (
            None if processing_stream == "NRT" else "presumed_vegetation_fire"
        ),
        "daynight": daynight,
        "nominal_resolution_m": resolution,
    }


def write_fixture_parquet(region_directory: Path, filename: str, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=CANONICAL_SCHEMA)
    pq.write_table(
        table,
        region_directory / filename,
        compression="zstd",
    )


class TestCompareDataset(unittest.TestCase):
    def test_filters_archive_platforms_and_years_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "processed" / "canonical_v1.1"
            region_directory = input_directory / "angola"
            output_directory = root / "reports" / "sensor_comparison"
            region_directory.mkdir(parents=True)

            write_fixture_parquet(
                region_directory,
                "terra_archive.parquet",
                [
                    make_row("1", "TERRA", date(2013, 1, 5), daynight="D", frp=10.0),
                    make_row("2", "TERRA", date(2013, 1, 5), daynight="N", frp=-0.5),
                ],
            )
            write_fixture_parquet(
                region_directory,
                "aqua_archive.parquet",
                [
                    make_row("3", "AQUA", date(2013, 1, 6), daynight="D", frp=20.0),
                    make_row(
                        "4",
                        "AQUA",
                        date(2013, 1, 7),
                        processing_stream="NRT",
                        frp=30.0,
                    ),
                ],
            )
            write_fixture_parquet(
                region_directory,
                "snpp_archive.parquet",
                [
                    make_row("5", "SNPP", date(2013, 1, 5), daynight="D", frp=5.0),
                    make_row("6", "SNPP", date(2013, 1, 6), daynight="N", frp=8.0),
                    make_row("7", "SNPP", date(2012, 12, 31), frp=12.0),
                    make_row("8", "SNPP", date(2026, 1, 1), frp=14.0),
                ],
            )
            write_fixture_parquet(
                region_directory,
                "noaa20_archive.parquet",
                [make_row("9", "NOAA-20", date(2013, 1, 5), frp=7.0)],
            )

            summary = compare_dataset(
                input_directory,
                output_directory,
                start_year=2013,
                end_year=2025,
                batch_size=2,
                grid_size_degrees=0.1,
            )

            self.assertEqual(summary["files_scanned"], 4)
            self.assertEqual(summary["rows_scanned"], 9)
            self.assertEqual(summary["archive_rows"], 8)
            self.assertEqual(summary["archive_target_platform_rows"], 7)
            self.assertEqual(summary["eligible_rows"], 5)
            self.assertEqual(summary["rows_excluded_non_archive"], 1)
            self.assertEqual(summary["rows_excluded_other_archive_platforms"], 1)
            self.assertEqual(
                summary["rows_excluded_outside_period_on_target_archive_platforms"],
                2,
            )
            self.assertEqual(
                summary["platform_observations"],
                {"MODIS_AQUA": 1, "MODIS_TERRA": 2, "VIIRS_SNPP": 2},
            )

            expected_outputs = [
                "monthly_platform_counts.csv",
                "monthly_sensor_comparison.csv",
                "monthly_frp_summary.csv",
                "monthly_spatial_grid_counts.csv",
                "comparison_summary.json",
            ]
            for filename in expected_outputs:
                self.assertTrue((output_directory / filename).is_file(), filename)

            with (output_directory / "monthly_platform_counts.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                platform_rows = list(csv.DictReader(csv_file))

            terra_row = next(
                row for row in platform_rows if row["platform"] == "TERRA"
            )
            self.assertEqual(terra_row["observations"], "2")
            self.assertEqual(terra_row["day_observations"], "1")
            self.assertEqual(terra_row["night_observations"], "1")
            self.assertEqual(terra_row["active_detection_days"], "1")
            self.assertEqual(terra_row["calendar_days"], "31")

            with (output_directory / "monthly_sensor_comparison.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                comparison_rows = list(csv.DictReader(csv_file))

            january_day = next(
                row
                for row in comparison_rows
                if row["region_id"] == "angola"
                and row["year"] == "2013"
                and row["month"] == "1"
                and row["daynight"] == "D"
            )
            self.assertEqual(january_day["modis_terra_observations"], "1")
            self.assertEqual(january_day["modis_aqua_observations"], "1")
            self.assertEqual(january_day["viirs_snpp_observations"], "1")
            self.assertEqual(
                january_day["terra_to_snpp_observation_count_ratio"], "1.0"
            )
            self.assertEqual(
                january_day["aqua_to_snpp_observation_count_ratio"], "1.0"
            )

            january_night = next(
                row
                for row in comparison_rows
                if row["region_id"] == "angola"
                and row["year"] == "2013"
                and row["month"] == "1"
                and row["daynight"] == "N"
            )
            self.assertEqual(january_night["modis_terra_observations"], "1")
            self.assertEqual(january_night["viirs_snpp_observations"], "1")

            with (output_directory / "monthly_frp_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                frp_rows = list(csv.DictReader(csv_file))

            terra_day = next(
                row
                for row in frp_rows
                if row["platform"] == "TERRA" and row["daynight"] == "D"
            )
            terra_night = next(
                row
                for row in frp_rows
                if row["platform"] == "TERRA" and row["daynight"] == "N"
            )
            self.assertEqual(terra_day["nonnegative_frp_count"], "1")
            self.assertEqual(terra_day["mean_nonnegative_frp_mw"], "10.0")
            self.assertEqual(terra_night["negative_frp_count"], "1")
            self.assertEqual(terra_night["nonnegative_frp_count"], "0")

            with (output_directory / "monthly_spatial_grid_counts.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                grid_rows = list(csv.DictReader(csv_file))

            terra_grid = next(
                row
                for row in grid_rows
                if row["platform"] == "TERRA" and row["daynight"] == "D"
            )
            self.assertEqual(terra_grid["grid_lat_index"], "-61")
            self.assertEqual(terra_grid["grid_lon_index"], "130")
            self.assertEqual(terra_grid["observations"], "1")

            with (output_directory / "comparison_summary.json").open(
                "r", encoding="utf-8"
            ) as json_file:
                saved_summary = json.load(json_file)

            self.assertFalse(
                saved_summary["methodology"]["unique_fire_events_identified"]
            )
            self.assertFalse(
                saved_summary["methodology"]["cross_sensor_matching_performed"]
            )

    def test_rejects_invalid_year_range(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            input_directory.mkdir()

            with self.assertRaisesRegex(ValueError, "valid year range"):
                compare_dataset(
                    input_directory,
                    root / "output",
                    start_year=2025,
                    end_year=2013,
                )

    def test_rejects_invalid_grid_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            input_directory.mkdir()

            with self.assertRaisesRegex(ValueError, "grid_size_degrees"):
                compare_dataset(
                    input_directory,
                    root / "output",
                    grid_size_degrees=0,
                )


if __name__ == "__main__":
    unittest.main()
