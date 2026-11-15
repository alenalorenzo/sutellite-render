import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA
from src.profile_data import profile_dataset


def make_modis_row(observation_id: str, record_index: int, frp: float) -> dict:
    return {
        "observation_id": observation_id,
        "region_id": "angola",
        "sensor": "MODIS",
        "platform": "TERRA",
        "satellite_raw": "Terra",
        "instrument_raw": "MODIS",
        "processing_stream": "ARCHIVE",
        "distribution_service": "NASA_FIRMS",
        "processing_provider": "UNIVERSITY_OF_MARYLAND",
        "source_collection": "MODIS_C61",
        "source_product_token_raw": "M-C61",
        "source_file": "angola_modis/fire_archive_M-C61_123456.csv",
        "source_record_index": record_index,
        "firms_version_raw": "6.03",
        "latitude_deg": -6.0 - record_index,
        "longitude_deg": 13.0 + record_index,
        "acq_date": date(2000, 11, 1),
        "acq_time_raw": "1006",
        "acq_time_hhmm_utc": "1006",
        "acq_datetime_utc": datetime(2000, 11, 1, 10, 6, tzinfo=timezone.utc),
        "modis_bt21_22_k": 315.5,
        "modis_bt31_k": 292.0,
        "viirs_bt_i4_k": None,
        "viirs_bt_i5_k": None,
        "frp_mw": frp,
        "scan_km": 1.9,
        "track_km": 1.3,
        "confidence_raw": "67",
        "confidence_system": "MODIS_PERCENT_0_100",
        "confidence_modis_pct": 67.0,
        "confidence_viirs_class": None,
        "detection_type_code": 0,
        "detection_type_label": "presumed_vegetation_fire",
        "daynight": "D",
        "nominal_resolution_m": 1000,
    }


class TestProfileDataset(unittest.TestCase):
    def test_profiles_fixture_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "processed" / "canonical_v1.1"
            region_directory = input_directory / "angola"
            output_directory = root / "reports" / "data_profiles"
            region_directory.mkdir(parents=True)

            table = pa.Table.from_pylist(
                [
                    make_modis_row("a" * 64, 1, 12.5),
                    make_modis_row("b" * 64, 2, -0.5),
                ],
                schema=CANONICAL_SCHEMA,
            )
            pq.write_table(
                table,
                region_directory / "fire_archive_M-C61_123456.parquet",
                compression="zstd",
            )

            report = profile_dataset(
                input_directory=input_directory,
                output_directory=output_directory,
                batch_size=1,
            )

            self.assertEqual(report["files_scanned"], 1)
            self.assertEqual(report["rows_scanned"], 2)
            self.assertEqual(report["counts"]["by_sensor"]["MODIS"], 2)
            self.assertEqual(report["frp_negative_observations"], 1)

            expected_outputs = [
                "profile_summary.json",
                "group_summary.csv",
                "yearly_observations.csv",
                "monthly_observations.csv",
                "field_completeness.csv",
                "categorical_distributions.csv",
                "frp_summary.csv",
                "spatial_bounds.csv",
            ]
            for filename in expected_outputs:
                self.assertTrue((output_directory / filename).is_file(), filename)

            with (output_directory / "frp_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                frp_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(frp_rows), 1)
            self.assertEqual(frp_rows[0]["negative_frp_count"], "1")
            self.assertEqual(frp_rows[0]["positive_frp_count"], "1")
            self.assertEqual(frp_rows[0]["nonnegative_frp_count"], "1")

            with (output_directory / "profile_summary.json").open(
                "r", encoding="utf-8"
            ) as json_file:
                saved_report = json.load(json_file)

            self.assertEqual(saved_report["rows_scanned"], 2)


if __name__ == "__main__":
    unittest.main()
