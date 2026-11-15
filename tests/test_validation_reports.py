
import csv
import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.validation_reports import (
    build_validation_report,
    create_file_result,
    export_validation_reports,
)


class TestCreateFileResult(unittest.TestCase):

    def test_creates_file_result_from_source_contract(self):
        contract = SimpleNamespace(
            region_id="angola",
            sensor="VIIRS",
            platform_group="SNPP",
            collection="VIIRS_C2",
            processing_stream="NRT",
            processing_provider="NASA_LANCE_FIRMS",
            source_id="123456",
        )

        result = create_file_result(
            contract,
            Path("fire_nrt_SV-C2_123456.csv"),
        )

        self.assertEqual(result["file"], "fire_nrt_SV-C2_123456.csv")
        self.assertEqual(result["region_id"], "angola")
        self.assertEqual(result["sensor"], "VIIRS")
        self.assertEqual(result["platform_group"], "SNPP")
        self.assertEqual(result["collection"], "VIIRS_C2")
        self.assertEqual(result["processing_stream"], "NRT")
        self.assertEqual(result["processing_provider"], "NASA_LANCE_FIRMS")
        self.assertEqual(result["source_id"], "123456")
        self.assertEqual(result["total_rows"], 0)
        self.assertEqual(result["valid_rows"], 0)
        self.assertEqual(result["invalid_rows"], 0)
        self.assertIsInstance(result["errors"], Counter)
        self.assertIsInstance(result["quality_flags"], Counter)


class TestBuildValidationReport(unittest.TestCase):

    def setUp(self):
        self.started_at = datetime(
            2026, 11, 7, 10, 0, tzinfo=timezone.utc
        )
        self.completed_at = datetime(
            2026, 11, 7, 11, 0, tzinfo=timezone.utc
        )

        self.file_results = [
            {
                "file": "fire_archive_M-C61_100.csv",
                "region_id": "angola",
                "sensor": "MODIS",
                "platform_group": "MODIS",
                "collection": "MODIS_C61",
                "processing_stream": "ARCHIVE",
                "processing_provider": "UNIVERSITY_OF_MARYLAND",
                "source_id": "100",
                "total_rows": 100,
                "valid_rows": 98,
                "invalid_rows": 2,
                "quality_flagged_rows": 1,
                "errors": Counter({
                    "latitude:missing": 2,
                }),
                "quality_flags": Counter({
                    "negative_frp": 1,
                }),
                "source_identity_errors": Counter(),
            },
            {
                "file": "fire_nrt_M-C61_101.csv",
                "region_id": "angola",
                "sensor": "MODIS",
                "platform_group": "MODIS",
                "collection": "MODIS_C61",
                "processing_stream": "NRT",
                "processing_provider": "NASA_LANCE_FIRMS",
                "source_id": "101",
                "total_rows": 100,
                "valid_rows": 100,
                "invalid_rows": 0,
                "quality_flagged_rows": 0,
                "errors": Counter(),
                "quality_flags": Counter(),
                "source_identity_errors": Counter(),
            },
            {
                "file": "fire_nrt_SV-C2_102.csv",
                "region_id": "angola",
                "sensor": "VIIRS",
                "platform_group": "SNPP",
                "collection": "VIIRS_C2",
                "processing_stream": "NRT",
                "processing_provider": "NASA_LANCE_FIRMS",
                "source_id": "102",
                "total_rows": 50,
                "valid_rows": 50,
                "invalid_rows": 0,
                "quality_flagged_rows": 1,
                "errors": Counter(),
                "quality_flags": Counter({
                    "negative_frp": 1,
                }),
                "source_identity_errors": Counter(),
            },
        ]

    def test_calculates_global_totals(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        summary = report["summary"]

        self.assertEqual(summary["files_processed"], 3)
        self.assertEqual(summary["total_rows"], 250)
        self.assertEqual(summary["valid_rows"], 248)
        self.assertEqual(summary["invalid_rows"], 2)
        self.assertEqual(summary["quality_flagged_rows"], 2)
        self.assertEqual(summary["valid_percentage"], 99.2)

    def test_aggregates_by_region(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        angola = report["by_region"]["angola"]

        self.assertEqual(angola["files"], 3)
        self.assertEqual(angola["total_rows"], 250)
        self.assertEqual(angola["valid_rows"], 248)
        self.assertEqual(angola["invalid_rows"], 2)

    def test_aggregates_by_sensor(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(
            report["by_sensor"]["MODIS"]["total_rows"],
            200,
        )
        self.assertEqual(
            report["by_sensor"]["VIIRS"]["total_rows"],
            50,
        )

    def test_aggregates_by_processing_stream(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(
            report["by_processing_stream"]["ARCHIVE"]["files"],
            1,
        )
        self.assertEqual(
            report["by_processing_stream"]["NRT"]["files"],
            2,
        )

    def test_aggregates_quality_flags(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(
            report["quality_flags"]["negative_frp"],
            2,
        )

    def test_aggregates_validation_errors(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(
            report["errors"]["latitude:missing"],
            2,
        )

    def test_preserves_source_identity_errors(self):
        self.file_results[0]["source_identity_errors"] = Counter({
            "Missing satellite identifier": 2,
        })

        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(
            report["source_identity_errors"][
                "Missing satellite identifier"
            ],
            2,
        )

    def test_handles_empty_results(self):
        report = build_validation_report(
            [],
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(report["summary"]["files_processed"], 0)
        self.assertEqual(report["summary"]["total_rows"], 0)
        self.assertEqual(report["summary"]["valid_rows"], 0)
        self.assertIsNone(report["summary"]["valid_percentage"])
        self.assertEqual(report["files"], [])

    def test_files_are_sorted_deterministically(self):
        report = build_validation_report(
            list(reversed(self.file_results)),
            self.started_at,
            self.completed_at,
        )

        filenames = [item["file"] for item in report["files"]]

        self.assertEqual(
            filenames,
            sorted(filenames),
        )

    def test_report_contains_timestamps_and_version(self):
        report = build_validation_report(
            self.file_results,
            self.started_at,
            self.completed_at,
        )

        self.assertEqual(report["report_version"], "1.0")
        self.assertEqual(
            report["validation_started_at_utc"],
            self.started_at.isoformat(),
        )
        self.assertEqual(
            report["validation_completed_at_utc"],
            self.completed_at.isoformat(),
        )
        self.assertIn("generated_at_utc", report)


class TestExportValidationReports(unittest.TestCase):

    def setUp(self):
        self.started_at = datetime(
            2026, 11, 7, 10, 0, tzinfo=timezone.utc
        )
        self.completed_at = datetime(
            2026, 11, 7, 11, 0, tzinfo=timezone.utc
        )

        file_result = {
            "file": "fire_nrt_SV-C2_123.csv",
            "region_id": "angola",
            "sensor": "VIIRS",
            "platform_group": "SNPP",
            "collection": "VIIRS_C2",
            "processing_stream": "NRT",
            "processing_provider": "NASA_LANCE_FIRMS",
            "source_id": "123",
            "total_rows": 10,
            "valid_rows": 10,
            "invalid_rows": 0,
            "quality_flagged_rows": 1,
            "errors": Counter(),
            "quality_flags": Counter({
                "negative_frp": 1,
            }),
            "source_identity_errors": Counter(),
        }

        self.report = build_validation_report(
            [file_result],
            self.started_at,
            self.completed_at,
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "reports"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exports_json_and_csv(self):
        json_path, csv_path = export_validation_reports(
            self.report,
            self.output_dir,
        )

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertEqual(json_path.suffix, ".json")
        self.assertEqual(csv_path.suffix, ".csv")

    def test_json_report_is_valid_and_complete(self):
        json_path, _ = export_validation_reports(
            self.report,
            self.output_dir,
        )

        with json_path.open(encoding="utf-8") as file:
            loaded = json.load(file)

        self.assertEqual(
            loaded["summary"]["total_rows"],
            10,
        )
        self.assertEqual(
            loaded["files"][0]["quality_flags"]["negative_frp"],
            1,
        )

    def test_csv_contains_one_row_per_file(self):
        _, csv_path = export_validation_reports(
            self.report,
            self.output_dir,
        )

        with csv_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], "fire_nrt_SV-C2_123.csv")
        self.assertEqual(rows[0]["total_rows"], "10")

        quality_flags = json.loads(rows[0]["quality_flags"])

        self.assertEqual(quality_flags["negative_frp"], 1)


if __name__ == "__main__":
    unittest.main()
