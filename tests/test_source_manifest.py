
"""Tests for the NASA FIRMS source-file manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest

from pathlib import Path

from src.source_contracts import ARCHIVE_HEADER
from src.source_manifest import (
    build_source_manifest,
    export_source_manifest,
)


def make_archive_row(
    satellite: str = "Terra",
) -> list[str]:
    """Return a valid sample MODIS ARCHIVE observation."""

    return [
        "-8.1",
        "13.2",
        "330.1",
        "1.0",
        "1.0",
        "2026-08-04",
        "0410",
        satellite,
        "MODIS",
        "80",
        "6.03",
        "310.0",
        "5.2",
        "D",
        "0",
    ]


class TestBuildSourceManifest(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temp_directory.name)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def write_archive(
        self,
        rows: list[list[str]] | None = None,
        header: tuple[str, ...] = ARCHIVE_HEADER,
    ) -> Path:
        """Create a sample MODIS ARCHIVE CSV."""

        folder = self.data_directory / "angola_modis"
        folder.mkdir(parents=True, exist_ok=True)

        file_path = (
            folder / "fire_archive_M-C61_123456.csv"
        )

        if rows is None:
            rows = [make_archive_row()]

        with file_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)

        return file_path

    def test_builds_manifest_for_valid_archive_file(self) -> None:
        file_path = self.write_archive(
            rows=[
                make_archive_row("Terra"),
                make_archive_row("Aqua"),
            ]
        )

        manifest = build_source_manifest(
            self.data_directory,
            generated_at_utc="2026-11-07T12:00:00Z",
        )

        self.assertEqual(manifest["manifest_version"], "1.0")
        self.assertTrue(manifest["snapshot_complete"])

        self.assertEqual(
            manifest["summary"]["files_discovered"],
            1,
        )

        record = manifest["files"][0]

        self.assertEqual(
            record["source_file"],
            "angola_modis/fire_archive_M-C61_123456.csv",
        )

        self.assertEqual(record["filename"], file_path.name)
        self.assertEqual(record["region_id"], "angola")
        self.assertEqual(record["sensor"], "MODIS")
        self.assertEqual(record["platform_group"], "MODIS")
        self.assertEqual(record["processing_stream"], "ARCHIVE")
        self.assertEqual(
            record["processing_provider"],
            "UNIVERSITY_OF_MARYLAND",
        )
        self.assertEqual(record["source_collection"], "MODIS_C61")
        self.assertEqual(record["source_id"], "123456")

        self.assertEqual(record["data_row_count"], 2)
        self.assertTrue(record["row_count_complete"])
        self.assertEqual(record["malformed_row_count"], 0)
        self.assertEqual(record["status"], "VALID")

    def test_calculates_exact_sha256_checksum(self) -> None:
        file_path = self.write_archive()

        manifest = build_source_manifest(self.data_directory)

        expected_checksum = hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()

        self.assertEqual(
            manifest["files"][0]["sha256"],
            expected_checksum,
        )

    def test_detects_multiple_platforms_in_modis_file(self) -> None:
        self.write_archive(
            rows=[
                make_archive_row("Terra"),
                make_archive_row("Aqua"),
                make_archive_row("Terra"),
            ]
        )

        manifest = build_source_manifest(self.data_directory)
        record = manifest["files"][0]

        self.assertEqual(
            record["platforms_detected"],
            ["AQUA", "TERRA"],
        )

        self.assertEqual(
            record["platform_counts"],
            {
                "AQUA": 1,
                "TERRA": 2,
            },
        )

    def test_counts_data_records_without_counting_header(self) -> None:
        self.write_archive(
            rows=[
                make_archive_row(),
                make_archive_row(),
                make_archive_row(),
            ]
        )

        manifest = build_source_manifest(self.data_directory)

        self.assertEqual(
            manifest["files"][0]["data_row_count"],
            3,
        )

        self.assertEqual(
            manifest["summary"]["total_data_rows"],
            3,
        )

    def test_snapshot_id_is_deterministic(self) -> None:
        self.write_archive()

        first = build_source_manifest(
            self.data_directory,
            generated_at_utc="2026-11-07T12:00:00Z",
        )

        second = build_source_manifest(
            self.data_directory,
            generated_at_utc="2026-11-07T13:00:00Z",
        )

        self.assertEqual(
            first["source_snapshot_id"],
            second["source_snapshot_id"],
        )

    def test_snapshot_id_changes_when_source_file_changes(self) -> None:
        file_path = self.write_archive()

        first = build_source_manifest(self.data_directory)

        with file_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(make_archive_row("Aqua"))

        second = build_source_manifest(self.data_directory)

        self.assertNotEqual(
            first["source_snapshot_id"],
            second["source_snapshot_id"],
        )

        self.assertNotEqual(
            first["files"][0]["sha256"],
            second["files"][0]["sha256"],
        )

    def test_header_mismatch_is_reported(self) -> None:
        self.write_archive(
            rows=[make_archive_row()[:-1]],
            header=ARCHIVE_HEADER[:-1],
        )

        manifest = build_source_manifest(self.data_directory)
        record = manifest["files"][0]

        self.assertEqual(record["header_status"], "INVALID")
        self.assertEqual(record["status"], "INVALID")
        self.assertIn("header_mismatch", record["errors"])
        self.assertIsNotNone(record["sha256"])

    def test_malformed_rows_are_counted_and_reported(self) -> None:
        self.write_archive(
            rows=[
                make_archive_row()[:-1],
            ]
        )

        manifest = build_source_manifest(self.data_directory)
        record = manifest["files"][0]

        self.assertEqual(record["data_row_count"], 1)
        self.assertEqual(record["malformed_row_count"], 1)
        self.assertEqual(record["row_structure_status"], "INVALID")
        self.assertEqual(record["status"], "INVALID")

    def test_empty_file_is_hashed_but_marked_invalid(self) -> None:
        folder = self.data_directory / "angola_modis"
        folder.mkdir(parents=True)

        file_path = (
            folder / "fire_archive_M-C61_123456.csv"
        )

        file_path.write_bytes(b"")

        manifest = build_source_manifest(self.data_directory)
        record = manifest["files"][0]

        self.assertEqual(record["data_row_count"], 0)
        self.assertTrue(record["row_count_complete"])
        self.assertIsNotNone(record["sha256"])
        self.assertEqual(record["status"], "INVALID")
        self.assertIn("empty_csv_file", record["errors"])

    def test_unrecognised_source_is_still_included(self) -> None:
        folder = self.data_directory / "unknown_folder"
        folder.mkdir(parents=True)

        file_path = folder / "unknown_file.csv"

        with file_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["latitude", "longitude"])
            writer.writerow(["10", "20"])

        manifest = build_source_manifest(self.data_directory)
        record = manifest["files"][0]

        self.assertEqual(manifest["summary"]["files_discovered"], 1)
        self.assertEqual(record["data_row_count"], 1)
        self.assertEqual(record["contract_status"], "INVALID")
        self.assertEqual(record["status"], "INVALID")
        self.assertIsNotNone(record["sha256"])

    def test_rejects_missing_directory(self) -> None:
        missing_directory = (
            self.data_directory / "does_not_exist"
        )

        with self.assertRaises(FileNotFoundError):
            build_source_manifest(missing_directory)


class TestExportSourceManifest(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temp_directory.name)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_exports_json_and_csv(self) -> None:
        folder = self.data_directory / "angola_modis"
        folder.mkdir()

        file_path = (
            folder / "fire_archive_M-C61_123456.csv"
        )

        with file_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(ARCHIVE_HEADER)
            writer.writerow(make_archive_row())

        manifest = build_source_manifest(self.data_directory)

        output_directory = self.data_directory / "reports"

        json_path, csv_path = export_source_manifest(
            manifest,
            output_directory,
        )

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        saved_json = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        self.assertEqual(
            saved_json["source_snapshot_id"],
            manifest["source_snapshot_id"],
        )

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["source_file"],
            "angola_modis/fire_archive_M-C61_123456.csv",
        )
        self.assertEqual(rows[0]["status"], "VALID")


if __name__ == "__main__":
    unittest.main()
