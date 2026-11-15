"""Unit tests for canonical FIRMS observation mapping."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from src.process_data import _map_observation, _observation_id, run_processing
from src.source_contracts import (
    ARCHIVE_HEADER,
    identify_source_contract,
    validate_source_identity,
)


SOURCE_SHA256 = "a" * 64


def make_raw_row(*, satellite: str, instrument: str, confidence: str, archive: bool) -> dict[str, str]:
    row = {
        "latitude": "-8.1234",
        "longitude": "13.4567",
        "brightness": "321.5",
        "scan": "1.2",
        "track": "1.4",
        "acq_date": "2026-11-07",
        "acq_time": "0350",
        "satellite": satellite,
        "instrument": instrument,
        "confidence": confidence,
        "version": "6.1NRT",
        "bright_t31": "301.25",
        "frp": "-0.66",
        "daynight": "n",
    }
    if archive:
        row["type"] = "0"
    return row


class TestObservationId(unittest.TestCase):
    def test_id_is_deterministic(self):
        first = _observation_id(SOURCE_SHA256, "angola_modis/file.csv", 1)
        second = _observation_id(SOURCE_SHA256, "angola_modis/file.csv", 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_id_changes_when_source_identity_changes(self):
        base = _observation_id(SOURCE_SHA256, "angola_modis/file.csv", 1)
        different_path = _observation_id(SOURCE_SHA256, "brasil_modis/file.csv", 1)
        different_index = _observation_id(SOURCE_SHA256, "angola_modis/file.csv", 2)
        different_checksum = _observation_id("b" * 64, "angola_modis/file.csv", 1)
        self.assertEqual(len({base, different_path, different_index, different_checksum}), 4)


class TestCanonicalMapping(unittest.TestCase):
    def test_modis_archive_mapping(self):
        source_path = Path("angola_modis/fire_archive_M-C61_838781.csv")
        contract = identify_source_contract(source_path)
        row = make_raw_row(
            satellite="Terra",
            instrument="MODIS",
            confidence="85",
            archive=True,
        )
        platform = validate_source_identity(contract, row["satellite"], row["instrument"])

        result = _map_observation(
            row,
            contract,
            platform,
            "angola_modis/fire_archive_M-C61_838781.csv",
            1,
            SOURCE_SHA256,
        )

        self.assertEqual(result["sensor"], "MODIS")
        self.assertEqual(result["platform"], "TERRA")
        self.assertEqual(result["modis_bt21_22_k"], 321.5)
        self.assertEqual(result["modis_bt31_k"], 301.25)
        self.assertIsNone(result["viirs_bt_i4_k"])
        self.assertIsNone(result["viirs_bt_i5_k"])
        self.assertEqual(result["confidence_modis_pct"], 85.0)
        self.assertIsNone(result["confidence_viirs_class"])
        self.assertEqual(result["detection_type_code"], 0)
        self.assertEqual(result["detection_type_label"], "presumed_vegetation_fire")
        self.assertEqual(result["daynight"], "N")
        self.assertEqual(result["acq_time_hhmm_utc"], "0350")
        self.assertEqual(result["frp_mw"], -0.66)

    def test_viirs_nrt_mapping_does_not_invent_detection_type(self):
        source_path = Path("angola_viirs_snpp/fire_nrt_SV-C2_838781.csv")
        contract = identify_source_contract(source_path)
        row = make_raw_row(
            satellite="SNPP",
            instrument="SNPP",
            confidence="n",
            archive=False,
        )
        platform = validate_source_identity(contract, row["satellite"], row["instrument"])

        result = _map_observation(
            row,
            contract,
            platform,
            "angola_viirs_snpp/fire_nrt_SV-C2_838781.csv",
            2,
            SOURCE_SHA256,
        )

        self.assertEqual(result["sensor"], "VIIRS")
        self.assertEqual(result["platform"], "SNPP")
        self.assertEqual(result["viirs_bt_i4_k"], 321.5)
        self.assertEqual(result["viirs_bt_i5_k"], 301.25)
        self.assertIsNone(result["modis_bt21_22_k"])
        self.assertIsNone(result["modis_bt31_k"])
        self.assertIsNone(result["confidence_modis_pct"])
        self.assertEqual(result["confidence_viirs_class"], "n")
        self.assertIsNone(result["detection_type_code"])
        self.assertIsNone(result["detection_type_label"])


class TestProcessingIntegration(unittest.TestCase):
    def test_run_processing_writes_canonical_parquet_and_quality_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            data_directory = temporary_root / "raw"
            project_root = temporary_root / "project"
            source_directory = data_directory / "angola_modis"
            source_directory.mkdir(parents=True)

            source_file = source_directory / "fire_archive_M-C61_123456.csv"
            raw_row = make_raw_row(
                satellite="Terra",
                instrument="MODIS",
                confidence="85",
                archive=True,
            )

            with source_file.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=ARCHIVE_HEADER)
                writer.writeheader()
                writer.writerow({field: raw_row[field] for field in ARCHIVE_HEADER})

            source_bytes = source_file.read_bytes()
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            relative_source_path = "angola_modis/fire_archive_M-C61_123456.csv"

            manifest_directory = project_root / "reports" / "source_manifests"
            manifest_directory.mkdir(parents=True)
            manifest_path = manifest_directory / "source_manifest_test.json"
            manifest = {
                "manifest_version": "1.0",
                "source_snapshot_id": "sha256:" + ("b" * 64),
                "snapshot_complete": True,
                "summary": {
                    "files_discovered": 1,
                    "total_data_rows": 1,
                    "all_row_counts_complete": True,
                    "all_file_checksums_complete": True,
                },
                "files": [
                    {
                        "source_file": relative_source_path,
                        "filename": source_file.name,
                        "file_size_bytes": len(source_bytes),
                        "sha256": source_sha256,
                        "data_row_count": 1,
                        "row_count_complete": True,
                        "status": "VALID",
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output_directory = project_root / "processed" / "fixture"
            result_directory = run_processing(
                data_directory=data_directory,
                project_root=project_root,
                manifest_path=manifest_path,
                output_directory=output_directory,
                batch_size=1,
            )

            canonical_path = (
                result_directory / "angola" / "fire_archive_M-C61_123456.parquet"
            )
            canonical_table = pq.read_table(canonical_path)
            quality_table = pq.read_table(result_directory / "quality_flags.parquet")
            report = json.loads(
                (result_directory / "processing_report.json").read_text(encoding="utf-8")
            )

            self.assertEqual(canonical_table.num_rows, 1)
            self.assertEqual(canonical_table.column("frp_mw").to_pylist(), [-0.66])
            self.assertEqual(quality_table.num_rows, 1)
            self.assertEqual(quality_table.column("flags").to_pylist(), [["negative_frp"]])
            self.assertEqual(report["summary"]["source_rows"], 1)
            self.assertEqual(report["summary"]["written_rows"], 1)
            self.assertEqual(report["summary"]["rejected_rows"], 0)
            self.assertTrue((result_directory / "rejected_records.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
