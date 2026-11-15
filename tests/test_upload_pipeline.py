from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from src.source_contracts import ARCHIVE_HEADER, NRT_HEADER
from src.upload_pipeline import (
    process_uploaded_firms_files,
    slugify_region,
)


def raw_row(*, satellite: str, instrument: str, confidence: str, archive: bool, longitude: str = "13.4567", frp: str = "12.5") -> dict[str, str]:
    row = {
        "latitude": "-8.1234",
        "longitude": longitude,
        "brightness": "321.5",
        "scan": "1.2",
        "track": "1.4",
        "acq_date": "2025-08-14",
        "acq_time": "0350",
        "satellite": satellite,
        "instrument": instrument,
        "confidence": confidence,
        "version": "2.0",
        "bright_t31": "301.25",
        "frp": frp,
        "daynight": "D",
    }
    if archive:
        row["type"] = "0"
    return row


def write_firms_file(path: Path, row: dict[str, str], *, archive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ARCHIVE_HEADER if archive else NRT_HEADER
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerow({key: row[key] for key in header})


class TestUploadPipeline(unittest.TestCase):
    def test_slugify_custom_region(self):
        self.assertEqual(slugify_region("Côte d’Ivoire"), "custom_cote_d_ivoire")
        self.assertEqual(slugify_region("Spain"), "custom_spain")

    def test_harmonises_multiple_satellites_into_one_canonical_parquet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modis = root / "fire_archive_M-C61_100001.csv"
            viirs = root / "fire_nrt_J1V-C2_100002.csv"
            write_firms_file(modis, raw_row(
                satellite="Terra", instrument="MODIS", confidence="85", archive=True
            ), archive=True)
            write_firms_file(viirs, raw_row(
                satellite="N20", instrument="VIIRS", confidence="n", archive=False, frp="-0.5"
            ), archive=False)

            output = root / "out"
            result = process_uploaded_firms_files(
                [modis, viirs],
                region_id="espana",
                region_name="España",
                output_dir=output,
            )

            table = pq.read_table(output / "harmonized_observations.parquet")
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(set(table.column("sensor").to_pylist()), {"MODIS", "VIIRS"})
            self.assertEqual(set(table.column("platform").to_pylist()), {"TERRA", "NOAA-20"})
            self.assertEqual(set(table.column("region_id").to_pylist()), {"espana"})
            quality = pq.read_table(output / "quality_flags.parquet")
            self.assertEqual(quality.num_rows, 1)
            self.assertEqual(result["summary"]["written_rows"], 2)
            self.assertEqual(result["summary"]["quality_flagged_rows"], 1)

    def test_custom_region_clips_points_to_uploaded_geojson(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "fire_archive_M-C61_100001.csv"
            write_firms_file(csv_path, raw_row(
                satellite="Aqua", instrument="MODIS", confidence="65", archive=True,
                longitude="13.4567"
            ), archive=True)
            boundary = {
                "type": "Polygon",
                "coordinates": [[[10, -10], [20, -10], [20, 0], [10, 0], [10, -10]]],
            }
            result = process_uploaded_firms_files(
                [csv_path],
                region_id="custom_test_area",
                region_name="Test area",
                output_dir=root / "out",
                boundary_geojson=json.dumps(boundary).encode("utf-8"),
            )
            table = pq.read_table(root / "out" / "harmonized_observations.parquet")
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table.column("region_id").to_pylist(), ["custom_test_area"])
            self.assertEqual(result["boundary_mode"], "geojson_clipped")
            self.assertTrue((root / "out" / "region_boundary.geojson").is_file())
            self.assertIsNotNone(result["boundary_geojson_sha256"])

    def test_custom_region_requires_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "fire_archive_M-C61_100001.csv"
            write_firms_file(csv_path, raw_row(
                satellite="Terra", instrument="MODIS", confidence="85", archive=True
            ), archive=True)
            with self.assertRaisesRegex(ValueError, "GeoJSON"):
                process_uploaded_firms_files(
                    [csv_path], region_id="custom_region", region_name="Custom",
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
