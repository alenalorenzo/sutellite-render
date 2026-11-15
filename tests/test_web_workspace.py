"""Regression tests for the shared calendar and spatial upload results."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.web_workspace import analyze_upload


class TestWebWorkspace(unittest.TestCase):
    def test_csv_satellite_and_region_views_share_the_same_counts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "firms.csv"
            path.write_text(
                "latitude,longitude,acq_date,instrument,satellite,frp\n"
                "-8,13,2025-08-01,VIIRS,N21,10\n"
                "-8,13,2025-08-02,VIIRS,NOAA-21,\n"
                "-8,13,2025-09-01,MODIS,Terra,20\n",
                encoding="utf-8",
            )
            result = analyze_upload(path, "fire_archive_example.csv", "Angola")

        series = {item["platform"]: item for item in result["series"]}
        self.assertEqual(set(series), {"NOAA-21", "TERRA"})
        self.assertEqual(series["NOAA-21"]["stream"], "ARCHIVE")
        region = series["NOAA-21"]["regions"][0]
        self.assertEqual(region["name"], "Angola")
        self.assertEqual(region["years"][2025]["months"][7]["detections"], 2)
        self.assertEqual(region["map"]["rec"]["n"], [2])
        self.assertEqual(region["map"]["rec"]["w"], [1])
        self.assertEqual(region["map"]["rec"]["f"], [100])

    def test_parquet_summary_compares_preceding_years_without_a_map(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.parquet"
            pd.DataFrame({
                "region": ["Test region"] * 4,
                "year": [2021, 2022, 2023, 2024],
                "month": [8] * 4,
                "detections": [2, 2, 2, 20],
            }).to_parquet(path)
            result = analyze_upload(path, path.name, "")

        region = result["series"][0]["regions"][0]
        self.assertIsNone(region["map"])
        self.assertEqual(region["years"][2024]["months"][7]["referenceYears"], 3)
        self.assertEqual(region["years"][2024]["months"][7]["median"], 2)
        self.assertIsNone(region["years"][2024]["months"][6]["detections"])
        self.assertEqual(region["monthlyFlags"], {"2024-8": {"referenceYears": 3, "median": 2.0}})

    def test_uploaded_cells_use_the_bundled_cover_when_available(self):
        with TemporaryDirectory() as directory, patch(
            "src.web_workspace.bundled_worldcover", return_value={"-18,30": 30}
        ):
            path = Path(directory) / "firms.csv"
            path.write_text(
                "latitude,longitude,acq_date,instrument,satellite\n"
                "-8.9,15.2,2025-08-01,VIIRS,N21\n"
                "-8.9,17.2,2025-08-01,VIIRS,N21\n",
                encoding="utf-8",
            )
            result = analyze_upload(path, path.name, "Angola")

        cells = result["series"][0]["regions"][0]["map"]["cells"]
        self.assertEqual(cells, [[-9.0, 15.0, 30], [-9.0, 17.0, 0]])

    def test_grid_summary_counts_every_detection_in_a_cell(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "grid.parquet"
            pd.DataFrame({
                "region_id": ["Grid region"] * 2,
                "sensor": ["SNPP"] * 2,
                "platform": ["NPP"] * 2,
                "year": [2024] * 2,
                "month": [8] * 2,
                "detections": [7, 3],
                "grid_lat": [14.1, 14.2],
                "grid_lon": [8.1, 8.2],
                "mean_frp": [5.0, 15.0],
            }).to_parquet(path)
            result = analyze_upload(path, path.name, "")

        source = result["series"][0]
        self.assertEqual((source["sensor"], source["platform"]), ("VIIRS", "SNPP"))
        region = source["regions"][0]
        self.assertEqual(region["years"][2024]["total"], 10)
        self.assertEqual(region["map"]["rec"]["n"], [10])
        self.assertEqual(region["map"]["rec"]["f"], [80])

    def test_incomplete_coordinate_pair_explains_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.csv"
            path.write_text("latitude,acq_date\n-8,2025-08-01\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both a latitude and a longitude"):
                analyze_upload(path, path.name, "Angola")
