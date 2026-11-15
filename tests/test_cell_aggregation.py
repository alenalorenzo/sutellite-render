from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA
from src.cell_aggregation import build_cell_products, detect_monthly_anomalies


def observation(*, platform: str, sensor: str, date: str, frp: float, lon: float = 13.4567) -> dict:
    from datetime import date as date_type, datetime, time, timezone

    parsed = date_type.fromisoformat(date)
    return {
        "observation_id": f"{platform}-{date}-{frp}",
        "region_id": "espana",
        "sensor": sensor,
        "platform": platform,
        "satellite_raw": platform,
        "instrument_raw": "MODIS" if sensor == "MODIS" else "VIIRS",
        "processing_stream": "ARCHIVE",
        "distribution_service": "NASA_FIRMS",
        "processing_provider": "TEST",
        "source_collection": "TEST",
        "source_product_token_raw": "TEST",
        "source_file": "test.csv",
        "source_record_index": 1,
        "firms_version_raw": "test",
        "latitude_deg": -8.1234,
        "longitude_deg": lon,
        "acq_date": parsed,
        "acq_time_raw": "0350",
        "acq_time_hhmm_utc": "0350",
        "acq_datetime_utc": datetime.combine(parsed, time(3, 50), tzinfo=timezone.utc),
        "modis_bt21_22_k": 321.0 if sensor == "MODIS" else None,
        "modis_bt31_k": 301.0 if sensor == "MODIS" else None,
        "viirs_bt_i4_k": 321.0 if sensor == "VIIRS" else None,
        "viirs_bt_i5_k": 301.0 if sensor == "VIIRS" else None,
        "frp_mw": frp,
        "scan_km": 1.0,
        "track_km": 1.0,
        "confidence_raw": "85" if sensor == "MODIS" else "n",
        "confidence_system": "MODIS_PERCENT_0_100" if sensor == "MODIS" else "VIIRS_LNH",
        "confidence_modis_pct": 85.0 if sensor == "MODIS" else None,
        "confidence_viirs_class": "n" if sensor == "VIIRS" else None,
        "detection_type_code": 0,
        "detection_type_label": "presumed_vegetation_fire",
        "daynight": "D",
        "nominal_resolution_m": 1000 if sensor == "MODIS" else 375,
    }


class TestCellAggregation(unittest.TestCase):
    def test_builds_cells_geojson_and_sensor_preserving_breakdown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "input.parquet"
            rows = [
                observation(platform="TERRA", sensor="MODIS", date="2020-08-01", frp=10.0),
                observation(platform="AQUA", sensor="MODIS", date="2020-08-02", frp=20.0),
                observation(platform="SNPP", sensor="VIIRS", date="2020-08-03", frp=-1.0),
            ]
            pq.write_table(pa.Table.from_pylist(rows, schema=CANONICAL_SCHEMA), input_file)
            output = root / "products"
            summary = build_cell_products([input_file], output, cell_degrees=0.5)
            cells = pd.read_parquet(output / "cell_aggregates.parquet")
            detail = pd.read_parquet(output / "cell_breakdown.parquet")
            daily = pd.read_parquet(output / "daily_activity.parquet")
            self.assertEqual(int(cells["observation_count"].sum()), 3)
            self.assertEqual(int(cells["modis_count"].sum()), 2)
            self.assertEqual(int(cells["viirs_count"].sum()), 1)
            self.assertEqual(int(cells["frp_negative_count"].sum()), 1)
            self.assertEqual(int(detail["observation_count"].sum()), 3)
            self.assertEqual(int(daily["observation_count"].sum()), 3)
            self.assertEqual(daily["activity_date"].nunique(), 3)
            self.assertEqual(summary["active_observation_days"], 3)
            self.assertTrue((cells["land_type"] == "No data").all())
            self.assertTrue((output / "map_cells.geojson").is_file())
            self.assertTrue((output / "daily_activity.csv").is_file())
            self.assertEqual(summary["input_observations"], 3)

    def test_anomaly_detection_uses_prior_observed_years_and_never_fills_gaps(self):
        monthly = pd.DataFrame({
            "region_id": ["espana"] * 4,
            "year": [2019, 2020, 2022, 2024],
            "month": [8] * 4,
            "sensor": ["VIIRS"] * 4,
            "platform": ["SNPP"] * 4,
            "processing_stream": ["ARCHIVE"] * 4,
            "observation_count": [10, 11, 9, 100],
        })
        result = detect_monthly_anomalies(monthly)
        final = result.loc[result["year"] == 2024].iloc[0]
        self.assertEqual(final["reference_years"], 3)
        self.assertTrue(final["is_anomaly"])
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
