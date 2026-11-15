import unittest

import numpy as np
import pandas as pd

from src.build_harmonized_calendar import (
    AQUA,
    SNPP,
    SOURCE_CALIBRATED,
    SOURCE_OBSERVED,
    SOURCE_UNCALIBRATED,
    TERRA,
    compute_calibration_factors,
    cross_validate_calibration,
    detect_platform_starts,
    harmonize_cells,
    month_index,
    summarize_validation,
)

STARTS = {
    "TERRA": int(month_index(2000, 11)),
    "AQUA": int(month_index(2002, 7)),
    "SNPP": int(month_index(2012, 1)),
}


def cell(region, year, month, daynight, aqua, terra, snpp, lat=10, lon=20):
    return {
        "region_id": region,
        "year": year,
        "month": month,
        "grid_lat_index": lat,
        "grid_lon_index": lon,
        "daynight": daynight,  # present in the source table, pooled by the code
        "grid_lat_center_deg": lat * 0.1 + 0.05,
        "grid_lon_center_deg": lon * 0.1 + 0.05,
        AQUA: aqua,
        TERRA: terra,
        SNPP: snpp,
    }


def calibration_rows(region="r1", month=1, daynight="D", aqua=10, terra=10,
                     snpp_by_year=None):
    snpp_by_year = snpp_by_year or {y: 40 for y in range(2013, 2023)}
    return [
        cell(region, y, month, daynight, aqua, terra, s)
        for y, s in snpp_by_year.items()
    ]


def monthly_factor(factors, config="TERRA_AQUA", month=1, region="r1"):
    return factors[
        (factors["level"] == "region_month")
        & (factors["modis_config"] == config)
        & (factors["month"] == month)
        & (factors["region_id"] == region)
    ].iloc[0]


class PlatformStartTests(unittest.TestCase):
    def test_detects_first_month_per_platform(self):
        totals = pd.DataFrame(
            [
                {"year": 2000, "month": 11, AQUA: 0, TERRA: 5, SNPP: 0},
                {"year": 2002, "month": 7, AQUA: 3, TERRA: 5, SNPP: 0},
                {"year": 2012, "month": 1, AQUA: 3, TERRA: 5, SNPP: 9},
            ]
        )
        self.assertEqual(detect_platform_starts(totals), STARTS)

    def test_missing_platform_raises(self):
        totals = pd.DataFrame(
            [{"year": 2000, "month": 11, AQUA: 0, TERRA: 5, SNPP: 0}]
        )
        with self.assertRaises(ValueError):
            detect_platform_starts(totals)


class CalibrationTests(unittest.TestCase):
    def test_constant_ratio_and_band_collapse(self):
        f = compute_calibration_factors(pd.DataFrame(calibration_rows()))
        row = monthly_factor(f)
        self.assertAlmostEqual(row["ratio"], 2.0)
        self.assertAlmostEqual(row["ratio_low"], 2.0)
        self.assertAlmostEqual(row["ratio_high"], 2.0)
        self.assertEqual(row["n_years"], 10)
        self.assertAlmostEqual(monthly_factor(f, config="TERRA")["ratio"], 4.0)

    def test_day_and_night_are_pooled(self):
        cells = pd.DataFrame(
            calibration_rows(daynight="D")
            + calibration_rows(
                daynight="N", snpp_by_year={y: 20 for y in range(2013, 2023)}
            )
        )
        # per year: MODIS 20 + 20 = 40, S-NPP 40 + 20 = 60
        self.assertAlmostEqual(
            monthly_factor(compute_calibration_factors(cells))["ratio"], 1.5
        )

    def test_band_brackets_central_ratio(self):
        snpp = {y: (30 if y % 2 else 50) for y in range(2013, 2023)}
        f = compute_calibration_factors(
            pd.DataFrame(calibration_rows(snpp_by_year=snpp))
        )
        row = monthly_factor(f)
        self.assertLess(row["ratio_low"], row["ratio"])
        self.assertGreater(row["ratio_high"], row["ratio"])

    def test_too_few_years_gives_nan(self):
        cells = pd.DataFrame(calibration_rows(snpp_by_year={2013: 40, 2014: 40}))
        f = compute_calibration_factors(cells, min_years=3)
        self.assertTrue(f["ratio"].isna().all())

    def test_unstable_band_gives_nan_for_month_but_not_region(self):
        wild = {y: (10 if y % 2 else 100) for y in range(2013, 2023)}
        cells = pd.DataFrame(
            calibration_rows(month=1, snpp_by_year=wild)
            + calibration_rows(month=2)
        )
        f = compute_calibration_factors(cells, max_band_ratio=3.0)
        self.assertTrue(np.isnan(monthly_factor(f, month=1)["ratio"]))
        self.assertAlmostEqual(monthly_factor(f, month=2)["ratio"], 2.0)
        region = f[(f["level"] == "region") & (f["modis_config"] == "TERRA_AQUA")]
        self.assertAlmostEqual(region.iloc[0]["ratio"], (550 + 400) / 400)

    def test_empty_window_raises(self):
        cells = pd.DataFrame([cell("r1", 2005, 1, "D", 1, 1, 0)])
        with self.assertRaises(ValueError):
            compute_calibration_factors(cells)


class HarmonizeTests(unittest.TestCase):
    def setUp(self):
        self.factors = compute_calibration_factors(pd.DataFrame(calibration_rows()))

    def run_harmonize(self, rows, factors=None):
        return harmonize_cells(
            pd.DataFrame(rows), self.factors if factors is None else factors, STARTS
        )

    def test_observed_period_passes_snpp_through(self):
        r = self.run_harmonize([cell("r1", 2015, 1, "D", 10, 10, 37)]).iloc[0]
        self.assertEqual(r["hotspots_harmonized"], 37)
        self.assertEqual(r["hotspots_low"], 37)
        self.assertEqual(r["hotspots_high"], 37)
        self.assertEqual(r["source"], SOURCE_OBSERVED)
        self.assertFalse(r["is_calibrated"])

    def test_terra_aqua_period_scaled_by_ratio(self):
        r = self.run_harmonize([cell("r1", 2005, 1, "D", 5, 5, 0)]).iloc[0]
        self.assertAlmostEqual(r["hotspots_harmonized"], 20.0)
        self.assertEqual(r["source"], SOURCE_CALIBRATED)
        self.assertTrue(r["is_calibrated"])
        self.assertEqual(r["modis_count"], 10)

    def test_terra_only_period_uses_terra_ratio(self):
        out = self.run_harmonize([cell("r1", 2001, 1, "D", 0, 5, 0)])
        self.assertAlmostEqual(out.iloc[0]["hotspots_harmonized"], 20.0)

    def test_day_night_rows_are_pooled_into_one_cell_month(self):
        out = self.run_harmonize(
            [cell("r1", 2005, 1, "D", 5, 5, 0), cell("r1", 2005, 1, "N", 5, 5, 0)]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["modis_count"], 20)
        self.assertAlmostEqual(out.iloc[0]["hotspots_harmonized"], 40.0)

    def test_unstable_month_uses_region_fallback(self):
        wild = {y: (10 if y % 2 else 100) for y in range(2013, 2023)}
        cells = pd.DataFrame(
            calibration_rows(month=1, snpp_by_year=wild)
            + calibration_rows(month=2)
        )
        factors = compute_calibration_factors(cells, max_band_ratio=3.0)
        r = self.run_harmonize(
            [cell("r1", 2005, 1, "D", 5, 5, 0)], factors
        ).iloc[0]
        self.assertAlmostEqual(r["hotspots_harmonized"], 10 * 950 / 400)
        self.assertEqual(r["source"], SOURCE_CALIBRATED)

    def test_region_without_factors_is_flagged_uncalibrated(self):
        r = self.run_harmonize([cell("other", 2005, 1, "D", 5, 5, 0)]).iloc[0]
        self.assertTrue(np.isnan(r["hotspots_harmonized"]))
        self.assertEqual(r["source"], SOURCE_UNCALIBRATED)

    def test_band_ordering_and_missing_columns(self):
        r = self.run_harmonize([cell("r1", 2005, 1, "D", 5, 5, 0)]).iloc[0]
        self.assertLessEqual(r["hotspots_low"], r["hotspots_harmonized"])
        self.assertLessEqual(r["hotspots_harmonized"], r["hotspots_high"])
        with self.assertRaises(ValueError):
            harmonize_cells(
                pd.DataFrame([{"region_id": "r1"}]), self.factors, STARTS
            )


class ValidationTests(unittest.TestCase):
    def test_constant_ratio_is_predicted_perfectly(self):
        detail = cross_validate_calibration(pd.DataFrame(calibration_rows()))
        self.assertEqual(len(detail), 20)  # 10 held-out years x 2 configs
        self.assertTrue((detail["abs_pct_error"] < 1e-9).all())
        summary = summarize_validation(detail)
        self.assertEqual(set(summary["modis_config"]), {"TERRA_AQUA", "TERRA"})
        self.assertTrue((summary["median_abs_pct_error"] < 1e-9).all())
        self.assertTrue((summary["band_coverage"] == 1.0).all())

    def test_error_reflects_year_to_year_variation(self):
        snpp = {y: (30 if y % 2 else 50) for y in range(2013, 2023)}
        detail = cross_validate_calibration(
            pd.DataFrame(calibration_rows(snpp_by_year=snpp))
        )
        both = detail[detail["modis_config"] == "TERRA_AQUA"]
        self.assertGreater(both["abs_pct_error"].median(), 0.1)

    def test_empty_window_raises(self):
        cells = pd.DataFrame([cell("r1", 2005, 1, "D", 1, 1, 0)])
        with self.assertRaises(ValueError):
            cross_validate_calibration(cells)


if __name__ == "__main__":
    unittest.main()