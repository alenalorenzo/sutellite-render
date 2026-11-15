import unittest

import pandas as pd

from src.export_map_cells import (
    build_payload,
    dominant_land,
    local_tile_set,
    plan_tiles,
    tile_name,
    tiles_for_coverage,
)


def fine(rows):
    return pd.DataFrame(rows, columns=["lat_i", "lon_i", "n", "lc"])


class LandTests(unittest.TestCase):
    def test_dominant_class_is_busiest_fine_cell(self):
        out = dominant_land(fine([(0, 0, 10, 10), (1, 1, 30, 40), (7, 0, 5, 30)]), 0.5)
        self.assertEqual(len(out), 2)
        a = out[(out.clat == 0) & (out.clon == 0)].iloc[0]
        self.assertEqual((a.n, a.lc), (40, 40))

    def test_negative_indices_floor_and_bad_size(self):
        out = dominant_land(fine([(-1, -1, 4, 0)]), 0.5)
        self.assertEqual((out.iloc[0].clat, out.iloc[0].clon), (-1, -1))
        with self.assertRaises(ValueError):
            dominant_land(fine([(0, 0, 1, 0)]), 0.25)

    def test_tile_name_and_plan(self):
        self.assertEqual(tile_name(-17.9, 12.1), "S18E012")
        self.assertEqual(tile_name(0.1, -0.1), "N00W003")
        plan = plan_tiles(fine([(1, 1, 5, 0), (-180, 120, 50, 0)]), {"N00E000"})
        self.assertEqual(plan.iloc[0].tile, "S18E012")
        self.assertTrue(plan[plan.tile == "N00E000"].iloc[0].local)

    def test_tiles_for_coverage(self):
        # 100 records in total: tile A (50, local), B (30), C (15), D (5)
        plan = pd.DataFrame({"tile": list("ABCD"), "records": [50, 30, 15, 5],
                             "local": [True, False, False, False]})
        self.assertEqual(tiles_for_coverage(plan, 0.5), 0)
        self.assertEqual(tiles_for_coverage(plan, 0.8), 1)
        self.assertEqual(tiles_for_coverage(plan, 0.95), 2)
        self.assertEqual(tiles_for_coverage(plan, 1.0), 3)

    def test_local_tile_set_missing_folder(self):
        self.assertEqual(local_tile_set(__import__("pathlib").Path("no/such/dir")), set())


class PayloadTests(unittest.TestCase):
    def test_columnar_records_roundtrip(self):
        f = fine([(0, 0, 10, 10), (1, 1, 30, 40)])
        agg = pd.DataFrame(
            [
                (0, 0, 2013, "SNPP", "ARCHIVE", 30, 30, 600.0),
                (0, 0, 2005, "AQUA", "ARCHIVE", 10, 0, 0.0),
                (0, 0, 2026, "NOAA-21", "NRT", 2, 2, 20.0),
            ],
            columns=["clat", "clon", "yr", "platform", "stream", "n", "n_frp", "frp_sum"],
        )
        p = build_payload(f, agg, 0.5, {"AQUA": "MODIS", "SNPP": "VIIRS", "NOAA-21": "VIIRS"})
        m = p["meta"]
        self.assertEqual(m["platforms"], ["AQUA", "SNPP", "NOAA-21"])
        self.assertEqual(m["streams"], ["ARCHIVE", "NRT"])
        self.assertEqual(m["years"], [2005, 2026])
        self.assertEqual(m["total_records"], 42)
        self.assertEqual(p["cells"], [[0.0, 0.0, 40]])
        rec = p["rec"]
        keys = dict(zip(rec["k"], zip(rec["n"], rec["f"])))
        self.assertEqual(keys[2013 * 100 + 1 * 10 + 0], (30, 200))  # SNPP, mean FRP 20.0
        self.assertEqual(keys[2005 * 100 + 0 * 10 + 0], (10, -1))   # AQUA, no FRP
        self.assertEqual(keys[2026 * 100 + 2 * 10 + 1], (2, 100))   # NOAA-21, NRT


if __name__ == "__main__":
    unittest.main()
