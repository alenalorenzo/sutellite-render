import unittest

from src.export_land_outline import land_rings


class LandRingsTests(unittest.TestCase):
    def test_polygon_and_multipolygon_outer_rings_only(self):
        gj = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
                    [[0, 0], [10, 0], [10, 10], [0, 0]],
                    [[2, 2], [3, 2], [3, 3], [2, 2]],  # hole, ignored
                ]}},
                {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [
                    [[[20.123, 5.987], [21, 5], [21, 6], [20.123, 5.987]]],
                    [[[30, 30], [31, 30], [31, 31], [30, 30]]],
                ]}},
                {"type": "Feature", "geometry": None},
            ],
        }
        rings = land_rings(gj)
        self.assertEqual(len(rings), 3)
        self.assertEqual(len(rings[0]), 4)
        self.assertEqual(rings[1][0], [20.12, 5.99])

    def test_no_polygons_raises(self):
        with self.assertRaises(ValueError):
            land_rings({"type": "FeatureCollection", "features": []})


if __name__ == "__main__":
    unittest.main()
