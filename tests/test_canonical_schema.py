import unittest

import pyarrow as pa

from src.canonical_schema import (
    CANONICAL_SCHEMA,
    CANONICAL_FIELD_NAMES,
    CONFIDENCE_SYSTEM_BY_SENSOR,
    DETECTION_TYPE_LABELS,
    NOMINAL_RESOLUTION_M_BY_SENSOR,
    validate_canonical_schema,
)


class TestCanonicalSchema(unittest.TestCase):

    def test_contains_35_fields(self):
        self.assertEqual(len(CANONICAL_SCHEMA), 35)

    def test_field_names_are_unique(self):
        self.assertEqual(
            len(CANONICAL_FIELD_NAMES),
            len(set(CANONICAL_FIELD_NAMES)),
        )

    def test_required_provenance_fields_are_non_nullable(self):
        required = [
            "observation_id",
            "region_id",
            "sensor",
            "platform",
            "source_file",
            "source_record_index",
        ]

        for name in required:
            with self.subTest(field=name):
                self.assertFalse(CANONICAL_SCHEMA.field(name).nullable)

    def test_optional_measurements_are_nullable(self):
        optional = [
            "modis_bt21_22_k",
            "modis_bt31_k",
            "viirs_bt_i4_k",
            "viirs_bt_i5_k",
            "frp_mw",
            "confidence_raw",
            "confidence_modis_pct",
            "confidence_viirs_class",
            "detection_type_code",
            "detection_type_label",
        ]

        for name in optional:
            with self.subTest(field=name):
                self.assertTrue(CANONICAL_SCHEMA.field(name).nullable)

    def test_sensor_specific_confidence_conventions(self):
        self.assertEqual(
            CONFIDENCE_SYSTEM_BY_SENSOR["MODIS"],
            "MODIS_PERCENT_0_100",
        )
        self.assertEqual(
            CONFIDENCE_SYSTEM_BY_SENSOR["VIIRS"],
            "VIIRS_LNH",
        )

    def test_nominal_resolution(self):
        self.assertEqual(NOMINAL_RESOLUTION_M_BY_SENSOR["MODIS"], 1000)
        self.assertEqual(NOMINAL_RESOLUTION_M_BY_SENSOR["VIIRS"], 375)

    def test_detection_type_labels(self):
        self.assertEqual(len(DETECTION_TYPE_LABELS), 4)
        self.assertIn(0, DETECTION_TYPE_LABELS)
        self.assertIn(3, DETECTION_TYPE_LABELS)

    def test_viirs_i5_mapping_is_provisional(self):
        metadata = CANONICAL_SCHEMA.field("viirs_bt_i5_k").metadata

        self.assertEqual(
            metadata[b"mapping_status"],
            b"PROVISIONAL",
        )

    def test_timestamp_is_utc(self):
        timestamp_type = CANONICAL_SCHEMA.field("acq_datetime_utc").type

        self.assertTrue(pa.types.is_timestamp(timestamp_type))
        self.assertEqual(str(timestamp_type.unit), "ms")
        self.assertEqual(timestamp_type.tz, "UTC")

    def test_accepts_identical_schema(self):
        validate_canonical_schema(CANONICAL_SCHEMA)

    def test_rejects_incompatible_schema(self):
        incompatible = pa.schema([
            pa.field("observation_id", pa.int64(), nullable=False)
        ])

        with self.assertRaises(ValueError):
            validate_canonical_schema(incompatible)

    def test_pixel_dimensions_are_non_nullable(self):
        for name in ("scan_km", "track_km"):
            with self.subTest(field=name):
                self.assertFalse(CANONICAL_SCHEMA.field(name).nullable)


if __name__ == "__main__":
    unittest.main()