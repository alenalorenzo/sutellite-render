"""Canonical Arrow schema for harmonised NASA FIRMS active-fire observations.

Schema reference:
    docs/data/CANONICAL_SCHEMA.md, Section 5

The schema preserves sensor-specific measurements and source provenance.
It does not imply that MODIS and VIIRS measurements are interchangeable.
"""

from __future__ import annotations

import pyarrow as pa


SCHEMA_NAME = "Sutellite Canonical Observation Schema"
SCHEMA_VERSION = "1.1"
SCHEMA_STATUS = "DRAFT"

CANONICAL_CRS = "EPSG:4326"
CANONICAL_TIMEZONE = "UTC"


# ---------------------------------------------------------------------
# Sensor-specific metadata
# ---------------------------------------------------------------------

CONFIDENCE_SYSTEM_BY_SENSOR = {
    "MODIS": "MODIS_PERCENT_0_100",
    "VIIRS": "VIIRS_LNH",
}

NOMINAL_RESOLUTION_M_BY_SENSOR = {
    "MODIS": 1000,
    "VIIRS": 375,
}

DETECTION_TYPE_LABELS = {
    0: "presumed_vegetation_fire",
    1: "active_volcano",
    2: "other_static_land_source",
    3: "offshore_detection",
}


# ---------------------------------------------------------------------
# Field construction helper
# ---------------------------------------------------------------------

def _field(
    name: str,
    data_type: pa.DataType,
    nullable: bool,
    description: str,
    *,
    unit: str | None = None,
    applies_to: str | None = None,
    mapping_status: str | None = None,
) -> pa.Field:
    """Create a canonical field with descriptive Arrow metadata."""

    metadata = {
        b"description": description.encode("utf-8"),
    }

    if unit is not None:
        metadata[b"unit"] = unit.encode("utf-8")

    if applies_to is not None:
        metadata[b"applies_to"] = applies_to.encode("utf-8")

    if mapping_status is not None:
        metadata[b"mapping_status"] = mapping_status.encode("utf-8")

    return pa.field(
        name,
        data_type,
        nullable=nullable,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Canonical observation schema
# ---------------------------------------------------------------------

CANONICAL_SCHEMA = pa.schema(
    [
        # -------------------------------------------------------------
        # 1. Identity and provenance
        # -------------------------------------------------------------

        _field(
            "observation_id",
            pa.string(),
            False,
            "Stable identifier for one source observation record.",
        ),
        _field(
            "region_id",
            pa.string(),
            False,
            "Canonical study-region identifier.",
        ),
        _field(
            "sensor",
            pa.string(),
            False,
            "Sensor identifier: MODIS or VIIRS.",
        ),
        _field(
            "platform",
            pa.string(),
            False,
            "Platform: TERRA, AQUA, SNPP, NOAA-20 or NOAA-21.",
        ),
        _field(
            "satellite_raw",
            pa.string(),
            False,
            "Original source satellite value.",
        ),
        _field(
            "instrument_raw",
            pa.string(),
            False,
            "Original source instrument value.",
        ),
        _field(
            "processing_stream",
            pa.string(),
            False,
            "Source processing stream: ARCHIVE or NRT.",
        ),
        _field(
            "distribution_service",
            pa.string(),
            False,
            "Distribution service. Expected value: NASA_FIRMS.",
        ),
        _field(
            "processing_provider",
            pa.string(),
            False,
            "Provider responsible for the processing stream.",
        ),
        _field(
            "source_collection",
            pa.string(),
            False,
            "Canonical source collection identifier.",
        ),
        _field(
            "source_product_token_raw",
            pa.string(),
            False,
            "Original product token extracted from the source filename.",
        ),
        _field(
            "source_file",
            pa.string(),
            False,
            "Relative path to the original source CSV.",
        ),
        _field(
            "source_record_index",
            pa.uint64(),
            False,
            "One-based source data-record index, excluding the header.",
        ),
        _field(
            "firms_version_raw",
            pa.string(),
            False,
            "Original source FIRMS version value.",
        ),

        # -------------------------------------------------------------
        # 2. Geographic fields
        # -------------------------------------------------------------

        _field(
            "latitude_deg",
            pa.float64(),
            False,
            "Latitude in decimal degrees, WGS 84.",
            unit="degree",
        ),
        _field(
            "longitude_deg",
            pa.float64(),
            False,
            "Longitude in decimal degrees, WGS 84.",
            unit="degree",
        ),

        # -------------------------------------------------------------
        # 3. Time fields
        # -------------------------------------------------------------

        _field(
            "acq_date",
            pa.date32(),
            False,
            "Original acquisition date.",
        ),
        _field(
            "acq_time_raw",
            pa.string(),
            False,
            "Original CSV acquisition-time value.",
        ),
        _field(
            "acq_time_hhmm_utc",
            pa.string(),
            False,
            "Normalised four-character UTC acquisition time.",
        ),
        _field(
            "acq_datetime_utc",
            pa.timestamp("ms", tz="UTC"),
            False,
            "Combined acquisition date and time in UTC.",
        ),

        # -------------------------------------------------------------
        # 4. Sensor-specific brightness temperatures
        # -------------------------------------------------------------

        _field(
            "modis_bt21_22_k",
            pa.float64(),
            True,
            "MODIS Band 21/22 brightness temperature.",
            unit="K",
            applies_to="MODIS",
        ),
        _field(
            "modis_bt31_k",
            pa.float64(),
            True,
            "MODIS Band 31 brightness temperature.",
            unit="K",
            applies_to="MODIS",
        ),
        _field(
            "viirs_bt_i4_k",
            pa.float64(),
            True,
            "VIIRS I4 brightness temperature.",
            unit="K",
            applies_to="VIIRS",
        ),
        _field(
            "viirs_bt_i5_k",
            pa.float64(),
            True,
            "Provisional VIIRS I5 archive-export alias mapping.",
            unit="K",
            applies_to="VIIRS",
            mapping_status="PROVISIONAL",
        ),

        # -------------------------------------------------------------
        # 5. Fire Radiative Power
        # -------------------------------------------------------------

        _field(
            "frp_mw",
            pa.float64(),
            True,
            "Pixel-integrated Fire Radiative Power.",
            unit="MW",
        ),

        # -------------------------------------------------------------
        # 6. Pixel dimensions
        # -------------------------------------------------------------

        _field(
            "scan_km",
            pa.float64(),
            False,
            "Pixel dimension in the scan direction.",
            unit="km",
        ),
        _field(
            "track_km",
            pa.float64(),
            False,
            "Pixel dimension in the track direction.",
            unit="km",
        ),

        # -------------------------------------------------------------
        # 7. Confidence
        # -------------------------------------------------------------

        _field(
            "confidence_raw",
            pa.string(),
            True,
            "Original source confidence value.",
        ),
        _field(
            "confidence_system",
            pa.string(),
            False,
            "Confidence convention used by the sensor.",
        ),
        _field(
            "confidence_modis_pct",
            pa.float64(),
            True,
            "MODIS confidence value, expressed as a percentage.",
            unit="%",
            applies_to="MODIS",
        ),
        _field(
            "confidence_viirs_class",
            pa.string(),
            True,
            "VIIRS categorical confidence: l, n or h.",
            applies_to="VIIRS",
        ),

        # -------------------------------------------------------------
        # 8. Detection type
        # -------------------------------------------------------------

        _field(
            "detection_type_code",
            pa.int8(),
            True,
            "Original ARCHIVE detection type code.",
        ),
        _field(
            "detection_type_label",
            pa.string(),
            True,
            "Documented label associated with the detection type code.",
        ),

        # -------------------------------------------------------------
        # 9. Day/night
        # -------------------------------------------------------------

        _field(
            "daynight",
            pa.string(),
            False,
            "Acquisition lighting indicator: D or N.",
        ),

        # -------------------------------------------------------------
        # 10. Nominal sensor resolution
        # -------------------------------------------------------------

        _field(
            "nominal_resolution_m",
            pa.uint16(),
            False,
            "Nominal sensor/product spatial resolution.",
            unit="m",
        ),
    ],
    metadata={
        b"schema_name": SCHEMA_NAME.encode("utf-8"),
        b"schema_version": SCHEMA_VERSION.encode("utf-8"),
        b"schema_status": SCHEMA_STATUS.encode("utf-8"),
        b"canonical_crs": CANONICAL_CRS.encode("utf-8"),
        b"canonical_timezone": CANONICAL_TIMEZONE.encode("utf-8"),
        b"source_document": b"docs/data/CANONICAL_SCHEMA.md",
        b"source_section": b"5",
    },
)


# Convenient ordered list of canonical column names.
CANONICAL_FIELD_NAMES = tuple(CANONICAL_SCHEMA.names)


# ---------------------------------------------------------------------
# Schema inspection and validation
# ---------------------------------------------------------------------

def validate_canonical_schema(candidate: pa.Schema) -> None:
    """Raise ValueError if a candidate schema differs from the canonical schema.

    Schema-level metadata is ignored for structural comparison.
    Field names, ordering, data types and nullability must match.
    """

    if not candidate.equals(CANONICAL_SCHEMA, check_metadata=False):
        raise ValueError(
            "The supplied Arrow schema does not match CANONICAL_SCHEMA."
        )