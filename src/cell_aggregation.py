"""Build spatial-cell, monthly, daily-calendar and GeoJSON products from canonical Parquet.

All quantities are source-observation statistics. The module never treats a
missing detection record as an observed zero and never pools sensor
measurements into a claim of sensor equivalence.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

LAND_CLASSES = {
    0: "No data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

REQUIRED_COLUMNS = {
    "region_id", "sensor", "platform", "processing_stream", "acq_date",
    "acq_datetime_utc", "latitude_deg", "longitude_deg", "frp_mw", "daynight",
    "confidence_modis_pct", "confidence_viirs_class", "detection_type_code",
}


def canonical_file_fingerprint(files: Iterable[Path], cell_degrees: float) -> str:
    """Fingerprint source files by resolved path, size, mtime and cell size."""
    digest = hashlib.sha256(f"SUTELLITE_CELL_PRODUCTS_V1|{cell_degrees:g}".encode())
    for path in sorted((Path(p).resolve() for p in files), key=lambda p: str(p).casefold()):
        stat = path.stat()
        digest.update(f"\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def detect_monthly_anomalies(
    monthly: pd.DataFrame,
    *,
    min_history_years: int = 3,
    robust_z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag unusually high monthly counts against prior observed same-month years.

    Grouping retains region, sensor, platform and stream separately. Absent
    month/year combinations are not inserted as zero observations.
    """
    required = {"region_id", "year", "month", "sensor", "platform", "processing_stream", "observation_count"}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"Monthly table is missing columns: {sorted(missing)}")
    if monthly.empty:
        result = monthly.copy()
        result["reference_years"] = pd.Series(dtype="int64")
        result["reference_median"] = pd.Series(dtype="float64")
        result["robust_z"] = pd.Series(dtype="float64")
        result["is_anomaly"] = pd.Series(dtype="bool")
        result["anomaly_status"] = pd.Series(dtype="object")
        return result

    group_cols = ["region_id", "month", "sensor", "platform", "processing_stream"]
    rows: list[dict[str, Any]] = []
    ordered = monthly.sort_values([*group_cols, "year"]).reset_index(drop=True)
    for _, group in ordered.groupby(group_cols, dropna=False, sort=False):
        history: list[tuple[int, float]] = []
        for record in group.to_dict("records"):
            value = float(record["observation_count"])
            prior = [v for year, v in history if year < int(record["year"])]
            row = dict(record)
            row["reference_years"] = len(prior)
            row["reference_median"] = np.nan
            row["robust_z"] = np.nan
            row["is_anomaly"] = False
            row["anomaly_status"] = "insufficient_history"
            if len(prior) >= min_history_years:
                median = float(np.median(prior))
                mad = float(np.median(np.abs(np.asarray(prior, dtype=float) - median)))
                scale = max(1.0, 1.4826 * mad)
                score = (value - median) / scale
                row["reference_median"] = median
                row["robust_z"] = score
                row["is_anomaly"] = bool(score >= robust_z_threshold and value > median)
                row["anomaly_status"] = "high_anomaly" if row["is_anomaly"] else "within_reference"
            rows.append(row)
            history.append((int(record["year"]), value))
    return pd.DataFrame(rows)


def _apply_worldcover(cells: pd.DataFrame, worldcover_dir: Path | None) -> pd.DataFrame:
    result = cells.copy()
    result["worldcover_2021_code"] = 0
    result["land_type"] = "No data"
    if worldcover_dir is None or not Path(worldcover_dir).is_dir():
        return result
    try:
        from geo.worldcover_lookup import anadir_worldcover
    except ImportError:
        return result

    sample = result[["cell_center_lat", "cell_center_lon"]].rename(columns={
        "cell_center_lat": "grid_lat", "cell_center_lon": "grid_lon"
    }).copy()
    try:
        joined = anadir_worldcover(sample, Path(worldcover_dir))
    except (ModuleNotFoundError, FileNotFoundError):
        return result
    result["worldcover_2021_code"] = joined["worldcover_2021_codigo"].fillna(0).astype(int)
    result["land_type"] = result["worldcover_2021_code"].map(LAND_CLASSES).fillna("No data")
    return result


def _json_safe_value(value: Any) -> Any:
    """Convert pandas/NumPy scalar values to strict JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    return value


def _to_geojson(cells: pd.DataFrame, output_path: Path, cell_degrees: float) -> None:
    """Write all cells as GeoJSON incrementally to avoid a second full-size copy."""
    metadata = {
        "cell_degrees": cell_degrees,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "counts_are": "source observation records, not unique fires",
        "missing_cell_is": (
            "no cell record in the detections dataset; not proof of zero activity "
            "or satellite non-observation"
        ),
    }
    columns = list(cells.columns)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(
            '{"type":"FeatureCollection","name":"Sutellite harmonized observation cells",'
            '"metadata":'
        )
        json.dump(metadata, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write(',"features":[')
        first = True
        for values in cells.itertuples(index=False, name=None):
            row = dict(zip(columns, values))
            south = max(-90.0, float(row["lat_index"]) * cell_degrees)
            west = max(-180.0, float(row["lon_index"]) * cell_degrees)
            north = min(90.0, south + cell_degrees)
            east = min(180.0, west + cell_degrees)
            feature = {
                "type": "Feature",
                "id": str(row["cell_id"]),
                "geometry": {"type": "Polygon", "coordinates": [[
                    [west, south], [east, south], [east, north], [west, north], [west, south]
                ]]},
                "properties": {key: _json_safe_value(value) for key, value in row.items()},
            }
            if not first:
                stream.write(",")
            json.dump(feature, stream, ensure_ascii=False, separators=(",", ":"))
            first = False
        stream.write("]}")


def build_cell_products(
    input_files: Iterable[Path],
    output_dir: Path,
    *,
    cell_degrees: float = 0.5,
    worldcover_dir: Path | None = None,
    use_hotspot_counts: bool = False,
) -> dict[str, Any]:
    """Aggregate canonical observations or weighted, geolocated hotspot counts."""
    source_paths = [Path(p).expanduser().resolve() for p in input_files]
    if not source_paths:
        raise FileNotFoundError("No canonical Parquet files were supplied.")
    missing_paths = [str(path) for path in source_paths if not path.is_file()]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        suffix = " …" if len(missing_paths) > 5 else ""
        raise FileNotFoundError(f"Canonical Parquet file(s) not found: {preview}{suffix}")
    files = [str(path) for path in source_paths]
    if not math.isfinite(cell_degrees) or not (0.1 <= cell_degrees <= 5.0):
        raise ValueError("Cell size must be between 0.1 and 5 degrees.")
    if abs(cell_degrees * 10 - round(cell_degrees * 10)) > 1e-8:
        raise ValueError("Cell size must be a multiple of 0.1 degrees.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        if use_hotspot_counts:
            con.execute("SET threads TO 2")
            con.execute("SET memory_limit TO '192MB'")
        schema = con.execute(
            "SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [files]
        ).fetchdf()
        required = REQUIRED_COLUMNS | {"hotspot_count"} if use_hotspot_counts else REQUIRED_COLUMNS
        missing = required - set(schema.columns)
        if missing:
            raise ValueError(f"Canonical Parquet is missing required columns: {sorted(missing)}")

        # Keep observations exactly on the north/east outer bounds inside
        # the final valid cell instead of generating zero-area polygons.
        lat_expr = (
            f"CAST(FLOOR(LEAST(latitude_deg, 89.999999999) "
            f"/ {cell_degrees:.10f}) AS INTEGER)"
        )
        lon_expr = (
            f"CAST(FLOOR(LEAST(longitude_deg, 179.999999999) "
            f"/ {cell_degrees:.10f}) AS INTEGER)"
        )
        base = "read_parquet(?, union_by_name = true)"
        def tally(condition: str = "") -> str:
            aggregate = "SUM(hotspot_count)" if use_hotspot_counts else "COUNT(*)"
            if condition:
                aggregate += f" FILTER (WHERE {condition})"
            return f"COALESCE({aggregate}, 0)::BIGINT" if use_hotspot_counts else aggregate

        power = "frp_mw * hotspot_count" if use_hotspot_counts else "frp_mw"
        power_sum = f"SUM({power}) FILTER (WHERE frp_mw >= 0)"
        power_mean = f"({power_sum}) / NULLIF({tally('frp_mw >= 0')}, 0)"

        cell_sql = f"""
            SELECT region_id, {lat_expr} AS lat_index, {lon_expr} AS lon_index,
                   {tally()} AS observation_count,
                   {tally("sensor = 'MODIS'")} AS modis_count,
                   {tally("sensor = 'VIIRS'")} AS viirs_count,
                   {tally("platform = 'TERRA'")} AS terra_count,
                   {tally("platform = 'AQUA'")} AS aqua_count,
                   {tally("platform = 'SNPP'")} AS snpp_count,
                   {tally("platform = 'NOAA-20'")} AS noaa20_count,
                   {tally("platform = 'NOAA-21'")} AS noaa21_count,
                   {tally("processing_stream = 'ARCHIVE'")} AS archive_count,
                   {tally("processing_stream = 'NRT'")} AS nrt_count,
                   {tally("daynight = 'D'")} AS daytime_count,
                   {tally("daynight = 'N'")} AS nighttime_count,
                   COUNT(DISTINCT acq_date) AS active_observation_days,
                   MIN(acq_date) AS first_detection_date,
                   MAX(acq_date) AS last_detection_date,
                   {tally('frp_mw IS NOT NULL')} AS frp_non_missing_count,
                   {tally('frp_mw IS NULL')} AS frp_missing_count,
                   {tally('frp_mw < 0')} AS frp_negative_count,
                   {tally('frp_mw = 0')} AS frp_zero_count,
                   {tally('frp_mw > 0')} AS frp_positive_count,
                   {tally('frp_mw >= 0')} AS frp_valid_count,
                   COALESCE({power_sum}, 0) AS frp_sum_mw,
                   {power_mean} AS frp_mean_mw,
                   MAX(frp_mw) FILTER (WHERE frp_mw >= 0) AS frp_max_mw,
                   AVG(confidence_modis_pct) AS modis_confidence_mean_pct,
                   COUNT(confidence_modis_pct) AS modis_confidence_n,
                   {tally("confidence_viirs_class = 'l'")} AS viirs_confidence_low_count,
                   {tally("confidence_viirs_class = 'n'")} AS viirs_confidence_nominal_count,
                   {tally("confidence_viirs_class = 'h'")} AS viirs_confidence_high_count,
                   {tally('detection_type_code = 0')} AS type_0_count,
                   {tally('detection_type_code = 1')} AS type_1_count,
                   {tally('detection_type_code = 2')} AS type_2_count,
                   {tally('detection_type_code = 3')} AS type_3_count
            FROM {base}
            WHERE latitude_deg BETWEEN -90 AND 90 AND longitude_deg BETWEEN -180 AND 180
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
        """
        cells = con.execute(cell_sql, [files]).fetchdf()

        detail_sql = f"""
            SELECT region_id, {lat_expr} AS lat_index, {lon_expr} AS lon_index,
                   YEAR(TRY_CAST(acq_date AS DATE)) AS year,
                   MONTH(TRY_CAST(acq_date AS DATE)) AS month,
                   sensor, platform, processing_stream, daynight,
                   {tally()} AS observation_count,
                   {tally("sensor = 'MODIS'")} AS modis_count,
                   {tally("sensor = 'VIIRS'")} AS viirs_count,
                   {tally("platform = 'TERRA'")} AS terra_count,
                   {tally("platform = 'AQUA'")} AS aqua_count,
                   {tally("platform = 'SNPP'")} AS snpp_count,
                   {tally("platform = 'NOAA-20'")} AS noaa20_count,
                   {tally("platform = 'NOAA-21'")} AS noaa21_count,
                   {tally("processing_stream = 'ARCHIVE'")} AS archive_count,
                   {tally("processing_stream = 'NRT'")} AS nrt_count,
                   {tally("daynight = 'D'")} AS daytime_count,
                   {tally("daynight = 'N'")} AS nighttime_count,
                   {tally('frp_mw IS NOT NULL')} AS frp_non_missing_count,
                   {tally('frp_mw IS NULL')} AS frp_missing_count,
                   {tally('frp_mw < 0')} AS frp_negative_count,
                   {tally('frp_mw = 0')} AS frp_zero_count,
                   {tally('frp_mw > 0')} AS frp_positive_count,
                   {tally('frp_mw >= 0')} AS frp_valid_count,
                   {power_sum} AS frp_sum_mw,
                   {power_mean} AS frp_mean_mw,
                   MAX(frp_mw) FILTER (WHERE frp_mw >= 0) AS frp_max_mw,
                   AVG(confidence_modis_pct) AS modis_confidence_mean_pct,
                   COUNT(confidence_modis_pct) AS modis_confidence_n,
                   {tally("confidence_viirs_class = 'l'")} AS viirs_confidence_low_count,
                   {tally("confidence_viirs_class = 'n'")} AS viirs_confidence_nominal_count,
                   {tally("confidence_viirs_class = 'h'")} AS viirs_confidence_high_count,
                   {tally('detection_type_code = 0')} AS type_0_count,
                   {tally('detection_type_code = 1')} AS type_1_count,
                   {tally('detection_type_code = 2')} AS type_2_count,
                   {tally('detection_type_code = 3')} AS type_3_count
            FROM {base}
            WHERE latitude_deg BETWEEN -90 AND 90 AND longitude_deg BETWEEN -180 AND 180
              AND TRY_CAST(acq_date AS DATE) IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
            ORDER BY 1, 2, 3, 4, 5, 6, 7, 8, 9
        """
        detail = con.execute(detail_sql, [files]).fetchdf()

        monthly_sql = f"""
            SELECT region_id, YEAR(TRY_CAST(acq_date AS DATE)) AS year,
                   MONTH(TRY_CAST(acq_date AS DATE)) AS month,
                   sensor, platform, processing_stream,
                   {tally()} AS observation_count,
                   {tally('frp_mw >= 0')} AS frp_valid_count,
                   {power_sum} AS frp_sum_mw,
                   {power_mean} AS frp_mean_mw
            FROM {base}
            WHERE TRY_CAST(acq_date AS DATE) IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5, 6
            ORDER BY 1, 2, 3, 4, 5, 6
        """
        monthly = con.execute(monthly_sql, [files]).fetchdf()
        anomalies = detect_monthly_anomalies(monthly)

        # Daily aggregates power the interactive month/day hotspot calendar.
        # These remain counts of source observation records, not unique fires.
        daily_sql = f"""
            SELECT region_id, TRY_CAST(acq_date AS DATE) AS activity_date,
                   sensor, platform, processing_stream,
                   {tally()} AS observation_count,
                   {tally('frp_mw >= 0')} AS frp_valid_count,
                   {power_sum} AS frp_sum_mw
            FROM {base}
            WHERE TRY_CAST(acq_date AS DATE) IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5
            ORDER BY 1, 2, 3, 4, 5
        """
        daily = con.execute(daily_sql, [files]).fetchdf()
        overall = con.execute(
            f"""SELECT {tally()} AS observation_count,
                       COUNT(DISTINCT region_id) AS region_count,
                       COUNT(DISTINCT TRY_CAST(acq_date AS DATE)) AS active_observation_days,
                       MIN(acq_date) AS first_date, MAX(acq_date) AS last_date
                FROM {base}""", [files]
        ).fetchdf().iloc[0].to_dict()
    finally:
        con.close()

    if cells.empty:
        raise ValueError("The input contains no observations with valid coordinates.")
    cells["cell_degrees"] = cell_degrees
    cells["cell_id"] = cells.apply(
        lambda r: f"{r['region_id']}:{int(r['lat_index'])}:{int(r['lon_index'])}", axis=1
    )
    cells["cell_south"] = cells["lat_index"] * cell_degrees
    cells["cell_west"] = cells["lon_index"] * cell_degrees
    cells["cell_center_lat"] = cells["cell_south"] + cell_degrees / 2
    cells["cell_center_lon"] = cells["cell_west"] + cell_degrees / 2
    cells = _apply_worldcover(cells, worldcover_dir)

    cells_path = output_dir / "cell_aggregates.parquet"
    detail_path = output_dir / "cell_breakdown.parquet"
    monthly_path = output_dir / "monthly_activity.parquet"
    monthly_csv = output_dir / "monthly_activity.csv"
    daily_path = output_dir / "daily_activity.parquet"
    daily_csv = output_dir / "daily_activity.csv"
    anomaly_path = output_dir / "monthly_anomalies.csv"
    geojson_path = output_dir / "map_cells.geojson"
    cells.to_parquet(cells_path, index=False, compression="zstd")
    detail.to_parquet(detail_path, index=False, compression="zstd")
    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    monthly.to_csv(monthly_csv, index=False)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    daily.to_csv(daily_csv, index=False)
    anomalies.to_csv(anomaly_path, index=False)

    # The downloadable GeoJSON contains every cell. The interactive UI
    # applies its own feature limit for browser performance.
    _to_geojson(cells, geojson_path, cell_degrees)
    map_cells = cells

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_files": len(files),
        "input_observations": int(overall["observation_count"] or 0),
        "regions": int(overall["region_count"] or 0),
        "active_observation_days": int(overall["active_observation_days"] or 0),
        "cells": int(len(cells)),
        "map_features": int(len(map_cells)),
        "cell_degrees": cell_degrees,
        "first_date": str(overall["first_date"]) if overall["first_date"] is not None else None,
        "last_date": str(overall["last_date"]) if overall["last_date"] is not None else None,
        "land_cover_known_cells": int(cells["worldcover_2021_code"].gt(0).sum()),
        "land_cover_note": "No data means no local WorldCover tile/sample was available; it does not mean a non-land class.",
        "anomaly_method": "One-sided robust MAD score against previous observed years for the same calendar month and same region/sensor/platform/stream; at least three historical observations/years required by available rows.",
        "daily_calendar_grain": "region/date/sensor/platform/processing_stream; counts are source observation records, not unique fires.",
        "interpretation": [
            "Counts are source observation records, not unique fire events.",
            "No observation-coverage denominator is available; absent records are not treated as zero activity.",
            "Sensor/platform/stream breakdowns remain separate. MODIS and VIIRS counts are not directly comparable.",
            "FRP summaries include only non-negative FRP values; missing, zero and negative FRP are counted separately.",
            "MODIS confidence percentages and VIIRS categorical confidence are kept as different fields.",
        ],
        "outputs": [p.name for p in (cells_path, detail_path, monthly_path, monthly_csv, daily_path, daily_csv, anomaly_path, geojson_path)],
    }
    summary_path = output_dir / "cell_processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**summary, "output_dir": str(output_dir), "files": {
        "cells_parquet": str(cells_path),
        "breakdown_parquet": str(detail_path),
        "monthly_parquet": str(monthly_path),
        "monthly_csv": str(monthly_csv),
        "daily_parquet": str(daily_path),
        "daily_csv": str(daily_csv),
        "anomalies_csv": str(anomaly_path),
        "geojson": str(geojson_path),
        "summary_json": str(summary_path),
    }}
