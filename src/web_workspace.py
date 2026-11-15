"""Prepare a shared calendar and spatial dataset from one uploaded file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_adapter import adapt_data, normalize_source_columns
from scripts.export_web_data import calendar_payload
from src.cell_aggregation import build_cell_products, detect_monthly_anomalies
from src.export_map_cells import PLATFORM_ORDER, build_payload

MAP_CELL_DEGREES = 0.5
WORLDCOVER_CELLS = Path(__file__).resolve().parents[1] / "web/data/spatial/worldcover-2021.json"
MAX_ROWS = 200_000
MAX_COLUMNS = 100
MAX_REGIONS = 32
MAX_EXACT_COUNT = 2**53 - 1
UNSPECIFIED = "UNSPECIFIED"
SENSOR_OF = {platform: "MODIS" if platform in {"TERRA", "AQUA"} else "VIIRS"
             for platform in PLATFORM_ORDER}
SOURCES = ["sensor", "platform", "processing_stream"]


def read_upload(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows > MAX_ROWS or len(parquet.schema_arrow) > MAX_COLUMNS:
                raise ValueError("Use at most 200,000 rows and 100 columns.")
            decoded_bytes = sum(parquet.metadata.row_group(i).total_byte_size
                                for i in range(parquet.num_row_groups))
            if decoded_bytes > 128 * 1024 * 1024:
                raise ValueError("The uncompressed Parquet exceeds 128 MiB.")
            frame = parquet.read().to_pandas()
    else:
        frame = pd.read_csv(path, nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError("Use at most 200,000 rows and 100 columns.")
    return normalize_source_columns(frame)


def read_source_fields(frame: pd.DataFrame, original_name: str) -> pd.DataFrame:
    """Use reported identities. Leave unknown sources explicitly unspecified."""
    series = {}
    for field in SOURCES:
        blank = pd.Series(UNSPECIFIED, index=frame.index, dtype="string")
        series[field] = frame.get(field, blank).replace("", pd.NA).fillna(UNSPECIFIED)
    known_sensor = series["platform"].map(SENSOR_OF)
    sensor = series["sensor"].where(series["sensor"] != UNSPECIFIED, known_sensor)
    series["sensor"] = sensor.fillna(UNSPECIFIED)
    if not series["sensor"].isin(["MODIS", "VIIRS", UNSPECIFIED]).all():
        raise ValueError("The sensor must be MODIS or VIIRS when supplied.")
    if not series["platform"].isin([*PLATFORM_ORDER, UNSPECIFIED]).all():
        raise ValueError("Unknown satellite. Use Terra, Aqua, S-NPP, NOAA-20 or NOAA-21.")
    if (known_sensor.notna() & known_sensor.ne(series["sensor"])).any():
        raise ValueError("A sensor does not match the reported satellite.")
    name = original_name.lower()
    if "processing_stream" not in frame and name.startswith(("fire_archive_", "fire_nrt_")):
        series["processing_stream"] = "ARCHIVE" if name.startswith("fire_archive_") else "NRT"
    if not pd.Series(series["processing_stream"]).isin(["ARCHIVE", "NRT", UNSPECIFIED]).all():
        raise ValueError("The processing stream must be ARCHIVE or NRT when supplied.")
    return pd.DataFrame(series)


def number_column(frame: pd.DataFrame, aliases: tuple[str, ...], low: float, high: float):
    source = next((frame[name] for name in aliases if name in frame), None)
    if source is None:
        return pd.Series(np.nan, index=frame.index)
    value = pd.to_numeric(source, errors="coerce")
    invalid = source.notna() & (value.isna() | ~value.between(low, high))
    if invalid.any():
        raise ValueError(f"Invalid {aliases[-1]} value: {source.loc[invalid].iloc[0]}.")
    return value.astype(float)


def spatial_rows(frame, counts, sources, latitude, longitude):
    """Keep the canonical fields used by the team's cell aggregation."""
    has_counts = any(name in frame for name in ("observations", "detections", "hotspot_count", "count"))
    frp_aliases = ("frp_mean_mw", "mean_frp", "frp_mw", "frp")
    # Raw hotspot power belongs to a detection. A monthly row needs a mean.
    power_columns = frp_aliases[:2] if has_counts else frp_aliases
    power = number_column(frame, power_columns, -1e8, 1e8)
    rows = sources.assign(
        region_id=counts["region"],
        acq_date=counts["date"].dt.date,
        acq_datetime_utc=counts["date"],
        latitude_deg=latitude,
        longitude_deg=longitude,
        frp_mw=power,
        daynight=frame.get("daynight", pd.Series("", index=frame.index)).astype("string").str.upper(),
        confidence_modis_pct=number_column(frame, ("confidence_modis_pct",), 0, 100),
        confidence_viirs_class=frame.get("confidence_viirs_class", pd.Series("", index=frame.index)).astype("string").str.lower(),
        detection_type_code=number_column(frame, ("detection_type_code",), 0, 3),
        hotspot_count=counts["hotspot_count"],
    )
    return rows[latitude.notna() & counts["hotspot_count"].gt(0)]


@lru_cache(maxsize=1)
def bundled_worldcover() -> dict:
    if WORLDCOVER_CELLS.is_file():
        return json.loads(WORLDCOVER_CELLS.read_text(encoding="utf-8"))
    return {}


def cells_payload(aggregated, land, sensor: str, platform: str, *, meta_extra=None) -> dict:
    """Serialize Ainhoa's monthly cells using her map's compact format."""
    grouped = aggregated.rename(columns={
        "lat_index": "clat", "lon_index": "clon", "year": "yr",
        "processing_stream": "stream", "observation_count": "n",
        "frp_valid_count": "n_frp", "frp_sum_mw": "frp_sum",
    })
    totals = grouped.groupby(["clat", "clon"], as_index=False)["n"].sum()
    fine = totals.merge(land, on=["clat", "clon"], how="left").rename(
        columns={"clat": "lat_i", "clon": "lon_i", "worldcover_2021_code": "lc"}
    )
    # The exporter accepts fine-cell indices on the 0.1 degree grid.
    fine[["lat_i", "lon_i"]] *= round(MAP_CELL_DEGREES / 0.1)
    return build_payload(fine, grouped, MAP_CELL_DEGREES, {platform: sensor}, meta_extra, monthly=True)


def monthly_flags(rows: pd.DataFrame, sensor: str, platform: str, stream: str) -> dict:
    monthly = rows.assign(year=rows["date"].dt.year, month=rows["date"].dt.month).groupby(
        ["region", "year", "month"], as_index=False
    )["hotspot_count"].sum().rename(columns={"region": "region_id", "hotspot_count": "observation_count"})
    stats = detect_monthly_anomalies(
        monthly.assign(sensor=sensor, platform=platform, processing_stream=stream)
    )
    return {
        key: {
            f"{row.year}-{row.month}": {
                "referenceYears": int(row.reference_years), "median": float(row.reference_median),
            }
            for row in group.itertuples()
        }
        for key, group in stats[stats["is_anomaly"]].groupby("region_id")
    }


def analyze_upload(path: Path, original_name: str, region: str, worldcover_dir: Path | None = None) -> dict:
    frame = read_upload(path).reset_index(drop=True)
    counts = adapt_data(frame, region=region or None, aggregate_months=False)
    if not counts["date"].dt.year.between(1900, 2100).all():
        raise ValueError("Use observation years between 1900 and 2100.")
    if any(len(label) > 80 for label in counts["region"]):
        raise ValueError("Region names must contain at most 80 characters.")
    if counts["region"].nunique() > MAX_REGIONS:
        raise ValueError("Use at most 32 regions per upload.")
    if counts["hotspot_count"].astype(object).sum() > MAX_EXACT_COUNT:
        raise ValueError("The total number of detections is too large.")
    sources = read_source_fields(frame, original_name)
    latitude = number_column(frame, ("latitude_deg", "grid_lat_center_deg", "grid_lat", "lat", "latitude"), -90, 90)
    longitude = number_column(frame, ("longitude_deg", "grid_lon_center_deg", "grid_lon", "lon", "lng", "longitude"), -180, 180)
    if (latitude.notna() != longitude.notna()).any():
        raise ValueError("Each mapped row needs both a latitude and a longitude.")
    spatial = spatial_rows(frame, counts, sources, latitude, longitude)

    by_source = {}
    land_groups = {}
    if not spatial.empty:
        source = path.parent / "spatial.parquet"
        spatial.to_parquet(source, index=False)
        products = build_cell_products(
            [source], path.parent / "analysis", cell_degrees=MAP_CELL_DEGREES,
            worldcover_dir=worldcover_dir, use_hotspot_counts=True,
        )["files"]
        detail = pd.read_parquet(products["breakdown_parquet"])
        breakdown_keys = [
            "region_id", *SOURCES, "lat_index", "lon_index", "year", "month",
        ]
        fields = ["observation_count", "frp_valid_count", "frp_sum_mw"]
        grouped = detail.groupby(breakdown_keys, as_index=False)[fields].sum(min_count=1)
        by_source = dict(tuple(grouped.groupby(SOURCES + ["region_id"], sort=False)))
        cells = pd.read_parquet(products["cells_parquet"], columns=[
            "region_id", "lat_index", "lon_index", "worldcover_2021_code",
        ]).rename(columns={"lat_index": "clat", "lon_index": "clon"})
        lookup = bundled_worldcover()
        missing = cells["worldcover_2021_code"].eq(0)
        cells.loc[missing, "worldcover_2021_code"] = [
            lookup.get(f"{lat},{lon}", 0)
            for lat, lon in cells.loc[missing, ["clat", "clon"]].itertuples(index=False, name=None)
        ]
        land_groups = dict(tuple(cells.groupby("region_id", sort=False)))
    output = []
    for key, source_rows in counts.join(sources).groupby(SOURCES, sort=True):
        sensor, platform, stream = key
        flags = monthly_flags(source_rows, sensor, platform, stream)
        calendar = calendar_payload(source_rows, fixed_demo_reference=False)
        regions = []
        for label, years in calendar.items():
            map_rows = by_source.get((*key, label))
            regions.append({
                "name": label,
                "years": years,
                "map": cells_payload(map_rows, land_groups[label], sensor, platform) if map_rows is not None else None,
                "monthlyFlags": flags.get(label, {}),
            })
        output.append({
            "sensor": sensor, "platform": platform, "stream": stream, "regions": regions,
        })

    return {"filename": original_name, "series": output}
