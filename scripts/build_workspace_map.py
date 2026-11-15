"""Build the Historical Calendar and 0.5° maps from the FIRMS archive and ESA WorldCover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scripts.export_web_data import calendar_payload, write_json
from src.build_harmonized_calendar import (
    AQUA, SNPP, TERRA, compute_calibration_factors, detect_platform_starts,
    harmonize_cells, month_index,
)
from src.source_contracts import REGION_SLUGS
from src.web_workspace import MAP_CELL_DEGREES, SENSOR_OF, cells_payload

YEARS = range(2000, 2026)
FIRST_FULL_SNPP = month_index(2012, 2)
HARMONIZED_INPUTS = {"TERRA": TERRA, "AQUA": AQUA, "SNPP": SNPP}
PLATFORMS = {*HARMONIZED_INPUTS, "NOAA-20", "NOAA-21"}
CELL_MONTH = ["region_id", "year", "month", "lat_index", "lon_index"]
LAND_DESCRIPTION = "ESA WorldCover 2021 v200, dominant valid overview class within each 0.5° cell"

SNPP_CELLS = """
    SELECT region_id, year, month,
           floor(grid_lat_index / ?)::INTEGER AS lat_index,
           floor(grid_lon_index / ?)::INTEGER AS lon_index,
           'SNPP' AS platform, sum(viirs_snpp_count)::BIGINT AS observation_count,
           sum(snpp_frp_rows)::BIGINT AS frp_valid_count, sum(snpp_frp) AS frp_sum_mw
    FROM read_parquet(?)
    WHERE region_id = ? AND year BETWEEN 2000 AND 2025 AND viirs_snpp_count > 0
    GROUP BY ALL
"""

CANONICAL_CELLS = """
    WITH records AS (
        SELECT region_id, year(acq_date)::INTEGER AS year, month(acq_date)::INTEGER AS month,
               least(179, floor(latitude_deg * 2))::INTEGER AS lat_index,
               least(359, floor(longitude_deg * 2))::INTEGER AS lon_index,
               platform, CASE WHEN isfinite(frp_mw) AND frp_mw >= 0 THEN frp_mw END AS frp
        FROM read_parquet(?, union_by_name = true, hive_partitioning = false)
        WHERE region_id = ? AND processing_stream = 'ARCHIVE'
          AND acq_date >= DATE '2000-01-01' AND acq_date < DATE '2026-01-01'
          AND latitude_deg BETWEEN -90 AND 90 AND longitude_deg BETWEEN -180 AND 180
          AND platform IN ('TERRA', 'AQUA', 'NOAA-20', 'NOAA-21')
    )
    SELECT region_id, year, month, lat_index, lon_index, platform,
           count(*) AS observation_count, count(frp) AS frp_valid_count, sum(frp) AS frp_sum_mw
    FROM records
    GROUP BY ALL
"""

AGGREGATED_CELLS = """
    SELECT region_id, year, month, lat_index, lon_index, platform,
           sum(observation_count)::BIGINT AS observation_count,
           sum(frp_valid_count)::BIGINT AS frp_valid_count, sum(frp_sum_mw) AS frp_sum_mw
    FROM read_parquet(?)
    WHERE region_id = ? AND year BETWEEN 2000 AND 2025
    GROUP BY ALL
"""


def balanced_counts(estimates: pd.DataFrame) -> pd.Series:
    """Round to whole cells while retaining the rounded regional total for each month."""
    order = estimates.sort_values(CELL_MONTH).copy()
    order["floor"] = np.floor(order["hotspots_harmonized"]).astype("int64")
    order["remainder"] = order["hotspots_harmonized"] - order["floor"]
    groups = ["region_id", "year", "month"]
    monthly = order.groupby(groups)
    rounded = np.rint(monthly["hotspots_harmonized"].transform("sum")).astype("int64")
    topups = rounded - monthly["floor"].transform("sum")
    rank = monthly["remainder"].rank(method="first", ascending=False)
    return (order["floor"] + rank.le(topups).astype("int64")).reindex(estimates.index)


def harmonized_rows(rows: pd.DataFrame) -> pd.DataFrame:
    modis_snpp = rows[rows["platform"].isin(HARMONIZED_INPUTS)]
    wide = modis_snpp.pivot(index=CELL_MONTH, columns="platform", values="observation_count")
    wide = wide.fillna(0).astype("int64").rename(columns=HARMONIZED_INPUTS).reset_index()
    starts = detect_platform_starts(wide)
    starts["SNPP"] = FIRST_FULL_SNPP
    width = MAP_CELL_DEGREES
    grid_keys = {"lat_index": "grid_lat_index", "lon_index": "grid_lon_index"}
    grid = wide.assign(
        grid_lat_center_deg=(wide["lat_index"] + 0.5) * width,
        grid_lon_center_deg=(wide["lon_index"] + 0.5) * width,
    ).rename(columns=grid_keys)
    factors = compute_calibration_factors(wide)
    estimates = harmonize_cells(grid, factors, starts).rename(
        columns={target: source for source, target in grid_keys.items()}
    )
    if estimates["hotspots_harmonized"].isna().any():
        raise ValueError("The harmonized series has gaps in its calibration factors.")
    estimates["observation_count"] = balanced_counts(estimates)
    raw_month = month_index(modis_snpp["year"], modis_snpp["month"])
    raw = modis_snpp[(raw_month < FIRST_FULL_SNPP) == modis_snpp["platform"].ne("SNPP")]
    power = raw.groupby(CELL_MONTH, as_index=False)[["frp_valid_count", "frp_sum_mw"]].sum(min_count=1)
    detected = estimates.loc[estimates["observation_count"] > 0, CELL_MONTH + ["observation_count"]]
    harmonized = detected.merge(power, on=CELL_MONTH, how="left", validate="one_to_one")
    harmonized["frp_valid_count"] = harmonized["frp_valid_count"].fillna(0).astype("int64")
    return harmonized


def export_series(region, platform, rows, cover, output) -> dict:
    harmonized = platform == "HARMONIZED"
    counts = rows.groupby(["year", "month"], as_index=False)["observation_count"].sum()
    counts = counts.assign(
        region=region, date=pd.to_datetime(counts.assign(day=1)[["year", "month", "day"]])
    ).rename(columns={"observation_count": "hotspot_count"})
    years = calendar_payload(counts, fixed_demo_reference=False, year_range=YEARS)[region]
    if harmonized:
        for year, period in years.items():
            for month, record in enumerate(period["months"], 1):
                if record["detections"] is not None and month_index(year, month) < FIRST_FULL_SNPP:
                    record["estimated"] = True

    land = rows[["lat_index", "lon_index"]].drop_duplicates().rename(
        columns={"lat_index": "clat", "lon_index": "clon"}
    )
    land["worldcover_2021_code"] = [
        cover.get(f"{lat},{lon}", 0) for lat, lon in land.itertuples(index=False, name=None)
    ]
    meta = {"land_cover": LAND_DESCRIPTION}
    if harmonized:
        meta["note"] = "MODIS estimates on the VIIRS S-NPP scale through January 2012; S-NPP observations afterward."
    name = region if harmonized else f"{region}-{platform}"
    payload = cells_payload(
        rows.assign(platform=platform, processing_stream="ARCHIVE"),
        land, SENSOR_OF.get(platform, "HARMONIZED"), platform, meta_extra=meta,
    )
    write_json(output / f"{name}.json", payload)
    return years


def build_maps(
    checkpoints: Path, worldcover: Path, output: Path,
    *, canonical: Path | None, archive_cells: Path | None,
) -> None:
    cover = json.loads(worldcover.read_text(encoding="utf-8"))
    ratio = round(MAP_CELL_DEGREES / 0.1)
    with duckdb.connect(config={"memory_limit": "512MB", "threads": 2}) as connection:
        for region in sorted(REGION_SLUGS):
            snpp = connection.execute(
                SNPP_CELLS, [ratio, ratio, str(checkpoints / f"{region}.parquet"), region]
            ).fetchdf()
            if canonical is not None:
                raw = sorted((canonical / region).glob("fire_archive_*.parquet"))
                other = connection.execute(CANONICAL_CELLS, [[str(path) for path in raw], region]).fetchdf()
            else:
                other = connection.execute(AGGREGATED_CELLS, [str(archive_cells / "*.parquet"), region]).fetchdf()
            rows = pd.concat([snpp, other], ignore_index=True)
            if set(rows["platform"]) != PLATFORMS:
                raise ValueError(f"Missing satellite archive data for {region}.")
            series = {platform: group for platform, group in rows.groupby("platform")}
            series["HARMONIZED"] = harmonized_rows(rows)
            calendar = {
                platform: export_series(region, platform, data, cover, output)
                for platform, data in series.items()
            }
            write_json(output / f"{region}-calendar.json", calendar)
            print(f"{region}: {len(calendar)} calendars and maps", flush=True)
    write_json(output / "worldcover-2021.json", cover)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True, help="Regional VIIRS S-NPP checkpoints.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--canonical", type=Path, help="Root of the canonical_v1.1 Parquets.")
    input_group.add_argument("--archive-cells", type=Path, help="Preaggregated MODIS and NOAA archive cell Parquets.")
    parser.add_argument("--worldcover", type=Path, required=True, help="WorldCover codes keyed by 0.5° cell indices.")
    parser.add_argument("--output", type=Path, default=Path("web/data/spatial"))
    args = parser.parse_args()
    build_maps(
        args.checkpoints, args.worldcover, args.output,
        canonical=args.canonical, archive_cells=args.archive_cells,
    )


if __name__ == "__main__":
    main()
