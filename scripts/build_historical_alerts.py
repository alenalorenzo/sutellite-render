"""Export the 2000–2025 anomaly map from the canonical Drive Parquet archive."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.build_harmonized_calendar import (
    AQUA,
    SNPP,
    TERRA,
    DEFAULT_CALIBRATION_START_YEAR,
    DEFAULT_CALIBRATION_END_YEAR,
    _attach_ratios,
    compute_calibration_factors,
    month_index,
)

START_YEAR = 2000
END_YEAR = 2025
# S-NPP starts on January 20. Use the complete MODIS month for January.
FIRST_FULL_SNPP = month_index(2012, 2)
FIRST_AQUA = month_index(2002, 7)
MIN_REFERENCE_YEARS = 3
REGIONS = ("angola", "australia", "brasil", "california", "espana", "indonesia", "rdc", "zambia")
CACHE_VERSION = 1
CELL_MONTH = ["grid_lat_index", "grid_lon_index", "month"]

MONTHLY_SQL = """
WITH records AS (
    SELECT region_id, year(acq_date)::INTEGER AS year, month(acq_date)::INTEGER AS month,
           least(899, floor(latitude_deg * 10))::INTEGER AS grid_lat_index,
           least(1799, floor(longitude_deg * 10))::INTEGER AS grid_lon_index,
           platform, CASE WHEN isfinite(frp_mw) AND frp_mw >= 0 THEN frp_mw END AS frp
    FROM read_parquet(?, union_by_name = true, hive_partitioning = false)
    WHERE processing_stream = 'ARCHIVE'
      AND acq_date >= DATE '2000-01-01' AND acq_date < DATE '2026-01-01'
      AND latitude_deg BETWEEN -90 AND 90 AND longitude_deg BETWEEN -180 AND 180
      AND ((sensor = 'MODIS' AND platform IN ('TERRA', 'AQUA'))
        OR (sensor = 'VIIRS' AND platform = 'SNPP'))
)
SELECT region_id, year, month, grid_lat_index, grid_lon_index,
       count(*) FILTER (WHERE platform = 'TERRA') AS modis_terra_count,
       count(*) FILTER (WHERE platform = 'AQUA') AS modis_aqua_count,
       count(*) FILTER (WHERE platform = 'SNPP') AS viirs_snpp_count,
       sum(frp) FILTER (WHERE platform <> 'SNPP') AS modis_frp,
       count(frp) FILTER (WHERE platform <> 'SNPP') AS modis_frp_rows,
       sum(frp) FILTER (WHERE platform = 'SNPP') AS snpp_frp,
       count(frp) FILTER (WHERE platform = 'SNPP') AS snpp_frp_rows
FROM records
GROUP BY ALL
"""


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def canonical_files(directory, region):
    folder = directory / region
    sources = (
        sorted(folder.glob("fire_archive_M-C61_*.parquet")),
        sorted(folder.glob("fire_archive_SV-C2_*.parquet")),
    )
    if any(not matches for matches in sources):
        raise FileNotFoundError(f"Missing the MODIS or VIIRS S-NPP archive Parquet files in {folder}")
    return sources[0] + sources[1]


def monthly_cells(connection, paths, checkpoint, region):
    source = {
        "version": CACHE_VERSION,
        "inputs": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths],
    }
    cache = checkpoint / f"{region}.parquet"
    fingerprint = checkpoint / f"{region}.json"
    if cache.is_file() and fingerprint.is_file():
        try:
            if json.loads(fingerprint.read_text(encoding="utf-8")) == source:
                cells = connection.read_parquet(str(cache)).df()
                print(f"{region}: using the saved monthly counts.", flush=True)
                return cells
        except Exception:
            print(f"{region}: unreadable checkpoint, recalculating.", flush=True)

    with tempfile.TemporaryDirectory(prefix="sutellite-parquets-") as folder:
        local = Path(folder)
        files = []
        for path in paths:
            target = local / path.name
            print(f"{region}: reading {path.name} ({path.stat().st_size / 1e6:,.0f} MB)", flush=True)
            shutil.copyfile(path, target)
            files.append(str(target))
        cells = connection.execute(MONTHLY_SQL, [files]).df()
        validate_counts(cells, region)
        connection.register("monthly_counts", cells)
        try:
            connection.table("monthly_counts").write_parquet(str(local / "monthly.parquet"))
        finally:
            connection.unregister("monthly_counts")
        temporary = cache.with_suffix(".parquet.tmp")
        shutil.copyfile(local / "monthly.parquet", temporary)
        temporary.replace(cache)
        write_json(fingerprint, source)
    return cells


def prepare_comparison(cells):
    totals = cells.groupby(["region_id", "year", "month"], as_index=False)[[TERRA, AQUA, SNPP]].sum()
    factors = compute_calibration_factors(totals)
    date = month_index(cells["year"], cells["month"])
    viirs = date >= FIRST_FULL_SNPP

    # Compare counts only where the reference instrument recorded a detection.
    # Sparse hotspot archives cannot distinguish missing coverage from a true zero.
    source_count = np.where(viirs, cells[SNPP], cells[TERRA] + cells[AQUA])
    active = cells.loc[source_count > 0, ["region_id", "year", *CELL_MONTH]].copy()
    active["viirs"] = viirs.loc[active.index]
    active["observed"] = source_count[source_count > 0]
    active["frp"] = np.where(active["viirs"], cells.loc[active.index, "snpp_frp"], cells.loc[active.index, "modis_frp"])
    active["frpRows"] = np.where(active["viirs"], cells.loc[active.index, "snpp_frp_rows"], cells.loc[active.index, "modis_frp_rows"])
    active["modis_config"] = np.where(date.loc[active.index] >= FIRST_AQUA, "TERRA_AQUA", "TERRA")
    active = _attach_ratios(active, factors)

    for label, factor in (("value", "ratio"), ("low", "ratio_low"), ("high", "ratio_high")):
        active[label] = active["observed"] * np.where(active["viirs"], 1, active[factor])
    return active.drop(columns=["ratio", "ratio_low", "ratio_high", "modis_config"])


def validate_counts(cells, region):
    if set(cells["region_id"]) != {region}:
        raise ValueError(f"Expected {region} records, got {cells['region_id'].unique()}.")
    if not cells["year"].between(START_YEAR, END_YEAR).all():
        raise ValueError(f"{region}: archive records fall outside {START_YEAR}–{END_YEAR}.")
    if not (cells[TERRA].sum() and cells[AQUA].sum() and cells[SNPP].sum()):
        raise ValueError(f"{region}: MODIS Terra, Aqua and S-NPP records are needed for calibration.")


def compare_cells(active):
    eligible = active.loc[active["value"].notna()].copy()
    eligible.sort_values([*CELL_MONTH, "year"], inplace=True)
    groups = eligible.groupby(CELL_MONTH, sort=False)
    eligible["years"] = groups.cumcount()
    same_cell_month = eligible["years"].gt(0)
    eligible["previousMax"] = groups["value"].cummax().shift().where(same_cell_month)
    eligible["previousUpper"] = groups["high"].cummax().shift().where(same_cell_month)
    new_record = ~same_cell_month | eligible["value"].gt(eligible["previousMax"])
    eligible["previousYear"] = eligible["year"].where(new_record).ffill().shift().where(same_cell_month)
    new_upper = ~same_cell_month | eligible["high"].gt(eligible["previousUpper"])
    eligible["upperYear"] = eligible["year"].where(new_upper).ffill().shift().where(same_cell_month)
    eligible["firstYear"] = groups["year"].transform("first")
    eligible["lastYear"] = groups["year"].shift()
    # A flag must exceed the previous ranges even after accounting for the
    # spread of the regional conversion ratio in the overlap years.
    comparable = eligible["years"].ge(MIN_REFERENCE_YEARS)
    flagged = comparable & eligible["low"].gt(eligible["previousUpper"] + 1e-6)
    return eligible.loc[comparable], eligible.loc[flagged]


def export_region(cells, directory, region):
    validate_counts(cells, region)
    active = prepare_comparison(cells)
    compared, flagged = compare_cells(active)
    compared_counts = compared.groupby(["year", "month"]).size().to_dict()
    flagged_counts = flagged.groupby(["year", "month"]).size().to_dict()
    periods = {}
    for (year, month), frame in active.groupby(["year", "month"]):
        source = "viirs" if frame["viirs"].iat[0] else "modis"
        status = "uncalibrated" if frame["value"].isna().all() else "available"
        periods.setdefault(int(year), {})[int(month)] = {
            "source": source,
            "status": status,
            "cells": len(frame),
            "compared": compared_counts.get((year, month), 0),
            "alerts": flagged_counts.get((year, month), 0),
        }

    region_dir = directory / region
    region_dir.mkdir()
    for (year, month), frame in flagged.groupby(["year", "month"]):
        year_dir = region_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        points = []
        for row in frame.sort_values("value", ascending=False).itertuples(index=False):
            points.append({
                "lat": (row.grid_lat_index + 0.5) / 10,
                "lon": (row.grid_lon_index + 0.5) / 10,
                "detections": int(row.observed),
                "value": round(row.value, 6),
                "low": round(row.low, 6),
                "high": round(row.high, 6),
                "historicalMax": round(row.previousMax, 6),
                "historicalUpper": round(row.previousUpper, 6),
                "historicalYear": int(row.previousYear),
                "historicalUpperYear": int(row.upperYear),
                "firstReferenceYear": int(row.firstYear),
                "lastReferenceYear": int(row.lastYear),
                "referenceYears": int(row.years),
                "frp": round(row.frp, 2) if pd.notna(row.frp) else None,
                "validFrp": int(row.frpRows),
            })
        write_json(year_dir / f"{month}.json", points)

    south = float(cells["grid_lat_index"].min()) / 10
    north = float(cells["grid_lat_index"].max() + 1) / 10
    west = float(cells["grid_lon_index"].min()) / 10
    east = float(cells["grid_lon_index"].max() + 1) / 10
    return {
        "bounds": [[south, west], [north, east]],
        "periods": periods,
        "yearsWithRecords": sorted(int(year) for year in cells["year"].unique()),
        "records": {symbol: int(cells[column].sum()) for symbol, column in (
            ("terra", TERRA), ("aqua", AQUA), ("snpp", SNPP),
        )},
        "alertCount": len(flagged),
    }


def write_manifest(directory, regions):
    write_json(directory / "index.json", {
        "schema": 1,
        "canonical": "canonical_v1.1",
        "startYear": START_YEAR,
        "endYear": END_YEAR,
        "calibrationYears": [DEFAULT_CALIBRATION_START_YEAR, DEFAULT_CALIBRATION_END_YEAR],
        "regions": regions,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    inputs = {region: canonical_files(args.canonical, region) for region in REGIONS}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)

    import duckdb

    with tempfile.TemporaryDirectory(prefix="sutellite-history-") as temporary:
        output = Path(temporary) / "historical-alerts"
        output.mkdir()
        inventory = {}
        with duckdb.connect(config={"memory_limit": "2GB", "threads": 2}) as connection:
            for region, paths in inputs.items():
                cells = monthly_cells(connection, paths, args.work_dir, region)
                data = export_region(cells, output, region)
                data["sourceFiles"] = [{"name": path.name, "bytes": path.stat().st_size} for path in paths]
                inventory[region] = data
                records = sum(data["records"].values())
                print(f"{region}: {len(data['yearsWithRecords'])} years, {records:,} canonical records, {data['alertCount']:,} flags.", flush=True)
                del cells
        write_manifest(output, inventory)
        archive = Path(temporary) / "historical-alerts.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            for file in sorted(output.rglob("*.json")):
                bundle.write(file, "web/data/historical-alerts/" + file.relative_to(output).as_posix())
        partial = args.output_zip.with_suffix(".zip.tmp")
        shutil.copyfile(archive, partial)
        partial.replace(args.output_zip)
    print(f"ZIP ready: {args.output_zip}", flush=True)


if __name__ == "__main__":
    main()
