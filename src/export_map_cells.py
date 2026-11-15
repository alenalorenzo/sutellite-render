"""Export map_cells.json (v2) for the map page, using ALL canonical records.

Nothing is filtered by year, sensor, satellite or stream: the page does that.
Records are aggregated to export cells (default 0.5 degrees) by year, platform
and processing stream, keeping the record count and the mean of non-negative FRP.

Land type: ESA WorldCover 2021 class of the 0.1-degree cell with the most
records inside each export cell (class 0 = no data). Land type needs the
WorldCover Map tiles. Without them every cell is class 0. The script reports
how much of the data has a land class and writes the tiles that are missing to
reports/map/worldcover_missing_tiles.csv (largest fire counts first).

Usage:
    python -m src.export_map_cells --input-dir processed/canonical_v1.1 \
        --worldcover-dir data/worldcover --output-file reports/map/map_cells.json
    # fetch missing tiles until 95% of the records have one (large downloads!):
    python -m src.export_map_cells ... --download-until 0.95 --cache-dir reports/map/cache
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FINE_DEG = 0.1
PLATFORM_ORDER = ["TERRA", "AQUA", "SNPP", "NOAA-20", "NOAA-21"]
STREAM_ORDER = ["ARCHIVE", "NRT"]
TILE_PATTERN = re.compile(r"([NS]\d{2}[EW]\d{3})_Map", re.IGNORECASE)
TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)


def tile_name(lat: float, lon: float) -> str:
    """WorldCover 3-degree tile that contains a point (same rule as geo/)."""

    south = int(math.floor(lat / 3) * 3)
    west = int(math.floor(lon / 3) * 3)
    return f"{'S' if south < 0 else 'N'}{abs(south):02d}{'W' if west < 0 else 'E'}{abs(west):03d}"


def _ratio(cell_deg: float) -> int:
    k = cell_deg / FINE_DEG
    if k < 1 or abs(k - round(k)) > 1e-9:
        raise ValueError("cell_deg must be a positive multiple of 0.1.")
    return int(round(k))


def dominant_land(fine: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    """Per export cell: total records and the land class of its busiest fine cell."""

    k = _ratio(cell_deg)
    for col in ("lat_i", "lon_i", "n"):
        if col not in fine.columns:
            raise ValueError(f"Missing column: {col}")
    d = fine.copy()
    d["lc"] = d["lc"].fillna(0).astype(int) if "lc" in d.columns else 0
    d["clat"] = np.floor_divide(d["lat_i"], k)
    d["clon"] = np.floor_divide(d["lon_i"], k)
    total = d.groupby(["clat", "clon"], as_index=False)["n"].sum()
    top = (
        d.sort_values(["n", "lat_i", "lon_i"], ascending=[False, True, True])
        .drop_duplicates(["clat", "clon"])[["clat", "clon", "lc"]]
    )
    out = total.merge(top, on=["clat", "clon"]).sort_values(["clat", "clon"])
    return out.reset_index(drop=True)


def _ordered(values: Iterable[str], preferred: list[str]) -> list[str]:
    present = set(values)
    return [v for v in preferred if v in present] + sorted(present - set(preferred))


def build_payload(
    fine: pd.DataFrame,
    agg: pd.DataFrame,
    cell_deg: float,
    sensor_of: dict[str, str],
    meta_extra: dict | None = None,
    *,
    monthly: bool = False,
) -> dict:
    """agg columns: clat, clon, yr, platform, stream, n, n_frp, frp_sum."""

    cells = dominant_land(fine, cell_deg)
    cells["idx"] = np.arange(len(cells))
    platforms = _ordered(agg["platform"], PLATFORM_ORDER)
    streams = _ordered(agg["stream"], STREAM_ORDER)
    if len(platforms) > 10 or len(streams) > 10:
        raise ValueError("Too many platforms or streams for the packed key.")

    m = agg.merge(cells[["clat", "clon", "idx"]], on=["clat", "clon"], how="inner")
    m["pl"] = m["platform"].map({p: i for i, p in enumerate(platforms)})
    m["st"] = m["stream"].map({s: i for i, s in enumerate(streams)})
    period = m["yr"].astype("int64")
    if monthly:
        period = period * 100 + m["month"].astype("int64")
    m["key"] = period * 100 + m["pl"] * 10 + m["st"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_frp = np.where(m["n_frp"] > 0, m["frp_sum"] / m["n_frp"], np.nan)
    m["f"] = np.where(np.isnan(mean_frp), -1, np.round(mean_frp * 10)).astype("int64")
    m = m.sort_values(["idx", "key"])

    meta = {
        "version": 2,
        "cell_deg": cell_deg,
        "platforms": platforms,
        "streams": streams,
        "sensor_of": {p: sensor_of.get(p, "MODIS" if p in ("TERRA", "AQUA") else "VIIRS")
                      for p in platforms},
        "years": [int(m["yr"].min()), int(m["yr"].max())],
        "total_records": int(m["n"].sum()),
        "frp": "mean of non-negative FRP (MW)",
        "land_cover": "ESA WorldCover 2021 (class of the fine cell with most records)",
        "note": "Counts are detection records, not unique fires; MODIS and VIIRS counts are not comparable.",
    }
    meta.update(meta_extra or {})
    if monthly:
        meta["period"] = "month"
    return {
        "meta": meta,
        "cells": [
            [round(float(a * cell_deg), 4), round(float(b * cell_deg), 4), int(c)]
            for a, b, c in zip(cells["clat"], cells["clon"], cells["lc"])
        ],
        "rec": {
            "c": m["idx"].astype("int64").tolist(),
            "k": m["key"].astype("int64").tolist(),
            "n": m["n"].astype("int64").tolist(),
            "f": m["f"].tolist(),
            **({"w": m["n_frp"].astype("int64").tolist()} if monthly else {}),
        },
    }


def plan_tiles(fine: pd.DataFrame, local_tiles: set[str]) -> pd.DataFrame:
    """Tiles needed by the data, with records, and whether they are on disk."""

    d = fine[["lat_i", "lon_i", "n"]].copy()
    d["tile"] = [
        tile_name(a * FINE_DEG, b * FINE_DEG) for a, b in zip(d["lat_i"], d["lon_i"])
    ]
    t = d.groupby("tile", as_index=False).agg(records=("n", "sum"))
    t["local"] = t["tile"].isin(local_tiles)
    t = t.sort_values("records", ascending=False).reset_index(drop=True)
    t["url"] = t["tile"].map(lambda x: TILE_URL.format(tile=x))
    return t


def tiles_for_coverage(plan: pd.DataFrame, target: float) -> int:
    """How many MISSING tiles (largest first) are needed to cover `target` of the records."""

    total = float(plan["records"].sum())
    if total <= 0:
        return 0
    have = float(plan.loc[plan["local"], "records"].sum()) / total
    if have >= target:
        return 0
    gain = plan.loc[~plan["local"], "records"].cumsum() / total
    reached = np.flatnonzero((have + gain.to_numpy()) >= target)
    return int(reached[0]) + 1 if len(reached) else int(len(gain))


def coverage_report(plan: pd.DataFrame) -> None:
    total = float(plan["records"].sum())
    have = float(plan.loc[plan["local"], "records"].sum()) / total if total else 0.0
    print(f"WorldCover tiles on disk cover {have:.1%} of the records.")
    for target in (0.5, 0.8, 0.9, 0.95, 0.99):
        need = tiles_for_coverage(plan, target)
        if need:
            print(f"  to reach {target:.0%}: download {need} more tiles")


def local_tile_set(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {m.group(1).upper() for p in folder.rglob("*.tif")
            if (m := TILE_PATTERN.search(p.name))}


def download_tiles(plan: pd.DataFrame, folder: Path, how_many: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    todo = plan[~plan["local"]].head(how_many)
    tiles = todo["tile"].astype(str).tolist()
    records = todo["records"].astype("int64").tolist()
    urls = todo["url"].astype(str).tolist()
    for tile, count, url in zip(tiles, records, urls):
        target = folder / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        part = target.with_suffix(".part")
        print(f"Downloading {tile} ({count:,} records)...", flush=True)
        try:
            urllib.request.urlretrieve(url, str(part))  # noqa: S310
            part.rename(target)
        except urllib.error.HTTPError as exc:
            print(f"  skipped {tile}: HTTP {exc.code} (probably no land tile)")
            part.unlink(missing_ok=True)


def query_all(input_dir: Path, cell_deg: float):
    import duckdb  # type: ignore[import-not-found]

    k = _ratio(cell_deg)
    files = sorted(str(p) for p in Path(input_dir).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {input_dir}/<region>/")
    con = duckdb.connect()
    src = "read_parquet(?, union_by_name = true)"
    valid = "latitude_deg BETWEEN -90 AND 90 AND longitude_deg BETWEEN -180 AND 180"
    fine = con.execute(
        f"""SELECT CAST(FLOOR(latitude_deg / 0.1) AS INTEGER) AS lat_i,
                   CAST(FLOOR(longitude_deg / 0.1) AS INTEGER) AS lon_i,
                   COUNT(*) AS n FROM {src} WHERE {valid} GROUP BY 1, 2""",
        [files],
    ).fetchdf()
    agg = con.execute(
        f"""SELECT CAST(FLOOR(FLOOR(latitude_deg / 0.1) / {k}) AS INTEGER) AS clat,
                   CAST(FLOOR(FLOOR(longitude_deg / 0.1) / {k}) AS INTEGER) AS clon,
                   CAST(YEAR(TRY_CAST(acq_date AS DATE)) AS INTEGER) AS yr,
                   platform, processing_stream AS stream,
                   COUNT(*) AS n,
                   COUNT(*) FILTER (WHERE frp_mw >= 0) AS n_frp,
                   COALESCE(SUM(frp_mw) FILTER (WHERE frp_mw >= 0), 0) AS frp_sum
            FROM {src} WHERE {valid} AND TRY_CAST(acq_date AS DATE) IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5""",
        [files],
    ).fetchdf()
    sensors = con.execute(
        f"SELECT DISTINCT platform, sensor FROM {src}", [files]
    ).fetchdf()
    return fine, agg, dict(zip(sensors["platform"], sensors["sensor"]))


def query_cached(input_dir: Path, cell_deg: float, cache_dir: Path | None):
    """query_all with an optional on-disk cache (needs pyarrow, like the project)."""

    if cache_dir is None:
        return query_all(input_dir, cell_deg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cell_deg:g}"
    f_fine, f_agg, f_sen = (cache_dir / f"fine.parquet", cache_dir / f"agg_{tag}.parquet",
                            cache_dir / "sensors.json")
    if f_fine.exists() and f_agg.exists() and f_sen.exists():
        print(f"Using cached query results from {cache_dir}", flush=True)
        return (pd.read_parquet(f_fine), pd.read_parquet(f_agg),
                json.loads(f_sen.read_text(encoding="utf-8")))
    fine, agg, sensors = query_all(input_dir, cell_deg)
    fine.to_parquet(f_fine)
    agg.to_parquet(f_agg)
    f_sen.write_text(json.dumps(sensors), encoding="utf-8")
    return fine, agg, sensors


def add_land_cover(fine: pd.DataFrame, worldcover_dir: Path) -> pd.DataFrame:
    anadir_worldcover = _load_worldcover_lookup()
    t = fine.copy()
    t["grid_lat"] = (t["lat_i"] * FINE_DEG).round(4)
    t["grid_lon"] = (t["lon_i"] * FINE_DEG).round(4)
    t = anadir_worldcover(t, worldcover_dir)
    t["lc"] = t["worldcover_2021_codigo"].fillna(0).astype(int)
    return t.drop(columns=["grid_lat", "grid_lon", "worldcover_2021_codigo",
                           "worldcover_2021_clase"])


def _load_worldcover_lookup():
    try:
        from geo.worldcover_lookup import (  # type: ignore[import-not-found]
            anadir_worldcover,
        )

        return anadir_worldcover
    except ModuleNotFoundError:
        pass
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    for path in [root / "geo" / "worldcover_lookup.py", *sorted(root.rglob("worldcover_lookup.py"))]:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("worldcover_lookup", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.anadir_worldcover
    raise FileNotFoundError("geo/worldcover_lookup.py not found in this repository.")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export map_cells.json (v2).")
    p.add_argument("--input-dir", type=Path, default=Path("processed/canonical_v1.1"))
    p.add_argument("--worldcover-dir", type=Path, default=None)
    p.add_argument("--download-tiles", type=int, default=0,
                   help="Download the N most important missing WorldCover tiles (large).")
    p.add_argument("--download-until", type=float, default=None, metavar="SHARE",
                   help="Download missing tiles until this share of records (e.g. 0.95) has a tile.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Cache the slow Parquet query here; reruns then take seconds.")
    p.add_argument("--cell-degrees", type=float, default=0.5)
    p.add_argument("--output-file", type=Path, default=Path("reports/map/map_cells.json"))
    a = p.parse_args(list(argv) if argv is not None else None)
    if a.output_file.exists():
        raise FileExistsError(f"{a.output_file} already exists.")
    if (a.download_tiles or a.download_until) and not a.worldcover_dir:
        raise SystemExit("Downloading tiles needs --worldcover-dir (where to save them).")
    if a.download_until is not None and not 0 < a.download_until <= 1:
        raise SystemExit("--download-until must be between 0 and 1.")

    print("Reading all canonical records (this can take several minutes)...", flush=True)
    fine, agg, sensor_of = query_cached(a.input_dir, a.cell_degrees, a.cache_dir)
    total = int(agg["n"].sum())
    print(f"Records: {total:,}; fine cells: {len(fine):,}")

    plan = plan_tiles(fine, local_tile_set(a.worldcover_dir) if a.worldcover_dir else set())
    coverage_report(plan)
    how_many = a.download_tiles
    if a.download_until is not None:
        how_many = max(how_many, tiles_for_coverage(plan, a.download_until))
    if how_many:
        print(f"Downloading up to {how_many} tiles (tens of MB each; ocean tiles are skipped).")
        download_tiles(plan, a.worldcover_dir, how_many)
        plan = plan_tiles(fine, local_tile_set(a.worldcover_dir))
        coverage_report(plan)

    if a.worldcover_dir and plan["local"].any():
        fine = add_land_cover(fine, a.worldcover_dir)
    known = float(fine.loc[fine["lc"] > 0, "n"].sum() / fine["n"].sum()) if "lc" in fine else 0.0
    print(f"Records with a land class: {known:.1%}")

    missing = plan[~plan["local"]]
    if len(missing):
        csv = a.output_file.parent / "worldcover_missing_tiles.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(csv, index=False)
        print(f"{len(missing)} WorldCover tiles missing; list written to {csv}")

    payload = build_payload(
        fine, agg, a.cell_degrees, sensor_of,
        {"land_cover_known_share": round(known, 4)},
    )
    a.output_file.parent.mkdir(parents=True, exist_ok=True)
    a.output_file.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(payload['cells']):,} cells and {len(payload['rec']['n']):,} "
          f"records ({total:,} detections) to {a.output_file}")


if __name__ == "__main__":
    main()
