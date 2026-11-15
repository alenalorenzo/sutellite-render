"""Compare MODIS Terra/Aqua and VIIRS SNPP in canonical NASA FIRMS data.

The analysis uses ARCHIVE observations from complete calendar years by default.
It keeps platforms separate and reports source-observation counts, not unique
fire events. MODIS-to-SNPP ratios are descriptive count ratios only; they are
not estimates of relative detection probability or corrected sensor bias.

Input layout:
    processed/canonical_v1.1/<region>/*.parquet

Outputs:
    monthly_platform_counts.csv
    monthly_sensor_comparison.csv
    monthly_frp_summary.csv
    monthly_spatial_grid_counts.csv
    comparison_summary.json

The script streams Parquet record batches and does not load the full dataset
into memory.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA, SCHEMA_VERSION


COMPARISON_VERSION = "1.0"
DEFAULT_INPUT_DIR = Path("processed/canonical_v1.1")
DEFAULT_OUTPUT_DIR = Path("reports/sensor_comparison")
DEFAULT_START_YEAR = 2013
DEFAULT_END_YEAR = 2025
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_GRID_SIZE_DEGREES = 0.1

REGIONS = {
    "angola",
    "australia",
    "brasil",
    "california",
    "espana",
    "indonesia",
    "rdc",
    "zambia",
}

PLATFORM_SENSOR = {
    "TERRA": "MODIS",
    "AQUA": "MODIS",
    "SNPP": "VIIRS",
}
TARGET_PLATFORMS = tuple(PLATFORM_SENSOR)

INPUT_COLUMNS = [
    "observation_id",
    "region_id",
    "sensor",
    "platform",
    "processing_stream",
    "acq_date",
    "latitude_deg",
    "longitude_deg",
    "frp_mw",
    "daynight",
]

MONTHLY_KEY_FIELDS = [
    "region_id",
    "platform",
    "_year",
    "_month",
    "daynight",
]
FRP_KEY_FIELDS = MONTHLY_KEY_FIELDS
GRID_KEY_FIELDS = [
    "region_id",
    "platform",
    "_year",
    "_month",
    "daynight",
    "_grid_lat_index",
    "_grid_lon_index",
]

# PyArrow's compute functions are dynamically exposed; this alias also keeps
# static type checkers from reporting false missing-attribute warnings.
pc: Any = _pc


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_count(table: pa.Table, keys: list[str]) -> list[dict]:
    """Return grouped observation counts as Python dictionaries."""
    if table.num_rows == 0:
        return []

    grouped = table.group_by(keys).aggregate([("observation_id", "count")])
    return grouped.to_pylist()


def _update_monthly_counts(
    table: pa.Table,
    monthly_daynight_counts: Counter,
) -> None:
    for row in _group_count(table, MONTHLY_KEY_FIELDS):
        key = (
            row["region_id"],
            row["platform"],
            int(row["_year"]),
            int(row["_month"]),
            row["daynight"],
        )
        monthly_daynight_counts[key] += int(row["observation_id_count"])


def _update_active_days(
    table: pa.Table,
    active_days: dict[tuple[str, str, int, int], set[date]],
) -> None:
    if table.num_rows == 0:
        return

    keys = ["region_id", "platform", "_year", "_month", "acq_date"]
    for row in _group_count(table, keys):
        key = (
            row["region_id"],
            row["platform"],
            int(row["_year"]),
            int(row["_month"]),
        )
        active_days[key].add(row["acq_date"])


def _update_spatial_grid(
    table: pa.Table,
    grid_counts: Counter,
    grid_size_degrees: float,
) -> None:
    if table.num_rows == 0:
        return

    lat_index = pc.cast(
        pc.floor(pc.divide(table["latitude_deg"], grid_size_degrees)),
        pa.int32(),
    )
    lon_index = pc.cast(
        pc.floor(pc.divide(table["longitude_deg"], grid_size_degrees)),
        pa.int32(),
    )

    grid_table = table.append_column("_grid_lat_index", lat_index)
    grid_table = grid_table.append_column("_grid_lon_index", lon_index)

    for row in _group_count(grid_table, GRID_KEY_FIELDS):
        key = (
            row["region_id"],
            row["platform"],
            int(row["_year"]),
            int(row["_month"]),
            row["daynight"],
            int(row["_grid_lat_index"]),
            int(row["_grid_lon_index"]),
        )
        grid_counts[key] += int(row["observation_id_count"])


def _update_frp_summary(
    table: pa.Table,
    frp_status_counts: Counter,
    frp_stats: dict[tuple[str, str, int, int, str], dict],
) -> None:
    if table.num_rows == 0:
        return

    frp = table["frp_mw"]
    negative_mask = pc.fill_null(pc.less(frp, 0), False)
    zero_mask = pc.fill_null(pc.equal(frp, 0), False)
    positive_mask = pc.fill_null(pc.greater(frp, 0), False)
    missing_mask = pc.is_null(frp)

    status_values = pc.if_else(
        missing_mask,
        "missing",
        pc.if_else(
            negative_mask,
            "negative",
            pc.if_else(zero_mask, "zero", "positive"),
        ),
    )
    status_table = table.append_column("_frp_status", status_values)
    status_grouped = status_table.group_by(
        FRP_KEY_FIELDS + ["_frp_status"]
    ).aggregate([("observation_id", "count")])

    for row in status_grouped.to_pylist():
        key = (
            row["region_id"],
            row["platform"],
            int(row["_year"]),
            int(row["_month"]),
            row["daynight"],
        )
        frp_status_counts[(key, row["_frp_status"])] += int(
            row["observation_id_count"]
        )

    nonnegative_mask = pc.fill_null(pc.greater_equal(frp, 0), False)
    nonnegative_table = table.filter(nonnegative_mask)
    if nonnegative_table.num_rows == 0:
        return

    squared_values = pc.multiply(
        nonnegative_table["frp_mw"],
        nonnegative_table["frp_mw"],
    )
    nonnegative_table = nonnegative_table.append_column(
        "_frp_squared", squared_values
    )

    grouped = nonnegative_table.group_by(FRP_KEY_FIELDS).aggregate(
        [
            ("frp_mw", "count"),
            ("frp_mw", "min"),
            ("frp_mw", "max"),
            ("frp_mw", "sum"),
            ("_frp_squared", "sum"),
        ]
    )

    for row in grouped.to_pylist():
        key = (
            row["region_id"],
            row["platform"],
            int(row["_year"]),
            int(row["_month"]),
            row["daynight"],
        )
        state = frp_stats.setdefault(
            key,
            {
                "count": 0,
                "sum": 0.0,
                "sum_squared": 0.0,
                "min": None,
                "max": None,
            },
        )

        state["count"] += int(row["frp_mw_count"])
        state["sum"] += float(row["frp_mw_sum"] or 0.0)
        state["sum_squared"] += float(row["_frp_squared_sum"] or 0.0)

        minimum = float(row["frp_mw_min"])
        maximum = float(row["frp_mw_max"])
        state["min"] = minimum if state["min"] is None else min(state["min"], minimum)
        state["max"] = maximum if state["max"] is None else max(state["max"], maximum)


def _build_monthly_platform_rows(
    monthly_daynight_counts: Counter,
    active_days: dict[tuple[str, str, int, int], set[date]],
) -> list[dict]:
    monthly_counts: Counter = Counter()
    day_counts: Counter = Counter()
    night_counts: Counter = Counter()

    for key, count in monthly_daynight_counts.items():
        region, platform, year, month, daynight = key
        month_key = (region, platform, year, month)
        monthly_counts[month_key] += count
        if daynight == "D":
            day_counts[month_key] += count
        elif daynight == "N":
            night_counts[month_key] += count

    rows = []
    for key in sorted(monthly_counts):
        region, platform, year, month = key
        observations = monthly_counts[key]
        days_in_month = calendar.monthrange(year, month)[1]
        rows.append(
            {
                "region_id": region,
                "sensor": PLATFORM_SENSOR[platform],
                "platform": platform,
                "year": year,
                "month": month,
                "observations": observations,
                "day_observations": day_counts[key],
                "night_observations": night_counts[key],
                "active_detection_days": len(active_days.get(key, set())),
                "calendar_days": days_in_month,
                "observations_per_calendar_day": round(
                    observations / days_in_month, 6
                ),
            }
        )

    return rows


def _build_sensor_comparison_rows(
    monthly_daynight_counts: Counter,
    start_year: int,
    end_year: int,
) -> list[dict]:
    rows = []
    for region in sorted(REGIONS):
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for daynight in ("D", "N"):
                    terra = monthly_daynight_counts.get(
                        (region, "TERRA", year, month, daynight), 0
                    )
                    aqua = monthly_daynight_counts.get(
                        (region, "AQUA", year, month, daynight), 0
                    )
                    snpp = monthly_daynight_counts.get(
                        (region, "SNPP", year, month, daynight), 0
                    )
                    rows.append(
                        {
                            "region_id": region,
                            "year": year,
                            "month": month,
                            "daynight": daynight,
                            "modis_terra_observations": terra,
                            "modis_aqua_observations": aqua,
                            "viirs_snpp_observations": snpp,
                            "terra_to_snpp_observation_count_ratio": (
                                round(terra / snpp, 6) if snpp > 0 else None
                            ),
                            "aqua_to_snpp_observation_count_ratio": (
                                round(aqua / snpp, 6) if snpp > 0 else None
                            ),
                        }
                    )
    return rows


def _build_frp_rows(
    frp_status_counts: Counter,
    frp_stats: dict[tuple[str, str, int, int, str], dict],
) -> list[dict]:
    all_keys = set(frp_stats)
    all_keys.update(key for key, _status in frp_status_counts)
    rows = []

    for key in sorted(all_keys):
        region, platform, year, month, daynight = key
        state = frp_stats.get(key)
        if state is None:
            count = 0
            mean = None
            stddev = None
            minimum = None
            maximum = None
        else:
            count = state["count"]
            mean = state["sum"] / count if count else None
            variance = (
                max(0.0, state["sum_squared"] / count - mean * mean)
                if count and mean is not None
                else None
            )
            stddev = math.sqrt(variance) if variance is not None else None
            minimum = state["min"]
            maximum = state["max"]

        rows.append(
            {
                "region_id": region,
                "sensor": PLATFORM_SENSOR[platform],
                "platform": platform,
                "year": year,
                "month": month,
                "daynight": daynight,
                "nonnegative_frp_count": count,
                "negative_frp_count": frp_status_counts[(key, "negative")],
                "zero_frp_count": frp_status_counts[(key, "zero")],
                "positive_frp_count": frp_status_counts[(key, "positive")],
                "missing_frp_count": frp_status_counts[(key, "missing")],
                "min_nonnegative_frp_mw": minimum,
                "mean_nonnegative_frp_mw": mean,
                "stddev_nonnegative_frp_mw": stddev,
                "max_nonnegative_frp_mw": maximum,
            }
        )

    return rows


def _write_grid_csv(
    path: Path,
    grid_counts: Counter,
    grid_size_degrees: float,
) -> None:
    fieldnames = [
        "region_id",
        "sensor",
        "platform",
        "year",
        "month",
        "daynight",
        "grid_lat_index",
        "grid_lon_index",
        "grid_lat_min_deg",
        "grid_lon_min_deg",
        "grid_size_degrees",
        "observations",
    ]

    # Write one row at a time to avoid constructing a second large list of
    # dictionaries for a potentially high-cardinality spatial grid.
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grid_counts):
            region, platform, year, month, daynight, lat_index, lon_index = key
            writer.writerow(
                {
                    "region_id": region,
                    "sensor": PLATFORM_SENSOR[platform],
                    "platform": platform,
                    "year": year,
                    "month": month,
                    "daynight": daynight,
                    "grid_lat_index": lat_index,
                    "grid_lon_index": lon_index,
                    "grid_lat_min_deg": round(lat_index * grid_size_degrees, 8),
                    "grid_lon_min_deg": round(lon_index * grid_size_degrees, 8),
                    "grid_size_degrees": grid_size_degrees,
                    "observations": grid_counts[key],
                }
            )


def compare_dataset(
    input_directory: Path,
    output_directory: Path,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
) -> dict:
    """Stream canonical Parquet files and write the first sensor comparison."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if start_year < 1900 or end_year > 2200 or start_year > end_year:
        raise ValueError("Provide a valid year range with start_year <= end_year")
    if not math.isfinite(grid_size_degrees) or not 0 < grid_size_degrees <= 180:
        raise ValueError("grid_size_degrees must be greater than 0 and at most 180")
    if not input_directory.is_dir():
        raise SystemExit(f"Canonical input directory not found: {input_directory}")

    parquet_files = sorted(
        path
        for path in input_directory.rglob("*.parquet")
        if path.parent.name in REGIONS
    )
    if not parquet_files:
        raise SystemExit(f"No regional Parquet files found in: {input_directory}")

    monthly_daynight_counts: Counter = Counter()
    active_days: dict[tuple[str, str, int, int], set[date]] = defaultdict(set)
    grid_counts: Counter = Counter()
    frp_status_counts: Counter = Counter()
    frp_stats: dict[tuple[str, str, int, int, str], dict] = {}
    rows_scanned = 0
    archive_rows = 0
    archive_target_platform_rows = 0
    eligible_rows = 0

    print("SUTELLITE - MODIS / VIIRS SENSOR COMPARISON")
    print("=" * 70)
    print(f"Parquet files discovered: {len(parquet_files):,}")
    print(f"Period: {start_year}-{end_year} (inclusive)")
    print("Processing stream: ARCHIVE")
    print("Platforms: TERRA, AQUA, SNPP")
    print(f"Spatial grid: {grid_size_degrees:g} degrees")
    print(f"Batch size: {batch_size:,}")
    print("Starting streaming comparison...\n")

    for file_number, path in enumerate(parquet_files, start=1):
        print(
            f"[{file_number}/{len(parquet_files)}] Processing: {path.name}",
            flush=True,
        )
        parquet_file = pq.ParquetFile(path)

        if not parquet_file.schema_arrow.equals(CANONICAL_SCHEMA, check_metadata=True):
            raise ValueError(
                f"Canonical schema mismatch in {path.relative_to(input_directory)}"
            )

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=INPUT_COLUMNS,
            use_threads=True,
        ):
            table = pa.Table.from_batches([batch])
            rows_scanned += table.num_rows

            archive_mask = pc.equal(table["processing_stream"], "ARCHIVE")
            archive_table = table.filter(archive_mask)
            archive_rows += archive_table.num_rows
            if archive_table.num_rows == 0:
                continue

            platform_mask = pc.is_in(
                archive_table["platform"],
                value_set=pa.array(TARGET_PLATFORMS, type=pa.string()),
            )
            platform_table = archive_table.filter(platform_mask)
            archive_target_platform_rows += platform_table.num_rows
            if platform_table.num_rows == 0:
                continue

            years = pc.year(platform_table["acq_date"])
            year_mask = pc.and_(
                pc.greater_equal(years, start_year),
                pc.less_equal(years, end_year),
            )
            eligible = platform_table.filter(year_mask)
            eligible_rows += eligible.num_rows
            if eligible.num_rows == 0:
                continue

            # Detect a canonical identity mismatch rather than silently grouping
            # a platform under the wrong sensor label.
            mismatch_mask = pc.or_(
                pc.or_(
                    pc.and_(
                        pc.equal(eligible["platform"], "TERRA"),
                        pc.not_equal(eligible["sensor"], "MODIS"),
                    ),
                    pc.and_(
                        pc.equal(eligible["platform"], "AQUA"),
                        pc.not_equal(eligible["sensor"], "MODIS"),
                    ),
                ),
                pc.and_(
                    pc.equal(eligible["platform"], "SNPP"),
                    pc.not_equal(eligible["sensor"], "VIIRS"),
                ),
            )
            if pc.any(mismatch_mask).as_py():
                raise ValueError("Sensor/platform mismatch in canonical data.")

            enriched = eligible.append_column("_year", pc.year(eligible["acq_date"]))
            enriched = enriched.append_column("_month", pc.month(eligible["acq_date"]))

            _update_monthly_counts(enriched, monthly_daynight_counts)
            _update_active_days(enriched, active_days)
            _update_spatial_grid(enriched, grid_counts, grid_size_degrees)
            _update_frp_summary(enriched, frp_status_counts, frp_stats)

    output_directory.mkdir(parents=True, exist_ok=True)

    sensor_platform_counts: Counter = Counter()
    for key, count in monthly_daynight_counts.items():
        _region, platform, _year, _month, _daynight = key
        sensor_platform_counts[(PLATFORM_SENSOR[platform], platform)] += count

    monthly_platform_rows = _build_monthly_platform_rows(
        monthly_daynight_counts,
        active_days,
    )
    comparison_rows = _build_sensor_comparison_rows(
        monthly_daynight_counts,
        start_year,
        end_year,
    )
    frp_rows = _build_frp_rows(frp_status_counts, frp_stats)

    _write_csv(
        output_directory / "monthly_platform_counts.csv",
        [
            "region_id",
            "sensor",
            "platform",
            "year",
            "month",
            "observations",
            "day_observations",
            "night_observations",
            "active_detection_days",
            "calendar_days",
            "observations_per_calendar_day",
        ],
        monthly_platform_rows,
    )
    _write_csv(
        output_directory / "monthly_sensor_comparison.csv",
        [
            "region_id",
            "year",
            "month",
            "daynight",
            "modis_terra_observations",
            "modis_aqua_observations",
            "viirs_snpp_observations",
            "terra_to_snpp_observation_count_ratio",
            "aqua_to_snpp_observation_count_ratio",
        ],
        comparison_rows,
    )
    _write_csv(
        output_directory / "monthly_frp_summary.csv",
        [
            "region_id",
            "sensor",
            "platform",
            "year",
            "month",
            "daynight",
            "nonnegative_frp_count",
            "negative_frp_count",
            "zero_frp_count",
            "positive_frp_count",
            "missing_frp_count",
            "min_nonnegative_frp_mw",
            "mean_nonnegative_frp_mw",
            "stddev_nonnegative_frp_mw",
            "max_nonnegative_frp_mw",
        ],
        frp_rows,
    )
    _write_grid_csv(
        output_directory / "monthly_spatial_grid_counts.csv",
        grid_counts,
        grid_size_degrees,
    )

    summary = {
        "comparison_version": COMPARISON_VERSION,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "canonical_schema_version": SCHEMA_VERSION,
        "input_directory_label": input_directory.name,
        "output_directory_label": output_directory.name,
        "files_scanned": len(parquet_files),
        "rows_scanned": rows_scanned,
        "archive_rows": archive_rows,
        "archive_target_platform_rows": archive_target_platform_rows,
        "eligible_rows": eligible_rows,
        "rows_excluded_non_archive": rows_scanned - archive_rows,
        "rows_excluded_other_archive_platforms": (
            archive_rows - archive_target_platform_rows
        ),
        "rows_excluded_outside_period_on_target_archive_platforms": (
            archive_target_platform_rows - eligible_rows
        ),
        "period": {
            "start_year": start_year,
            "end_year": end_year,
            "full_calendar_year_window": True,
        },
        "filters": {
            "processing_stream": "ARCHIVE",
            "platforms": PLATFORM_SENSOR,
            "regions": sorted(REGIONS),
        },
        "platform_observations": {
            f"{sensor}_{platform}": count
            for (sensor, platform), count in sorted(sensor_platform_counts.items())
        },
        "spatial_grid": {
            "size_degrees": grid_size_degrees,
            "grouped_grid_rows_written": len(grid_counts),
            "alignment": "Global grid aligned to integer multiples of grid size from 0 degrees",
            "interpretation": (
                "Angular grid cells are not equal-area; the physical east-west "
                "cell width varies with latitude. Counts are source observations, "
                "not matched fire events."
            ),
        },
        "methodology": {
            "unit_of_count": "One canonical source observation record",
            "unique_fire_events_identified": False,
            "cross_sensor_matching_performed": False,
            "cross_platform_deduplication_performed": False,
            "within_platform_duplicate_audit_performed": False,
            "valid_observation_opportunity_denominator_available": False,
            "confidence_filter_applied": False,
            "frp_statistics": (
                "Negative FRP values are counted separately and excluded from "
                "nonnegative FRP summary statistics. Means and standard deviations "
                "are descriptive and may be affected by right-skew and outliers."
            ),
            "calendar_day_normalisation": (
                "Observations per calendar day divide the observation count by the "
                "number of calendar days in the month; this is not normalised by "
                "valid satellite observation opportunities. Active detection days "
                "are dates containing at least one detection, not valid-observation days."
            ),
            "observation_count_ratios": (
                "MODIS Terra/SNPP and MODIS Aqua/SNPP ratios are descriptive ratios "
                "of source-observation counts only. They are not detection-rate, "
                "fire-event, or sensor-bias correction factors. A blank ratio means "
                "there were no SNPP detections in that comparison group."
            ),
        },
        "limitations": [
            "The MODIS and VIIRS platforms have different spatial resolutions and detection algorithms.",
            "Terra, Aqua and SNPP have different overpass times and observation opportunities.",
            "Active-fire detections alone do not provide a denominator for valid satellite observation opportunities.",
            "A source observation is not necessarily a unique fire event.",
            "The 0.1-degree grid is exploratory and does not make sensor footprints equivalent.",
            "FRP percentiles are not calculated in this first comparison version.",
        ],
        "outputs": [
            "monthly_platform_counts.csv",
            "monthly_sensor_comparison.csv",
            "monthly_frp_summary.csv",
            "monthly_spatial_grid_counts.csv",
            "comparison_summary.json",
        ],
    }

    with (output_directory / "comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    print()
    print("SUTELLITE - SENSOR COMPARISON COMPLETE")
    print("=" * 58)
    print(f"Files scanned:                       {len(parquet_files):,}")
    print(f"Canonical rows scanned:              {rows_scanned:,}")
    print(f"ARCHIVE rows:                        {archive_rows:,}")
    print(f"Target-platform ARCHIVE rows:        {archive_target_platform_rows:,}")
    print(f"Eligible rows ({start_year}-{end_year}):              {eligible_rows:,}")
    print(f"Spatial grid groups written:         {len(grid_counts):,}")
    print(f"Output directory:                    {output_directory}")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MODIS Terra/Aqua and VIIRS SNPP using ARCHIVE canonical "
            "observations. Counts are source observations, not unique fire events."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Canonical Parquet directory (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First included calendar year (default: {DEFAULT_START_YEAR}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last included calendar year (default: {DEFAULT_END_YEAR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=DEFAULT_GRID_SIZE_DEGREES,
        help=(
            "Exploratory latitude/longitude grid size in degrees "
            f"(default: {DEFAULT_GRID_SIZE_DEGREES})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare_dataset(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        batch_size=args.batch_size,
        grid_size_degrees=args.grid_size_degrees,
    )


if __name__ == "__main__":
    main()
