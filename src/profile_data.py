from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA, SCHEMA_VERSION

# PyArrow exposes compute kernels dynamically; its typing stubs
# do not declare every available compute function.
pc: Any = _pc

PROFILE_VERSION = "1.0"
DEFAULT_INPUT_DIR = Path("processed/canonical_v1.1")
DEFAULT_OUTPUT_DIR = Path("reports/data_profiles")
DEFAULT_BATCH_SIZE = 65_536
EXPECTED_FILE_COUNT = 64
EXPECTED_ROW_COUNT = 231_359_744

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

GROUP_FIELDS = ["region_id", "sensor", "platform", "processing_stream"]
CATEGORICAL_FIELDS = [
    "daynight",
    "confidence_system",
    "confidence_viirs_class",
    "confidence_raw",
    "detection_type_code",
    "detection_type_label",
    "firms_version_raw",
    "distribution_service",
    "processing_provider",
    "source_collection",
]


def _scalar_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _update_count_groups(
    table: pa.Table,
    keys: list[str],
    counter: Counter,
) -> None:
    if table.num_rows == 0:
        return

    grouped = table.group_by(keys).aggregate([("observation_id", "count")])
    for row in grouped.to_pylist():
        key = tuple(row[field] for field in keys)
        counter[key] += int(row["observation_id_count"])


def _update_group_summary(table: pa.Table, groups: dict) -> None:
    grouped = table.group_by(GROUP_FIELDS).aggregate(
        [
            ("observation_id", "count"),
            ("acq_datetime_utc", "min"),
            ("acq_datetime_utc", "max"),
        ]
    )

    for row in grouped.to_pylist():
        key = tuple(row[field] for field in GROUP_FIELDS)
        state = groups.setdefault(
            key,
            {
                "observations": 0,
                "first_acquisition_utc": None,
                "last_acquisition_utc": None,
            },
        )
        state["observations"] += int(row["observation_id_count"])

        first_seen = row["acq_datetime_utc_min"]
        last_seen = row["acq_datetime_utc_max"]

        if first_seen is not None and (
            state["first_acquisition_utc"] is None
            or first_seen < state["first_acquisition_utc"]
        ):
            state["first_acquisition_utc"] = first_seen

        if last_seen is not None and (
            state["last_acquisition_utc"] is None
            or last_seen > state["last_acquisition_utc"]
        ):
            state["last_acquisition_utc"] = last_seen


def _update_spatial_bounds(table: pa.Table, bounds: dict) -> None:
    grouped = table.group_by(["region_id", "sensor", "platform"]).aggregate(
        [
            ("latitude_deg", "min"),
            ("latitude_deg", "max"),
            ("longitude_deg", "min"),
            ("longitude_deg", "max"),
        ]
    )

    for row in grouped.to_pylist():
        key = (row["region_id"], row["sensor"], row["platform"])
        state = bounds.setdefault(
            key,
            {
                "min_latitude_deg": None,
                "max_latitude_deg": None,
                "min_longitude_deg": None,
                "max_longitude_deg": None,
            },
        )

        source_fields = {
            "min_latitude_deg": "latitude_deg_min",
            "max_latitude_deg": "latitude_deg_max",
            "min_longitude_deg": "longitude_deg_min",
            "max_longitude_deg": "longitude_deg_max",
        }
        for output_field, source_field in source_fields.items():
            value = row[source_field]
            if value is None:
                continue
            if state[output_field] is None:
                state[output_field] = value
            elif output_field.startswith("min_"):
                state[output_field] = min(state[output_field], value)
            else:
                state[output_field] = max(state[output_field], value)


def _update_frp_profile(table: pa.Table, frp_status_counts: Counter, frp_stats: dict) -> None:
    frp = table["frp_mw"]

    is_missing = pc.is_null(frp)
    is_negative = pc.fill_null(pc.less(frp, 0), False)
    is_zero = pc.fill_null(pc.equal(frp, 0), False)
    is_positive = pc.fill_null(pc.greater(frp, 0), False)

    status = pc.if_else(
        is_missing,
        pa.scalar("missing"),
        pc.if_else(
            is_negative,
            pa.scalar("negative"),
            pc.if_else(is_zero, pa.scalar("zero"), pa.scalar("positive")),
        ),
    )

    status_table = table.append_column("_frp_status", status)
    grouped_status = status_table.group_by(
        ["sensor", "platform", "_frp_status"]
    ).aggregate([("observation_id", "count")])

    for row in grouped_status.to_pylist():
        key = (row["sensor"], row["platform"], row["_frp_status"])
        frp_status_counts[key] += int(row["observation_id_count"])

    nonnegative_mask = pc.and_(pc.is_valid(frp), pc.greater_equal(frp, 0))
    nonnegative_table = table.filter(nonnegative_mask)

    if nonnegative_table.num_rows == 0:
        return

    squared = pc.multiply(
        nonnegative_table["frp_mw"],
        nonnegative_table["frp_mw"],
    )
    nonnegative_table = nonnegative_table.append_column("_frp_squared", squared)

    grouped_stats = nonnegative_table.group_by(["sensor", "platform"]).aggregate(
        [
            ("frp_mw", "count"),
            ("frp_mw", "min"),
            ("frp_mw", "max"),
            ("frp_mw", "sum"),
            ("_frp_squared", "sum"),
        ]
    )

    for row in grouped_stats.to_pylist():
        key = (row["sensor"], row["platform"])
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _build_field_completeness(
    total_rows: int,
    sensor_rows: Counter,
    field_missing_by_sensor: Counter,
) -> list[dict]:
    rows: list[dict] = []
    sensors = sorted(sensor_rows)

    for field in CANONICAL_SCHEMA:
        missing_all = sum(
            field_missing_by_sensor[(sensor, field.name)]
            for sensor in sensors
        )
        scopes = [("ALL", total_rows, missing_all)]

        for sensor in sensors:
            scopes.append(
                (
                    sensor,
                    sensor_rows[sensor],
                    field_missing_by_sensor[(sensor, field.name)],
                )
            )

        for scope, row_count, missing in scopes:
            non_missing = row_count - missing
            completeness = (
                (non_missing / row_count) * 100.0 if row_count else 0.0
            )
            rows.append(
                {
                    "scope": scope,
                    "field": field.name,
                    "data_type": str(field.type),
                    "nullable": field.nullable,
                    "rows": row_count,
                    "non_missing": non_missing,
                    "missing": missing,
                    "completeness_pct": round(completeness, 6),
                }
            )

    return rows


def profile_dataset(
    input_directory: Path,
    output_directory: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if not input_directory.is_dir():
        raise SystemExit(f"Canonical input directory not found: {input_directory}")

    parquet_files = sorted(
        path
        for path in input_directory.rglob("*.parquet")
        if path.parent.name in REGIONS
    )
    if not parquet_files:
        raise SystemExit(f"No regional Parquet files found in: {input_directory}")

    group_summary: dict = {}
    spatial_bounds: dict = {}
    yearly_counts: Counter = Counter()
    monthly_counts: Counter = Counter()
    categorical_counts: Counter = Counter()
    frp_status_counts: Counter = Counter()
    frp_stats: dict = {}
    field_missing_by_sensor: Counter = Counter()
    sensor_rows: Counter = Counter()
    total_rows = 0

    schema_fields = [field.name for field in CANONICAL_SCHEMA]
    completeness_fields = [name for name in schema_fields if name != "sensor"]

    print("SUTELLITE - CANONICAL DATA PROFILING")
    print("=" * 70)
    print(f"Parquet files discovered: {len(parquet_files)}")
    print(f"Batch size: {batch_size:,}")
    print("Starting streaming profile...\n")

    for file_number, path in enumerate(parquet_files, start=1):
        print(f"[{file_number}/{len(parquet_files)}] Processing: {path.name}", flush=True)
        parquet_file = pq.ParquetFile(path)

        if not parquet_file.schema_arrow.equals(CANONICAL_SCHEMA, check_metadata=True):
            raise ValueError(
                f"Canonical schema mismatch in {path.relative_to(input_directory)}"
            )

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=schema_fields,
            use_threads=True,
        ):
            table = pa.Table.from_batches([batch])
            batch_rows = table.num_rows
            total_rows += batch_rows

            _update_group_summary(table, group_summary)
            _update_spatial_bounds(table, spatial_bounds)

            year_values = pc.year(table["acq_datetime_utc"])
            month_values = pc.month(table["acq_datetime_utc"])
            time_table = table.append_column("_year", year_values)
            time_table = time_table.append_column("_month", month_values)

            _update_count_groups(
                time_table,
                GROUP_FIELDS + ["_year"],
                yearly_counts,
            )
            _update_count_groups(
                time_table,
                GROUP_FIELDS + ["_year", "_month"],
                monthly_counts,
            )

            # One grouped aggregation per batch calculates completeness by sensor.
            completeness_aggregations = [
                (name, "count") for name in completeness_fields
            ]
            completeness_table = table.group_by(["sensor"]).aggregate(
                completeness_aggregations
            )

            for row in completeness_table.to_pylist():
                sensor = row["sensor"]
                rows_for_sensor = int(row["observation_id_count"])
                sensor_rows[sensor] += rows_for_sensor

                for field_name in completeness_fields:
                    non_missing = int(row[f"{field_name}_count"])
                    field_missing_by_sensor[(sensor, field_name)] += (
                        rows_for_sensor - non_missing
                    )

            # Categorical distributions are grouped in Arrow, not counted row by row.
            for field_name in CATEGORICAL_FIELDS:
                valid_table = table.filter(pc.is_valid(table[field_name]))
                if valid_table.num_rows == 0:
                    continue

                grouped = valid_table.group_by(
                    ["sensor", field_name]
                ).aggregate([("observation_id", "count")])

                for row in grouped.to_pylist():
                    key = (row["sensor"], field_name, str(row[field_name]))
                    categorical_counts[key] += int(row["observation_id_count"])

            _update_frp_profile(table, frp_status_counts, frp_stats)

    output_directory.mkdir(parents=True, exist_ok=True)

    group_rows = []
    sensor_counts: Counter = Counter()
    stream_counts: Counter = Counter()
    platform_counts: Counter = Counter()
    region_counts: Counter = Counter()
    global_first = None
    global_last = None

    for key, state in sorted(group_summary.items()):
        region, sensor, platform, stream = key
        first_seen = state["first_acquisition_utc"]
        last_seen = state["last_acquisition_utc"]

        group_rows.append(
            {
                "region_id": region,
                "sensor": sensor,
                "platform": platform,
                "processing_stream": stream,
                "observations": state["observations"],
                "first_acquisition_utc": _iso_utc(first_seen),
                "last_acquisition_utc": _iso_utc(last_seen),
            }
        )

        sensor_counts[sensor] += state["observations"]
        stream_counts[stream] += state["observations"]
        platform_counts[platform] += state["observations"]
        region_counts[region] += state["observations"]

        if first_seen is not None and (global_first is None or first_seen < global_first):
            global_first = first_seen
        if last_seen is not None and (global_last is None or last_seen > global_last):
            global_last = last_seen

    yearly_rows = [
        {
            "region_id": key[0],
            "sensor": key[1],
            "platform": key[2],
            "processing_stream": key[3],
            "year": key[4],
            "observations": count,
        }
        for key, count in sorted(yearly_counts.items())
    ]

    monthly_rows = [
        {
            "region_id": key[0],
            "sensor": key[1],
            "platform": key[2],
            "processing_stream": key[3],
            "year": key[4],
            "month": key[5],
            "observations": count,
        }
        for key, count in sorted(monthly_counts.items())
    ]

    completeness_rows = _build_field_completeness(
        total_rows,
        sensor_rows,
        field_missing_by_sensor,
    )

    categorical_rows = [
        {
            "sensor": key[0],
            "field": key[1],
            "value": key[2],
            "observations": count,
        }
        for key, count in sorted(categorical_counts.items())
    ]

    frp_rows = []
    frp_group_keys = sorted(
        set(frp_stats)
        | {
            (sensor, platform)
            for sensor, platform, _status in frp_status_counts
        }
    )

    for sensor, platform in frp_group_keys:
        stats = frp_stats.get(
            (sensor, platform),
            {"count": 0, "sum": 0.0, "sum_squared": 0.0, "min": None, "max": None},
        )
        count = stats["count"]
        mean = stats["sum"] / count if count else None
        variance = (
            max(0.0, (stats["sum_squared"] / count) - (mean * mean))
            if count and mean is not None
            else None
        )

        frp_rows.append(
            {
                "sensor": sensor,
                "platform": platform,
                "nonnegative_frp_count": count,
                "negative_frp_count": frp_status_counts[(sensor, platform, "negative")],
                "zero_frp_count": frp_status_counts[(sensor, platform, "zero")],
                "positive_frp_count": frp_status_counts[(sensor, platform, "positive")],
                "missing_frp_count": frp_status_counts[(sensor, platform, "missing")],
                "min_nonnegative_frp_mw": stats["min"],
                "mean_nonnegative_frp_mw": mean,
                "stddev_nonnegative_frp_mw": math.sqrt(variance) if variance is not None else None,
                "max_nonnegative_frp_mw": stats["max"],
            }
        )

    spatial_rows = [
        {
            "region_id": key[0],
            "sensor": key[1],
            "platform": key[2],
            **state,
        }
        for key, state in sorted(spatial_bounds.items())
    ]

    _write_csv(
        output_directory / "group_summary.csv",
        [
            "region_id", "sensor", "platform", "processing_stream",
            "observations", "first_acquisition_utc", "last_acquisition_utc",
        ],
        group_rows,
    )
    _write_csv(
        output_directory / "yearly_observations.csv",
        ["region_id", "sensor", "platform", "processing_stream", "year", "observations"],
        yearly_rows,
    )
    _write_csv(
        output_directory / "monthly_observations.csv",
        [
            "region_id", "sensor", "platform", "processing_stream",
            "year", "month", "observations",
        ],
        monthly_rows,
    )
    _write_csv(
        output_directory / "field_completeness.csv",
        [
            "scope", "field", "data_type", "nullable", "rows",
            "non_missing", "missing", "completeness_pct",
        ],
        completeness_rows,
    )
    _write_csv(
        output_directory / "categorical_distributions.csv",
        ["sensor", "field", "value", "observations"],
        categorical_rows,
    )
    _write_csv(
        output_directory / "frp_summary.csv",
        [
            "sensor", "platform", "nonnegative_frp_count", "negative_frp_count",
            "zero_frp_count", "positive_frp_count", "missing_frp_count",
            "min_nonnegative_frp_mw", "mean_nonnegative_frp_mw",
            "stddev_nonnegative_frp_mw", "max_nonnegative_frp_mw",
        ],
        frp_rows,
    )
    _write_csv(
        output_directory / "spatial_bounds.csv",
        [
            "region_id", "sensor", "platform", "min_latitude_deg",
            "max_latitude_deg", "min_longitude_deg", "max_longitude_deg",
        ],
        spatial_rows,
    )

    report = {
        "profile_version": PROFILE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "canonical_schema_version": SCHEMA_VERSION,
        "input_directory": input_directory.as_posix(),
        "files_scanned": len(parquet_files),
        "rows_scanned": total_rows,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "file_count_matches_expected": len(parquet_files) == EXPECTED_FILE_COUNT,
        "row_count_matches_expected": total_rows == EXPECTED_ROW_COUNT,
        "acquisition_date_range_utc": {
            "first": _iso_utc(global_first),
            "last": _iso_utc(global_last),
        },
        "counts": {
            "by_sensor": dict(sorted(sensor_counts.items())),
            "by_platform": dict(sorted(platform_counts.items())),
            "by_processing_stream": dict(sorted(stream_counts.items())),
            "by_region": dict(sorted(region_counts.items())),
        },
        "frp_negative_observations": sum(
            count for (sensor, platform, status), count in frp_status_counts.items()
            if status == "negative"
        ),
        "notes": [
            "Counts represent source observations, not unique fire events.",
            "FRP descriptive statistics exclude negative FRP values; negative values are reported separately.",
            "Spatial bounds are bounding boxes, not exact sensor footprints.",
            "ARCHIVE and NRT overlap and cross-sensor duplicate analysis are not performed by this profiler.",
            "FRP means and standard deviations are computed over nonnegative FRP observations.",
        ],
        "outputs": {
            "group_summary": "group_summary.csv",
            "yearly_observations": "yearly_observations.csv",
            "monthly_observations": "monthly_observations.csv",
            "field_completeness": "field_completeness.csv",
            "categorical_distributions": "categorical_distributions.csv",
            "frp_summary": "frp_summary.csv",
            "spatial_bounds": "spatial_bounds.csv",
        },
    }

    with (output_directory / "profile_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    print("\n" + "=" * 70)
    print("CANONICAL PROFILE COMPLETED")
    print("=" * 70)
    print(f"Files scanned: {len(parquet_files):,}")
    print(f"Observations scanned: {total_rows:,}")
    print(f"Date range: {_iso_utc(global_first)} to {_iso_utc(global_last)}")
    print(f"MODIS observations: {sensor_counts['MODIS']:,}")
    print(f"VIIRS observations: {sensor_counts['VIIRS']:,}")
    print(f"Negative FRP observations: {report['frp_negative_observations']:,}")
    print(f"Reports written to: {output_directory}")

    if len(parquet_files) != EXPECTED_FILE_COUNT:
        print("WARNING: discovered Parquet file count differs from expected count.")
    if total_rows != EXPECTED_ROW_COUNT:
        print("WARNING: profiled row count differs from the source inventory.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-profile Sutellite canonical Parquet observations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Canonical Parquet directory (default: processed/canonical_v1.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Report output directory (default: reports/data_profiles).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per Arrow batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    profile_dataset(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
