
"""Prepare a Spain-only MODIS Aqua / VIIRS SNPP harmonisation dataset.

The experiment uses ARCHIVE observations, a shared monthly 0.1-degree grid,
and separate day/night groups. Counts represent source observations, not
unique fire events.

The outer join creates the union of cell-month groups with at least one
sensor detection. A missing count is filled with zero and accompanied by a
presence flag. This does not imply that the corresponding satellite had
observation coverage for that cell-month.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FEATURES_VERSION = "1.0"

DEFAULT_INPUT_DIR = Path("processed/canonical_v1.1/espana")
DEFAULT_OUTPUT_DIR = Path("processed/harmonization_spain")

DEFAULT_MODIS_FILE = DEFAULT_INPUT_DIR / "fire_archive_M-C61_838802.parquet"
DEFAULT_VIIRS_FILE = DEFAULT_INPUT_DIR / "fire_archive_SV-C2_838805.parquet"

DEFAULT_START_YEAR = 2013
DEFAULT_END_YEAR = 2025
DEFAULT_GRID_SIZE_DEGREES = 0.1

REGION_ID = "espana"
PROCESSING_STREAM = "ARCHIVE"

MODIS_SENSOR = "MODIS"
MODIS_PLATFORM = "AQUA"
VIIRS_SENSOR = "VIIRS"
VIIRS_PLATFORM = "SNPP"

REQUIRED_COLUMNS = (
    "observation_id",
    "region_id",
    "sensor",
    "platform",
    "processing_stream",
    "latitude_deg",
    "longitude_deg",
    "acq_date",
    "daynight",
)

KEY_COLUMNS = (
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "daynight",
)

OUTPUT_COLUMNS = (
    *KEY_COLUMNS,
    "modis_count",
    "viirs_count",
    "modis_present",
    "viirs_present",
)


def _validate_configuration(
    start_year: int,
    end_year: int,
    grid_size_degrees: float,
) -> None:
    """Validate the time range and spatial grid configuration."""

    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) cannot exceed end_year ({end_year})."
        )

    if not np.isfinite(grid_size_degrees) or grid_size_degrees <= 0:
        raise ValueError("grid_size_degrees must be a finite positive number.")

    if grid_size_degrees > 180:
        raise ValueError("grid_size_degrees cannot exceed 180 degrees.")


def _read_sensor_observations(
    file_path: Path,
    *,
    expected_sensor: str,
    expected_platform: str,
    start_year: int,
    end_year: int,
    grid_size_degrees: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate, filter and aggregate one sensor's observations."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Input Parquet file not found: {file_path}")

    parquet_file = pq.ParquetFile(file_path)
    available_columns = set(parquet_file.schema_arrow.names)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    table = pq.read_table(file_path, columns=list(REQUIRED_COLUMNS))
    observations = table.to_pandas()

    stats: dict[str, Any] = {
        "file": file_path.name,
        "rows_read": int(len(observations)),
    }

    region_mask = observations["region_id"].eq(REGION_ID)
    observations = observations.loc[region_mask].copy()
    stats["rows_region"] = int(len(observations))

    archive_mask = observations["processing_stream"].eq(PROCESSING_STREAM)
    observations = observations.loc[archive_mask].copy()
    stats["rows_archive"] = int(len(observations))

    platform_mask = (
        observations["sensor"].eq(expected_sensor)
        & observations["platform"].eq(expected_platform)
    )
    observations = observations.loc[platform_mask].copy()
    stats["rows_target_platform"] = int(len(observations))

    dates = pd.to_datetime(observations["acq_date"], errors="coerce")

    if dates.isna().any():
        invalid_count = int(dates.isna().sum())
        raise ValueError(
            f"{file_path.name} contains {invalid_count} invalid acquisition "
            "dates among target-platform ARCHIVE observations."
        )

    observations["year"] = dates.dt.year.astype("int16")
    observations["month"] = dates.dt.month.astype("int8")

    year_mask = observations["year"].between(start_year, end_year)
    observations = observations.loc[year_mask].copy()
    stats["rows_in_year_range"] = int(len(observations))

    if observations.empty:
        raise ValueError(
            f"No eligible {expected_platform} observations found in "
            f"{file_path.name} for {REGION_ID}, {PROCESSING_STREAM}, "
            f"and years {start_year}-{end_year}."
        )

    if observations["observation_id"].isna().any():
        raise ValueError(
            f"{file_path.name} contains null observation IDs in eligible rows."
        )

    duplicate_mask = observations["observation_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"{file_path.name} contains {duplicate_count} rows with "
            "duplicated observation IDs among eligible observations."
        )

    invalid_daynight = ~observations["daynight"].isin(("D", "N"))
    if invalid_daynight.any():
        values = sorted(
            str(value)
            for value in observations.loc[invalid_daynight, "daynight"].unique()
        )
        raise ValueError(
            f"{file_path.name} contains invalid daynight values: {values}. "
            "Only 'D' and 'N' are accepted."
        )

    for column, minimum, maximum in (
        ("latitude_deg", -90.0, 90.0),
        ("longitude_deg", -180.0, 180.0),
    ):
        numeric_values = pd.to_numeric(
            observations[column],
            errors="coerce",
        )

        values_array = numeric_values.to_numpy(dtype=np.float64)

        invalid_values = (
            ~np.isfinite(values_array)
            | (values_array < minimum)
            | (values_array > maximum)
        )

        if invalid_values.any():
            invalid_count = int(invalid_values.sum())
            raise ValueError(
                f"{file_path.name} contains {invalid_count} invalid "
                f"{column} values among eligible observations."
            )

        observations[column] = numeric_values.astype(np.float64)

    # Keep the exact globally aligned floor-based grid convention used by
    # src.compare_sensors. In particular, negative coordinates must use floor.
    observations["grid_lat_index"] = np.floor(
        observations["latitude_deg"] / grid_size_degrees
    ).astype(np.int32)

    observations["grid_lon_index"] = np.floor(
        observations["longitude_deg"] / grid_size_degrees
    ).astype(np.int32)

    grouped = (
        observations.groupby(
            list(KEY_COLUMNS),
            sort=True,
            observed=True,
        )
        .size()
        .rename("detection_count")
        .reset_index()
    )

    stats["eligible_observations"] = int(len(observations))
    stats["aggregated_cell_months"] = int(len(grouped))

    return grouped, stats


def prepare_harmonization_dataset(
    modis_file: Path = DEFAULT_MODIS_FILE,
    viirs_file: Path = DEFAULT_VIIRS_FILE,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the shared monthly grid dataset and an audit summary.

    A row exists when at least one of the sensors has detections for the
    corresponding cell, month and day/night period. Missing sensor counts
    are zero-filled, while presence flags retain the distinction between
    an observed detection group and an absent detection group.
    """

    _validate_configuration(
        start_year,
        end_year,
        grid_size_degrees,
    )

    modis_groups, modis_stats = _read_sensor_observations(
        modis_file,
        expected_sensor=MODIS_SENSOR,
        expected_platform=MODIS_PLATFORM,
        start_year=start_year,
        end_year=end_year,
        grid_size_degrees=grid_size_degrees,
    )

    viirs_groups, viirs_stats = _read_sensor_observations(
        viirs_file,
        expected_sensor=VIIRS_SENSOR,
        expected_platform=VIIRS_PLATFORM,
        start_year=start_year,
        end_year=end_year,
        grid_size_degrees=grid_size_degrees,
    )

    modis_groups = modis_groups.rename(
        columns={"detection_count": "modis_count"}
    )
    viirs_groups = viirs_groups.rename(
        columns={"detection_count": "viirs_count"}
    )

    merged = modis_groups.merge(
        viirs_groups,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
    )

    merged["modis_present"] = merged["modis_count"].notna()
    merged["viirs_present"] = merged["viirs_count"].notna()

    merged["modis_count"] = merged["modis_count"].fillna(0).astype("int64")
    merged["viirs_count"] = merged["viirs_count"].fillna(0).astype("int64")

    dataset = (
        merged.loc[:, list(OUTPUT_COLUMNS)]
        .sort_values(list(KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )

    if dataset.empty:
        raise ValueError("The harmonisation dataset is empty.")

    both_present = dataset["modis_present"] & dataset["viirs_present"]
    modis_only = dataset["modis_present"] & ~dataset["viirs_present"]
    viirs_only = ~dataset["modis_present"] & dataset["viirs_present"]

    summary: dict[str, Any] = {
        "features_version": FEATURES_VERSION,
        "region_id": REGION_ID,
        "processing_stream": PROCESSING_STREAM,
        "modis_sensor": MODIS_SENSOR,
        "modis_platform": MODIS_PLATFORM,
        "viirs_sensor": VIIRS_SENSOR,
        "viirs_platform": VIIRS_PLATFORM,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "temporal_unit": "calendar_month",
        "daynight_values": ["D", "N"],
        "grid_size_degrees": float(grid_size_degrees),
        "grid_definition": (
            "floor(latitude_deg / grid_size_degrees), "
            "floor(longitude_deg / grid_size_degrees)"
        ),
        "input": {
            "modis": modis_stats,
            "viirs": viirs_stats,
        },
        "dataset": {
            "rows": int(len(dataset)),
            "modis_detection_observations": int(dataset["modis_count"].sum()),
            "viirs_detection_observations": int(dataset["viirs_count"].sum()),
            "groups_with_both_sensors": int(both_present.sum()),
            "groups_with_modis_only": int(modis_only.sum()),
            "groups_with_viirs_only": int(viirs_only.sum()),
            "groups_with_zero_filled_modis_count": int((~dataset["modis_present"]).sum()),
            "groups_with_zero_filled_viirs_count": int((~dataset["viirs_present"]).sum()),
        },
        "interpretation": {
            "count_unit": "source_observation",
            "population": "union_of_sensor_active_cell_month_groups",
            "missing_count": (
                "No detection record for that sensor and group; "
                "not proof of no fire or satellite observation coverage."
            ),
            "limitations": [
                "The dataset does not contain satellite observation opportunity or coverage.",
                "Counts are detections, not unique fire events.",
                "The 0.1-degree grid is angular and not equal-area.",
                "MODIS and VIIRS confidence values are not combined.",
                "FRP is not used as a cross-sensor feature or target.",
            ],
        },
    }

    return dataset, summary


def export_harmonization_dataset(
    dataset: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Export the prepared dataset and its summary."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    parquet_path = output_directory / "spain_monthly_cell_counts.parquet"
    csv_path = output_directory / "spain_monthly_cell_counts.csv"
    summary_path = output_directory / "feature_summary.json"

    table = pa.Table.from_pandas(dataset, preserve_index=False)
    pq.write_table(table, parquet_path, compression="zstd")

    dataset.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)

    return {
        "parquet": parquet_path,
        "csv": csv_path,
        "summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare monthly 0.1-degree MODIS Aqua / VIIRS SNPP "
            "ARCHIVE detection counts for the Spain harmonisation pilot."
        )
    )

    parser.add_argument(
        "--modis-file",
        type=Path,
        default=DEFAULT_MODIS_FILE,
        help=f"MODIS canonical Parquet file (default: {DEFAULT_MODIS_FILE}).",
    )
    parser.add_argument(
        "--viirs-file",
        type=Path,
        default=DEFAULT_VIIRS_FILE,
        help=f"VIIRS canonical Parquet file (default: {DEFAULT_VIIRS_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
    )
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=DEFAULT_GRID_SIZE_DEGREES,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset, summary = prepare_harmonization_dataset(
        modis_file=args.modis_file,
        viirs_file=args.viirs_file,
        start_year=args.start_year,
        end_year=args.end_year,
        grid_size_degrees=args.grid_size_degrees,
    )

    output_paths = export_harmonization_dataset(
        dataset,
        summary,
        args.output_dir,
    )

    print("SPAIN HARMONISATION FEATURE PREPARATION")
    print(f"Dataset rows: {len(dataset):,}")
    print(
        "MODIS Aqua observations: "
        f"{summary['dataset']['modis_detection_observations']:,}"
    )
    print(
        "VIIRS SNPP observations: "
        f"{summary['dataset']['viirs_detection_observations']:,}"
    )
    print(
        "Groups with both sensors: "
        f"{summary['dataset']['groups_with_both_sensors']:,}"
    )
    print(
        "MODIS-only groups: "
        f"{summary['dataset']['groups_with_modis_only']:,}"
    )
    print(
        "VIIRS-only groups: "
        f"{summary['dataset']['groups_with_viirs_only']:,}"
    )

    for label, path in output_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
