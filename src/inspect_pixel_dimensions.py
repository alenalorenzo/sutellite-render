import csv
import math
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv


FIELDS = ("scan", "track")


def identify_sensor(folder_name: str) -> str:
    if folder_name.endswith("_modis"):
        return "MODIS"

    if folder_name.endswith("_viirs_snpp"):
        return "VIIRS_SNPP"

    if folder_name.endswith("_viirs_20"):
        return "VIIRS_NOAA20"

    if folder_name.endswith("_viirs_21"):
        return "VIIRS_NOAA21"

    return "UNKNOWN"


def identify_product(filename: str) -> str:
    if filename.startswith("fire_archive"):
        return "ARCHIVE"

    if filename.startswith("fire_nrt"):
        return "NRT"

    return "UNKNOWN"


def new_stats():
    return {
        "files": 0,
        "rows": 0,
        "missing_header": {field: 0 for field in FIELDS},
        "missing": {field: 0 for field in FIELDS},
        "invalid": {field: 0 for field in FIELDS},
        "min": {field: None for field in FIELDS},
        "max": {field: None for field in FIELDS},
    }


def inspect_value(value, field, stats):
    if value is None or not value.strip():
        stats["missing"][field] += 1
        return

    try:
        number = float(value)
    except ValueError:
        stats["invalid"][field] += 1
        return

    if not math.isfinite(number):
        stats["invalid"][field] += 1
        return

    current_min = stats["min"][field]
    current_max = stats["max"][field]

    if current_min is None or number < current_min:
        stats["min"][field] = number

    if current_max is None or number > current_max:
        stats["max"][field] = number


def main():
    load_dotenv()

    data_dir = os.getenv("SUTELLITE_DATA_DIR")

    if not data_dir:
        raise SystemExit("ERROR: SUTELLITE_DATA_DIR is not configured.")

    raw_dir = Path(data_dir)

    if not raw_dir.is_dir():
        raise SystemExit("ERROR: Configured data directory does not exist.")

    stats_by_group = defaultdict(new_stats)
    files_seen = 0
    unknown_files = []

    print("SUTELLITE - PIXEL DIMENSION INSPECTION")
    print("=" * 75)
    print("Inspecting scan and track across all raw CSV files.")
    print("Raw files will not be modified.")
    print()

    for file_path in sorted(raw_dir.rglob("*.csv")):
        sensor = identify_sensor(file_path.parent.name)
        product = identify_product(file_path.name)

        if sensor == "UNKNOWN" or product == "UNKNOWN":
            unknown_files.append(file_path.name)
            continue

        stats = stats_by_group[(sensor, product)]
        stats["files"] += 1
        files_seen += 1

        print(f"Processing: {file_path.name}")

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            headers = reader.fieldnames or []

            for field in FIELDS:
                if field not in headers:
                    stats["missing_header"][field] += 1

            for row in reader:
                stats["rows"] += 1

                for field in FIELDS:
                    if field in headers:
                        inspect_value(row.get(field), field, stats)

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)
    print(f"Recognised files processed: {files_seen}")

    if unknown_files:
        print(f"Unrecognised CSV files: {len(unknown_files)}")
        for filename in unknown_files:
            print(f"  - {filename}")

    for (sensor, product), stats in sorted(stats_by_group.items()):
        print()
        print(f"{sensor} / {product}")
        print("-" * 75)
        print(f"Files: {stats['files']}")
        print(f"Rows:  {stats['rows']:,}")

        for field in FIELDS:
            print(f"\n  {field.upper()}")
            print(f"    Missing headers: {stats['missing_header'][field]}")
            print(f"    Missing values:  {stats['missing'][field]:,}")
            print(f"    Invalid values:  {stats['invalid'][field]:,}")
            print(f"    Minimum:         {stats['min'][field]}")
            print(f"    Maximum:         {stats['max'][field]}")

    print()
    print("=" * 75)
    print("INSPECTION COMPLETE")


if __name__ == "__main__":
    main()