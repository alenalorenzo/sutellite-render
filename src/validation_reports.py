"""Build machine-readable reports for raw FIRMS data validation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_VERSION = "1.0"


def create_file_result(contract: Any, file_path: Path) -> dict:
    """Create an independent statistics record for one source file."""

    return {
        "file": file_path.name,
        "region_id": contract.region_id,
        "sensor": contract.sensor,
        "platform_group": contract.platform_group,
        "collection": contract.collection,
        "processing_stream": contract.processing_stream,
        "processing_provider": contract.processing_provider,
        "source_id": contract.source_id,
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "quality_flagged_rows": 0,
        "errors": Counter(),
        "quality_flags": Counter(),
        "source_identity_errors": Counter(),
    }


def _serialise_counter(counter: Counter) -> dict[str, int]:
    """Convert a Counter into a JSON-compatible dictionary."""

    return dict(sorted(counter.items()))


def _aggregate(
    file_results: list[dict],
    group_fields: tuple[str, ...],
) -> dict[str, dict]:
    """Aggregate file results by one or more metadata fields.

    Tuple keys are used internally so combinations cannot collide because
    of a separator appearing inside one of the field values. Single-field
    groups retain their existing readable keys (for example, ``AF``).
    Multi-field groups use a JSON-encoded tuple as their dictionary key.
    """

    groups = defaultdict(
        lambda: {
            "files": 0,
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "quality_flagged_rows": 0,
            "errors": Counter(),
            "quality_flags": Counter(),
        }
    )

    for result in file_results:
        key = tuple(result[field] for field in group_fields)
        group = groups[key]

        group["files"] += 1

        for field in (
            "total_rows",
            "valid_rows",
            "invalid_rows",
            "quality_flagged_rows",
        ):
            group[field] += result[field]

        group["errors"].update(result["errors"])
        group["quality_flags"].update(result["quality_flags"])

    output = {}

    sort_key = lambda item: tuple(
        (type(value).__name__, repr(value)) for value in item[0]
    )

    for key_values, group in sorted(groups.items(), key=sort_key):
        if len(key_values) == 1 and isinstance(key_values[0], str):
            output_key = key_values[0]
        else:
            output_key = json.dumps(
                key_values,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        output[output_key] = {
            "files": group["files"],
            "total_rows": group["total_rows"],
            "valid_rows": group["valid_rows"],
            "invalid_rows": group["invalid_rows"],
            "quality_flagged_rows": group["quality_flagged_rows"],
            "valid_percentage": (
                round(
                    group["valid_rows"] / group["total_rows"] * 100,
                    4,
                )
                if group["total_rows"]
                else None
            ),
            "errors": _serialise_counter(group["errors"]),
            "quality_flags": _serialise_counter(
                group["quality_flags"]
            ),
        }

    return output


def build_validation_report(
    file_results: list[dict],
    started_at: datetime,
    completed_at: datetime,
) -> dict:
    """Build the complete JSON-compatible validation report."""

    total_rows = sum(item["total_rows"] for item in file_results)
    valid_rows = sum(item["valid_rows"] for item in file_results)

    totals = {
        "files_processed": len(file_results),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": sum(
            item["invalid_rows"] for item in file_results
        ),
        "quality_flagged_rows": sum(
            item["quality_flagged_rows"] for item in file_results
        ),
        "valid_percentage": (
            round(valid_rows / total_rows * 100, 4)
            if total_rows > 0
            else None
        ),
    }

    all_errors = Counter()
    all_quality_flags = Counter()
    all_identity_errors = Counter()

    for item in file_results:
        all_errors.update(item["errors"])
        all_quality_flags.update(item["quality_flags"])
        all_identity_errors.update(item["source_identity_errors"])

    serialised_files = []

    for item in sorted(
        file_results,
        key=lambda result: (
            result["region_id"],
            result["sensor"],
            result["platform_group"],
            result["processing_stream"],
            result["file"],
        ),
    ):
        serialised_files.append(
            {
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {
                        "errors",
                        "quality_flags",
                        "source_identity_errors",
                    }
                },
                "errors": _serialise_counter(item["errors"]),
                "quality_flags": _serialise_counter(
                    item["quality_flags"]
                ),
                "source_identity_errors": _serialise_counter(
                    item["source_identity_errors"]
                ),
            }
        )

    return {
        "report_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_started_at_utc": started_at.isoformat(),
        "validation_completed_at_utc": completed_at.isoformat(),
        "summary": totals,
        "errors": _serialise_counter(all_errors),
        "quality_flags": _serialise_counter(all_quality_flags),
        "source_identity_errors": _serialise_counter(
            all_identity_errors
        ),
        "by_region": _aggregate(file_results, ("region_id",)),
        "by_sensor": _aggregate(file_results, ("sensor",)),
        "by_processing_stream": _aggregate(
            file_results,
            ("processing_stream",),
        ),
        "by_platform": _aggregate(
            file_results,
            ("platform_group",),
        ),
        "files": serialised_files,
    }


def export_validation_reports(
    report: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV validation reports without touching raw data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    json_path = output_dir / f"validation_report_{timestamp}.json"
    csv_path = output_dir / f"validation_files_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8", newline="") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    fieldnames = [
        "file",
        "region_id",
        "sensor",
        "platform_group",
        "collection",
        "processing_stream",
        "processing_provider",
        "source_id",
        "total_rows",
        "valid_rows",
        "invalid_rows",
        "quality_flagged_rows",
        "errors",
        "quality_flags",
        "source_identity_errors",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in report["files"]:
            row = dict(item)

            for field in (
                "errors",
                "quality_flags",
                "source_identity_errors",
            ):
                row[field] = json.dumps(
                    row[field],
                    ensure_ascii=False,
                    sort_keys=True,
                )

            writer.writerow(row)

    return json_path, csv_path
