
"""Build a reproducible source-file manifest for NASA FIRMS CSV data.

The manifest records file-level provenance, SHA-256 checksums,
CSV headers, record counts and source-contract metadata.

Raw source files are read only. The implementation uses streaming
I/O and does not load complete CSV files into memory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.audit_sources import discover_csv_files
from src.source_contracts import (
    identify_platform,
    identify_source_contract,
)
from src.validate_data import get_data_directory


MANIFEST_VERSION = "1.0"
HASH_CHUNK_SIZE = 1024 * 1024

CSV_EXPORT_FIELDS = (
    "source_snapshot_id",
    "source_file",
    "filename",
    "file_size_bytes",
    "sha256",
    "data_row_count",
    "row_count_complete",
    "malformed_row_count",
    "raw_header",
    "region_id",
    "sensor",
    "platform_group",
    "platforms_expected",
    "platforms_detected",
    "platform_counts",
    "unmapped_satellite_values",
    "processing_stream",
    "processing_provider",
    "source_collection",
    "source_product_token_raw",
    "source_id",
    "downloaded_at_utc",
    "contract_status",
    "header_status",
    "row_structure_status",
    "status",
    "errors",
)

class _HashingRawReader(io.RawIOBase):
    def __init__(self, raw_stream):
        super().__init__()
        self._raw_stream = raw_stream
        self._hasher = hashlib.sha256()
        self._bytes_hashed = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int | None:
        data = self._raw_stream.read(len(buffer))

        if not data:
            return 0

        self._hasher.update(data)
        buffer[:len(data)] = data
        self._bytes_hashed += len(data)

        return len(data)

    @property
    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    @property
    def bytes_hashed(self) -> int:
        return self._bytes_hashed

    def close(self) -> None:
        if not self.closed:
            try:
                self._raw_stream.close()
            finally:
                super().close()

def _utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _inspect_and_hash_csv(
    file_path: Path,
    expected_header: tuple[str, ...] | None,
) -> dict:
    """Read one CSV incrementally while calculating its checksum.

    Counts CSV data records rather than physical lines. This correctly
    handles quoted fields containing line breaks.

    The binary hashing layer ensures that the checksum represents
    the exact original file bytes, including its original line endings
    and any UTF-8 BOM.
    """

    file_stat_before = file_path.stat()
    file_size_bytes = file_stat_before.st_size

    errors = []
    header = []
    data_row_count = 0
    malformed_row_count = 0
    row_count_complete = True

    platform_counts: Counter[str] = Counter()
    unmapped_satellite_values: Counter[str] = Counter()

    hashing_reader = None
    buffered_reader = None
    text_stream = None

    try:
        with file_path.open("rb") as raw_stream:
            hashing_reader = _HashingRawReader(raw_stream)

            buffered_reader = io.BufferedReader(
                hashing_reader,
                buffer_size=HASH_CHUNK_SIZE,
            )

            text_stream = io.TextIOWrapper(
                buffered_reader,
                encoding="utf-8-sig",
                newline="",
            )

            try:
                reader = csv.reader(text_stream, strict=True)

                header = next(reader, None)

                if header is None:
                    errors.append("empty_csv_file")

                elif not header:
                    errors.append("empty_csv_header")

                else:
                    if len(header) != len(set(header)):
                        errors.append("duplicate_column_names")

                    if (
                        expected_header is not None
                        and tuple(header) != expected_header
                    ):
                        errors.append("header_mismatch")

                    satellite_index = (
                        header.index("satellite")
                        if "satellite" in header
                        else None
                    )

                    for row in reader:
                        data_row_count += 1

                        if len(row) != len(header):
                            malformed_row_count += 1

                        if (
                            satellite_index is None
                            or satellite_index >= len(row)
                        ):
                            continue

                        satellite_raw = row[satellite_index].strip()

                        if not satellite_raw:
                            unmapped_satellite_values["<MISSING>"] += 1
                            continue

                        try:
                            platform = identify_platform(satellite_raw)
                        except ValueError:
                            unmapped_satellite_values[
                                satellite_raw
                            ] += 1
                        else:
                            platform_counts[platform] += 1

            except (csv.Error, UnicodeError, OSError) as exc:
                row_count_complete = False
                errors.append(
                    f"csv_read_error:{type(exc).__name__}"
                )

            finally:
                # Drain unread bytes so the checksum still covers the
                # entire source file if CSV parsing stopped prematurely.
                try:
                    while buffered_reader.read(HASH_CHUNK_SIZE):
                        pass

                except (OSError, ValueError) as exc:
                    errors.append(
                        f"checksum_read_error:{type(exc).__name__}"
                    )

                try:
                    text_stream.close()
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"stream_close_error:{type(exc).__name__}"
                    )

    except OSError as exc:
        errors.append(f"file_read_error:{type(exc).__name__}")
        row_count_complete = False

    sha256 = None

    if hashing_reader is not None:
        if hashing_reader.bytes_hashed == file_size_bytes:
            sha256 = hashing_reader.hexdigest
        else:
            errors.append("incomplete_file_checksum")

    # Detect ordinary file modifications during manifest generation.
    try:
        file_stat_after = file_path.stat()

        if (
            file_stat_before.st_size != file_stat_after.st_size
            or file_stat_before.st_mtime_ns
            != file_stat_after.st_mtime_ns
        ):
            errors.append("source_file_changed_during_manifest_generation")
            sha256 = None

    except OSError:
        errors.append("source_file_stat_error")
        sha256 = None

    if malformed_row_count:
        errors.append(
            f"malformed_data_rows:{malformed_row_count}"
        )

    if not row_count_complete:
        row_structure_status = "INCOMPLETE"
    elif malformed_row_count:
        row_structure_status = "INVALID"
    else:
        row_structure_status = "VALID"

    if expected_header is None:
        header_status = "NOT_CHECKED"
    elif header and tuple(header) == expected_header:
        header_status = "VALID"
    else:
        header_status = "INVALID"

    return {
        "file_size_bytes": file_size_bytes,
        "sha256": sha256,
        "raw_header": header,
        "data_row_count": data_row_count,
        "row_count_complete": row_count_complete,
        "malformed_row_count": malformed_row_count,
        "platform_counts": dict(sorted(platform_counts.items())),
        "unmapped_satellite_values": dict(
            sorted(unmapped_satellite_values.items())
        ),
        "header_status": header_status,
        "row_structure_status": row_structure_status,
        "errors": errors,
    }


def _build_file_record(
    file_path: Path,
    data_directory: Path,
) -> dict:
    """Build the manifest entry for one source CSV."""

    # build_source_manifest() validates that every resolved input path is
    # inside data_directory before this function is called.
    relative_path = file_path.relative_to(data_directory).as_posix()

    errors = []
    contract = None

    try:
        contract = identify_source_contract(file_path)

    except ValueError as exc:
        errors.append(f"source_contract_invalid:{exc}")

    expected_header = (
        contract.expected_header
        if contract is not None
        else None
    )

    try:
        inspection = _inspect_and_hash_csv(
            file_path,
            expected_header,
        )

    except OSError as exc:
        inspection = {
            "file_size_bytes": None,
            "sha256": None,
            "raw_header": [],
            "data_row_count": 0,
            "row_count_complete": False,
            "malformed_row_count": 0,
            "platform_counts": {},
            "unmapped_satellite_values": {},
            "header_status": (
                "NOT_CHECKED"
                if expected_header is None
                else "INVALID"
            ),
            "row_structure_status": "INCOMPLETE",
            "errors": [
                f"file_inspection_error:{type(exc).__name__}"
            ],
        }

    errors.extend(inspection["errors"])

    contract_status = (
        "VALID" if contract is not None else "INVALID"
    )

    if errors or inspection["sha256"] is None:
        status = "INVALID"
    else:
        status = "VALID"

    record = {
        "source_snapshot_id": None,
        "source_file": relative_path,
        "filename": file_path.name,
        "downloaded_at_utc": None,
        "contract_status": contract_status,
        "status": status,
        "errors": errors,
    }

    record.update(inspection)

    if contract is not None:
        record.update(
            {
                "region_id": contract.region_id,
                "sensor": contract.sensor,
                "platform_group": contract.platform_group,
                "platforms_expected": list(
                    contract.expected_platforms
                ),
                "processing_stream": contract.processing_stream,
                "processing_provider": contract.processing_provider,
                "source_collection": contract.collection,
                "source_product_token_raw": contract.product_token,
                "source_id": contract.source_id,
            }
        )

    else:
        record.update(
            {
                "region_id": None,
                "sensor": None,
                "platform_group": None,
                "platforms_expected": [],
                "processing_stream": None,
                "processing_provider": None,
                "source_collection": None,
                "source_product_token_raw": None,
                "source_id": None,
            }
        )

    record["platforms_detected"] = sorted(
        record["platform_counts"].keys()
    )

    return record


def _calculate_snapshot_id(file_records: list[dict]) -> str | None:
    """Calculate a deterministic ID for the complete source snapshot."""

    if not file_records:
        return None

    if any(record["sha256"] is None for record in file_records):
        return None

    hasher = hashlib.sha256()
    hasher.update(b"SUTELLITE_SOURCE_SNAPSHOT_V1\n")

    for record in sorted(
        file_records,
        key=lambda item: item["source_file"],
    ):
        hasher.update(
            record["source_file"].encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(bytes.fromhex(record["sha256"]))
        hasher.update(b"\n")

    return f"sha256:{hasher.hexdigest()}"


def build_source_manifest(
    data_directory: Path,
    csv_files: list[Path] | None = None,
    generated_at_utc: str | None = None,
) -> dict:
    """Build a complete file-level manifest for a source dataset."""

    data_directory = Path(data_directory).resolve()

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_directory}"
        )

    if not data_directory.is_dir():
        raise NotADirectoryError(
            f"Data path is not a directory: {data_directory}"
        )

    if csv_files is None:
        csv_files = discover_csv_files(data_directory)

    # Resolve and validate every input path before inspecting any files.
    # This prevents paths outside the configured source root from entering
    # the manifest and avoids leaking absolute paths into source_file.
    resolved_csv_files = []
    seen_paths = set()

    for path in csv_files:
        resolved_path = Path(path).resolve()

        try:
            resolved_path.relative_to(data_directory)
        except ValueError as exc:
            raise ValueError(
                "Every source CSV must be located inside the data directory: "
                f"{resolved_path.name}"
            ) from exc

        if resolved_path in seen_paths:
            raise ValueError(
                f"Duplicate source CSV path supplied: {resolved_path.name}"
            )

        seen_paths.add(resolved_path)
        resolved_csv_files.append(resolved_path)

    csv_files = sorted(
        resolved_csv_files,
        key=lambda path: path.as_posix(),
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV source files were found."
        )

    file_records = [
        _build_file_record(file_path, data_directory)
        for file_path in csv_files
    ]

    snapshot_id = _calculate_snapshot_id(file_records)

    for record in file_records:
        record["source_snapshot_id"] = snapshot_id

    total_rows = sum(
        record["data_row_count"]
        for record in file_records
    )

    total_file_size = sum(
        record["file_size_bytes"] or 0
        for record in file_records
    )

    files_hashed = sum(
        record["sha256"] is not None
        for record in file_records
    )

    files_valid = sum(
        record["status"] == "VALID"
        for record in file_records
    )

    files_invalid = len(file_records) - files_valid

    all_row_counts_complete = all(
        record["row_count_complete"]
        for record in file_records
    )

    total_malformed_rows = sum(
        record["malformed_row_count"]
        for record in file_records
    )

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "source_root_name": data_directory.name,
        "hash_algorithm": "SHA-256",
        "source_snapshot_id": snapshot_id,
        # This describes checksum completeness for the supplied file list.
        # It does not prove that every expected source file was discovered,
        # nor that every CSV passed structural inspection.
        "snapshot_hash_complete": snapshot_id is not None,
        # Backward-compatible alias retained for existing consumers.
        # Its meaning is identical to snapshot_hash_complete.
        "snapshot_complete": snapshot_id is not None,
        "snapshot_completeness_basis": (
            "ALL_SUPPLIED_FILES_HAVE_SHA256"
        ),
        "summary": {
            "files_discovered": len(file_records),
            "files_hashed": files_hashed,
            "files_valid": files_valid,
            "files_invalid": files_invalid,
            "total_data_rows": total_rows,
            "total_file_size_bytes": total_file_size,
            "total_malformed_rows": total_malformed_rows,
            "all_row_counts_complete": all_row_counts_complete,
            "all_file_checksums_complete": (
                files_hashed == len(file_records)
            ),
        },
        "files": file_records,
    }


def export_source_manifest(
    manifest: dict,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Export the manifest as JSON and CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    json_path = (
        output_directory
        / f"source_manifest_{timestamp}.json"
    )

    csv_path = (
        output_directory
        / f"source_manifest_files_{timestamp}.csv"
    )

    json_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    nested_fields = {
        "raw_header",
        "platforms_expected",
        "platforms_detected",
        "platform_counts",
        "unmapped_satellite_values",
        "errors",
    }

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_EXPORT_FIELDS,
        )

        writer.writeheader()

        for record in manifest["files"]:
            row = {}

            for field in CSV_EXPORT_FIELDS:
                value = record.get(field)

                if field in nested_fields:
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                    )

                row[field] = value

            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    """Generate source manifests for the configured raw dataset."""

    data_directory = get_data_directory()

    csv_files = discover_csv_files(data_directory)

    print("SUTELLITE - SOURCE FILE MANIFEST")
    print("=" * 55)
    print(f"CSV files discovered: {len(csv_files):,}")
    print("Calculating checksums and inspecting CSV records...")
    print()

    manifest = build_source_manifest(
        data_directory=data_directory,
        csv_files=csv_files,
    )

    project_root = Path(__file__).resolve().parents[1]

    output_directory = (
        project_root
        / "reports"
        / "source_manifests"
    )

    json_path, csv_path = export_source_manifest(
        manifest,
        output_directory,
    )

    summary = manifest["summary"]

    print("SOURCE MANIFEST SUMMARY")
    print("=" * 55)
    print(f"Files discovered:          {summary['files_discovered']:,}")
    print(f"Files hashed:              {summary['files_hashed']:,}")
    print(f"Valid files:               {summary['files_valid']:,}")
    print(f"Invalid files:             {summary['files_invalid']:,}")
    print(f"Total data rows:           {summary['total_data_rows']:,}")
    print(f"Total file size:           {summary['total_file_size_bytes']:,} bytes")
    print(f"Malformed rows:            {summary['total_malformed_rows']:,}")
    print(f"Row counts complete:       {summary['all_row_counts_complete']}")
    print(f"Checksums complete:        {summary['all_file_checksums_complete']}")
    print(
        "Snapshot hash complete:   "
        f"{manifest['snapshot_hash_complete']}"
    )
    print(f"Source snapshot ID:        {manifest['source_snapshot_id']}")

    print()
    print(f"JSON manifest: {json_path}")
    print(f"CSV manifest:  {csv_path}")

    if (
        summary["files_invalid"] > 0
        or not summary["all_row_counts_complete"]
        or not summary["all_file_checksums_complete"]
    ):
        print()
        print("The manifest contains incomplete or invalid entries.")
        raise SystemExit(1)

    print()
    print("Source manifest generated successfully.")


if __name__ == "__main__":
    main()
