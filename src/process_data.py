"""Convert raw NASA FIRMS CSV observations into canonical Parquet files.

The processor reads one source CSV at a time and buffers only a bounded
number of observations. Raw files are never modified. Rows that fail the
implemented validation rules are written to a JSONL rejection sidecar rather
than silently discarded. Non-fatal quality flags are stored separately and
linked to observations by their deterministic observation_id.

Run from the repository root:
    python -m src.process_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.canonical_schema import (
    CANONICAL_SCHEMA,
    CONFIDENCE_SYSTEM_BY_SENSOR,
    DETECTION_TYPE_LABELS,
    NOMINAL_RESOLUTION_M_BY_SENSOR,
    SCHEMA_STATUS,
    SCHEMA_VERSION,
)
from src.source_contracts import (
    SourceIdentityError,
    identify_source_contract,
    validate_source_identity,
)
from src.validate_data import (
    get_data_directory,
    get_row_quality_flags,
    validate_row,
)


DEFAULT_BATCH_SIZE = 10_000
HASH_BUFFER_SIZE = 1024 * 1024
DISTRIBUTION_SERVICE = "NASA_FIRMS"

QUALITY_FLAGS_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_record_index", pa.uint64(), nullable=False),
        pa.field("record_status", pa.string(), nullable=False),
        pa.field("flags", pa.list_(pa.string()), nullable=False),
    ]
)

PROCESSING_FILE_FIELDS = (
    "source_file",
    "region_id",
    "sensor",
    "platform_group",
    "processing_stream",
    "source_rows",
    "written_rows",
    "rejected_rows",
    "quality_flagged_rows",
    "sha256",
    "errors",
    "quality_flags",
    "source_identity_errors",
)


class _HashingRawReader(io.RawIOBase):
    """Hash exact source bytes while exposing a streaming binary reader."""

    def __init__(self, raw_stream: Any) -> None:
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
        buffer[: len(data)] = data
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _observation_id(
    source_sha256: str,
    relative_source_path: str,
    source_record_index: int,
) -> str:
    """Create a deterministic ID from source checksum, path and row index."""

    payload = (
        "SUTELLITE_OBSERVATION_V1\0"
        f"{source_sha256}\0"
        f"{relative_source_path}\0"
        f"{source_record_index}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value.strip())


def _load_source_manifest(
    project_root: Path,
    manifest_path: Path | None,
) -> tuple[dict, dict[str, dict], Path]:
    """Load the latest source manifest and verify its inventory metadata."""

    if manifest_path is None:
        manifest_directory = project_root / "reports" / "source_manifests"
        candidates = sorted(manifest_directory.glob("source_manifest_*.json"))
        if not candidates:
            raise FileNotFoundError(
                "No source manifest was found in reports/source_manifests. "
                "Run `python -m src.source_manifest` first."
            )
        manifest_path = candidates[-1]

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    summary = manifest.get("summary", {})
    if not manifest.get("source_snapshot_id"):
        raise ValueError("The source manifest has no complete snapshot ID.")
    if not manifest.get("snapshot_complete"):
        raise ValueError("The source manifest snapshot is not complete.")
    if not summary.get("all_row_counts_complete"):
        raise ValueError("The source manifest contains incomplete row counts.")
    if not summary.get("all_file_checksums_complete"):
        raise ValueError("The source manifest contains incomplete checksums.")

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("The source manifest contains no file records.")

    records_by_path: dict[str, dict] = {}
    for record in records:
        relative_path = record.get("source_file")
        checksum = record.get("sha256")
        if not relative_path or not checksum:
            raise ValueError("A source manifest record lacks its path or SHA-256.")
        if relative_path in records_by_path:
            raise ValueError(f"Duplicate source path in manifest: {relative_path}")
        if record.get("status") != "VALID":
            raise ValueError(
                f"Source file is not marked VALID in the manifest: {relative_path}"
            )
        if not record.get("row_count_complete"):
            raise ValueError(
                f"Source row count is incomplete in the manifest: {relative_path}"
            )
        records_by_path[relative_path] = record

    return manifest, records_by_path, manifest_path


def _map_observation(
    row: dict[str, str],
    contract: Any,
    platform: str,
    relative_source_path: str,
    source_record_index: int,
    source_sha256: str,
) -> dict[str, Any]:
    """Map one validated raw FIRMS row to the canonical Arrow schema."""

    sensor = contract.sensor
    acq_date = date.fromisoformat(row["acq_date"].strip())
    acq_time_raw = row["acq_time"]
    acq_time_hhmm = acq_time_raw.strip().zfill(4)
    hour = int(acq_time_hhmm[:2])
    minute = int(acq_time_hhmm[2:])
    acq_datetime = datetime.combine(
        acq_date,
        time(hour=hour, minute=minute),
        tzinfo=timezone.utc,
    )

    brightness = _float_or_none(row.get("brightness"))
    bright_t31 = _float_or_none(row.get("bright_t31"))

    modis_bt21_22 = brightness if sensor == "MODIS" else None
    modis_bt31 = bright_t31 if sensor == "MODIS" else None
    viirs_bt_i4 = brightness if sensor == "VIIRS" else None
    viirs_bt_i5 = bright_t31 if sensor == "VIIRS" else None

    confidence_raw = row.get("confidence")
    confidence_modis = None
    confidence_viirs = None
    if confidence_raw is not None and confidence_raw.strip():
        if sensor == "MODIS":
            confidence_modis = float(confidence_raw.strip())
        elif sensor == "VIIRS":
            confidence_viirs = confidence_raw.strip().lower()

    detection_type_code = None
    detection_type_label = None
    if contract.processing_stream == "ARCHIVE":
        detection_type_code = int(row["type"].strip())
        detection_type_label = DETECTION_TYPE_LABELS[detection_type_code]

    return {
        "observation_id": _observation_id(
            source_sha256,
            relative_source_path,
            source_record_index,
        ),
        "region_id": contract.region_id,
        "sensor": sensor,
        "platform": platform,
        "satellite_raw": row["satellite"],
        "instrument_raw": row["instrument"],
        "processing_stream": contract.processing_stream,
        "distribution_service": DISTRIBUTION_SERVICE,
        "processing_provider": contract.processing_provider,
        "source_collection": contract.collection,
        "source_product_token_raw": contract.product_token,
        "source_file": relative_source_path,
        "source_record_index": source_record_index,
        "firms_version_raw": row["version"],
        "latitude_deg": float(row["latitude"].strip()),
        "longitude_deg": float(row["longitude"].strip()),
        "acq_date": acq_date,
        "acq_time_raw": acq_time_raw,
        "acq_time_hhmm_utc": acq_time_hhmm,
        "acq_datetime_utc": acq_datetime,
        "modis_bt21_22_k": modis_bt21_22,
        "modis_bt31_k": modis_bt31,
        "viirs_bt_i4_k": viirs_bt_i4,
        "viirs_bt_i5_k": viirs_bt_i5,
        "frp_mw": _float_or_none(row.get("frp")),
        "scan_km": float(row["scan"].strip()),
        "track_km": float(row["track"].strip()),
        "confidence_raw": confidence_raw,
        "confidence_system": CONFIDENCE_SYSTEM_BY_SENSOR[sensor],
        "confidence_modis_pct": confidence_modis,
        "confidence_viirs_class": confidence_viirs,
        "detection_type_code": detection_type_code,
        "detection_type_label": detection_type_label,
        "daynight": row["daynight"].strip().upper(),
        "nominal_resolution_m": NOMINAL_RESOLUTION_M_BY_SENSOR[sensor],
    }


def _write_arrow_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    batch_size: int,
) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table, row_group_size=batch_size)
    rows.clear()


def _process_source_file(
    file_path: Path,
    data_directory: Path,
    manifest_record: dict,
    output_directory: Path,
    quality_writer: pq.ParquetWriter,
    rejected_file: Any,
    batch_size: int,
    global_errors: Counter,
    global_quality_flags: Counter,
    global_identity_errors: Counter,
) -> dict:
    """Stream, validate, map and write one source CSV."""

    contract = identify_source_contract(file_path)
    relative_source_path = file_path.resolve().relative_to(
        data_directory.resolve()
    ).as_posix()

    expected_size = manifest_record.get("file_size_bytes")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"Source file size differs from manifest: {relative_source_path}")

    expected_sha256 = manifest_record["sha256"]
    expected_rows = manifest_record["data_row_count"]

    output_file = output_directory / contract.region_id / f"{file_path.stem}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_errors: Counter = Counter()
    file_quality_flags: Counter = Counter()
    file_source_identity_errors: Counter = Counter()
    source_rows = 0
    written_rows = 0
    rejected_rows = 0
    quality_flagged_rows = 0
    canonical_batch: list[dict[str, Any]] = []
    quality_batch: list[dict[str, Any]] = []

    with file_path.open("rb") as raw_stream:
        hashing_reader = _HashingRawReader(raw_stream)
        buffered_reader = io.BufferedReader(
            hashing_reader,
            buffer_size=HASH_BUFFER_SIZE,
        )
        text_stream = io.TextIOWrapper(
            buffered_reader,
            encoding="utf-8-sig",
            newline="",
        )

        try:
            csv_reader = csv.reader(text_stream, strict=True)
            header = next(csv_reader, None)
            if header is None or tuple(header) != contract.expected_header:
                raise ValueError(
                    f"Source header differs from its contract: {relative_source_path}"
                )

            with pq.ParquetWriter(
                output_file,
                CANONICAL_SCHEMA,
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
            ) as canonical_writer:
                for values in csv_reader:
                    source_rows += 1
                    row_index = source_rows
                    row = dict(zip(header, values))
                    observation_id = _observation_id(
                        expected_sha256,
                        relative_source_path,
                        row_index,
                    )

                    quality_flags = get_row_quality_flags(row)
                    errors: list[str] = []
                    platform: str | None = None

                    if quality_flags:
                        quality_flagged_rows += 1
                        file_quality_flags.update(quality_flags)
                        global_quality_flags.update(quality_flags)

                    if len(values) != len(header):
                        errors.append("malformed_row")
                    else:
                        errors.extend(validate_row(row, contract.sensor))

                        raw_version = row.get("version")
                        if raw_version is None or not raw_version.strip():
                            errors.append("version:missing")

                        try:
                            platform = validate_source_identity(
                                contract,
                                row.get("satellite"),
                                row.get("instrument"),
                            )
                        except SourceIdentityError as exc:
                            errors.append("source_identity:invalid")
                            file_source_identity_errors[exc.code] += 1
                            global_identity_errors[exc.code] += 1

                    if quality_flags:
                        quality_batch.append(
                            {
                                "observation_id": observation_id,
                                "source_file": relative_source_path,
                                "source_record_index": row_index,
                                "record_status": "REJECTED" if errors else "VALID",
                                "flags": quality_flags,
                            }
                        )

                    if errors:
                        rejected_rows += 1
                        file_errors.update(errors)
                        global_errors.update(errors)
                        rejected_record = {
                            "observation_id": observation_id,
                            "source_file": relative_source_path,
                            "source_record_index": row_index,
                            "errors": errors,
                            "raw_header": header,
                            "raw_values": values,
                        }
                        rejected_file.write(
                            json.dumps(rejected_record, ensure_ascii=False) + "\n"
                        )
                    else:
                        if platform is None:
                            raise RuntimeError(
                                "A row passed validation without a canonical platform."
                            )
                        canonical_batch.append(
                            _map_observation(
                                row=row,
                                contract=contract,
                                platform=platform,
                                relative_source_path=relative_source_path,
                                source_record_index=row_index,
                                source_sha256=expected_sha256,
                            )
                        )
                        written_rows += 1

                    if len(canonical_batch) >= batch_size:
                        _write_arrow_rows(
                            canonical_writer,
                            canonical_batch,
                            CANONICAL_SCHEMA,
                            batch_size,
                        )

                    if len(quality_batch) >= batch_size:
                        _write_arrow_rows(
                            quality_writer,
                            quality_batch,
                            QUALITY_FLAGS_SCHEMA,
                            batch_size,
                        )

                _write_arrow_rows(
                    canonical_writer,
                    canonical_batch,
                    CANONICAL_SCHEMA,
                    batch_size,
                )
                if written_rows == 0:
                    canonical_writer.write_table(
                        pa.Table.from_pylist([], schema=CANONICAL_SCHEMA)
                    )

            _write_arrow_rows(
                quality_writer,
                quality_batch,
                QUALITY_FLAGS_SCHEMA,
                batch_size,
            )

        finally:
            text_stream.close()

        actual_sha256 = hashing_reader.hexdigest
        actual_bytes_hashed = hashing_reader.bytes_hashed

    if actual_bytes_hashed != file_path.stat().st_size:
        raise ValueError(f"Checksum did not cover the complete file: {relative_source_path}")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Source SHA-256 differs from manifest: {relative_source_path}")
    if source_rows != expected_rows:
        raise ValueError(
            f"Source row count differs from manifest for {relative_source_path}: "
            f"expected {expected_rows:,}, received {source_rows:,}"
        )

    return {
        "source_file": relative_source_path,
        "region_id": contract.region_id,
        "sensor": contract.sensor,
        "platform_group": contract.platform_group,
        "processing_stream": contract.processing_stream,
        "source_rows": source_rows,
        "written_rows": written_rows,
        "rejected_rows": rejected_rows,
        "quality_flagged_rows": quality_flagged_rows,
        "sha256": actual_sha256,
        "errors": dict(sorted(file_errors.items())),
        "quality_flags": dict(sorted(file_quality_flags.items())),
        "source_identity_errors": dict(sorted(file_source_identity_errors.items())),
    }


def _write_processing_reports(
    output_directory: Path,
    report: dict,
) -> tuple[Path, Path]:
    json_path = output_directory / "processing_report.json"
    csv_path = output_directory / "processing_files.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PROCESSING_FILE_FIELDS)
        writer.writeheader()

        for item in report["files"]:
            row = dict(item)
            row["errors"] = json.dumps(
                row["errors"], ensure_ascii=False, sort_keys=True
            )
            row["quality_flags"] = json.dumps(
                row["quality_flags"], ensure_ascii=False, sort_keys=True
            )
            row["source_identity_errors"] = json.dumps(
                row["source_identity_errors"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(row)

    return json_path, csv_path


def run_processing(
    data_directory: Path,
    project_root: Path,
    manifest_path: Path | None = None,
    output_directory: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Run the full conversion and return the completed dataset directory."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    data_directory = Path(data_directory).resolve()
    project_root = Path(project_root).resolve()
    if output_directory is None:
        output_directory = project_root / "processed" / f"canonical_v{SCHEMA_VERSION}"
    output_directory = Path(output_directory).resolve()

    if not data_directory.is_dir():
        raise NotADirectoryError(f"Raw data directory does not exist: {data_directory}")
    if output_directory == data_directory or data_directory in output_directory.parents:
        raise ValueError("Processed output must not be inside the raw data directory.")
    if output_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_directory}. "
            "Choose a new output directory or move the previous result first."
        )

    manifest, manifest_records, resolved_manifest_path = _load_source_manifest(
        project_root,
        manifest_path,
    )

    source_files = sorted(data_directory.rglob("*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No raw CSV files found in {data_directory}")

    discovered_relative_paths = {
        file_path.resolve().relative_to(data_directory).as_posix()
        for file_path in source_files
    }
    manifest_relative_paths = set(manifest_records)
    if discovered_relative_paths != manifest_relative_paths:
        missing_from_manifest = sorted(discovered_relative_paths - manifest_relative_paths)
        missing_from_disk = sorted(manifest_relative_paths - discovered_relative_paths)
        raise ValueError(
            "Raw CSV inventory differs from the source manifest. "
            f"Unlisted files: {missing_from_manifest[:5]}; "
            f"missing files: {missing_from_disk[:5]}"
        )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = output_directory.with_name(
        f".{output_directory.name}.staging-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    staging_directory.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now()
    total_source_rows = 0
    total_written_rows = 0
    total_rejected_rows = 0
    total_quality_flagged_rows = 0
    global_errors: Counter = Counter()
    global_quality_flags: Counter = Counter()
    global_identity_errors: Counter = Counter()
    file_results: list[dict] = []

    quality_path = staging_directory / "quality_flags.parquet"
    rejected_path = staging_directory / "rejected_records.jsonl"

    try:
        with pq.ParquetWriter(
            quality_path,
            QUALITY_FLAGS_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        ) as quality_writer, rejected_path.open("w", encoding="utf-8") as rejected_file:
            for index, file_path in enumerate(source_files, start=1):
                relative_path = file_path.resolve().relative_to(data_directory).as_posix()
                print(f"[{index}/{len(source_files)}] Processing {relative_path}")

                result = _process_source_file(
                    file_path=file_path,
                    data_directory=data_directory,
                    manifest_record=manifest_records[relative_path],
                    output_directory=staging_directory,
                    quality_writer=quality_writer,
                    rejected_file=rejected_file,
                    batch_size=batch_size,
                    global_errors=global_errors,
                    global_quality_flags=global_quality_flags,
                    global_identity_errors=global_identity_errors,
                )
                file_results.append(result)
                total_source_rows += result["source_rows"]
                total_written_rows += result["written_rows"]
                total_rejected_rows += result["rejected_rows"]
                total_quality_flagged_rows += result["quality_flagged_rows"]

            if total_quality_flagged_rows == 0:
                quality_writer.write_table(
                    pa.Table.from_pylist([], schema=QUALITY_FLAGS_SCHEMA)
                )

        expected_total_rows = manifest["summary"]["total_data_rows"]
        if total_source_rows != expected_total_rows:
            raise ValueError(
                "Processed input row count differs from the source manifest: "
                f"expected {expected_total_rows:,}, received {total_source_rows:,}"
            )
        if total_source_rows != total_written_rows + total_rejected_rows:
            raise RuntimeError("Source rows do not reconcile with written and rejected rows.")

        completed_at = _utc_now()
        report = {
            "processing_version": "1.0",
            "generated_at_utc": completed_at,
            "processing_started_at_utc": started_at,
            "processing_completed_at_utc": completed_at,
            "canonical_schema_name": "Sutellite Canonical Observation Schema",
            "canonical_schema_version": SCHEMA_VERSION,
            "canonical_schema_status": SCHEMA_STATUS,
            "source_snapshot_id": manifest["source_snapshot_id"],
            "source_manifest_file": resolved_manifest_path.name,
            "hash_algorithm": "SHA-256",
            "checksum_verification": "PASSED",
            "summary": {
                "files_processed": len(file_results),
                "source_rows": total_source_rows,
                "written_rows": total_written_rows,
                "rejected_rows": total_rejected_rows,
                "quality_flagged_rows": total_quality_flagged_rows,
                "errors": dict(sorted(global_errors.items())),
                "quality_flags": dict(sorted(global_quality_flags.items())),
                "source_identity_errors": dict(sorted(global_identity_errors.items())),
            },
            "files": sorted(file_results, key=lambda item: item["source_file"]),
            "outputs": {
                "canonical_observations": "<region>/<source_filename>.parquet",
                "quality_flags": "quality_flags.parquet",
                "rejected_records": "rejected_records.jsonl",
                "processing_report": "processing_report.json",
                "processing_files": "processing_files.csv",
            },
        }

        _write_processing_reports(staging_directory, report)
        staging_directory.replace(output_directory)

    except Exception:
        print(f"Processing failed. Partial output was retained at: {staging_directory}")
        raise

    print()
    print("SUTELLITE - CANONICAL PROCESSING COMPLETE")
    print("=" * 58)
    print(f"Files processed:          {len(file_results):,}")
    print(f"Source rows:              {total_source_rows:,}")
    print(f"Canonical rows written:   {total_written_rows:,}")
    print(f"Rejected rows:            {total_rejected_rows:,}")
    print(f"Quality-flagged rows:     {total_quality_flagged_rows:,}")
    print(f"Output directory:         {output_directory}")
    return output_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert validated NASA FIRMS CSV files into canonical Parquet."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Source manifest JSON. Defaults to the newest report in reports/source_manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to processed/canonical_v1.1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum observations held in memory per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_directory = get_data_directory()
    run_processing(
        data_directory=data_directory,
        project_root=project_root,
        manifest_path=args.manifest,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
