"""Stream uploaded NASA FIRMS CSV files into the Sutellite canonical Parquet schema.

This upload path reuses the repository's source contracts, row validation and
canonical observation mapper. It does not infer satellite identity from row
counts or convert missing sensor observations into zeroes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src.canonical_schema import CANONICAL_SCHEMA, SCHEMA_STATUS, SCHEMA_VERSION
from src.process_data import (
    QUALITY_FLAGS_SCHEMA,
    _map_observation,
    _observation_id,
    _write_arrow_rows,
)
from src.source_contracts import (
    FILENAME_PATTERN,
    SourceIdentityError,
    identify_source_contract,
    validate_source_identity,
)
from src.validate_data import get_row_quality_flags, validate_row

BUCKET_SUFFIX_BY_TOKEN = {
    "M-C61": "_modis",
    "SV-C2": "_viirs_snpp",
    "J1V-C2": "_viirs_20",
    "J2V-C2": "_viirs_21",
}
BUILT_IN_REGIONS = {
    "angola": "Angola",
    "australia": "Australia",
    "brasil": "Brasil",
    "california": "California",
    "espana": "España",
    "indonesia": "Indonesia",
    "rdc": "República Democrática del Congo",
    "zambia": "Zambia",
}


def slugify_region(name: str) -> str:
    """Create a stable ASCII identifier for a user-defined region."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", name.replace("’", "'")).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        raise ValueError("Enter a region name containing letters or numbers.")
    if slug in BUILT_IN_REGIONS:
        slug = f"custom_{slug}"
    return f"custom_{slug}" if not slug.startswith("custom_") else slug


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_for_upload(region_id: str, filename: str):
    match = FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(
            f"Unsupported filename '{filename}'. Use NASA FIRMS names such as "
            "fire_archive_M-C61_123456.csv, fire_nrt_SV-C2_123456.csv, "
            "fire_archive_J1V-C2_123456.csv or fire_nrt_J2V-C2_123456.csv."
        )
    _, token, _ = match.groups()
    suffix = BUCKET_SUFFIX_BY_TOKEN[token.upper()]
    # The established source-contract resolver accepts the eight built-in
    # region slugs. Resolve through a built-in placeholder, then set the chosen
    # upload region explicitly (custom regions are handled by the upload UI).
    resolver_region = region_id if region_id in BUILT_IN_REGIONS else "espana"
    contract = identify_source_contract(Path(f"{resolver_region}{suffix}") / Path(filename).name)
    return replace(contract, region_id=region_id)


def load_geojson_boundary(raw_geojson: bytes):
    """Parse and validate a WGS84 GeoJSON Polygon/MultiPolygon boundary."""
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - dependency installed by app requirements
        raise RuntimeError("Custom-region clipping requires Shapely. Install requirements.txt.") from exc

    try:
        payload = json.loads(raw_geojson.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The region boundary is not valid UTF-8 GeoJSON.") from exc

    kind = payload.get("type")
    if kind == "FeatureCollection":
        features = payload.get("features", [])
        geometries = [shape(f["geometry"]) for f in features if f.get("geometry")]
        geometry = unary_union(geometries) if geometries else None
    elif kind == "Feature":
        geometry = shape(payload["geometry"]) if payload.get("geometry") else None
    else:
        geometry = shape(payload)

    if geometry is None or geometry.is_empty:
        raise ValueError("GeoJSON does not contain a polygon boundary.")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Custom region GeoJSON must contain a Polygon or MultiPolygon.")
    if not geometry.is_valid:
        raise ValueError("Custom region GeoJSON has invalid polygon geometry.")
    min_x, min_y, max_x, max_y = geometry.bounds
    if min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90:
        raise ValueError("GeoJSON coordinates must be longitude/latitude in EPSG:4326.")
    return geometry


def process_uploaded_firms_files(
    source_files: Iterable[Path],
    *,
    region_id: str,
    region_name: str,
    output_dir: Path,
    boundary_geojson: bytes | None = None,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    """Validate and stream one or more raw FIRMS CSVs to canonical Parquet.

    Uploaded files must use the product/stream filename convention used by
    NASA FIRMS. For built-in regions, files are assumed to have already been
    spatially filtered to that region. Custom regions require GeoJSON and are
    clipped to the supplied boundary.
    """
    files = [Path(p).resolve() for p in source_files]
    if not files:
        raise ValueError("Upload at least one NASA FIRMS CSV file.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if len({p.name.casefold() for p in files}) != len(files):
        raise ValueError("Uploaded filenames must be unique.")
    for path in files:
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {path.name}")

    is_custom = region_id not in BUILT_IN_REGIONS
    boundary = load_geojson_boundary(boundary_geojson) if is_custom and boundary_geojson else None
    if is_custom and boundary is None:
        raise ValueError("Upload a GeoJSON Polygon/MultiPolygon boundary for a custom region.")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    canonical_path = staging / "harmonized_observations.parquet"
    quality_path = staging / "quality_flags.parquet"
    rejected_path = staging / "rejected_records.jsonl"
    source_results: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    errors_total: Counter[str] = Counter()
    quality_total: Counter[str] = Counter()
    platform_total: Counter[str] = Counter()
    sensor_total: Counter[str] = Counter()
    written_batch: list[dict[str, Any]] = []
    quality_batch: list[dict[str, Any]] = []
    writer = pq.ParquetWriter(canonical_path, CANONICAL_SCHEMA, compression="zstd")
    quality_writer = pq.ParquetWriter(quality_path, QUALITY_FLAGS_SCHEMA, compression="zstd")
    writer_closed = False
    quality_writer_closed = False

    try:
        with rejected_path.open("w", encoding="utf-8") as rejected_stream:
            for path in files:
                contract = _contract_for_upload(region_id, path.name)
                expected_header = contract.expected_header
                file_sha256 = _sha256_file(path)
                relative_source = f"uploads/{path.name}"
                file_counts: Counter[str] = Counter()
                file_errors: Counter[str] = Counter()
                file_flags: Counter[str] = Counter()

                with path.open("r", encoding="utf-8-sig", newline="") as csv_stream:
                    reader = csv.DictReader(csv_stream, strict=True)
                    actual_header = tuple(reader.fieldnames or ())
                    if actual_header != expected_header:
                        raise ValueError(
                            f"Header mismatch in {path.name}. Expected {expected_header}; "
                            f"received {actual_header}. Confirm this is a raw FIRMS ARCHIVE/NRT export."
                        )

                    for row_index, row in enumerate(reader, start=1):
                        file_counts["source_rows"] += 1
                        totals["source_rows"] += 1
                        flags = get_row_quality_flags(row)
                        if flags:
                            file_counts["quality_flagged_rows"] += 1
                            totals["quality_flagged_rows"] += 1
                            file_flags.update(flags)
                            quality_total.update(flags)

                        row_errors: list[str] = []
                        if None in row or any(value is None for value in row.values()):
                            row_errors.append("malformed_csv_row")
                        else:
                            row_errors.extend(validate_row(row, contract.sensor))
                            raw_version = row.get("version")
                            if raw_version is None or not raw_version.strip():
                                row_errors.append("version:missing")

                        platform = None
                        if not row_errors:
                            try:
                                platform = validate_source_identity(
                                    contract, row.get("satellite"), row.get("instrument")
                                )
                            except SourceIdentityError as exc:
                                row_errors.append(f"source_identity:{exc.code}")

                        quality_status = "REJECTED" if row_errors else "VALID"
                        written = None
                        if row_errors:
                            file_counts["rejected_rows"] += 1
                            totals["rejected_rows"] += 1
                            file_errors.update(row_errors)
                            errors_total.update(row_errors)
                            rejected_stream.write(json.dumps({
                                "source_file": relative_source,
                                "source_record_index": row_index,
                                "errors": row_errors,
                                "raw_values": row,
                            }, ensure_ascii=False) + "\n")
                        else:
                            outside_boundary = False
                            if boundary is not None:
                                from shapely.geometry import Point
                                point = Point(float(row["longitude"]), float(row["latitude"]))
                                outside_boundary = not boundary.covers(point)
                            if outside_boundary:
                                file_counts["outside_custom_boundary"] += 1
                                totals["outside_custom_boundary"] += 1
                                quality_status = "OUTSIDE_CUSTOM_BOUNDARY"
                            else:
                                written = _map_observation(
                                    row=row,
                                    contract=contract,
                                    platform=platform,
                                    relative_source_path=relative_source,
                                    source_record_index=row_index,
                                    source_sha256=file_sha256,
                                )
                                written_batch.append(written)
                                platform_total[platform] += 1
                                sensor_total[contract.sensor] += 1
                                file_counts["written_rows"] += 1
                                totals["written_rows"] += 1

                        if flags:
                            quality_batch.append({
                                "observation_id": (
                                    written["observation_id"] if written is not None else _observation_id(
                                        file_sha256, relative_source, row_index
                                    )
                                ),
                                "source_file": relative_source,
                                "source_record_index": row_index,
                                "record_status": quality_status,
                                "flags": flags,
                            })

                        if len(written_batch) >= batch_size:
                            _write_arrow_rows(writer, written_batch, CANONICAL_SCHEMA, batch_size)
                        if len(quality_batch) >= batch_size:
                            _write_arrow_rows(quality_writer, quality_batch, QUALITY_FLAGS_SCHEMA, batch_size)

                source_results.append({
                    "source_file": path.name,
                    "sha256": file_sha256,
                    "sensor": contract.sensor,
                    "platform_group": contract.platform_group,
                    "processing_stream": contract.processing_stream,
                    **{key: int(file_counts[key]) for key in (
                        "source_rows", "written_rows", "rejected_rows",
                        "quality_flagged_rows", "outside_custom_boundary"
                    )},
                    "errors": dict(sorted(file_errors.items())),
                    "quality_flags": dict(sorted(file_flags.items())),
                })

            _write_arrow_rows(writer, written_batch, CANONICAL_SCHEMA, batch_size)
            _write_arrow_rows(quality_writer, quality_batch, QUALITY_FLAGS_SCHEMA, batch_size)
            if totals["written_rows"] == 0:
                writer.write_table(pa.Table.from_pylist([], schema=CANONICAL_SCHEMA))
            if totals["quality_flagged_rows"] == 0:
                quality_writer.write_table(pa.Table.from_pylist([], schema=QUALITY_FLAGS_SCHEMA))

        writer.close()
        writer_closed = True
        quality_writer.close()
        quality_writer_closed = True

        report = {
            "application": "Sutellite",
            "processing_type": "user_upload",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "canonical_schema_version": SCHEMA_VERSION,
            "canonical_schema_status": SCHEMA_STATUS,
            "region_id": region_id,
            "region_name": region_name,
            "boundary_mode": "geojson_clipped" if boundary is not None else "pre_filtered_upload_assumed",
            "boundary_crs": "EPSG:4326" if boundary is not None else None,
            "boundary_geojson_sha256": hashlib.sha256(boundary_geojson).hexdigest() if boundary_geojson else None,
            "summary": {
                "files_uploaded": len(files),
                **{key: int(totals[key]) for key in (
                    "source_rows", "written_rows", "rejected_rows",
                    "quality_flagged_rows", "outside_custom_boundary"
                )},
                "errors": dict(sorted(errors_total.items())),
                "quality_flags": dict(sorted(quality_total.items())),
                "platform_counts": dict(sorted(platform_total.items())),
                "sensor_counts": dict(sorted(sensor_total.items())),
            },
            "files": source_results,
            "limitations": [
                "Rows are FIRMS detection observations, not unique fire events.",
                "No sensor observation-coverage data are inferred from detections.",
                "MODIS and VIIRS confidence systems and nominal resolutions remain sensor-specific.",
                "Negative FRP observations are retained and quality-flagged; non-negative FRP summaries exclude them.",
            ],
        }
        (staging / "processing_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        region_metadata = {
            "region_id": region_id,
            "region_name": region_name,
            "boundary_mode": report["boundary_mode"],
            "boundary_crs": report["boundary_crs"],
            "boundary_geojson_sha256": report["boundary_geojson_sha256"],
        }
        (staging / "region.json").write_text(
            json.dumps(region_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if boundary_geojson is not None:
            (staging / "region_boundary.geojson").write_bytes(boundary_geojson)
        staging.replace(output_dir)
        return {**report, "output_dir": str(output_dir)}
    except Exception:
        if not writer_closed:
            writer.close()
        if not quality_writer_closed:
            quality_writer.close()
        raise
