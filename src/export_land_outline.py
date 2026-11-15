"""Export a light world land outline (land.json) for the map page.

The map page cannot download base maps itself, so this script downloads the
Natural Earth land polygons (public domain) once and writes the outer rings to
a small JSON file that you load into the page with "Base map (land.json)".

Usage:
    python -m src.export_land_outline
    python -m src.export_land_outline --resolution 50m      # more detail
    python -m src.export_land_outline --source path/to/ne_110m_land.geojson
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_{resolution}_land.geojson"
)


def land_rings(geojson: dict[str, Any], decimals: int = 2) -> list[list[list[float]]]:
    """Outer rings of every Polygon/MultiPolygon as [[lon, lat], ...]."""

    if geojson.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") or {} for f in geojson.get("features", [])]
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry") or {}]
    else:
        geometries = [geojson]

    rings: list[list[list[float]]] = []
    for geometry in geometries:
        kind = geometry.get("type")
        coords = geometry.get("coordinates") or []
        polygons = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
        for polygon in polygons:
            if not polygon:
                continue
            outer = polygon[0]  # holes are ignored on purpose
            rings.append(
                [[round(float(p[0]), decimals), round(float(p[1]), decimals)] for p in outer]
            )
    if not rings:
        raise ValueError("No Polygon or MultiPolygon geometries found.")
    return rings


def read_source(source: str) -> dict[str, Any]:
    if source.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(source).read_text(encoding="utf-8"))


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export land.json for the map page.")
    p.add_argument("--resolution", choices=["110m", "50m"], default="110m")
    p.add_argument("--source", default=None, help="URL or local GeoJSON path.")
    p.add_argument("--output-file", type=Path, default=Path("reports/map/land.json"))
    a = p.parse_args(list(argv) if argv is not None else None)
    if a.output_file.exists():
        raise FileExistsError(f"{a.output_file} already exists.")

    source = a.source or URL_TEMPLATE.format(resolution=a.resolution)
    rings = land_rings(read_source(source))
    a.output_file.parent.mkdir(parents=True, exist_ok=True)
    a.output_file.write_text(
        json.dumps({"type": "land", "source": "Natural Earth", "rings": rings},
                   separators=(",", ":")),
        encoding="utf-8",
    )
    points = sum(len(r) for r in rings)
    print(f"Wrote {len(rings)} outlines ({points:,} points) to {a.output_file}")


if __name__ == "__main__":
    main()
