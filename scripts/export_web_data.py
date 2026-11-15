"""Export Sutellite's existing calendar and alert data for the static atlas.

Run from the repository root with the demo's Python environment:
    python -m scripts.export_web_data
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from components.calendar_view import MONTH_LABELS, compute_monthly_statistics
from data_adapter import load_data


SUMMARY = Path("reports/sensor_comparison/monthly_platform_counts.csv")
MODIS_STATUS = {
    "Superficie quemada detectada": "burned",
    "Sin quemadura detectada": "unburned",
    "Sin datos MODIS validos": "no-data",
    "Sin datos MODIS válidos": "no-data",
}


def number_or_none(value):
    return value if math.isfinite(value) else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def calendar_data(source):
    observations = load_data(
        str(source / SUMMARY), sensor="VIIRS", platform="SNPP", processing_stream="ARCHIVE"
    )
    observations = observations[observations["date"].dt.year.between(2013, 2025)]
    return calendar_payload(observations)


def calendar_payload(observations, *, fixed_demo_reference=True, year_range=None):
    """Serialize the calendar view for either the bundled archive or an upload."""
    calendar = {}
    for region, region_data in observations.groupby("region"):
        monthly = (
            region_data.assign(date=region_data["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("date", as_index=False)["hotspot_count"].sum(min_count=1)
        )
        years = {}
        display_years = sorted(monthly["date"].dt.year.unique()) if year_range is None else year_range
        for year in display_years:
            _, statistics, total, peak, reference = compute_monthly_statistics(
                monthly, year, fixed_demo_reference=fixed_demo_reference
            )
            months = [
                {
                    "detections": number_or_none(row["Detections"]),
                    "referenceYears": row["Valid Reference Years"],
                    "median": number_or_none(row["Historical Median"]),
                    "mean": number_or_none(row["Historical Mean"]),
                    "minimum": number_or_none(row["Historical Min"]),
                    "maximum": number_or_none(row["Historical Max"]),
                    "difference": number_or_none(row["Difference vs Median (%)"]),
                }
                for row in statistics.to_dict("records")
            ]
            years[int(year)] = {
                "total": total,
                "peakMonth": MONTH_LABELS.index(peak) + 1 if peak else None,
                "reference": reference if reference != "None" else None,
                "months": months,
            }
        calendar[region] = years
    return calendar


def alert_data(source):
    with (source / "alertas.geojson").open(encoding="utf-8") as file:
        collection = json.load(file)

    alerts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    locations = set()
    total = 0
    for feature in collection["features"]:
        properties = feature["properties"]
        if properties["resultado"] != "Alerta de aumento":
            continue

        lon, lat = feature["geometry"]["coordinates"]
        region, year, month = (
            properties["region_id"],
            properties["year"],
            properties["month"],
        )
        if year not in (2024, 2025) or not 1 <= month <= 12:
            raise ValueError(f"Unexpected alert date {year}-{month} in {region}")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Invalid alert coordinates in {region}: {lat}, {lon}")

        key = (region, year, month, lat, lon)
        if key in locations:
            raise ValueError(f"Duplicate alert grid cell: {key}")
        locations.add(key)

        status = MODIS_STATUS[properties["validacion_modis"]]
        if (
            properties["coincide_modis"] != (status == "burned")
            or properties["modis_datos_validos"] != (status != "no-data")
        ):
            raise ValueError(f"Inconsistent MODIS flags for {key}")

        cover = properties["worldcover_2021_codigo"]
        alerts[region][year][month].append(
            {
                "lat": lat,
                "lon": lon,
                "detections": properties["detections"],
                "historicalMax": properties["maximo_historico"],
                "frp": properties["frp_total_mw"],
                "landCover": int(cover) if cover is not None else None,
                "modis": status,
                "burnedHectares": properties["hectareas_quemadas"],
            }
        )
        total += 1

    return alerts, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("web/data"))
    args = parser.parse_args()

    calendar = calendar_data(args.source)
    alerts, total = alert_data(args.source)
    if set(calendar) != set(alerts):
        raise ValueError("Calendar and alert files cover different regions")

    write_json(args.output / "calendar.json", calendar)
    for region, periods in alerts.items():
        write_json(args.output / "alerts" / f"{region}.json", periods)

    print(f"Exported {len(calendar)} regions and {total:,} alert points to {args.output}")


if __name__ == "__main__":
    main()
