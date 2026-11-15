"""Sutellite: globe-first MODIS/VIIRS active-fire exploration app."""

from __future__ import annotations

import calendar
import html
import json
import math
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

from src.cell_aggregation import build_cell_products, canonical_file_fingerprint
from src.upload_pipeline import BUILT_IN_REGIONS, process_uploaded_firms_files, slugify_region

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "processed" / "canonical_v1.1"
CELL_SIZES = (0.1, 0.5, 1.0)
MAP_FEATURE_LIMIT = 25_000
MAX_BROWSER_DOWNLOAD_BYTES = 250 * 1024 * 1024

# Region markers are selectable entry points on the globe. ISO-3 country
# polygons are highlighted where the study area is a whole country; California
# is represented by its own marker because a country polygon would be misleading.
REGION_INFO: dict[str, dict[str, Any]] = {
    "angola": {"label": "Angola", "lat": -12.2, "lon": 17.8, "iso3": "AGO", "area": "Southern Africa"},
    "australia": {"label": "Australia", "lat": -25.0, "lon": 133.0, "iso3": "AUS", "area": "Oceania"},
    "brasil": {"label": "Brasil", "lat": -10.5, "lon": -52.0, "iso3": "BRA", "area": "South America"},
    "california": {"label": "California", "lat": 36.5, "lon": -119.5, "iso3": None, "area": "North America"},
    "espana": {"label": "España", "lat": 40.2, "lon": -3.7, "iso3": "ESP", "area": "Europe"},
    "indonesia": {"label": "Indonesia", "lat": -2.5, "lon": 118.0, "iso3": "IDN", "area": "Southeast Asia"},
    "rdc": {"label": "RDC", "lat": -3.5, "lon": 23.5, "iso3": "COD", "area": "Central Africa"},
    "zambia": {"label": "Zambia", "lat": -13.4, "lon": 27.8, "iso3": "ZMB", "area": "Southern Africa"},
}

st.set_page_config(page_title="Sutellite | Fire observation explorer", page_icon="S", layout="wide")
load_dotenv(PROJECT_ROOT / ".env")


# -----------------------------------------------------------------------------
# Shared styling and navigation
# -----------------------------------------------------------------------------


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="st-"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: radial-gradient(ellipse at 50% -20%, #20364b 0%, #0b1726 50%, #07111d 100%); }
        [data-testid="stHeader"] { background: rgba(7,17,29,0.88); }
        [data-testid="stSidebar"] { background: #0d1b2a; }
        .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
        .sutellite-brand { color: #f3f7fb; font-size: 0.82rem; font-weight: 800; letter-spacing: .24em; text-transform: uppercase; }
        .sutellite-kicker { color: #9fb8ca; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; }
        .sutellite-title { color: #f4f8fc; font-size: clamp(2rem, 4vw, 3.45rem); font-weight: 800; line-height: 1.03; letter-spacing: -.055em; margin: .45rem 0 .8rem 0; }
        .sutellite-subtitle { color: #a9bdcc; font-size: 1rem; max-width: 760px; line-height: 1.7; }
        .sutellite-panel { background: linear-gradient(145deg, rgba(22,42,60,.96), rgba(12,27,42,.95)); border: 1px solid rgba(139,174,195,.18); border-radius: 18px; padding: 1.1rem 1.25rem; }
        .sutellite-muted { color: #91a9ba; font-size: .82rem; }
        .sutellite-region-name { color: #f2f7fb; font-size: 1.05rem; font-weight: 750; }
        .sutellite-region-meta { color: #98adbd; font-size: .73rem; letter-spacing: .08em; text-transform: uppercase; }
        div[data-testid="stMetric"] { background: rgba(20,39,56,.72); border: 1px solid rgba(130,164,188,.18); padding: .8rem 1rem; border-radius: 13px; }
        div[data-testid="stMetricLabel"] { color: #9eb5c5; }
        div[data-testid="stMetricValue"] { color: #f4f8fb; }
        div[data-testid="stExpander"] { border: 1px solid rgba(130,164,188,.2); border-radius: 14px; background: rgba(14,30,45,.78); }
        div.stButton > button { border-radius: 11px; border: 1px solid rgba(135,171,194,.28); background: rgba(25,48,67,.88); color: #eaf3f8; font-weight: 700; min-height: 2.7rem; }
        div.stButton > button:hover { border-color: #ff9858; color: #fff; }
        div.stButton > button[kind="primary"] { background: linear-gradient(100deg,#ef7548,#ffad60); border: 0; color: #111a22; }
        div[data-baseweb="tab-list"] { gap: .35rem; }
        div[data-baseweb="tab"] { background: rgba(24,44,62,.8); border-radius: 9px 9px 0 0; }
        div[data-baseweb="tab"] p { color: #c2d2dc; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #8fa8ba; }
        hr { border-color: rgba(133,166,188,.15); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _brand_bar() -> None:
    st.markdown(
        '<div class="sutellite-brand">SUTELLITE <span style="color:#ff9a5d">/</span> ACTIVE FIRE OBSERVATION EXPLORER</div>',
        unsafe_allow_html=True,
    )


def _dataset_dir() -> Path:
    configured = st.session_state.get(
        "canonical_dataset_path",
        os.getenv("SUTELLITE_CANONICAL_DIR", str(DEFAULT_CANONICAL_DIR)),
    )
    return Path(str(configured)).expanduser()


def _canonical_files(dataset_dir: Path, region_id: str | None = None) -> list[Path]:
    if not dataset_dir.is_dir():
        return []
    files = sorted(dataset_dir.glob("*/*.parquet"))
    if region_id and region_id != "all":
        files = [path for path in files if path.parent.name == region_id]
    return [path for path in files if path.name != "quality_flags.parquet"]


def _build_cache_dir(files: list[Path], cell_degrees: float, region_id: str) -> Path:
    fingerprint = canonical_file_fingerprint(files, cell_degrees)
    return PROJECT_ROOT / ".cache" / "sutellite" / f"{region_id}_{cell_degrees:g}_{fingerprint}"


def _get_or_build_products(
    files: list[Path],
    *,
    cell_degrees: float,
    region_id: str,
    worldcover_dir: Path | None,
    force: bool = False,
) -> Path:
    output = _build_cache_dir(files, cell_degrees, region_id)
    required = [
        "cell_aggregates.parquet", "cell_breakdown.parquet", "monthly_activity.parquet",
        "daily_activity.parquet", "monthly_activity.csv", "monthly_anomalies.csv",
        "map_cells.geojson", "cell_processing_summary.json",
    ]
    if force and output.exists():
        shutil.rmtree(output)
    if not all((output / name).is_file() for name in required):
        output.parent.mkdir(parents=True, exist_ok=True)
        build_cell_products(files, output, cell_degrees=cell_degrees, worldcover_dir=worldcover_dir)
    return output


# -----------------------------------------------------------------------------
# Globe landing page
# -----------------------------------------------------------------------------


def _globe_figure() -> go.Figure:
    country_regions = [(key, info) for key, info in REGION_INFO.items() if info.get("iso3")]
    countries = [info["iso3"] for _, info in country_regions]
    country_custom = [[key, info["label"]] for key, info in country_regions]
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=countries,
            customdata=country_custom,
            z=[1] * len(countries),
            locationmode="ISO-3",
            colorscale=[[0, "#28516a"], [1, "#28516a"]],
            showscale=False,
            marker_line_color="#5b8298",
            marker_line_width=0.8,
            hovertemplate="<b>%{customdata[1]}</b><br>Click to open study area<extra></extra>",
            name="Study areas",
        )
    )
    region_ids = list(REGION_INFO)
    lats = [REGION_INFO[key]["lat"] for key in region_ids]
    lons = [REGION_INFO[key]["lon"] for key in region_ids]
    labels = [REGION_INFO[key]["label"] for key in region_ids]
    areas = [REGION_INFO[key]["area"] for key in region_ids]
    custom = [[key, REGION_INFO[key]["label"]] for key in region_ids]
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=labels,
            customdata=custom,
            mode="markers+text",
            textposition=["top center", "bottom center", "top center", "top center", "bottom center", "bottom center", "top center", "bottom center"],
            textfont={"color": "#e9f3f8", "size": 11, "family": "Manrope"},
            marker={
                "size": 14,
                "color": "#ff995b",
                "line": {"color": "#fff0df", "width": 1.4},
                "opacity": 0.98,
            },
            hovertemplate="<b>%{customdata[1]}</b><br>%{text}<extra>Open study area</extra>",
            name="Open a study area",
        )
    )
    fig.update_layout(
        height=610,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        clickmode="event+select",
        geo={
            "projection": {"type": "orthographic", "rotation": {"lon": -20, "lat": 12, "roll": 0}},
            "showcoastlines": True,
            "coastlinecolor": "#66869a",
            "showland": True,
            "landcolor": "#1b3448",
            "showocean": True,
            "oceancolor": "#0b1c2b",
            "showlakes": True,
            "lakecolor": "#0b1c2b",
            "showcountries": True,
            "countrycolor": "#3a5a70",
            "bgcolor": "rgba(0,0,0,0)",
            "showframe": False,
        },
    )
    return fig


def _selected_globe_region(event: Any) -> str | None:
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event.get("selection", {}).get("points", [])
        except Exception:
            points = []
    for point in points or []:
        custom = point.get("customdata") if isinstance(point, dict) else None
        if isinstance(custom, (list, tuple)) and custom:
            candidate = str(custom[0])
            if candidate in REGION_INFO:
                return candidate
        if isinstance(custom, str) and custom in REGION_INFO:
            return custom
    return None


def _open_region(region_id: str, region_name: str | None = None) -> None:
    st.session_state["selected_region"] = region_id
    st.session_state["selected_region_label"] = region_name or REGION_INFO.get(region_id, {}).get("label", region_id)
    st.rerun()


def _render_region_cards() -> None:
    ids = list(REGION_INFO)
    for row_start in range(0, len(ids), 4):
        columns = st.columns(4)
        for column, region_id in zip(columns, ids[row_start:row_start + 4]):
            info = REGION_INFO[region_id]
            with column:
                st.markdown(
                    f'<div class="sutellite-panel"><div class="sutellite-region-meta">{info["area"]}</div>'
                    f'<div class="sutellite-region-name">{info["label"]}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Explore region ↗", key=f"open_region_{region_id}", use_container_width=True):
                    _open_region(region_id)


def _render_landing_page() -> None:
    _brand_bar()
    st.markdown('<div class="sutellite-kicker" style="margin-top:2.2rem">Satellite fire observations · eight study areas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sutellite-title">One planet.<br/>Eight windows into fire activity.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sutellite-subtitle">Explore harmonised MODIS and VIIRS active-fire observations. Select a study area on the globe to open its spatial cells, statistics, anomaly analysis and daily hotspot calendar.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="sutellite-kicker" style="margin:.8rem 0 .4rem">Select a study area on the globe</div>', unsafe_allow_html=True)
        event = st.plotly_chart(
            _globe_figure(),
            key="sutellite_region_globe",
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": False, "scrollZoom": False},
        )
        selected = _selected_globe_region(event)
        if selected:
            _open_region(selected)
        st.caption("Click one of the glowing region markers, or choose a region below. Country outlines are contextual; California is shown as a study-area marker.")
    with right:
        st.markdown('<div class="sutellite-panel" style="margin-top:1.3rem">', unsafe_allow_html=True)
        st.markdown("**SUTELLITE EXPLORER**")
        st.markdown("The same workspace is available for all eight study regions.")
        st.markdown('<div class="sutellite-muted">MODIS · Terra / Aqua<br/>VIIRS · Suomi-NPP / NOAA-20 / NOAA-21<br/>ARCHIVE and NRT processing streams</div>', unsafe_allow_html=True)
        st.markdown("---")
        dataset = st.text_input(
            "Canonical Parquet dataset folder",
            value=str(_dataset_dir()),
            key="canonical_dataset_path",
            help="Point this to the shared canonical_v1.1 folder. The large Parquet dataset is not bundled in GitHub.",
        )
        if Path(dataset).expanduser().is_dir():
            st.success("Canonical dataset folder found")
        else:
            st.warning("Canonical dataset folder not found on this machine")
        st.markdown("---")
        st.markdown("**Bring your own FIRMS data**")
        st.caption("Upload raw NASA FIRMS CSVs to create canonical Parquet and spatial cell products for a built-in or custom region.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Study regions")
    _render_region_cards()
    show_upload = bool(st.session_state.pop("show_upload_on_landing", False))
    with st.expander("Upload FIRMS CSVs / add a custom region", expanded=show_upload):
        _render_upload_panel()


# -----------------------------------------------------------------------------
# Region data and map
# -----------------------------------------------------------------------------


def _query_detail(
    detail_path: Path,
    *,
    region_id: str,
    sensors: list[str],
    platforms: list[str],
    streams: list[str],
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    if not sensors or not platforms or not streams:
        return pd.DataFrame()
    where = ["year BETWEEN ? AND ?"]
    params: list[object] = [str(detail_path), int(year_min), int(year_max)]
    if region_id != "all":
        where.append("region_id = ?")
        params.append(region_id)
    for column, values in (("sensor", sensors), ("platform", platforms), ("processing_stream", streams)):
        where.append(f"{column} IN ({','.join('?' for _ in values)})")
        params.extend(values)
    sql = f"SELECT * FROM read_parquet(?) WHERE {' AND '.join(where)}"
    with duckdb.connect(database=":memory:") as con:
        return con.execute(sql, params).fetchdf()


def _aggregate_filtered_cells(detail: pd.DataFrame, cells_all: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    keys = ["region_id", "lat_index", "lon_index"]
    sum_fields = [
        "observation_count", "modis_count", "viirs_count", "terra_count", "aqua_count",
        "snpp_count", "noaa20_count", "noaa21_count", "archive_count", "nrt_count",
        "daytime_count", "nighttime_count", "frp_non_missing_count", "frp_missing_count",
        "frp_negative_count", "frp_zero_count", "frp_positive_count", "frp_valid_count",
        "frp_sum_mw", "modis_confidence_n", "viirs_confidence_low_count",
        "viirs_confidence_nominal_count", "viirs_confidence_high_count",
        "type_0_count", "type_1_count", "type_2_count", "type_3_count",
    ]
    grouped = detail.groupby(keys, as_index=False)[sum_fields].sum(min_count=1)
    grouped["frp_sum_mw"] = grouped["frp_sum_mw"].fillna(0)
    grouped["frp_max_mw"] = detail.groupby(keys)["frp_max_mw"].max().to_numpy()
    weighted = detail.copy()
    weighted["_confidence_weighted"] = (
        weighted["modis_confidence_mean_pct"].fillna(0) * weighted["modis_confidence_n"].fillna(0)
    )
    conf = weighted.groupby(keys, as_index=False)["_confidence_weighted"].sum()
    grouped = grouped.merge(conf, on=keys, how="left")
    grouped["modis_confidence_mean_pct"] = np.where(
        grouped["modis_confidence_n"] > 0,
        grouped["_confidence_weighted"] / grouped["modis_confidence_n"],
        np.nan,
    )
    grouped["frp_mean_mw"] = np.where(
        grouped["frp_valid_count"] > 0,
        grouped["frp_sum_mw"] / grouped["frp_valid_count"],
        np.nan,
    )
    output = grouped.merge(
        cells_all[[*keys, "cell_id", "cell_center_lat", "cell_center_lon", "land_type", "worldcover_2021_code"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return output.sort_values("observation_count", ascending=False).reset_index(drop=True)


def _render_map(cells: pd.DataFrame, metric: str, cell_degrees: float, map_mode: str, region_id: str) -> None:
    if cells.empty:
        info = REGION_INFO.get(region_id, {"lat": 0, "lon": 0, "label": region_id})
        st.info("No detection cells match the selected filters. Missing cells are not evidence of zero fire activity or satellite non-observation.")
        st.pydeck_chart(
            pdk.Deck(
                layers=[],
                initial_view_state=pdk.ViewState(latitude=float(info["lat"]), longitude=float(info["lon"]), zoom=4, pitch=0),
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            ),
            use_container_width=True,
        )
        return

    shown = cells.head(MAP_FEATURE_LIMIT).copy()
    if len(cells) > MAP_FEATURE_LIMIT:
        st.warning(f"The map displays the {MAP_FEATURE_LIMIT:,} most active cells for performance. The downloaded files contain all cells.")
    value = pd.to_numeric(shown[metric], errors="coerce").fillna(0).to_numpy(dtype=float)
    scale = np.log1p(np.maximum(value, 0))
    vmax = float(np.quantile(scale, 0.98)) if len(scale) else 1.0
    vmax = max(vmax, 1.0)
    colors = []
    heights = []
    for val in scale:
        t = float(np.clip(val / vmax, 0, 1))
        colors.append([int(242 - 95 * t), int(165 - 95 * t), int(91 - 28 * t), 190])
        heights.append(max(80, min(9000, float(math.log1p(max(val, 0)) * 1800))))

    features = []
    for (_, row), color, height in zip(shown.iterrows(), colors, heights):
        south = max(-90.0, float(row["lat_index"]) * cell_degrees)
        west = max(-180.0, float(row["lon_index"]) * cell_degrees)
        north = min(90.0, south + cell_degrees)
        east = min(180.0, west + cell_degrees)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, south], [east, south], [east, north], [west, north], [west, south]
            ]]},
            "properties": {
                "region": row["region_id"], "cell_id": row["cell_id"],
                "detections": int(row["observation_count"]),
                "modis_count": int(row["modis_count"]), "viirs_count": int(row["viirs_count"]),
                "terra_count": int(row["terra_count"]), "aqua_count": int(row["aqua_count"]),
                "snpp_count": int(row["snpp_count"]), "noaa20_count": int(row["noaa20_count"]),
                "noaa21_count": int(row["noaa21_count"]),
                "daytime_count": int(row["daytime_count"]), "nighttime_count": int(row["nighttime_count"]),
                "mean_frp_mw": None if pd.isna(row["frp_mean_mw"]) else round(float(row["frp_mean_mw"]), 2),
                "land_type": row.get("land_type", "No data") or "No data",
                "height": height, "color": color,
            },
        })
    geojson = {"type": "FeatureCollection", "features": features}
    center_lat = float(shown["cell_center_lat"].mean())
    center_lon = float(shown["cell_center_lon"].mean())
    pitch = 0 if map_mode == "2D" else 48
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=(map_mode == "2.5D"),
        get_elevation="properties.height",
        elevation_scale=1,
        get_fill_color="properties.color",
        get_line_color=[75, 93, 105, 110],
        line_width_min_pixels=0.35,
        auto_highlight=True,
        opacity=0.84,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=4.4, pitch=pitch, bearing=0),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"html": (
                "<b>{region} · {cell_id}</b><br/>Observation records: {detections}"
                "<br/>MODIS / VIIRS: {modis_count} / {viirs_count}"
                "<br/>Terra / Aqua: {terra_count} / {aqua_count}"
                "<br/>SNPP / NOAA-20 / NOAA-21: {snpp_count} / {noaa20_count} / {noaa21_count}"
                "<br/>Day / night: {daytime_count} / {nighttime_count}"
                "<br/>Mean non-negative FRP: {mean_frp_mw} MW"
                "<br/>WorldCover: {land_type}"
            )},
        ),
        use_container_width=True,
    )
    st.caption(
        f"Grid resolution: {cell_degrees:g}° · Colour: {metric.replace('_', ' ')} · {map_mode}. "
        "Cells are angular grid bins containing detections; they are not fire perimeters or satellite coverage footprints."
    )


def _render_hotspot_calendar(
    daily_path: Path,
    region_id: str,
    sensors: list[str],
    platforms: list[str],
    streams: list[str],
) -> None:
    if not daily_path.is_file():
        st.info("Daily activity output is missing. Rebuild the region's spatial products to generate the hotspot calendar.")
        return
    daily = pd.read_parquet(daily_path)
    if daily.empty:
        st.info("No daily observation records are available.")
        return
    daily["activity_date"] = pd.to_datetime(daily["activity_date"], errors="coerce")
    daily = daily.dropna(subset=["activity_date"])
    daily = daily[
        daily["region_id"].astype(str).eq(region_id)
        & daily["sensor"].isin(sensors)
        & daily["platform"].isin(platforms)
        & daily["processing_stream"].isin(streams)
    ]
    if daily.empty:
        st.info("No daily activity matches the selected sensor, platform and stream filters.")
        return
    daily["year"] = daily["activity_date"].dt.year
    daily["month"] = daily["activity_date"].dt.month
    years = sorted(daily["year"].unique().tolist())
    c1, c2 = st.columns(2)
    year = c1.selectbox("Calendar year", years, index=len(years) - 1, key=f"calendar_year_{region_id}")
    months = sorted(daily.loc[daily["year"].eq(year), "month"].unique().tolist())
    month = c2.selectbox("Month", months, index=len(months) - 1, format_func=lambda m: calendar.month_name[int(m)], key=f"calendar_month_{region_id}")
    selected = daily[daily["year"].eq(year) & daily["month"].eq(month)]
    counts = selected.groupby(selected["activity_date"].dt.day)["observation_count"].sum().to_dict()
    month_grid = calendar.monthcalendar(int(year), int(month))
    max_count = max([int(v) for v in counts.values()], default=0)
    html_days = []
    for weekday in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        html_days.append(f'<div style="color:#8fa9bc;font-size:.72rem;text-align:center;padding:.35rem">{weekday}</div>')
    for week in month_grid:
        for day in week:
            if day == 0:
                html_days.append('<div style="min-height:76px;border-radius:9px;background:rgba(255,255,255,.015)"></div>')
                continue
            count = int(counts.get(day, 0))
            intensity = 0 if count == 0 or max_count == 0 else math.sqrt(count / max_count)
            red = int(40 + 190 * intensity)
            green = int(54 + 66 * (1 - intensity))
            blue = int(73 + 24 * (1 - intensity))
            bg = "#142638" if count == 0 else f"rgb({red},{green},{blue})"
            sub = "No records" if count == 0 else f"{count:,} records"
            html_days.append(
                f'<div title="{int(year)}-{int(month):02d}-{day:02d}: {count:,} observation records" '
                f'style="min-height:76px;border:1px solid rgba(170,195,210,.15);border-radius:9px;padding:.55rem;background:{bg};">'
                f'<div style="font-size:.75rem;color:#f2f6f9;font-weight:700">{day}</div>'
                f'<div style="font-family:monospace;font-size:.68rem;color:#fff;opacity:.92;margin-top:.55rem">{sub}</div></div>'
            )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px">'
        + "".join(html_days) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Colour intensity represents the number of source observation records on each date. "
        "A blank/zero-record day does not prove there was no fire or that a satellite observed the area and detected nothing."
    )


def _find_ai_prediction_file(region_id: str) -> Path | None:
    configured = os.getenv("SUTELLITE_AI_RESULTS_DIR", "").strip()
    if not configured:
        return None
    folder = Path(configured).expanduser()
    if not folder.is_dir():
        return None
    candidates = sorted(folder.glob(f"*{region_id}*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _render_ai_detections(region_id: str) -> None:
    prediction_file = _find_ai_prediction_file(region_id)
    if prediction_file is None:
        st.info(
            "No AI prediction output is configured for this region. The repository contains the experimental "
            "monthly anomaly inference code, but the trained regional model artifacts and prediction CSVs are "
            "stored separately and are not bundled with this website."
        )
        st.markdown(
            "**To connect model outputs:** set `SUTELLITE_AI_RESULTS_DIR` in `.env` to the folder containing "
            "the region's prediction CSV (for example, `predicciones_angola.csv`)."
        )
        st.warning(
            "The statistical anomaly flags in the Anomalies section are not AI predictions. They are shown separately "
            "to avoid presenting one method as the other."
        )
        return
    try:
        predictions = pd.read_csv(prediction_file)
    except Exception as exc:
        st.error(f"Could not read AI prediction file {prediction_file.name}: {exc}")
        return
    if "region_id" in predictions.columns:
        predictions = predictions[predictions["region_id"].astype(str).eq(region_id)]
    if predictions.empty:
        st.info(f"The configured file {prediction_file.name} contains no predictions for {region_id}.")
        return
    st.caption(f"Prediction source: {prediction_file.name}")
    if "resultado" in predictions.columns:
        summary = predictions["resultado"].fillna("Unknown").value_counts().rename_axis("Model output").reset_index(name="Rows")
        st.dataframe(summary, hide_index=True, use_container_width=True)
    if "puntuacion" in predictions.columns:
        predictions["puntuacion"] = pd.to_numeric(predictions["puntuacion"], errors="coerce")
    preferred = [
        "region_id", "year", "month", "grid_lat", "grid_lon", "detections",
        "media_referencia", "maximo_historico", "resultado", "puntuacion", "worldcover_2021_clase",
    ]
    display = [name for name in preferred if name in predictions.columns]
    st.dataframe(predictions[display].head(1000), hide_index=True, use_container_width=True)
    st.caption("These are outputs from the configured experimental model. They are estimates/flags, not verified fire events or operational alerts.")


def _render_dashboard(products_dir: Path, region_id: str, title: str, key_prefix: str) -> None:
    cells_path = products_dir / "cell_aggregates.parquet"
    detail_path = products_dir / "cell_breakdown.parquet"
    monthly_path = products_dir / "monthly_activity.parquet"
    daily_path = products_dir / "daily_activity.parquet"
    anomalies_path = products_dir / "monthly_anomalies.csv"
    summary_path = products_dir / "cell_processing_summary.json"
    needed = [cells_path, detail_path, monthly_path, daily_path, anomalies_path, summary_path]
    if not all(path.is_file() for path in needed):
        st.warning("Some region products are missing. Rebuild the spatial products to enable the full workspace and calendar.")
        return

    cells_all = pd.read_parquet(cells_path)
    monthly_all = pd.read_parquet(monthly_path)
    anomalies_all = pd.read_csv(anomalies_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sensor_options = sorted(monthly_all["sensor"].dropna().astype(str).unique().tolist())
    platform_options = sorted(monthly_all["platform"].dropna().astype(str).unique().tolist())
    stream_options = sorted(monthly_all["processing_stream"].dropna().astype(str).unique().tolist())
    years = sorted(monthly_all["year"].dropna().astype(int).unique().tolist())
    if not years:
        st.info("No dated observations are available in this region.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observation records", f"{summary['input_observations']:,}")
    k2.metric("Detection cells", f"{summary['cells']:,}")
    k3.metric("Active observation days", f"{int(summary.get('active_observation_days', 0)):,}")
    k4.metric("Land-cover labelled cells", f"{summary['land_cover_known_cells']:,}")

    with st.expander("Filters · sensor, satellite, stream and period", expanded=False):
        f1, f2, f3 = st.columns(3)
        selected_sensors = f1.multiselect("Sensors", sensor_options, default=sensor_options, key=f"{key_prefix}_sensors")
        selected_platforms = f2.multiselect("Satellites", platform_options, default=platform_options, key=f"{key_prefix}_platforms")
        selected_streams = f3.multiselect("Processing stream", stream_options, default=stream_options, key=f"{key_prefix}_streams")
        if len(years) > 1:
            selected_years = st.slider(
                "Acquisition years", min_value=years[0], max_value=years[-1],
                value=(years[0], years[-1]), key=f"{key_prefix}_years",
            )
        else:
            selected_years = (years[0], years[0])
            st.caption(f"Available year: {years[0]}")

    detail = _query_detail(
        detail_path,
        region_id=region_id,
        sensors=selected_sensors,
        platforms=selected_platforms,
        streams=selected_streams,
        year_min=int(selected_years[0]),
        year_max=int(selected_years[1]),
    )
    if detail.empty:
        st.info("No observations match the selected filters.")
        filtered_cells = pd.DataFrame()
        total_observations = 0
        frp_mean = np.nan
    else:
        filtered_cells = _aggregate_filtered_cells(detail, cells_all)
        total_observations = int(detail["observation_count"].sum())
        valid_frp = int(detail["frp_valid_count"].sum())
        frp_sum = float(detail["frp_sum_mw"].fillna(0).sum())
        frp_mean = frp_sum / valid_frp if valid_frp else np.nan

    selected_anomalies = anomalies_all[
        anomalies_all["region_id"].astype(str).eq(region_id)
        & anomalies_all["sensor"].isin(selected_sensors)
        & anomalies_all["platform"].isin(selected_platforms)
        & anomalies_all["processing_stream"].isin(selected_streams)
        & anomalies_all["year"].between(selected_years[0], selected_years[1])
    ]
    anomaly_count = int(selected_anomalies["is_anomaly"].fillna(False).astype(bool).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered observations", f"{total_observations:,}")
    m2.metric("Cells with detections", f"{len(filtered_cells):,}")
    m3.metric("Mean non-negative FRP", "—" if not np.isfinite(frp_mean) else f"{frp_mean:.2f} MW")
    m4.metric("Statistical anomalies", f"{anomaly_count:,}")

    map_controls = st.columns([1.25, 1, 2])
    map_metric = map_controls[0].selectbox(
        "Cell colour", ["observation_count", "frp_mean_mw"],
        format_func=lambda value: "Observation records" if value == "observation_count" else "Mean non-negative FRP (MW)",
        key=f"{key_prefix}_map_metric",
    )
    map_mode = map_controls[1].radio("Map view", ["2D", "2.5D"], horizontal=True, key=f"{key_prefix}_map_mode")
    map_controls[2].markdown('<div class="sutellite-muted" style="padding-top:2rem">Hover over a cell to inspect its counts, platform split, FRP and land-cover label.</div>', unsafe_allow_html=True)
    _render_map(filtered_cells, map_metric, float(summary["cell_degrees"]), map_mode, region_id)

    with st.expander("Analysis & monitoring", expanded=False):
        tab_stats, tab_anomaly, tab_ai, tab_calendar = st.tabs(
            ["Statistics", "Anomalies", "AI detections", "Hotspot calendar"]
        )
        with tab_stats:
            monthly = monthly_all[
                monthly_all["region_id"].astype(str).eq(region_id)
                & monthly_all["sensor"].isin(selected_sensors)
                & monthly_all["platform"].isin(selected_platforms)
                & monthly_all["processing_stream"].isin(selected_streams)
                & monthly_all["year"].between(selected_years[0], selected_years[1])
            ].copy()
            trend = monthly.groupby(["year", "month"], as_index=False)["observation_count"].sum()
            if not trend.empty:
                trend["date"] = pd.to_datetime(
                    trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2) + "-01"
                )
                fig = px.area(trend, x="date", y="observation_count", markers=True, title="Monthly observation records")
                fig.update_traces(line_color="#ff955a", fillcolor="rgba(255,149,90,.18)")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,37,53,.55)",
                    font_color="#dce8ef", xaxis_title="Acquisition month", yaxis_title="Source observation records",
                )
                st.plotly_chart(fig, use_container_width=True)
            split = detail.groupby(["sensor", "platform", "processing_stream"], as_index=False)["observation_count"].sum() if not detail.empty else pd.DataFrame()
            st.markdown("#### Sensor, satellite and stream breakdown")
            if not split.empty:
                st.dataframe(split.sort_values("observation_count", ascending=False), width="stretch", hide_index=True)
            if not filtered_cells.empty:
                modis_n = int(filtered_cells["modis_confidence_n"].sum())
                if modis_n:
                    modis_mean = float((filtered_cells["modis_confidence_mean_pct"].fillna(0) * filtered_cells["modis_confidence_n"].fillna(0)).sum() / modis_n)
                    st.metric("Mean MODIS confidence · percentage scale", f"{modis_mean:.1f}%")
                viirs_total = int(filtered_cells[["viirs_confidence_low_count", "viirs_confidence_nominal_count", "viirs_confidence_high_count"]].sum().sum())
                if viirs_total:
                    st.dataframe(pd.DataFrame({
                        "VIIRS confidence class": ["Low", "Nominal", "High"],
                        "Observation records": [
                            int(filtered_cells["viirs_confidence_low_count"].sum()),
                            int(filtered_cells["viirs_confidence_nominal_count"].sum()),
                            int(filtered_cells["viirs_confidence_high_count"].sum()),
                        ],
                    }), hide_index=True, use_container_width=True)
            st.caption("MODIS percentage confidence and VIIRS low/nominal/high confidence are different systems and are not directly comparable.")

        with tab_anomaly:
            alerts = selected_anomalies[selected_anomalies["is_anomaly"].fillna(False).astype(bool)]
            if alerts.empty:
                st.info("No high monthly statistical anomalies were flagged for these filters, or the history is too short.")
            else:
                columns = [
                    "region_id", "year", "month", "sensor", "platform", "processing_stream",
                    "observation_count", "reference_years", "reference_median", "robust_z",
                ]
                st.dataframe(alerts.sort_values("robust_z", ascending=False)[columns], width="stretch", hide_index=True)
            st.caption(
                "Exploratory one-sided robust score against earlier observed values for the same calendar month, "
                "region, sensor, satellite and stream. At least three earlier observations are required. "
                "Missing months are not filled with zero. These are statistical flags, not fire alerts."
            )

        with tab_ai:
            _render_ai_detections(region_id)

        with tab_calendar:
            st.markdown("#### Daily hotspot activity")
            _render_hotspot_calendar(daily_path, region_id, selected_sensors, selected_platforms, selected_streams)

    with st.expander("Cell data & downloads", expanded=False):
        if not filtered_cells.empty:
            columns = [
                "region_id", "cell_id", "cell_center_lat", "cell_center_lon", "observation_count",
                "modis_count", "viirs_count", "terra_count", "aqua_count", "snpp_count",
                "noaa20_count", "noaa21_count", "archive_count", "nrt_count", "daytime_count", "nighttime_count",
                "frp_non_missing_count", "frp_missing_count", "frp_negative_count", "frp_zero_count",
                "frp_positive_count", "frp_valid_count", "frp_sum_mw", "frp_mean_mw", "frp_max_mw",
                "modis_confidence_mean_pct", "modis_confidence_n", "viirs_confidence_low_count",
                "viirs_confidence_nominal_count", "viirs_confidence_high_count", "land_type", "worldcover_2021_code",
            ]
            st.dataframe(filtered_cells[[column for column in columns if column in filtered_cells.columns]].head(1000), width="stretch", hide_index=True)
            st.caption("Showing up to 1,000 cells. The complete cell-level Parquet and GeoJSON outputs are available below.")
        _download_file(products_dir / "map_cells.geojson", "Download map cells · GeoJSON", "application/geo+json")
        _download_file(products_dir / "cell_aggregates.parquet", "Download cell aggregates · Parquet", "application/octet-stream")
        _download_file(products_dir / "cell_breakdown.parquet", "Download cell breakdown · Parquet", "application/octet-stream")
        _download_file(products_dir / "monthly_activity.parquet", "Download monthly activity · Parquet", "application/octet-stream")
        _download_file(products_dir / "daily_activity.parquet", "Download daily calendar data · Parquet", "application/octet-stream")
        _download_file(products_dir / "monthly_anomalies.csv", "Download anomaly results · CSV", "text/csv")
        _download_file(products_dir / "cell_processing_summary.json", "Download processing summary · JSON", "application/json")


def _download_file(path: Path, label: str, mime: str) -> None:
    if not path.is_file():
        return
    size = path.stat().st_size
    if size > MAX_BROWSER_DOWNLOAD_BYTES:
        st.warning(f"{path.name} is {size / (1024 ** 2):,.1f} MiB, above the 250 MiB browser-download limit. It remains saved locally at:")
        st.code(str(path), language=None)
        return
    st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime, key=f"download_{path.parent.name}_{path.name}")


def _render_empty_region_map(region_id: str) -> None:
    info = REGION_INFO.get(region_id, {"lat": 0, "lon": 0})
    st.pydeck_chart(
        pdk.Deck(
            layers=[],
            initial_view_state=pdk.ViewState(latitude=float(info["lat"]), longitude=float(info["lon"]), zoom=4, pitch=0),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        ),
        use_container_width=True,
    )


def _render_region_workspace(region_id: str) -> None:
    info = REGION_INFO.get(region_id)
    label = st.session_state.get("selected_region_label") or (info["label"] if info else region_id)
    back, name, tag = st.columns([0.8, 3.3, 1.2])
    if back.button("← Globe", key="return_to_globe"):
        st.session_state.pop("selected_region", None)
        st.session_state.pop("selected_region_label", None)
        st.rerun()
    safe_label = html.escape(str(label))
    area_label = html.escape(str(info["area"] if info else "CUSTOM REGION"))
    name.markdown(f'<div class="sutellite-kicker">STUDY AREA / {area_label}</div><div class="sutellite-title" style="font-size:2.35rem;margin:.25rem 0">{safe_label}</div>', unsafe_allow_html=True)
    tag.markdown('<div style="text-align:right;padding-top:1rem;color:#ffad74;font-family:monospace;font-size:.75rem">MODIS + VIIRS<br/>CELL EXPLORER</div>', unsafe_allow_html=True)

    dataset_dir = _dataset_dir()
    with st.expander("Dataset connection & map resolution", expanded=False):
        path_text = st.text_input("Canonical dataset folder", value=str(dataset_dir), key=f"region_dataset_path_{region_id}")
        if path_text:
            st.session_state["canonical_dataset_path"] = path_text
        cell_degrees = st.selectbox("Spatial grid-cell size", CELL_SIZES, index=1, format_func=lambda value: f"{value:g}°", key=f"region_cell_size_{region_id}")

    dataset_dir = _dataset_dir()
    files = _canonical_files(dataset_dir, region_id)
    last_region = st.session_state.get("last_upload_region")
    upload_products_text = st.session_state.get("last_upload_products") if last_region == region_id else None
    if upload_products_text and Path(upload_products_text).is_dir():
        st.success("Using the most recent CSV upload for this region.")
        _render_dashboard(Path(upload_products_text), region_id, label, f"region_{region_id}_upload")
        return

    if not files:
        _render_empty_region_map(region_id)
        st.warning(f"No canonical Parquet files were found for {label} in {dataset_dir}.")
        st.markdown(
            "Upload the FIRMS CSV files for this region from the globe page, or connect the shared `canonical_v1.1` folder. "
            "The website does not invent empty cells or infer that an unrepresented area had no fire."
        )
        if st.button("Go to CSV upload", type="primary", key=f"go_upload_{region_id}"):
            st.session_state["show_upload_on_landing"] = True
            st.session_state.pop("selected_region", None)
            st.rerun()
        return

    st.caption(f"Connected to {len(files):,} canonical Parquet files · source dataset is read-only.")
    expected_products = _build_cache_dir(files, cell_degrees, region_id)
    worldcover_value = os.getenv("SUTELLITE_WORLDCOVER_DIR", "").strip()
    worldcover_dir = Path(worldcover_value).expanduser() if worldcover_value else None
    if not all((expected_products / filename).is_file() for filename in ["cell_aggregates.parquet", "daily_activity.parquet", "monthly_anomalies.csv"]):
        st.markdown('<div class="sutellite-panel"><b>Spatial products need to be prepared for this region.</b><br/><span class="sutellite-muted">This reads the canonical Parquet files and builds the map cells, monthly statistics, daily calendar and anomaly outputs.</span></div>', unsafe_allow_html=True)
        if st.button("Prepare region explorer", type="primary", key=f"prepare_region_{region_id}"):
            with st.spinner("Aggregating canonical observations into spatial cells and daily activity..."):
                try:
                    _get_or_build_products(files, cell_degrees=cell_degrees, region_id=region_id, worldcover_dir=worldcover_dir, force=True)
                    st.session_state[f"prepared_region_{region_id}"] = str(expected_products)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not prepare this region: {exc}")
        return
    _render_dashboard(expected_products, region_id, label, f"region_{region_id}")


# -----------------------------------------------------------------------------
# FIRMS upload flow
# -----------------------------------------------------------------------------


def _render_upload_panel() -> None:
    st.write(
        "Upload NASA FIRMS ARCHIVE/NRT CSV files. Sutellite validates the source contract, maps accepted rows to "
        "the canonical schema and creates cell, monthly and daily products."
    )
    mode = st.radio("Region type", ["One of the eight built-in regions", "My own region"], horizontal=True, key="upload_region_mode")
    boundary_bytes = None
    if mode == "One of the eight built-in regions":
        label = st.selectbox("Study region", list(BUILT_IN_REGIONS.values()), key="upload_builtin_region")
        region_id = next(key for key, value in BUILT_IN_REGIONS.items() if value == label)
        region_name = label
        st.info("For built-in regions, the uploaded files must already be filtered to that region. Official study-area boundary polygons are not bundled in this prototype.")
    else:
        region_name = st.text_input("Custom region name", key="upload_custom_region_name")
        boundary_upload = st.file_uploader("Region boundary · GeoJSON Polygon/MultiPolygon · EPSG:4326", type=["geojson", "json"], key="upload_boundary")
        region_id = slugify_region(region_name) if region_name.strip() else "custom_region"
        if boundary_upload is not None:
            boundary_bytes = boundary_upload.getvalue()
        st.caption("Custom detections are clipped to the uploaded WGS84 GeoJSON boundary.")

    uploaded = st.file_uploader(
        "NASA FIRMS CSV files · select multiple platforms/streams",
        type=["csv"], accept_multiple_files=True, key="firms_csv_uploads",
        help="FIRMS ARCHIVE/NRT files for MODIS M-C61, VIIRS S-NPP SV-C2, NOAA-20 J1V-C2 and NOAA-21 J2V-C2.",
    )
    cell_degrees = st.selectbox("Output grid-cell size", CELL_SIZES, index=1, format_func=lambda value: f"{value:g}°", key="upload_cell_size")
    st.warning("The uploader processes only the files supplied. It does not download missing satellites or infer satellite observation coverage. For large historical collections, use the batch pipeline.")

    if st.button("Validate, harmonise and build region", type="primary", disabled=not uploaded, key="run_upload_pipeline"):
        if mode == "My own region" and (not region_name.strip() or boundary_bytes is None):
            st.error("Enter a custom region name and upload its GeoJSON boundary.")
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = PROJECT_ROOT / "processed" / "uploads" / run_id
        raw_dir = run_dir / "_uploaded_sources"
        raw_dir.mkdir(parents=True, exist_ok=False)
        saved_paths: list[Path] = []
        try:
            with st.spinner("Saving uploads and validating source files..."):
                for uploaded_file in uploaded:
                    target = raw_dir / Path(uploaded_file.name).name
                    target.write_bytes(uploaded_file.getvalue())
                    saved_paths.append(target)
                processing = process_uploaded_firms_files(
                    saved_paths,
                    region_id=region_id,
                    region_name=region_name,
                    output_dir=run_dir / "canonical",
                    boundary_geojson=boundary_bytes,
                )
            platform_counts = processing["summary"].get("platform_counts", {})
            st.write("Platforms present in this upload:", platform_counts or "None")
            if processing["summary"]["written_rows"] == 0:
                st.error("No valid observations were written. Inspect the validation report and rejected-row details.")
                st.json(processing["summary"])
                return
            worldcover_value = os.getenv("SUTELLITE_WORLDCOVER_DIR", "").strip()
            worldcover_dir = Path(worldcover_value).expanduser() if worldcover_value and mode != "My own region" else None
            spatial_dir = run_dir / "spatial"
            with st.spinner("Building spatial cells, monthly statistics, daily calendar and anomalies..."):
                spatial = build_cell_products(
                    [run_dir / "canonical" / "harmonized_observations.parquet"],
                    spatial_dir,
                    cell_degrees=cell_degrees,
                    worldcover_dir=worldcover_dir,
                )
            shutil.rmtree(raw_dir, ignore_errors=True)
            st.session_state["last_upload_products"] = str(spatial_dir)
            st.session_state["last_upload_canonical"] = str(run_dir / "canonical")
            st.session_state["last_upload_processing"] = processing
            st.session_state["last_upload_spatial"] = spatial
            st.session_state["last_upload_region"] = region_id
            st.session_state["selected_region"] = region_id
            st.session_state["selected_region_label"] = region_name
            st.success(
                f"Harmonisation complete: {processing['summary']['written_rows']:,} accepted records; "
                f"{spatial['cells']:,} detection cells."
            )
            expected_platforms = {"TERRA", "AQUA", "SNPP", "NOAA-20", "NOAA-21"}
            absent = sorted(expected_platforms - set(platform_counts))
            if absent:
                st.info("Not present in this upload: " + ", ".join(absent) + ". This does not mean those satellites observed zero fires.")
            st.rerun()
        except Exception as exc:
            st.error(f"Upload pipeline failed: {exc}")
            st.caption(f"The run folder was retained for diagnosis: {run_dir}")


def _render_methodology_footer() -> None:
    with st.expander("Methodology & interpretation", expanded=False):
        st.markdown(
            "- **Harmonisation:** source CSV rows are mapped into the versioned Sutellite canonical Parquet schema. "
            "Satellite/platform identity and source-specific confidence fields are preserved.\n"
            "- **Cells:** detections are aggregated into angular latitude/longitude grid cells. A cell is not a fire perimeter.\n"
            "- **Calendar:** colours represent daily source observation-record counts. They are not unique fire counts.\n"
            "- **Anomalies:** the Anomalies section uses an exploratory historical robust-score method. It is not an emergency alert.\n"
            "- **AI detections:** prediction outputs appear only when a trained-model result file is configured.\n"
            "- **Missing values:** no detection record is not proof of zero activity or satellite non-observation.\n"
            "- **Land cover:** ESA WorldCover labels are shown only where local tiles are configured and available; otherwise the UI shows No data.\n"
            "- **Sensor comparisons:** MODIS and VIIRS confidence systems differ and should not be compared as if they shared one probability scale."
        )


def main() -> None:
    _inject_styles()
    selected_region = st.session_state.get("selected_region")
    if selected_region:
        _brand_bar()
        _render_region_workspace(str(selected_region))
        _render_methodology_footer()
    else:
        _render_landing_page()
        _render_methodology_footer()


if __name__ == "__main__":
    main()
