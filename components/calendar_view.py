"""
Historical Calendar Visualization Component.

Renders monthly aggregated heatmaps, trend curves, summary metrics,
and historical baseline statistics while preserving missing observation gaps.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

MONTH_LABELS: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _get_reference_subset(
    monthly_pivot: pd.DataFrame, selected_year: int, fixed_demo_reference: bool
) -> Tuple[pd.DataFrame, str]:
    """
    Determine the historical reference years for baseline statistics.

    For 2024 and 2025, uses the fixed 2013-2023 baseline window (if available).
    For other years, uses all available prior years (< selected_year).
    """
    available_years = sorted([int(y) for y in monthly_pivot.index.tolist()])

    if fixed_demo_reference and selected_year in (2024, 2025):
        ref_years = [y for y in available_years if 2013 <= y <= 2023]
        period_label = "2013-2023"
    else:
        ref_years = [y for y in available_years if y < selected_year]
        period_label = f"{min(ref_years)}-{max(ref_years)}" if ref_years else "None"

    return monthly_pivot.loc[ref_years], period_label


def compute_monthly_statistics(
    region_df: pd.DataFrame, selected_year: int, *, fixed_demo_reference: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int], Optional[str], str]:
    """
    Compute the year-by-month matrix and descriptive baseline statistics.
    """
    df = region_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_pivot = (
        df.groupby(["year", "month"])["hotspot_count"]
        .sum(min_count=1)
        .unstack(level=1)
        .reindex(columns=range(1, 13))
    )

    current_map: Dict[int, float] = {}
    if selected_year in monthly_pivot.index:
        row_series = monthly_pivot.loc[selected_year]
        for m_idx in range(1, 13):
            raw_v = row_series[m_idx]
            current_map[m_idx] = float(raw_v) if pd.notna(raw_v) else float("nan")
    else:
        for m_idx in range(1, 13):
            current_map[m_idx] = float("nan")

    valid_months = {m: v for m, v in current_map.items() if not np.isnan(v)}
    if valid_months:
        total_detections: Optional[int] = int(sum(valid_months.values()))
        peak_month_idx = max(valid_months, key=lambda k: valid_months[k])
        peak_month: Optional[str] = MONTH_LABELS[peak_month_idx - 1]
    else:
        total_detections = None
        peak_month = None

    ref_pivot, ref_period_label = _get_reference_subset(monthly_pivot, selected_year, fixed_demo_reference)

    rows = []
    for m_idx in range(1, 13):
        m_name = MONTH_LABELS[m_idx - 1]
        curr_val = current_map[m_idx]

        ref_col = (
            ref_pivot[m_idx].dropna()
            if m_idx in ref_pivot.columns
            else pd.Series(dtype=float)
        )
        valid_years_count = int(len(ref_col))

        if valid_years_count >= 3:
            med_val = float(ref_col.median())
            mean_val = float(ref_col.mean())
            min_val = float(ref_col.min())
            max_val = float(ref_col.max())

            if not np.isnan(curr_val) and med_val > 0:
                diff_pct = ((curr_val - med_val) / med_val) * 100.0
            else:
                diff_pct = float("nan")
        else:
            med_val = float("nan")
            mean_val = float("nan")
            min_val = float("nan")
            max_val = float("nan")
            diff_pct = float("nan")

        rows.append(
            {
                "Month": m_name,
                "Detections": curr_val,
                "Reference Period": ref_period_label,
                "Valid Reference Years": valid_years_count,
                "Historical Median": med_val,
                "Historical Mean": mean_val,
                "Historical Min": min_val,
                "Historical Max": max_val,
                "Difference vs Median (%)": diff_pct,
            }
        )

    stats_df = pd.DataFrame(rows)
    return monthly_pivot, stats_df, total_detections, peak_month, ref_period_label


def render_calendar(region_df: pd.DataFrame, selected_year: int) -> None:
    """
    Render the historical calendar view for a selected region and year.
    """
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown("## Historical Activity Calendar")

    if region_df.empty:
        st.warning("No data available for the selected region.")
        return

    monthly_pivot, stats_df, total_detections, peak_month, ref_period_label = (
        compute_monthly_statistics(region_df, selected_year)
    )

    all_years = [str(y) for y in sorted(monthly_pivot.index.tolist())]

    # 1. Heatmap (gaps preserved as empty cells)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=monthly_pivot.to_numpy(dtype=float),
            x=MONTH_LABELS,
            y=all_years,
            colorscale="YlOrRd",
            colorbar=dict(title="Detections"),
            hoverongaps=False,
        )
    )
    heatmap_fig.update_layout(
        title="Monthly Detection Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis=dict(type="category"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    st.plotly_chart(heatmap_fig, width="stretch")
    st.caption("Empty cells indicate months with no recorded observations (not zero detections).")

    # 2. Trend Analysis for Selected Year
    st.markdown(f"### Trend Analysis ({selected_year})")

    year_values = stats_df["Detections"].tolist()
    valid_months_count = int(stats_df["Detections"].notna().sum())

    col1, col2 = st.columns(2)
    if total_detections is not None and peak_month is not None:
        total_label = (
            f"Total Detections ({selected_year})"
            if valid_months_count == 12
            else f"Sum of Available Months ({valid_months_count}/12)"
        )
        col1.metric(total_label, f"{total_detections:,}")
        col2.metric("Peak Activity Month", peak_month)
    else:
        col1.metric(f"Total Detections ({selected_year})", "No data")
        col2.metric("Peak Activity Month", "No data")

    line_fig = go.Figure(
        data=go.Scatter(
            x=MONTH_LABELS,
            y=year_values,
            mode="lines+markers",
            name=str(selected_year),
            line=dict(color="#d95f02", width=3),
            marker=dict(size=8),
            connectgaps=False,
        )
    )
    line_fig.update_layout(
        title=f"Monthly Detections in {selected_year}",
        xaxis_title="Month",
        yaxis_title="Detection Count",
        margin=dict(l=40, r=40, t=40, b=40),
        height=340,
    )
    st.plotly_chart(line_fig, width="stretch")

    # 3. Historical Baseline Comparison Table
    st.markdown("### Historical Baseline & Statistics Comparison")
    st.caption(
        f"Reference period: {ref_period_label}. A minimum of 3 observed years per month is required "
        "to compute baseline statistics. Explicit zeros count as valid observations."
    )

    display_df = pd.DataFrame()
    display_df["Month"] = stats_df["Month"]
    display_df[f"Detections ({selected_year})"] = stats_df["Detections"].apply(
        lambda v: f"{int(v):,}" if pd.notna(v) else "No data"
    )
    display_df["Reference Period"] = stats_df["Reference Period"]
    display_df["Years with Data"] = stats_df["Valid Reference Years"]

    for col in ["Historical Median", "Historical Mean", "Historical Min", "Historical Max"]:
        display_df[col] = stats_df[col].apply(
            lambda v: f"{v:,.1f}" if pd.notna(v) else "Insufficient history"
        )

    display_df["Difference vs Median"] = stats_df.apply(
        lambda r: (
            f"{r['Difference vs Median (%)']:+.1f}%"
            if pd.notna(r["Difference vs Median (%)"])
            else ("Insufficient history" if r["Valid Reference Years"] < 3 else "")
        ),
        axis=1,
    )

    st.dataframe(display_df, hide_index=True, width="stretch")

    # Export clean numeric CSV (empty cells for NaN, real zeros preserved)
    export_df = stats_df.rename(columns={"Detections": f"Detections ({selected_year})"})
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Monthly Summary (CSV)",
        data=csv_bytes,
        file_name=f"historical_summary_{selected_year}.csv",
        mime="text/csv",
    )
