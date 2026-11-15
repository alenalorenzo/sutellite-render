"""
Data Adapter Module for Satellite Historical Analysis.

Harmonizes satellite detection datasets across three supported schemas:
1. Platform monthly summary (year, month, region_id, observations)
2. Monthly spatial grid (year, month, detections)
3. Canonical observation-level records (acq_date, region_id, processing_stream)
Also supports raw NASA FIRMS Parquet/CSV files with instrument/satellite aliases.
"""

from typing import Any, Optional
import pandas as pd


PLATFORM_ALIASES = {
    "NOAA20": "NOAA-20", "N20": "NOAA-20", "J1": "NOAA-20", "JPSS1": "NOAA-20",
    "NOAA21": "NOAA-21", "N21": "NOAA-21", "J2": "NOAA-21", "JPSS2": "NOAA-21",
    "1": "NOAA-20", "1.0": "NOAA-20", "2": "NOAA-21", "2.0": "NOAA-21",
    "SUOMINPP": "SNPP", "NPP": "SNPP", "N": "SNPP", "T": "TERRA", "A": "AQUA",
}


def _source_key(values):
    return values.astype("string").str.upper().str.replace(r"[-_\s]+", "", regex=True)


def normalize_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and FIRMS aliases while retaining spatial columns."""
    frame = frame.copy(deep=False)
    frame.columns = [str(col).lower().strip() for col in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("The file has duplicate column names.")
    for raw, canonical in (("instrument", "sensor"), ("satellite", "platform")):
        if canonical not in frame and raw in frame:
            frame[canonical] = frame[raw]
    if "sensor" in frame:
        frame["sensor"] = _source_key(frame["sensor"]).replace({"SNPP": "VIIRS", "NPP": "VIIRS"})
    if "platform" in frame:
        frame["platform"] = _source_key(frame["platform"]).replace(PLATFORM_ALIASES)
    if "processing_stream" in frame:
        frame["processing_stream"] = _source_key(frame["processing_stream"])
    return frame


def load_data(
    uploaded_file: Any,
    region: Optional[str] = None,
    sensor: Optional[str] = "VIIRS",
    platform: Optional[str] = "SNPP",
    processing_stream: Optional[str] = "ARCHIVE",
) -> pd.DataFrame:
    """
    Load and validate a CSV or Parquet file into a standardized monthly/daily schema.

    Args:
        uploaded_file: File path (str/Path) or Streamlit UploadedFile object.
        region: Explicit region identifier required when the dataset lacks a region column.
        sensor: Sensor filter for multi-sensor datasets (default: 'VIIRS').
        platform: Platform filter for multi-platform datasets (default: 'SNPP').
        processing_stream: Stream filter for canonical records (default: 'ARCHIVE').

    Returns:
        pd.DataFrame: Standardized DataFrame with columns ['region', 'date', 'hotspot_count'].

    Raises:
        ValueError: If schema is ambiguous, required fields are missing, or values are invalid.
    """
    file_name: str = getattr(uploaded_file, "name", str(uploaded_file))

    if file_name.lower().endswith(".parquet"):
        df: pd.DataFrame = pd.read_parquet(uploaded_file)
    else:
        df: pd.DataFrame = pd.read_csv(uploaded_file)

    return adapt_data(df, region, sensor, platform, processing_stream)


def adapt_data(
    frame: pd.DataFrame,
    region: Optional[str] = None,
    sensor: Optional[str] = None,
    platform: Optional[str] = None,
    processing_stream: Optional[str] = None,
    *,
    aggregate_months: bool = True,
) -> pd.DataFrame:
    """Validate calendar rows. The spatial workflow can keep one row per input."""
    df = normalize_source_columns(frame)

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    # Resolve region column or explicit parameter
    region_col = next(
        (c for c in ["region_id", "region", "country", "adm0"] if c in df.columns),
        None,
    )
    if region_col:
        if df[region_col].isna().any() or (df[region_col].astype(str).str.strip() == "").any():
            raise ValueError(f"Invalid data: empty or null values found in '{region_col}'.")
        df["region"] = df[region_col].astype(str).str.strip()
    elif region and str(region).strip():
        df["region"] = str(region).strip()
    else:
        raise ValueError(
            "Missing region identifier: dataset lacks 'region_id'/'region' and no explicit region was provided."
        )

    for column, selected in (("sensor", sensor), ("platform", platform), ("processing_stream", processing_stream)):
        if column in df and selected:
            key = _source_key(pd.Series([selected])).iat[0]
            if column == "platform":
                key = _source_key(pd.Series([PLATFORM_ALIASES.get(key, key)])).iat[0]
            elif column == "sensor" and key in {"SNPP", "NPP"}:
                key = "VIIRS"
            df = df[_source_key(df[column]) == key].copy()

    if df.empty:
        return pd.DataFrame(columns=["region", "date", "hotspot_count"])

    is_monthly_schema = "year" in df.columns and "month" in df.columns
    is_canonical_schema = "acq_date" in df.columns or "date" in df.columns

    if is_monthly_schema:
        years = pd.to_numeric(df["year"], errors="coerce")
        months = pd.to_numeric(df["month"], errors="coerce")

        if years.isna().any() or (years % 1 != 0).any():
            bad_val = df.loc[years.isna() | (years % 1 != 0), "year"].iloc[0]
            raise ValueError(f"Invalid year value detected: '{bad_val}'.")

        if months.isna().any() or (months % 1 != 0).any() or (~months.between(1, 12)).any():
            bad_val = df.loc[
                months.isna() | (months % 1 != 0) | (~months.between(1, 12)), "month"
            ].iloc[0]
            raise ValueError(f"Invalid month value detected (must be 1-12): '{bad_val}'.")

        date_strings = (
            years.astype(int).astype(str)
            + "-"
            + months.astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        df["date"] = pd.to_datetime(date_strings, format="%Y-%m-%d", errors="raise")

        count_col = next(
            (c for c in ["observations", "detections", "hotspot_count", "count"] if c in df.columns),
            None,
        )
        if not count_col:
            raise ValueError(
                "Ambiguous monthly schema: missing count column ('observations', 'detections', 'hotspot_count', or 'count')."
            )

        df["hotspot_count"] = _validate_non_negative_integers(df[count_col], count_col)
        if not aggregate_months:
            return df[["region", "date", "hotspot_count"]].reset_index(drop=True)

        result = (
            df.groupby(["region", "date"], as_index=False)
            .agg(hotspot_count=("hotspot_count", "sum"))
            .sort_values(["region", "date"])
        )
        return result[["region", "date", "hotspot_count"]].reset_index(drop=True)

    elif is_canonical_schema:
        date_col = "acq_date" if "acq_date" in df.columns else "date"
        raw_dates = df[date_col]
        parsed_dates = pd.to_datetime(raw_dates.astype("string"), errors="coerce", format="mixed", utc=True)

        if parsed_dates.isna().any():
            bad_date = raw_dates.loc[parsed_dates.isna()].iloc[0]
            raise ValueError(f"Invalid date value detected in '{date_col}': '{bad_date}'.")

        df["date"] = parsed_dates.dt.tz_convert(None)

        count_col = next(
            (c for c in ["hotspot_count", "count", "observations", "detections"] if c in df.columns),
            None,
        )
        if count_col:
            df["hotspot_count"] = _validate_non_negative_integers(df[count_col], count_col)
        else:
            df["hotspot_count"] = 1

        return df[["region", "date", "hotspot_count"]].reset_index(drop=True)

    else:
        raise ValueError(
            "Unrecognized dataset schema: expected ('year', 'month') or ('acq_date'/'date')."
        )


def _validate_non_negative_integers(series: pd.Series, col_name: str) -> pd.Series:
    """Validate that a series contains only non-negative integers without nulls or text."""
    if series.isna().any():
        raise ValueError(f"Invalid count in '{col_name}': empty or null values are not allowed.")

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.isna().any():
        bad_val = series.loc[numeric_series.isna()].iloc[0]
        raise ValueError(f"Invalid count in '{col_name}': non-numeric text '{bad_val}' detected.")

    if (numeric_series < 0).any():
        bad_val = numeric_series.loc[numeric_series < 0].iloc[0]
        raise ValueError(f"Invalid count in '{col_name}': negative value '{bad_val}' detected.")

    if (numeric_series % 1 != 0).any():
        bad_val = numeric_series.loc[numeric_series % 1 != 0].iloc[0]
        raise ValueError(f"Invalid count in '{col_name}': fractional value '{bad_val}' detected.")

    if (numeric_series >= 2**63).any():
        raise ValueError(f"Invalid count in '{col_name}': use integers below 2⁶³.")
    return numeric_series.astype("int64")
