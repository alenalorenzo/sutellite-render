"""Build a harmonized MODIS/VIIRS monthly hotspot series for the calendar app.

The output is a single, consistent series of *VIIRS S-NPP-equivalent* detection
records per 0.1-degree cell and calendar month, from the first MODIS Terra data
(Nov 2000) to the latest month in the input.

Method (deliberately simple and auditable):

* Reference platform: VIIRS S-NPP only. NOAA-20 and NOAA-21 are NOT added, so
  the series does not jump when more VIIRS satellites come online.
* From the first S-NPP month onward, the series is the observed S-NPP count.
* Before that, MODIS counts are scaled by a regional calibration ratio
  (S-NPP / MODIS) estimated on the overlap years, per region, calendar month and
  (day and night pooled, because night-time MODIS counts are too small for a
  stable ratio). Months with only Terra use a Terra-only ratio. Where a
  region/month ratio has too few overlap years or an unstable band, the
  region-level ratio is used instead.
* An uncertainty band comes from the 10th/90th percentile of the annual ratios
  in the overlap period.
* A leave-one-year-out cross-validation on the overlap years reports how well
  the ratio predicts regional monthly S-NPP totals (calibration_validation.csv).

This is NOT the Ridge/Poisson count-mapping model. That model uses calendar
year and coordinates as features and was trained on 2012-2022, so it must not be
extrapolated to 2000-2011. A scalar regional ratio can be, with a visible flag.

Limitations (keep them in any report or presentation):

* Counts are source observation records, not unique fires.
* One ratio per region/month/day-night is applied to every cell in the region;
  cell-level values before 2012 are proportional allocations, not estimates.
  Use the series aggregated over an area of interest.
* The ratio is not constant with fire activity or over MODIS sensor aging.
* After S-NPP starts, a cell with MODIS records but no S-NPP record counts 0.
* No observation-opportunity denominator, no cross-sensor matching.

Usage:
    python -m src.build_harmonized_calendar \
        --dataset-file processed/harmonization_global/global_monthly_cell_counts.parquet \
        --output-dir processed/harmonized_calendar
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HARMONIZATION_VERSION = "1.1.0"

DEFAULT_DATASET_FILE = Path(
    "processed/harmonization_global/global_monthly_cell_counts.parquet"
)
DEFAULT_OUTPUT_DIR = Path("processed/harmonized_calendar")
OUTPUT_FILE_NAME = "harmonized_cell_month.parquet"

DEFAULT_CALIBRATION_START_YEAR = 2013
DEFAULT_CALIBRATION_END_YEAR = 2022
DEFAULT_MIN_CALIBRATION_YEARS = 3
# Month ratios whose 10th-90th percentile band is wider than this factor are
# treated as unstable and replaced by the region-level ratio.
DEFAULT_MAX_BAND_RATIO = 3.0

AQUA = "modis_aqua_count"
TERRA = "modis_terra_count"
SNPP = "viirs_snpp_count"

CELL_KEYS = [
    "region_id",
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "grid_lat_center_deg",
    "grid_lon_center_deg",
]
INPUT_COLUMNS = [
    "region_id",
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "grid_lat_center_deg",
    "grid_lon_center_deg",
    AQUA,
    TERRA,
    SNPP,
]
OUTPUT_COLUMNS = [
    "region_id",
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "grid_lat_center_deg",
    "grid_lon_center_deg",
    "hotspots_harmonized",
    "hotspots_low",
    "hotspots_high",
    "modis_count",
    "snpp_count",
    "is_calibrated",
    "source",
]
FACTOR_COLUMNS = [
    "level",
    "region_id",
    "month",
    "modis_config",
    "ratio",
    "ratio_low",
    "ratio_high",
    "n_years",
]
# month == 0 in the factor table means "all calendar months" (fallback level).
LEVEL_MONTHLY = "region_month"
LEVEL_FALLBACK = "region"

SOURCE_OBSERVED = "VIIRS_SNPP_OBSERVED"
SOURCE_CALIBRATED = "MODIS_CALIBRATED"
SOURCE_UNCALIBRATED = "UNCALIBRATED"

LIMITATIONS = [
    "Counts are source observation records, not unique fire events.",
    "Reference series is VIIRS S-NPP only; NOAA-20/21 are not added.",
    "Before S-NPP starts, MODIS counts are scaled by a regional "
    "month ratio estimated on the overlap years; use area aggregates.",
    "The ratio is not constant with fire activity or over sensor aging.",
    "No observation-opportunity denominator and no cross-sensor matching.",
    "The 0.1-degree grid is angular, not equal-area.",
]


def month_index(year: Any, month: Any) -> Any:
    """Return a monotonic integer month index (works on scalars and Series)."""

    return year * 12 + (month - 1)


def _month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


# --------------------------------------------------------------------------
# Pure logic (no Parquet I/O): platform starts, calibration, harmonization
# --------------------------------------------------------------------------


def detect_platform_starts(totals: pd.DataFrame) -> dict[str, int]:
    """Find the first month index with detections for Terra, Aqua and S-NPP.

    ``totals`` needs year, month and the three platform count columns. It may be
    already summed over cells; the global start is what matters, so it must
    cover all regions, not a single one.
    """

    required = {"year", "month", AQUA, TERRA, SNPP}
    missing = sorted(required - set(totals.columns))
    if missing:
        raise ValueError(f"Missing columns for platform start detection: {missing}")

    idx = month_index(totals["year"], totals["month"])
    starts: dict[str, int] = {}
    for platform, column in (("TERRA", TERRA), ("AQUA", AQUA), ("SNPP", SNPP)):
        present = idx[totals[column] > 0]
        if present.empty:
            raise ValueError(f"No detections found for platform {platform}.")
        starts[platform] = int(present.min())
    return starts


def _ratio_stats(
    frame: pd.DataFrame,
    keys: list[str],
    num: str,
    den: str,
    min_years: int,
    max_band_ratio: float | None = None,
) -> pd.DataFrame:
    """Pooled ratio plus 10th/90th percentile of annual ratios per key group.

    ``frame`` must hold one row per ``keys`` + ``year``. Groups with fewer than
    ``min_years`` years, or (when ``max_band_ratio`` is given) with an unstable
    band, get NaN ratios.
    """

    out_columns = [*keys, "n_years", "ratio", "ratio_low", "ratio_high"]
    use = frame[frame[den] > 0].copy()
    if use.empty:
        return pd.DataFrame(columns=out_columns)
    use["_annual_ratio"] = use[num] / use[den]
    out = (
        use.groupby(keys)
        .agg(
            n_years=("year", "nunique"),
            _num=(num, "sum"),
            _den=(den, "sum"),
            _p10=("_annual_ratio", lambda s: float(s.quantile(0.10))),
            _p90=("_annual_ratio", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
    )
    out["ratio"] = out["_num"] / out["_den"]
    out["ratio_low"] = np.minimum(out["_p10"], out["ratio"])
    out["ratio_high"] = np.maximum(out["_p90"], out["ratio"])
    bad = out["n_years"] < min_years
    if max_band_ratio is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            band = out["ratio_high"] / out["ratio_low"]
        bad |= (out["ratio_low"] <= 0) | (band > max_band_ratio)
    out.loc[bad, ["ratio", "ratio_low", "ratio_high"]] = np.nan
    return out.loc[:, out_columns]


def compute_calibration_factors(
    cells: pd.DataFrame,
    *,
    start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    min_years: int = DEFAULT_MIN_CALIBRATION_YEARS,
    max_band_ratio: float = DEFAULT_MAX_BAND_RATIO,
    exclude_years: Iterable[int] = (),
) -> pd.DataFrame:
    """Estimate S-NPP / MODIS ratios on the overlap years.

    Day and night are pooled. Returns a table with FACTOR_COLUMNS. Rows with
    ``month == 0`` are the region-level fallback factors. ``exclude_years`` is
    used by the leave-one-year-out validation.
    """

    window = cells[(cells["year"] >= start_year) & (cells["year"] <= end_year)]
    window = window[~window["year"].isin(list(exclude_years))]
    if window.empty:
        raise ValueError(
            f"No cell rows in calibration window {start_year}-{end_year}."
        )

    monthly = window.groupby(["region_id", "month", "year"], as_index=False)[
        [AQUA, TERRA, SNPP]
    ].sum()
    monthly["modis_both"] = monthly[AQUA] + monthly[TERRA]
    fallback = window.groupby(["region_id", "year"], as_index=False)[
        [AQUA, TERRA, SNPP]
    ].sum()
    fallback["modis_both"] = fallback[AQUA] + fallback[TERRA]

    tables: list[pd.DataFrame] = []
    for config, den in (("TERRA_AQUA", "modis_both"), ("TERRA", TERRA)):
        m = _ratio_stats(
            monthly, ["region_id", "month"], SNPP, den, min_years, max_band_ratio
        )
        m["level"] = LEVEL_MONTHLY
        m["modis_config"] = config
        f = _ratio_stats(fallback, ["region_id"], SNPP, den, min_years)
        f["level"] = LEVEL_FALLBACK
        f["month"] = 0
        f["modis_config"] = config
        tables.extend([m, f])

    factors = pd.concat(tables, ignore_index=True)
    factors["month"] = factors["month"].astype(int)
    return (
        factors.loc[:, FACTOR_COLUMNS]
        .sort_values(["level", "region_id", "modis_config", "month"])
        .reset_index(drop=True)
    )


def _attach_ratios(
    frame: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """Merge month-level factors, then fill gaps from the region-level ones."""

    ratio_cols = ["ratio", "ratio_low", "ratio_high"]
    monthly = factors[factors["month"] > 0].drop(columns=["level", "n_years"])
    fallback = factors[factors["month"] == 0].drop(
        columns=["level", "month", "n_years"]
    )
    out = frame.merge(
        monthly, on=["region_id", "month", "modis_config"], how="left"
    ).merge(
        fallback,
        on=["region_id", "modis_config"],
        how="left",
        suffixes=("", "_fb"),
    )
    for col in ratio_cols:
        out[col] = out[col].fillna(out[f"{col}_fb"])
    return out.drop(columns=[f"{c}_fb" for c in ratio_cols])


def harmonize_cells(
    cells: pd.DataFrame, factors: pd.DataFrame, starts: dict[str, int]
) -> pd.DataFrame:
    """Turn platform counts into the harmonized cell-month series."""

    missing = sorted(set(INPUT_COLUMNS) - set(cells.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    # Day and night are pooled before calibration.
    d = (
        cells.loc[:, INPUT_COLUMNS]
        .groupby(CELL_KEYS, as_index=False)[[AQUA, TERRA, SNPP]]
        .sum()
    )
    idx = month_index(d["year"], d["month"])
    d["modis_config"] = np.where(idx >= starts["AQUA"], "TERRA_AQUA", "TERRA")
    d["modis_count"] = d[AQUA] + d[TERRA]
    d = _attach_ratios(d, factors)

    # Recomputed after the merges, which do not preserve row order.
    observed = month_index(d["year"], d["month"]) >= starts["SNPP"]
    snpp = d[SNPP].astype(float)
    modis = d["modis_count"].astype(float)
    d["hotspots_harmonized"] = np.where(observed, snpp, modis * d["ratio"])
    d["hotspots_low"] = np.where(observed, snpp, modis * d["ratio_low"])
    d["hotspots_high"] = np.where(observed, snpp, modis * d["ratio_high"])
    uncalibrated = ~observed & d["ratio"].isna()

    d["snpp_count"] = d[SNPP].astype("int64")
    d["modis_count"] = d["modis_count"].astype("int64")
    d["is_calibrated"] = ~observed
    d["source"] = np.where(
        uncalibrated,
        SOURCE_UNCALIBRATED,
        np.where(observed, SOURCE_OBSERVED, SOURCE_CALIBRATED),
    )
    return d.loc[:, OUTPUT_COLUMNS]


# --------------------------------------------------------------------------
# Validation: leave-one-year-out on the overlap years
# --------------------------------------------------------------------------


def cross_validate_calibration(
    cells: pd.DataFrame,
    *,
    start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    min_years: int = DEFAULT_MIN_CALIBRATION_YEARS,
    max_band_ratio: float = DEFAULT_MAX_BAND_RATIO,
) -> pd.DataFrame:
    """Predict held-out regional monthly S-NPP totals from MODIS counts.

    For each overlap year, factors are re-estimated without that year and used
    to predict that year's S-NPP total per region and month. The unit is the
    region/month aggregate, the level at which the ratios are meant to be used.
    """

    window = cells[(cells["year"] >= start_year) & (cells["year"] <= end_year)]
    if window.empty:
        raise ValueError(
            f"No cell rows in validation window {start_year}-{end_year}."
        )
    totals = window.groupby(["region_id", "month", "year"], as_index=False)[
        [AQUA, TERRA, SNPP]
    ].sum()

    parts: list[pd.DataFrame] = []
    for year in sorted(totals["year"].unique()):
        factors = compute_calibration_factors(
            window,
            start_year=start_year,
            end_year=end_year,
            min_years=min_years,
            max_band_ratio=max_band_ratio,
            exclude_years=[int(year)],
        )
        held_out = totals[totals["year"] == year]
        for config, modis in (
            ("TERRA_AQUA", held_out[AQUA] + held_out[TERRA]),
            ("TERRA", held_out[TERRA]),
        ):
            frame = held_out[["region_id", "month", "year", SNPP]].copy()
            frame["modis_config"] = config
            frame["modis_count"] = modis.to_numpy()
            frame = _attach_ratios(frame, factors)
            frame["actual"] = frame[SNPP].astype(float)
            frame["predicted"] = frame["modis_count"] * frame["ratio"]
            frame["low"] = frame["modis_count"] * frame["ratio_low"]
            frame["high"] = frame["modis_count"] * frame["ratio_high"]
            parts.append(
                frame[
                    [
                        "region_id",
                        "modis_config",
                        "month",
                        "year",
                        "actual",
                        "predicted",
                        "low",
                        "high",
                    ]
                ]
            )

    detail = pd.concat(parts, ignore_index=True)
    inside = (detail["actual"] >= detail["low"]) & (
        detail["actual"] <= detail["high"]
    )
    # 1.0 / 0.0, or NaN when no prediction was possible.
    detail["within_band"] = np.where(
        detail["predicted"].isna(), np.nan, inside.astype(float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        detail["abs_pct_error"] = np.where(
            detail["actual"] > 0,
            (detail["predicted"] - detail["actual"]).abs() / detail["actual"],
            np.nan,
        )
        detail["signed_pct_error"] = np.where(
            detail["actual"] > 0,
            (detail["predicted"] - detail["actual"]) / detail["actual"],
            np.nan,
        )
    return detail


def summarize_validation(detail: pd.DataFrame) -> pd.DataFrame:
    """Median/90th-percentile error and band coverage per region and config."""

    grouped = detail.groupby(["region_id", "modis_config"])
    out = grouped.agg(
        n_region_months=("abs_pct_error", "count"),
        median_abs_pct_error=("abs_pct_error", "median"),
        median_signed_pct_error=("signed_pct_error", "median"),
        band_coverage=("within_band", "mean"),
    )
    out["p90_abs_pct_error"] = grouped["abs_pct_error"].quantile(0.90)
    out = out.reset_index()
    return out.loc[
        :,
        [
            "region_id",
            "modis_config",
            "n_region_months",
            "median_abs_pct_error",
            "p90_abs_pct_error",
            "median_signed_pct_error",
            "band_coverage",
        ],
    ]


# --------------------------------------------------------------------------
# Parquet I/O (needs pyarrow, which the project already requires)
# --------------------------------------------------------------------------


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pyarrow is required. Install it with: pip install -r requirements.txt"
        ) from exc
    return pa, pq


def list_regions(dataset_file: Path) -> list[str]:
    _, pq = _require_pyarrow()
    column = pq.read_table(dataset_file, columns=["region_id"]).column("region_id")
    return sorted({str(v) for v in column.unique().to_pylist() if v is not None})


def read_global_platform_totals(dataset_file: Path) -> pd.DataFrame:
    """Sum platform counts by year and month over ALL regions, batch by batch."""

    _, pq = _require_pyarrow()
    parquet = pq.ParquetFile(dataset_file)
    parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(
        columns=["year", "month", AQUA, TERRA, SNPP], batch_size=1_000_000
    ):
        frame = batch.to_pandas()
        parts.append(frame.groupby(["year", "month"], as_index=False).sum())
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["year", "month"], as_index=False)
        .sum()
    )


def read_region_cells(dataset_file: Path, region_id: str) -> pd.DataFrame:
    _, pq = _require_pyarrow()
    table = pq.read_table(
        dataset_file,
        columns=INPUT_COLUMNS,
        filters=[("region_id", "=", region_id)],
    )
    return table.to_pandas()


def _round_or_none(value: Any) -> float | None:
    return None if pd.isna(value) else round(float(value), 4)


def build_harmonized_calendar(
    dataset_file: Path,
    output_dir: Path,
    *,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    min_calibration_years: int = DEFAULT_MIN_CALIBRATION_YEARS,
    max_band_ratio: float = DEFAULT_MAX_BAND_RATIO,
) -> dict[str, Any]:
    """Run the full build, region by region, and write the three outputs."""

    pa, pq = _require_pyarrow()
    dataset_file = Path(dataset_file)
    output_dir = Path(output_dir)
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Move it or choose a different --output-dir."
        )

    starts = detect_platform_starts(read_global_platform_totals(dataset_file))
    regions = list_regions(dataset_file)

    output_dir.mkdir(parents=True)
    parquet_path = output_dir / OUTPUT_FILE_NAME
    writer = None
    schema = None
    factor_tables: list[pd.DataFrame] = []
    validation_tables: list[pd.DataFrame] = []
    region_stats: dict[str, dict[str, int]] = {}

    try:
        for region in regions:
            cells = read_region_cells(dataset_file, region)
            factors = compute_calibration_factors(
                cells,
                start_year=calibration_start_year,
                end_year=calibration_end_year,
                min_years=min_calibration_years,
                max_band_ratio=max_band_ratio,
            )
            validation_tables.append(
                cross_validate_calibration(
                    cells,
                    start_year=calibration_start_year,
                    end_year=calibration_end_year,
                    min_years=min_calibration_years,
                    max_band_ratio=max_band_ratio,
                )
            )
            result = harmonize_cells(cells, factors, starts)
            result = result.sort_values(
                ["grid_lat_index", "grid_lon_index", "year", "month"]
            )
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
            writer.write_table(table.cast(schema))

            factor_tables.append(factors)
            region_stats[region] = {
                "cell_month_rows": int(len(result)),
                "calibrated_rows": int(result["is_calibrated"].sum()),
                "uncalibrated_rows": int(
                    (result["source"] == SOURCE_UNCALIBRATED).sum()
                ),
            }
    finally:
        if writer is not None:
            writer.close()

    all_factors = pd.concat(factor_tables, ignore_index=True)
    all_factors.to_csv(output_dir / "calibration_factors.csv", index=False)

    detail = pd.concat(validation_tables, ignore_index=True)
    detail.to_csv(output_dir / "calibration_validation_detail.csv", index=False)
    validation = summarize_validation(detail)
    validation.to_csv(output_dir / "calibration_validation.csv", index=False)
    overall: dict[str, dict[str, float | int | None]] = {}
    for config in ("TERRA_AQUA", "TERRA"):
        sub = summarize_validation(detail[detail["modis_config"] == config].assign(
            region_id="ALL"
        ))
        row = sub.iloc[0]
        overall[config] = {
            "n_region_months": int(row["n_region_months"]),
            "median_abs_pct_error": _round_or_none(row["median_abs_pct_error"]),
            "p90_abs_pct_error": _round_or_none(row["p90_abs_pct_error"]),
            "band_coverage": _round_or_none(row["band_coverage"]),
        }

    summary = {
        "harmonization_version": HARMONIZATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_file_name": dataset_file.name,
        "reference_platform": "VIIRS S-NPP",
        "series_start": _month_label(min(starts.values())),
        "platform_first_month": {k: _month_label(v) for k, v in starts.items()},
        "calibration": {
            "start_year": calibration_start_year,
            "end_year": calibration_end_year,
            "min_years": min_calibration_years,
            "max_band_ratio": max_band_ratio,
            "day_night": "pooled",
            "uncertainty_band": "10th-90th percentile of annual ratios",
            "factor_levels": [LEVEL_MONTHLY, LEVEL_FALLBACK],
        },
        "validation": {
            "method": "leave-one-year-out on the overlap years, "
            "region/month S-NPP totals",
            "overall": overall,
        },
        "regions": region_stats,
        "output_files": [
            OUTPUT_FILE_NAME,
            "calibration_factors.csv",
            "calibration_validation.csv",
            "calibration_validation_detail.csv",
        ],
        "limitations": LIMITATIONS,
    }
    with (output_dir / "harmonization_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the harmonized (S-NPP-equivalent) monthly hotspot series."
    )
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calibration-start-year", type=int, default=DEFAULT_CALIBRATION_START_YEAR
    )
    parser.add_argument(
        "--calibration-end-year", type=int, default=DEFAULT_CALIBRATION_END_YEAR
    )
    parser.add_argument(
        "--min-calibration-years", type=int, default=DEFAULT_MIN_CALIBRATION_YEARS
    )
    parser.add_argument(
        "--max-band-ratio", type=float, default=DEFAULT_MAX_BAND_RATIO
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_harmonized_calendar(
        args.dataset_file,
        args.output_dir,
        calibration_start_year=args.calibration_start_year,
        calibration_end_year=args.calibration_end_year,
        min_calibration_years=args.min_calibration_years,
        max_band_ratio=args.max_band_ratio,
    )
    total = sum(r["cell_month_rows"] for r in summary["regions"].values())
    print(f"Wrote {total:,} cell-month rows to {args.output_dir}")
    for config, stats in summary["validation"]["overall"].items():
        print(
            f"Validation {config}: median abs error "
            f"{stats['median_abs_pct_error']}, band coverage "
            f"{stats['band_coverage']} (n={stats['n_region_months']})"
        )


if __name__ == "__main__":
    main()