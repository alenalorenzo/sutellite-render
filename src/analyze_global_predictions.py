"""Generate diagnostics for the global v2 sensor-harmonisation test predictions.

Expected input columns are produced by src/sensor_harmonization_global.py:
region_id, year, month, grid_lat_index, grid_lon_index, daynight,
viirs_count, prediction_naive_modis,
prediction_region_month_daynight_mean, prediction_ridge_smearing.

The script evaluates aggregate detection-record counts in paired MODIS/VIIRS
region-month-grid-day/night groups. It does not evaluate unique fire events.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("sutellite.global_diagnostics")

DEFAULT_INPUT = Path(
    "processed/harmonization_global/model/test_predictions.parquet"
)

DEFAULT_OUTPUT = Path(
    "processed/harmonization_global/model/diagnostics"
)

ACTUAL = "viirs_count"

PREDICTIONS = {
    "naive_modis": "prediction_naive_modis",
    "region_month_daynight_mean": "prediction_region_month_daynight_mean",
    "ridge_smearing": "prediction_ridge_smearing",
}

KEY_COLUMNS = [
    "region_id",
    "year",
    "month",
    "grid_lat_index",
    "grid_lon_index",
    "daynight",
]

GROUP_REPORTS = {
    "by_region.csv": ["region_id"],
    "by_month.csv": ["month"],
    "by_daynight.csv": ["daynight"],
    "by_region_month.csv": ["region_id", "month"],
    "by_region_month_daynight.csv": [
        "region_id",
        "month",
        "daynight",
    ],
    "by_year.csv": ["year"],
}


def _require_columns(df: pd.DataFrame) -> None:
    """Raise an informative error when required columns are missing."""
    required = KEY_COLUMNS + [ACTUAL, *PREDICTIONS.values()]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction file is missing required columns: "
            + ", ".join(missing)
        )


def _metric_aggregation_spec() -> dict[str, tuple[str, str]]:
    """Return aggregation instructions used by grouped_metrics."""
    spec: dict[str, tuple[str, str]] = {
        "n": (ACTUAL, "size"),
        "mean_actual": (ACTUAL, "mean"),
        "total_actual": (ACTUAL, "sum"),
    }

    for model in PREDICTIONS:
        spec[f"{model}_mae"] = (
            f"{model}_abs_error",
            "mean",
        )

        spec[f"{model}_mse"] = (
            f"{model}_squared_error",
            "mean",
        )

        spec[f"{model}_bias"] = (
            f"{model}_error",
            "mean",
        )

        spec[f"{model}_absolute_error_sum"] = (
            f"{model}_abs_error",
            "sum",
        )

    return spec


def _add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed, absolute and squared prediction errors."""
    result = df.copy()

    actual = (
        pd.to_numeric(result[ACTUAL], errors="coerce")
        .astype(float)
    )

    for model, prediction_column in PREDICTIONS.items():
        predicted = (
            pd.to_numeric(
                result[prediction_column],
                errors="coerce",
            )
            .astype(float)
        )

        error = predicted - actual

        result[f"{model}_error"] = error
        result[f"{model}_abs_error"] = error.abs()
        result[f"{model}_squared_error"] = error.pow(2)

    return result


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row containing comparable metrics for each model."""
    if df.empty:
        raise ValueError(
            "No valid prediction rows are available for metric calculation."
        )

    enriched = _add_error_columns(df)

    actual_total = float(enriched[ACTUAL].sum())

    result: dict[str, float | int] = {
        "n": int(len(enriched)),
        "mean_actual": float(enriched[ACTUAL].mean()),
        "total_actual": actual_total,
    }

    for model in PREDICTIONS:
        error = enriched[f"{model}_error"]

        absolute_error = enriched[f"{model}_abs_error"]

        squared_error = enriched[f"{model}_squared_error"]

        result[f"{model}_mae"] = float(
            absolute_error.mean()
        )

        result[f"{model}_rmse"] = float(
            np.sqrt(squared_error.mean())
        )

        result[f"{model}_bias"] = float(
            error.mean()
        )

        result[f"{model}_wape"] = (
            float(absolute_error.sum() / actual_total)
            if actual_total > 0
            else np.nan
        )

    return pd.DataFrame([result])


def grouped_metrics(
    df: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    """Calculate metrics per group without GroupBy.apply compatibility issues."""
    columns = list(group_columns)

    if not columns:
        return calculate_metrics(df)

    enriched = _add_error_columns(df)

    grouped = (
        enriched.groupby(
            columns,
            dropna=False,
            observed=True,
        )
        .agg(**_metric_aggregation_spec())
        .reset_index()
    )

    for model in PREDICTIONS:
        grouped[f"{model}_rmse"] = np.sqrt(
            grouped.pop(f"{model}_mse")
        )

        denominator = grouped["total_actual"].replace(
            0,
            np.nan,
        )

        grouped[f"{model}_wape"] = (
            grouped.pop(f"{model}_absolute_error_sum")
            / denominator
        )

    return grouped


def build_quality_report(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Summarize data-quality checks and return a valid-row mask.

    Returns:
        A tuple containing:
        - A one-row DataFrame with data-quality statistics.
        - A NumPy boolean array identifying rows valid for metrics.
    """
    required = KEY_COLUMNS + [ACTUAL, *PREDICTIONS.values()]

    numeric_columns = [ACTUAL, *PREDICTIONS.values()]

    numeric = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    complete_mask = df[required].notna().all(axis=1)

    finite_mask = np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all(axis=1)

    positive_actual = numeric[ACTUAL].gt(0).fillna(False)

    nonnegative_predictions = {
        model: numeric[column].ge(0).fillna(False)
        for model, column in PREDICTIONS.items()
    }

    report = {
        "rows_in_file": int(len(df)),
        "missing_required_cells": int(
            df[required].isna().sum().sum()
        ),
        "rows_missing_required_values": int(
            (~complete_mask).sum()
        ),
        "rows_with_non_finite_numeric_values": int(
            (~finite_mask).sum()
        ),
        "duplicate_full_grain_rows_after_first": int(
            df.duplicated(KEY_COLUMNS).sum()
        ),
        "negative_actual_count_rows": int(
            (numeric[ACTUAL] < 0).sum()
        ),
        "zero_actual_count_rows": int(
            (numeric[ACTUAL] == 0).sum()
        ),
    }

    for model, mask in nonnegative_predictions.items():
        missing_prediction_count = int(
            numeric[PREDICTIONS[model]].isna().sum()
        )

        report[f"negative_{model}_prediction_rows"] = int(
            (~mask).sum() - missing_prediction_count
        )

    # A paired supervised example must contain at least one VIIRS detection.
    valid_mask = (
        complete_mask.to_numpy()
        & finite_mask
        & positive_actual.to_numpy()
    )

    for mask in nonnegative_predictions.values():
        valid_mask &= mask.to_numpy()

    report["valid_rows_used_for_metrics"] = int(
        valid_mask.sum()
    )

    report["rows_excluded_from_metrics"] = int(
        len(df) - valid_mask.sum()
    )

    return pd.DataFrame([report]), valid_mask


def _activity_level(actual: pd.Series) -> pd.Series:
    """Classify observations by actual VIIRS detection count."""
    # Paired groups should have positive VIIRS counts. Including zero in the
    # lowest bin makes the diagnostic robust if an unexpected zero is present.
    return pd.cut(
        actual,
        bins=[-np.inf, 1, 5, np.inf],
        labels=[
            "Low (1)",
            "Medium (2-5)",
            "High (6+)",
        ],
        right=True,
        include_lowest=True,
        ordered=True,
    )


def _write_error_distribution(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write error-distribution statistics for each model."""
    rows = []

    for model in PREDICTIONS:
        error = df[f"{model}_error"].astype(float)

        absolute = error.abs()

        rows.append(
            {
                "model": model,
                "n": int(error.notna().sum()),
                "mean_error_bias": float(error.mean()),
                "median_error": float(error.median()),
                "mae": float(absolute.mean()),
                "error_p90": float(error.quantile(0.90)),
                "error_p95": float(error.quantile(0.95)),
                "error_p99": float(error.quantile(0.99)),
                "absolute_error_p90": float(
                    absolute.quantile(0.90)
                ),
                "absolute_error_p95": float(
                    absolute.quantile(0.95)
                ),
                "absolute_error_p99": float(
                    absolute.quantile(0.99)
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False,
    )


def _write_markdown_summary(
    metrics: pd.DataFrame,
    quality: pd.DataFrame,
    out: Path,
) -> None:
    """Write a human-readable summary of the diagnostic results."""
    row = metrics.iloc[0]

    quality_row = quality.iloc[0]

    lines = [
        "# Global sensor-harmonisation test diagnostics",
        "",
        (
            "This report is generated from the paired-group test "
            "prediction file. Counts are source detection records, "
            "not unique fire events."
        ),
        "",
        f"- Rows used for metrics: {int(row['n']):,}",
        (
            "- Rows in prediction file: "
            f"{int(quality_row['rows_in_file']):,}"
        ),
        (
            "- Rows excluded from metrics: "
            f"{int(quality_row['rows_excluded_from_metrics']):,}"
        ),
        "",
        "## Overall test metrics",
        "",
        "| Model | MAE | RMSE | Signed bias | WAPE |",
        "|---|---:|---:|---:|---:|",
    ]

    for model in PREDICTIONS:
        lines.append(
            f"| {model} "
            f"| {row[f'{model}_mae']:.2f} "
            f"| {row[f'{model}_rmse']:.2f} "
            f"| {row[f'{model}_bias']:+.2f} "
            f"| {row[f'{model}_wape']:.2%} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Residuals are calculated as prediction minus observed "
            "VIIRS count. Negative signed bias indicates underprediction; "
            "positive signed bias indicates overprediction. WAPE is the "
            "sum of absolute errors divided by the sum of observed counts."
        ),
        "",
        (
            "These metrics describe the paired-detection test population "
            "only. They do not validate predictions for MODIS-only groups, "
            "establish unique-fire counts, or account for satellite "
            "observation opportunity."
        ),
        "",
        (
            "See the CSV files in this directory for regional, seasonal, "
            "day/night, region-by-month, region-by-month-by-day/night, "
            "activity-level, and error-tail diagnostics."
        ),
    ]

    out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_analysis(
    input_file: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run all diagnostics and save their output files."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {input_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Reading predictions from %s",
        input_file,
    )

    raw = pd.read_parquet(input_file)

    _require_columns(raw)

    quality_df, valid_mask = build_quality_report(raw)

    quality_path = output_dir / "data_quality.csv"

    quality_df.to_csv(
        quality_path,
        index=False,
    )

    if quality_df.iloc[0]["duplicate_full_grain_rows_after_first"]:
        LOGGER.warning(
            (
                "Found %s duplicate full-grain rows; metrics include them. "
                "Inspect data_quality.csv."
            ),
            int(
                quality_df.iloc[0][
                    "duplicate_full_grain_rows_after_first"
                ]
            ),
        )

    valid = raw.loc[valid_mask].copy()

    if valid.empty:
        raise ValueError(
            "No complete, finite, non-negative rows remain. "
            "See data_quality.csv."
        )

    valid[ACTUAL] = (
        pd.to_numeric(
            valid[ACTUAL],
            errors="raise",
        )
        .astype(float)
    )

    for column in PREDICTIONS.values():
        valid[column] = (
            pd.to_numeric(
                valid[column],
                errors="raise",
            )
            .astype(float)
        )

    enriched = _add_error_columns(valid)

    outputs: dict[str, Path] = {
        "data_quality": quality_path,
    }

    # Overall metrics.
    overall = calculate_metrics(valid)

    outputs["overall_metrics"] = (
        output_dir / "overall_metrics.csv"
    )

    overall.to_csv(
        outputs["overall_metrics"],
        index=False,
    )

    # Grouped reports.
    for filename, keys in GROUP_REPORTS.items():
        report = grouped_metrics(
            valid,
            keys,
        )

        if "month" in keys:
            report = report.sort_values(
                keys,
                kind="stable",
            )

        output_key = Path(filename).stem

        outputs[output_key] = output_dir / filename

        report.to_csv(
            outputs[output_key],
            index=False,
        )

    # Activity-level report.
    activity = valid.copy()

    activity["activity_level"] = _activity_level(
        activity[ACTUAL]
    )

    activity_report = grouped_metrics(
        activity.dropna(subset=["activity_level"]),
        ["activity_level"],
    )

    outputs["by_activity_level"] = (
        output_dir / "by_activity_level.csv"
    )

    activity_report.to_csv(
        outputs["by_activity_level"],
        index=False,
    )

    # Largest Ridge errors.
    # Keep prediction fields and grouping keys so outliers are traceable.
    enriched["ridge_error"] = enriched["ridge_smearing_error"]

    enriched["ridge_absolute_error"] = (
        enriched["ridge_smearing_abs_error"]
    )

    outputs["largest_ridge_errors"] = (
        output_dir / "largest_ridge_errors.csv"
    )

    (
        enriched.nlargest(
            100,
            "ridge_absolute_error",
        )
        .sort_values(
            "ridge_absolute_error",
            ascending=False,
        )
        .to_csv(
            outputs["largest_ridge_errors"],
            index=False,
        )
    )

    # Error distributions.
    outputs["error_distribution"] = (
        output_dir / "error_distribution.csv"
    )

    _write_error_distribution(
        enriched,
        outputs["error_distribution"],
    )

    # Markdown summary.
    outputs["report"] = (
        output_dir / "diagnostic_summary.md"
    )

    _write_markdown_summary(
        overall,
        quality_df,
        outputs["report"],
    )

    LOGGER.info(
        "Analysed %s valid rows; outputs saved to %s",
        len(valid),
        output_dir,
    )

    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate grouped diagnostics for Sutellite "
            "global v2 predictions."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    """Run the diagnostics command-line interface."""
    args = parse_args()

    logging.basicConfig(
        level=(
            logging.INFO
            if args.verbose
            else logging.WARNING
        ),
        format="%(levelname)s: %(message)s",
    )

    try:
        outputs = run_analysis(
            args.input_file,
            args.output_dir,
        )

    except (
        FileNotFoundError,
        ValueError,
        ImportError,
        OSError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(
        f"Diagnostics written to: {args.output_dir}"
    )

    for label, path in outputs.items():
        print(f"  {label}: {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())