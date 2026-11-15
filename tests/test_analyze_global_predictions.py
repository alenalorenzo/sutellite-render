import numpy as np
import pandas as pd

from src.analyze_global_predictions import (
    ACTUAL,
    KEY_COLUMNS,
    PREDICTIONS,
    build_quality_report,
    calculate_metrics,
    grouped_metrics,
    run_analysis,
)


def sample_predictions() -> pd.DataFrame:
    data = {
        "region_id": ["angola", "angola", "brasil", "brasil"],
        "year": [2025, 2025, 2025, 2025],
        "month": [1, 1, 2, 2],
        "grid_lat_index": [10, 11, 20, 21],
        "grid_lon_index": [30, 31, 40, 41],
        "daynight": ["D", "N", "D", "N"],
        ACTUAL: [10, 20, 5, 15],
        PREDICTIONS["naive_modis"]: [5, 10, 2, 7],
        PREDICTIONS["region_month_daynight_mean"]: [8, 18, 6, 12],
        PREDICTIONS["ridge_smearing"]: [11, 18, 4, 17],
    }

    return pd.DataFrame(data)


def test_overall_metrics_use_prediction_minus_actual_bias() -> None:
    metrics = calculate_metrics(sample_predictions()).iloc[0]

    assert metrics["n"] == 4
    assert np.isclose(metrics["ridge_smearing_mae"], 1.5)
    assert np.isclose(metrics["ridge_smearing_bias"], 0.0)
    assert metrics["ridge_smearing_rmse"] > 0
    assert 0 <= metrics["ridge_smearing_wape"] < 1


def test_region_month_grouping_returns_each_region_month_pair() -> None:
    result = grouped_metrics(
        sample_predictions(),
        ["region_id", "month"],
    )

    assert len(result) == 2

    assert set(zip(result["region_id"], result["month"])) == {
        ("angola", 1),
        ("brasil", 2),
    }


def test_quality_report_flags_duplicates_and_negative_values() -> None:
    df = sample_predictions()

    # Add a duplicate full-grain row.
    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    # Introduce one invalid negative actual count.
    df.loc[1, ACTUAL] = -1

    report, valid = build_quality_report(df)

    assert report.iloc[0]["duplicate_full_grain_rows_after_first"] == 1
    assert report.iloc[0]["negative_actual_count_rows"] == 1
    assert valid.sum() == 4


def test_all_required_group_keys_are_used_for_duplicate_detection() -> None:
    df = sample_predictions()

    assert set(KEY_COLUMNS).issubset(df.columns)

    report, _ = build_quality_report(df)

    assert report.iloc[0]["duplicate_full_grain_rows_after_first"] == 0


def test_run_analysis_writes_region_month_report(
    tmp_path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "predictions.parquet"
    input_file.touch()

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda _path: sample_predictions(),
    )

    output_dir = tmp_path / "diagnostics"

    outputs = run_analysis(input_file, output_dir)

    assert "by_region_month" in outputs

    report = pd.read_csv(
        output_dir / "by_region_month.csv",
    )

    assert len(report) == 2

    assert set(report.columns).issuperset(
        {
            "region_id",
            "month",
            "ridge_smearing_mae",
            "ridge_smearing_bias",
            "ridge_smearing_wape",
        }
    )

    assert (output_dir / "diagnostic_summary.md").exists()