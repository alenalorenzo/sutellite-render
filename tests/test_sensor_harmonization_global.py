"""Tests for the full-dataset harmonisation experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sensor_harmonization_global import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    _global_aggregation_sql,
    build_model_features,
    evaluate_predictions,
    fit_ridge_smearing,
    predict_global_counts,
    predict_region_month_daynight_mean,
    prepare_paired_splits,
    validate_global_dataset,
)


def make_dataset() -> pd.DataFrame:
    rows = []
    for year in range(2013, 2026):
        for region_index, region in enumerate(("region_a", "region_b")):
            for month in range(1, 13):
                for lat_index in (0, 1):
                    for lon_index in (0, 1):
                        for daynight in ("D", "N"):
                            aqua = 1 + ((year + month + lat_index) % 8)
                            terra = (year + lon_index + region_index) % 4
                            modis = aqua + terra
                            viirs = 2 * aqua + terra + (month % 3) + region_index
                            rows.append(
                                {
                                    "region_id": region,
                                    "year": year,
                                    "month": month,
                                    "grid_lat_index": lat_index,
                                    "grid_lon_index": lon_index,
                                    "daynight": daynight,
                                    "grid_lat_center_deg": (lat_index + 0.5) * 0.1,
                                    "grid_lon_center_deg": (lon_index + 0.5) * 0.1,
                                    "modis_aqua_count": aqua,
                                    "modis_terra_count": terra,
                                    "viirs_snpp_count": viirs,
                                    "viirs_noaa20_count": 0,
                                    "viirs_noaa21_count": 0,
                                    "modis_count": modis,
                                    "viirs_count": viirs,
                                    "modis_present": True,
                                    "viirs_present": True,
                                }
                            )

    # One-sided groups must be retained in the feature dataset but excluded
    # from supervised fitting because absence is not a confirmed zero.
    rows.append(
        {
            "region_id": "region_a", "year": 2025, "month": 1,
            "grid_lat_index": 20, "grid_lon_index": 20, "daynight": "D",
            "grid_lat_center_deg": 2.05, "grid_lon_center_deg": 2.05,
            "modis_aqua_count": 4, "modis_terra_count": 0,
            "viirs_snpp_count": 0, "viirs_noaa20_count": 0,
            "viirs_noaa21_count": 0, "modis_count": 4, "viirs_count": 0,
            "modis_present": True, "viirs_present": False,
        }
    )
    rows.append(
        {
            "region_id": "region_b", "year": 2025, "month": 2,
            "grid_lat_index": 21, "grid_lon_index": 21, "daynight": "N",
            "grid_lat_center_deg": 2.15, "grid_lon_center_deg": 2.15,
            "modis_aqua_count": 0, "modis_terra_count": 0,
            "viirs_snpp_count": 6, "viirs_noaa20_count": 0,
            "viirs_noaa21_count": 0, "modis_count": 0, "viirs_count": 6,
            "modis_present": False, "viirs_present": True,
        }
    )
    return pd.DataFrame(rows, columns=list(DATASET_COLUMNS))


def test_validate_global_dataset_accepts_consistent_platform_totals():
    data = validate_global_dataset(make_dataset())
    assert len(data) > 0
    assert data["modis_count"].equals(
        data["modis_aqua_count"] + data["modis_terra_count"]
    )


def test_validate_global_dataset_rejects_inconsistent_totals():
    data = make_dataset()
    # Assign an explicit Python int so Pylance does not infer a broad scalar union.
    data.loc[0, "modis_count"] = int(data.loc[0, "modis_count"]) + 1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_global_dataset(data)


def test_paired_splits_are_chronological_and_exclude_one_sided_groups():
    training, validation, test, counts = prepare_paired_splits(make_dataset())
    assert training["year"].max() <= 2022
    assert validation["year"].between(2023, 2024).all()
    assert test["year"].eq(2025).all()
    assert counts["one_sided_groups_excluded"] == 2
    assert len(training) + len(validation) + len(test) == counts["paired_groups"]


def test_features_do_not_use_viirs_target():
    data = validate_global_dataset(make_dataset().head(10))
    before = build_model_features(data)
    changed = data.copy()
    changed["viirs_count"] *= 100
    after = build_model_features(changed)
    pd.testing.assert_frame_equal(before, after)
    assert list(before.columns) == list(FEATURE_COLUMNS)


def test_ridge_fit_and_smearing_produce_finite_nonnegative_predictions():
    data = validate_global_dataset(make_dataset())
    training = data.loc[data["year"].le(2022) & data["modis_present"] & data["viirs_present"]]
    estimator, smearing = fit_ridge_smearing(training, cv_folds=3)
    predictions = np.exp(estimator.predict(build_model_features(training.head(20)))) * smearing - 1
    assert np.isfinite(smearing) and smearing > 0
    assert np.isfinite(predictions).all()
    assert (predictions >= -1e-9).all()


def test_region_month_baseline_uses_training_means():
    data = validate_global_dataset(make_dataset())
    training = data.loc[data["year"].le(2022)]
    prediction = data.loc[data["year"].eq(2025)].head(5)
    values = predict_region_month_daynight_mean(training, prediction)
    assert len(values) == len(prediction)
    assert np.isfinite(values).all()


def test_poisson_gradient_boosting_returns_nonnegative_counts():
    from src.sensor_harmonization_global import fit_poisson_gradient_boosting

    data = validate_global_dataset(make_dataset())
    training = data.loc[data["year"].le(2022) & data["modis_present"] & data["viirs_present"]]
    estimator = fit_poisson_gradient_boosting(training)
    # sklearn's estimator stubs can expose predict() as an ndarray-or-tuple
    # union. Normalise the runtime result to a numeric ndarray for type checkers.
    predictions = np.asarray(
        estimator.predict(build_model_features(training.head(20))),
        dtype=np.float64,
    )
    assert np.isfinite(predictions).all()
    assert np.greater_equal(predictions, 0.0).all()

def test_inference_flags_modis_only_as_outside_validated_population():
    data = validate_global_dataset(make_dataset())
    training = data.loc[data["year"].le(2022)]
    estimator, smearing = fit_ridge_smearing(training, cv_folds=3)
    artifact = {
        "model_version": "2.0.0",
        "model_name": "ridge_smearing",
        "estimator": estimator,
        "smearing_factor": smearing,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "viirs_count",
    }
    with pytest.raises(ValueError, match="No eligible"):
        predict_global_counts(
            data.loc[data["modis_present"] & ~data["viirs_present"]],
            artifact,
        )
    predictions = predict_global_counts(
        data.loc[data["modis_present"]],
        artifact,
        include_modis_only=True,
    )
    assert "model_population_status" in predictions
    assert predictions["model_population_status"].str.contains("OUTSIDE_VALIDATED_POPULATION").any()



def test_feature_only_inference_requires_explicit_unvalidated_mode():
    data = validate_global_dataset(make_dataset())
    training = data.loc[data["year"].le(2022)]
    estimator, smearing = fit_ridge_smearing(training, cv_folds=3)
    artifact = {
        "model_version": "2.0.0",
        "model_name": "ridge_smearing",
        "estimator": estimator,
        "smearing_factor": smearing,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "viirs_count",
    }
    modis_only_inputs = data.loc[data["modis_present"]].drop(
        columns=[
            "viirs_count", "viirs_present", "viirs_snpp_count",
            "viirs_noaa20_count", "viirs_noaa21_count",
        ]
    )
    with pytest.raises(ValueError, match="VIIRS truth is absent"):
        predict_global_counts(modis_only_inputs, artifact)

    estimates = predict_global_counts(
        modis_only_inputs,
        artifact,
        include_modis_only=True,
    )
    assert estimates["viirs_count_observed"].isna().all()
    assert (~estimates["is_within_paired_detection_population"]).all()
    assert estimates["model_population_status"].eq("OUTSIDE_VALIDATED_POPULATION").all()

def test_sql_aggregation_covers_all_sensor_platforms_and_excludes_nrt():
    sql = _global_aggregation_sql(
        [Path("/tmp/modis.parquet"), Path("/tmp/viirs.parquet")],
        grid_size_degrees=0.1,
    )
    assert "processing_stream = 'ARCHIVE'" in sql
    assert "NOAA-20" in sql and "NOAA-21" in sql and "SNPP" in sql
    assert "modis_terra_count" in sql and "modis_aqua_count" in sql
    assert "viirs_count" in sql
    assert "read_parquet" in sql


def test_evaluation_reports_mae_rmse_bias_and_wape():
    data = validate_global_dataset(make_dataset())
    test = data.loc[data["year"].eq(2025) & data["modis_present"] & data["viirs_present"]].head(20).copy()
    test["prediction_test"] = test["viirs_count"].astype(float) + 2.0
    metrics = evaluate_predictions(
        test,
        split_name="test",
        prediction_columns=["prediction_test"],
    )
    overall = metrics.loc[metrics["scope"].eq("overall")].iloc[0]
    assert overall["mae"] == pytest.approx(2.0)
    assert overall["signed_bias"] == pytest.approx(2.0)
    assert np.isfinite(overall["rmse"])
    assert np.isfinite(overall["wape"])
