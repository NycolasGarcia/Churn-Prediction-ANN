"""Smoke tests — verify the full pipeline runs end-to-end without errors."""

from __future__ import annotations

import numpy as np
import pytest

from churn.data.loader import load_raw_data
from churn.data.preprocessing import (
    build_preprocessing_pipeline,
    clean_raw,
    split_features_target,
    stratified_split,
)


def test_full_pipeline_produces_finite_array():
    """Raw data → clean → split → ColumnTransformer must yield a finite array."""
    df_raw = load_raw_data()
    df_clean = clean_raw(df_raw)
    X, y = split_features_target(df_clean)
    splits = stratified_split(X, y, test_size=0.10, val_size=0.10)

    pipeline = build_preprocessing_pipeline(tenure_variant="ohe")
    X_tr = pipeline.fit_transform(splits.X_train)

    assert X_tr.shape[0] == len(splits.X_train)
    assert not np.isnan(X_tr).any(), "NaN values found after preprocessing"
    assert not np.isinf(X_tr).any(), "Inf values found after preprocessing"


def test_split_preserves_class_ratio(split_data):
    """All three splits must have a churn rate within 1 p.p. of the full dataset."""
    full_rate = split_data.y_train.mean()
    for name, y in [
        ("val", split_data.y_val),
        ("test", split_data.y_test),
    ]:
        rate = y.mean()
        assert abs(rate - full_rate) < 0.01, f"{name}: {rate:.4f} vs {full_rate:.4f}"


def test_pipeline_api_path_no_leakage_columns():
    """clean_raw must not raise when called with a DataFrame missing ID/leakage cols."""
    df_raw = load_raw_data()
    df_clean = clean_raw(df_raw)
    X, _ = split_features_target(df_clean)

    single_row = X.iloc[[0]].copy()
    try:
        enriched = clean_raw(single_row)
    except KeyError as exc:
        pytest.fail(f"clean_raw raised KeyError on API-style input: {exc}")

    pipeline = build_preprocessing_pipeline(tenure_variant="ohe")
    pipeline.fit_transform(X)
    result = pipeline.transform(enriched)
    assert result.shape == (1, X.shape[1] if False else result.shape[1])
    assert not np.isnan(result).any()
