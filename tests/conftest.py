"""Shared pytest fixtures.

``split_data``    — full 80/10/10 stratified split; cached at session scope.
``api_client``    — FastAPI TestClient with model loaded from local MLflow;
                    cached at session scope so the heavy MLflow load runs once.
``valid_payload`` — dict matching CustomerInput schema for a high-risk customer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.data.loader import load_raw_data
from churn.data.preprocessing import (
    SplitData,
    clean_raw,
    split_features_target,
    stratified_split,
)


@pytest.fixture(scope="session")
def split_data() -> SplitData:
    """80/10/10 stratified split, computed once per test session."""
    df_raw = load_raw_data()
    df_clean = clean_raw(df_raw)
    X, y = split_features_target(df_clean)
    return stratified_split(X, y, test_size=0.10, val_size=0.10)


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """TestClient backed by the real FastAPI app — loads the MLflow model once."""
    from churn.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """High-risk customer payload that satisfies all CustomerInput constraints."""
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.5,
        "total_charges": 171.0,
        "cltv": 3200,
    }
