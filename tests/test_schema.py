"""Schema validation tests for CustomerInput and PredictionOutput."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from churn.api.schemas import CustomerInput, PredictionOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base() -> dict:
    """Minimal valid CustomerInput payload."""
    return {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "One year",
        "paperless_billing": "No",
        "payment_method": "Mailed check",
        "monthly_charges": 55.0,
        "total_charges": 1320.0,
        "cltv": 4500,
    }


# ---------------------------------------------------------------------------
# CustomerInput — valid cases
# ---------------------------------------------------------------------------

def test_valid_input_is_accepted():
    customer = CustomerInput(**_base())
    assert customer.tenure_months == 24
    assert customer.monthly_charges == 55.0


def test_to_raw_dict_maps_column_names():
    raw = CustomerInput(**_base()).to_raw_dict()
    assert "Tenure Months" in raw
    assert "Monthly Charges" in raw
    assert "tenure_months" not in raw


def test_senior_citizen_yes_and_no_accepted():
    for val in ("Yes", "No"):
        c = CustomerInput(**{**_base(), "senior_citizen": val})
        assert c.senior_citizen == val


# ---------------------------------------------------------------------------
# CustomerInput — invalid cases
# ---------------------------------------------------------------------------

def test_negative_tenure_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "tenure_months": -1})


def test_tenure_above_120_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "tenure_months": 121})


def test_negative_monthly_charges_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "monthly_charges": -10.0})


def test_invalid_gender_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "gender": "Other"})


def test_invalid_contract_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "contract": "Weekly"})


def test_invalid_internet_service_rejected():
    with pytest.raises(ValidationError):
        CustomerInput(**{**_base(), "internet_service": "5G"})


def test_missing_required_field_rejected():
    payload = _base()
    del payload["monthly_charges"]
    with pytest.raises(ValidationError):
        CustomerInput(**payload)


# ---------------------------------------------------------------------------
# PredictionOutput — valid cases
# ---------------------------------------------------------------------------

def test_prediction_output_valid():
    out = PredictionOutput(
        churn_probability=0.73,
        churn_prediction=True,
        risk_level="high",
        model_version="1.0.0",
        threshold_used=0.27,
    )
    assert out.churn_probability == 0.73
    assert out.risk_level == "high"


def test_prediction_output_probability_bounds():
    for prob in (0.0, 0.5, 1.0):
        out = PredictionOutput(
            churn_probability=prob,
            churn_prediction=prob >= 0.5,
            risk_level="medium",
            model_version="1.0.0",
            threshold_used=0.5,
        )
        assert 0.0 <= out.churn_probability <= 1.0


def test_prediction_output_rejects_probability_above_one():
    with pytest.raises(ValidationError):
        PredictionOutput(
            churn_probability=1.01,
            churn_prediction=True,
            risk_level="high",
            model_version="1.0.0",
            threshold_used=0.27,
        )
