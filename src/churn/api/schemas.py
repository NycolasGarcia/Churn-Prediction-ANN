"""Pydantic v2 schemas for the /predict endpoint.

``CustomerInput`` mirrors the raw features available after removing identifiers,
leakage and geographic columns (see ``preprocessing.clean_raw``). Field names
use snake_case; the API layer maps them back to the original column names before
calling ``clean_raw``.

``PredictionOutput`` carries the probability, binary decision, risk tier and
model version so downstream consumers can log or display all relevant context.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Raw customer features sent to POST /predict."""

    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure_months: int = Field(ge=0, le=120)
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0.0)
    total_charges: float = Field(ge=0.0)
    cltv: int = Field(ge=0)

    def to_raw_dict(self) -> dict:
        """Return a dict with original column names expected by ``clean_raw``."""
        return {
            "Gender": self.gender,
            "Senior Citizen": self.senior_citizen,
            "Partner": self.partner,
            "Dependents": self.dependents,
            "Tenure Months": self.tenure_months,
            "Phone Service": self.phone_service,
            "Multiple Lines": self.multiple_lines,
            "Internet Service": self.internet_service,
            "Online Security": self.online_security,
            "Online Backup": self.online_backup,
            "Device Protection": self.device_protection,
            "Tech Support": self.tech_support,
            "Streaming TV": self.streaming_tv,
            "Streaming Movies": self.streaming_movies,
            "Contract": self.contract,
            "Paperless Billing": self.paperless_billing,
            "Payment Method": self.payment_method,
            "Monthly Charges": self.monthly_charges,
            "Total Charges": self.total_charges,
            "CLTV": self.cltv,
        }

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
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
        "total_charges": 1024.0,
        "cltv": 3200,
    }}}


class PredictionOutput(BaseModel):
    """Response payload from POST /predict."""

    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: bool
    risk_level: Literal["low", "medium", "high"]
    model_version: str
    threshold_used: float


class HealthOutput(BaseModel):
    """Response payload from GET /health."""

    status: Literal["ok", "degraded"]
    model_version: str
    model_loaded: bool
    timestamp: str
