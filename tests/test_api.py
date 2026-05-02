"""FastAPI endpoint tests — /health and /predict."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"


def test_health_model_loaded(api_client):
    body = api_client.get("/health").json()
    assert body["model_loaded"] is True


def test_health_has_timestamp(api_client):
    body = api_client.get("/health").json()
    assert "timestamp" in body
    assert len(body["timestamp"]) > 0


def test_health_model_version(api_client):
    from churn.config import MODEL_VERSION
    body = api_client.get("/health").json()
    assert body["model_version"] == MODEL_VERSION


# ---------------------------------------------------------------------------
# /predict — valid input
# ---------------------------------------------------------------------------

def test_predict_returns_200(api_client, valid_payload):
    response = api_client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_probability_in_range(api_client, valid_payload):
    body = api_client.post("/predict", json=valid_payload).json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_prediction_is_bool(api_client, valid_payload):
    body = api_client.post("/predict", json=valid_payload).json()
    assert isinstance(body["churn_prediction"], bool)


def test_predict_risk_level_valid(api_client, valid_payload):
    body = api_client.post("/predict", json=valid_payload).json()
    assert body["risk_level"] in ("low", "medium", "high")


def test_predict_model_version_present(api_client, valid_payload):
    from churn.config import MODEL_VERSION
    body = api_client.post("/predict", json=valid_payload).json()
    assert body["model_version"] == MODEL_VERSION


def test_predict_threshold_used(api_client, valid_payload):
    from churn.config import DEPLOY_THRESHOLD
    body = api_client.post("/predict", json=valid_payload).json()
    assert body["threshold_used"] == DEPLOY_THRESHOLD


def test_predict_high_risk_customer(api_client, valid_payload):
    """Month-to-month + 2 months tenure + Fiber optic should yield high probability."""
    body = api_client.post("/predict", json=valid_payload).json()
    assert body["churn_probability"] > 0.4, (
        f"Expected high-risk customer to have p>0.4, got {body['churn_probability']}"
    )


# ---------------------------------------------------------------------------
# /predict — invalid input (422 Unprocessable Entity)
# ---------------------------------------------------------------------------

def test_predict_negative_tenure_returns_422(api_client, valid_payload):
    bad = {**valid_payload, "tenure_months": -1}
    response = api_client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_gender_returns_422(api_client, valid_payload):
    bad = {**valid_payload, "gender": "Unknown"}
    response = api_client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(api_client, valid_payload):
    bad = {k: v for k, v in valid_payload.items() if k != "monthly_charges"}
    response = api_client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_contract_returns_422(api_client, valid_payload):
    bad = {**valid_payload, "contract": "Daily"}
    response = api_client.post("/predict", json=bad)
    assert response.status_code == 422
