"""FastAPI application for churn prediction inference.

Endpoints
---------
GET  /health   — liveness check; reports model load status.
POST /predict  — returns churn probability, binary decision and risk tier.

Model loading
-------------
At startup the app searches MLflow for the canonical run (``MLFLOW_RUN_NAME``),
downloads the PyTorch model and the fitted sklearn preprocessor, and stores
them in ``app.state``. All subsequent requests are served from memory with
no MLflow I/O on the hot path.

The preprocessor handles the same ``clean_raw`` → ``ColumnTransformer``
transformation used during training, so inference is byte-for-byte identical
to the notebook evaluation.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from churn.api.middleware import LatencyLoggingMiddleware
from churn.api.schemas import CustomerInput, HealthOutput, PredictionOutput
from churn.config import (
    DEPLOY_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
    MODEL_VERSION,
    ROOT_DIR,
)
from churn.data.preprocessing import clean_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_model_from_mlflow() -> tuple[Any, Any]:
    """Load MLP model and preprocessor from the canonical MLflow run.

    Returns:
        Tuple of (torch_model_in_eval_mode, fitted_sklearn_preprocessor).

    Raises:
        RuntimeError: If the run is not found or artifacts are missing.
    """
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{MLFLOW_RUN_NAME}'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"MLflow run '{MLFLOW_RUN_NAME}' not found in experiment "
            f"'{MLFLOW_EXPERIMENT_NAME}'. Re-run 04_mlp.ipynb to register it."
        )

    run_id = runs.iloc[0]["run_id"]
    logger.info("Loading model from MLflow run %s (%s)", MLFLOW_RUN_NAME, run_id)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/mlp_model")
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        pp_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/preprocessor.joblib",
            dst_path=tmpdir,
        )
        preprocessor = joblib.load(pp_path)

    logger.info("Model and preprocessor loaded successfully")
    return model, preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts once at startup; release at shutdown."""
    model, preprocessor = _load_model_from_mlflow()
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.model_loaded = True
    logger.info("API ready — model_version=%s threshold=%.2f", MODEL_VERSION, DEPLOY_THRESHOLD)
    yield
    app.state.model_loaded = False
    logger.info("API shutting down")


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts churn probability for telecom customers using a PyTorch MLP.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.add_middleware(LatencyLoggingMiddleware)


@app.get("/health", response_model=HealthOutput, tags=["ops"])
def health() -> HealthOutput:
    """Liveness check. Returns model load status and current timestamp."""
    return HealthOutput(
        status="ok" if getattr(app.state, "model_loaded", False) else "degraded",
        model_version=MODEL_VERSION,
        model_loaded=getattr(app.state, "model_loaded", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict(customer: CustomerInput) -> PredictionOutput:
    """Return churn probability and risk tier for a single customer.

    The input is validated by Pydantic before reaching this function.
    A 503 is returned if the model was not loaded successfully at startup.
    """
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    row_df = pd.DataFrame([customer.to_raw_dict()])
    enriched = clean_raw(row_df)

    X = np.asarray(
        app.state.preprocessor.transform(enriched),
        dtype=np.float32,
    )
    tensor = torch.as_tensor(X)

    with torch.no_grad():
        logit = app.state.model(tensor)
        probability = float(torch.sigmoid(logit).squeeze())

    prediction = probability >= DEPLOY_THRESHOLD

    logger.info(
        "prediction issued",
        extra={
            "probability": round(probability, 4),
            "prediction": prediction,
            "threshold": DEPLOY_THRESHOLD,
        },
    )

    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=bool(prediction),
        risk_level=_risk_level(probability),
        model_version=MODEL_VERSION,
        threshold_used=DEPLOY_THRESHOLD,
    )
