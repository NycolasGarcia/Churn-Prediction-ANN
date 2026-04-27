"""Tests for ``src/churn/training/tracking.py`` - smoke, schema, and API.

Light by design. The pure pieces (``compute_classification_metrics``) get
proper coverage here; the orchestrator ``log_baseline_cv_run`` is
exercised end-to-end by ``notebooks/03_baseline.ipynb`` in
sub-checkpoint 2.4 because mocking the full mlflow + sklearn stack would
test the mocks rather than the integration.

Tests that touch MLflow point ``setup_mlflow`` at a per-test temp
directory so the project's real ``./mlruns/`` is never written to.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pytest

from churn.config import MLFLOW_EXPERIMENT_NAME
from churn.training.tracking import (
    METRIC_KEYS,
    compute_classification_metrics,
    setup_mlflow,
)

# ---------------------------------------------------------------------------
# Smoke - pure metric helper runs end-to-end on synthetic data
# ---------------------------------------------------------------------------


def test_compute_classification_metrics_smoke() -> None:
    """Random binary targets + random probas: helper returns a dict."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.uniform(0.0, 1.0, size=200)
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert isinstance(metrics, dict)
    assert len(metrics) == len(METRIC_KEYS)


def test_compute_classification_metrics_handles_constant_predictions() -> None:
    """Dummy-style constant 0 predictions must not raise (zero_division=0)."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.full_like(y_true, fill_value=0.0, dtype=float)
    y_pred = np.zeros_like(y_true)

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


# ---------------------------------------------------------------------------
# Schema - return shape, key set, value ranges
# ---------------------------------------------------------------------------


def test_metrics_dict_has_expected_keys() -> None:
    """Returned dict keys are exactly METRIC_KEYS - no extras, no omissions."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.7, 0.9, 0.6, 0.3, 0.8, 0.4])
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == set(METRIC_KEYS)


def test_metrics_values_are_floats_in_unit_interval() -> None:
    """All metrics return floats in [0, 1]."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.7, 0.9, 0.6, 0.3, 0.8, 0.4])
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"
        assert 0.0 <= value <= 1.0, f"{key}={value} outside [0, 1]"


# ---------------------------------------------------------------------------
# API - setup_mlflow respects custom tracking URIs and is idempotent
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tracking_uri(tmp_path, monkeypatch) -> str:
    """Point MLflow at ``tmp_path/mlruns`` for the duration of one test.

    ``mlflow.set_tracking_uri`` mutates module-level state; the fixture
    captures the previous URI and restores it on teardown so other tests
    do not inherit a stale temp path.
    """
    previous_uri = mlflow.get_tracking_uri()
    uri = (tmp_path / "mlruns").as_uri()
    yield uri
    mlflow.set_tracking_uri(previous_uri)


def test_setup_mlflow_uses_custom_tracking_uri(isolated_tracking_uri) -> None:
    """setup_mlflow honours the tracking_uri argument without touching ./mlruns."""
    setup_mlflow(tracking_uri=isolated_tracking_uri)

    assert mlflow.get_tracking_uri() == isolated_tracking_uri
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.name == MLFLOW_EXPERIMENT_NAME


def test_setup_mlflow_is_idempotent(isolated_tracking_uri) -> None:
    """Calling setup_mlflow twice with the same args does not raise."""
    setup_mlflow(tracking_uri=isolated_tracking_uri)
    setup_mlflow(tracking_uri=isolated_tracking_uri)

    assert mlflow.get_tracking_uri() == isolated_tracking_uri
