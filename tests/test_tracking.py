"""Tests for ``src/churn/training/tracking.py`` — smoke, schema, and API.

Light by design. The pure pieces (``compute_classification_metrics``) get
proper coverage here; the orchestrators are each given a targeted smoke and
schema test that verifies the MLflow contract (run completes, expected keys
logged) using small synthetic data. End-to-end integration with real data is
covered by ``notebooks/03_baseline.ipynb`` and ``notebooks/04_mlp.ipynb``.

Tests that touch MLflow point ``setup_mlflow`` at a per-test temp directory
so the project's real ``./mlruns/`` is never written to.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest

from churn.config import MLFLOW_EXPERIMENT_NAME
from churn.data.preprocessing import build_preprocessing_pipeline
from churn.training.tracking import (
    METRIC_KEYS,
    compute_classification_metrics,
    log_mlp_cv_run,
    setup_mlflow,
)

# ---------------------------------------------------------------------------
# Smoke — pure metric helper runs end-to-end on synthetic data
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
# Schema — return shape, key set, value ranges
# ---------------------------------------------------------------------------


def test_metrics_dict_has_expected_keys() -> None:
    """Returned dict keys are exactly METRIC_KEYS — no extras, no omissions."""
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
# API — setup_mlflow respects custom tracking URIs and is idempotent
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tracking_uri(tmp_path, monkeypatch) -> str:  # noqa: ARG001
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


# ---------------------------------------------------------------------------
# Fixtures — synthetic split used by the MLP CV orchestrator tests
# ---------------------------------------------------------------------------

# Categorical columns after clean_raw, with their valid string categories.
# Mirrors BINARY_COLUMNS + MULTICLASS_COLUMNS in preprocessing.py so that
# build_preprocessing_pipeline can be used as the build_preprocessor factory.
_CHURN_CATEGORIES: dict[str, list[str]] = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["Yes", "No"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "Phone Service": ["Yes", "No"],
    "Paperless Billing": ["Yes", "No"],
    "Online Security": ["Yes", "No"],
    "Online Backup": ["Yes", "No"],
    "Device Protection": ["Yes", "No"],
    "Tech Support": ["Yes", "No"],
    "Streaming TV": ["Yes", "No"],
    "Streaming Movies": ["Yes", "No"],
    "Multiple Lines": ["Yes", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Payment Method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def _make_churn_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Return a DataFrame with the real post-clean column schema, randomly populated.

    Args:
        rng: NumPy random generator for reproducibility.
        n: Number of rows to generate.

    Returns:
        DataFrame with numeric and categorical columns matching the schema
        expected by :func:`~churn.data.preprocessing.build_preprocessing_pipeline`.
    """
    data: dict[str, object] = {
        "Tenure Months": rng.integers(1, 72, size=n),
        "Monthly Charges": rng.uniform(18.0, 120.0, size=n),
        "Total Charges": rng.uniform(18.0, 8000.0, size=n),
        "CLTV": rng.integers(2000, 6000, size=n),
    }
    for col, choices in _CHURN_CATEGORIES.items():
        data[col] = rng.choice(choices, size=n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def synthetic_churn_split():
    """Return ``(X_train, X_val, y_train, y_val)`` matching the real column schema.

    Sized small (200 train / 50 val) so MLP CV smoke tests complete in seconds
    on CPU. Module-scoped because the DataFrames are read-only across all tests.
    The ~26 % positive rate mirrors the real class imbalance.
    """
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    X_train = _make_churn_df(rng, n_train)
    X_val = _make_churn_df(rng, n_val)
    y_train = pd.Series(rng.choice([0, 1], size=n_train, p=[0.74, 0.26]).astype(int))
    y_val = pd.Series(rng.choice([0, 1], size=n_val, p=[0.74, 0.26]).astype(int))
    return X_train, X_val, y_train, y_val


# Shared kwargs that cap training time for every MLP test: 5 epochs maximum,
# patience=3, 3 CV folds (vs the project default of 5). Quality does not
# matter here — only that the full orchestration path executes without error.
_MLP_FAST_KWARGS: dict[str, int] = {"max_epochs": 5, "patience": 3}
_CV_FOLDS_FAST: int = 3


# ---------------------------------------------------------------------------
# Smoke — log_mlp_cv_run completes end-to-end
# ---------------------------------------------------------------------------


def test_log_mlp_cv_run_smoke(
    isolated_tracking_uri,
    synthetic_churn_split,
) -> None:
    """log_mlp_cv_run runs without error and returns a non-empty run_id string."""
    X_train, X_val, y_train, y_val = synthetic_churn_split
    setup_mlflow(tracking_uri=isolated_tracking_uri)

    run_id = log_mlp_cv_run(
        model_name="mlp_smoke",
        build_preprocessor=build_preprocessing_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_kwargs=_MLP_FAST_KWARGS,
        cv_folds=_CV_FOLDS_FAST,
    )

    assert isinstance(run_id, str)
    assert len(run_id) > 0


# ---------------------------------------------------------------------------
# Schema — logged run satisfies the ADR-008 metric and param contract
# ---------------------------------------------------------------------------


def test_log_mlp_cv_run_logs_expected_metric_keys(
    isolated_tracking_uri,
    synthetic_churn_split,
) -> None:
    """Logged run has ``<key>_mean`` and ``holdout_val_<key>`` for every METRIC_KEYS."""
    X_train, X_val, y_train, y_val = synthetic_churn_split
    setup_mlflow(tracking_uri=isolated_tracking_uri)

    run_id = log_mlp_cv_run(
        model_name="mlp_schema_metrics",
        build_preprocessor=build_preprocessing_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_kwargs=_MLP_FAST_KWARGS,
        cv_folds=_CV_FOLDS_FAST,
    )

    logged_metrics = set(mlflow.get_run(run_id).data.metrics.keys())
    for key in METRIC_KEYS:
        assert f"{key}_mean" in logged_metrics, f"missing {key!r}_mean"
        assert f"holdout_val_{key}" in logged_metrics, f"missing holdout_val_{key!r}"


def test_log_mlp_cv_run_logs_canonical_params(
    isolated_tracking_uri,
    synthetic_churn_split,
) -> None:
    """Logged run contains all canonical MLP hyperparameter keys."""
    X_train, X_val, y_train, y_val = synthetic_churn_split
    setup_mlflow(tracking_uri=isolated_tracking_uri)

    run_id = log_mlp_cv_run(
        model_name="mlp_schema_params",
        build_preprocessor=build_preprocessing_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_kwargs=_MLP_FAST_KWARGS,
        cv_folds=_CV_FOLDS_FAST,
    )

    logged_params = set(mlflow.get_run(run_id).data.params.keys())
    required = {
        "model_type",
        "seed",
        "dataset_version",
        "hidden_dims",
        "dropout_rates",
        "batch_size",
        "learning_rate",
        "patience",
        "best_epoch",
        "stopped_early",
    }
    for param in required:
        assert param in logged_params, f"missing canonical param {param!r}"
