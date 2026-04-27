"""Tests for ``src/churn/models/baseline.py`` — smoke, schema, and API.

Three categories:

- **Smoke** — the pipeline fits and predicts end-to-end without errors.
- **Schema** — outputs have the expected shape, dtype, and value range.
- **API** — the module's public contract (factory output structure and
  classifier configuration) is what consumers depend on.

The Phase-4 tests in ``test_smoke.py`` / ``test_schema.py`` / ``test_api.py``
target the deployed MLP and the FastAPI service. These baseline tests live
separately because the failure modes are different (training pipeline vs.
inference pipeline).
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import SEED
from churn.models.baseline import build_dummy_baseline, build_logreg_baseline

# ---------------------------------------------------------------------------
# Smoke — end-to-end fit/predict runs cleanly
# ---------------------------------------------------------------------------


def test_dummy_baseline_fits_and_predicts(split_data) -> None:
    """Dummy pipeline fits on train and predicts on val without raising."""
    pipe = build_dummy_baseline()
    pipe.fit(split_data.X_train, split_data.y_train)
    pipe.predict(split_data.X_val)
    pipe.predict_proba(split_data.X_val)


def test_logreg_baseline_fits_and_predicts(split_data) -> None:
    """LogReg pipeline fits on train and predicts on val without raising."""
    pipe = build_logreg_baseline()
    pipe.fit(split_data.X_train, split_data.y_train)
    pipe.predict(split_data.X_val)
    pipe.predict_proba(split_data.X_val)


# ---------------------------------------------------------------------------
# Schema — shapes, dtypes, value ranges
# ---------------------------------------------------------------------------


def test_dummy_output_schema(split_data) -> None:
    """Dummy: predictions all 0, proba is (n, 2) with rows summing to 1."""
    pipe = build_dummy_baseline()
    pipe.fit(split_data.X_train, split_data.y_train)
    n_val = len(split_data.X_val)

    y_pred = pipe.predict(split_data.X_val)
    assert y_pred.shape == (n_val,)
    assert set(y_pred.tolist()) <= {0, 1}
    assert (y_pred == 0).all(), "most_frequent on this data must always predict 0"

    y_proba = pipe.predict_proba(split_data.X_val)
    assert y_proba.shape == (n_val, 2)
    assert ((y_proba >= 0) & (y_proba <= 1)).all()
    np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)


def test_logreg_output_schema(split_data) -> None:
    """LogReg: predictions in {0, 1}, proba is (n, 2) summing to 1."""
    pipe = build_logreg_baseline()
    pipe.fit(split_data.X_train, split_data.y_train)
    n_val = len(split_data.X_val)

    y_pred = pipe.predict(split_data.X_val)
    assert y_pred.shape == (n_val,)
    assert set(y_pred.tolist()) <= {0, 1}
    assert {0, 1}.issubset(set(y_pred.tolist())), (
        "LogReg should produce both classes on this dataset"
    )

    y_proba = pipe.predict_proba(split_data.X_val)
    assert y_proba.shape == (n_val, 2)
    assert ((y_proba >= 0) & (y_proba <= 1)).all()
    np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# API — factory contract: pipeline structure and classifier configuration
# ---------------------------------------------------------------------------


def test_builders_return_sklearn_pipelines() -> None:
    """Builders return Pipeline with named 'preprocessor' and 'classifier' steps."""
    for builder in (build_dummy_baseline, build_logreg_baseline):
        pipe = builder()
        assert isinstance(pipe, Pipeline)
        assert list(pipe.named_steps) == ["preprocessor", "classifier"]


def test_dummy_classifier_uses_most_frequent_strategy() -> None:
    """Dummy is wired with strategy='most_frequent' and the project SEED."""
    clf = build_dummy_baseline().named_steps["classifier"]
    assert isinstance(clf, DummyClassifier)
    assert clf.strategy == "most_frequent"
    assert clf.random_state == SEED


def test_logreg_classifier_uses_balanced_class_weight() -> None:
    """LogReg is wired with class_weight='balanced' and the project SEED."""
    clf = build_logreg_baseline().named_steps["classifier"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced"
    assert clf.random_state == SEED


def test_logreg_is_deterministic(split_data) -> None:
    """Two independent fits with the same SEED produce identical predictions."""
    pipe_a = build_logreg_baseline()
    pipe_b = build_logreg_baseline()
    pipe_a.fit(split_data.X_train, split_data.y_train)
    pipe_b.fit(split_data.X_train, split_data.y_train)
    proba_a = pipe_a.predict_proba(split_data.X_val)
    proba_b = pipe_b.predict_proba(split_data.X_val)
    np.testing.assert_array_equal(proba_a, proba_b)
