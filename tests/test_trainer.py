"""Tests for ``src/churn/training/trainer.py`` — smoke, schema, and API.

The trainer is exercised on small synthetic data so the suite stays
fast (sub-second per test). End-to-end training on the real dataset is
covered indirectly by the MLflow integration in sub-checkpoint 3.3.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from churn.config import SEED
from churn.models.mlp import ChurnMLP
from churn.training.trainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    EpochMetrics,
    TrainingResult,
    _compute_pos_weight,
    train_mlp,
)


def _make_separable_data(
    n_train: int = 256,
    n_val: int = 96,
    n_features: int = 8,
    pos_rate: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny linearly-near-separable problem the MLP can learn quickly."""
    rng = np.random.default_rng(seed)
    n_pos = int(n_train * pos_rate)
    n_neg = n_train - n_pos
    X_train = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n_neg, n_features)),
            rng.normal(1.0, 1.0, size=(n_pos, n_features)),
        ]
    ).astype(np.float32)
    y_train = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(np.float32)

    n_pos_v = int(n_val * pos_rate)
    n_neg_v = n_val - n_pos_v
    X_val = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n_neg_v, n_features)),
            rng.normal(1.0, 1.0, size=(n_pos_v, n_features)),
        ]
    ).astype(np.float32)
    y_val = np.concatenate([np.zeros(n_neg_v), np.ones(n_pos_v)]).astype(np.float32)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_train_mlp_runs_end_to_end() -> None:
    """A 5-epoch run on synthetic data returns a TrainingResult."""
    X_train, y_train, X_val, y_val = _make_separable_data()
    model = ChurnMLP(n_features=X_train.shape[1])
    result = train_mlp(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=5,
        patience=10,
    )
    assert isinstance(result, TrainingResult)
    assert isinstance(result.model, ChurnMLP)
    assert len(result.history) == 5


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_history_entries_have_finite_metrics() -> None:
    """Every EpochMetrics row carries finite floats."""
    X_train, y_train, X_val, y_val = _make_separable_data()
    model = ChurnMLP(n_features=X_train.shape[1])
    result = train_mlp(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=4,
        patience=10,
    )
    for entry in result.history:
        assert isinstance(entry, EpochMetrics)
        assert np.isfinite(entry.train_loss)
        assert np.isfinite(entry.val_loss)
        assert 0.0 <= entry.val_auc <= 1.0


def test_best_epoch_is_within_history_range() -> None:
    """``best_epoch`` is one of the executed epochs."""
    X_train, y_train, X_val, y_val = _make_separable_data()
    model = ChurnMLP(n_features=X_train.shape[1])
    result = train_mlp(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=8,
        patience=10,
    )
    epochs = [e.epoch for e in result.history]
    assert result.best_epoch in epochs
    assert result.best_val_loss == min(e.val_loss for e in result.history)


# ---------------------------------------------------------------------------
# API — pos_weight, defaults, early stopping, determinism
# ---------------------------------------------------------------------------


def test_compute_pos_weight_matches_neg_over_pos() -> None:
    """Auto pos_weight is ``num_neg / num_pos``."""
    y = np.array([0, 0, 0, 1, 1])
    assert _compute_pos_weight(y) == pytest.approx(3 / 2)


def test_compute_pos_weight_raises_when_no_positives() -> None:
    """Auto pos_weight refuses an all-negative training set."""
    with pytest.raises(ValueError, match="no positive samples"):
        _compute_pos_weight(np.zeros(10))


def test_default_hyperparams_match_spec() -> None:
    """Defaults match training spec (batch=32, max=100, patience=10)."""
    assert DEFAULT_BATCH_SIZE == 32
    assert DEFAULT_MAX_EPOCHS == 100
    assert DEFAULT_PATIENCE == 10


def test_early_stopping_triggers_on_stagnant_val_loss() -> None:
    """When val_loss never improves, trainer stops after ``patience`` epochs."""
    # Frozen X_val/y_val that the model can't fit better over time —
    # easiest way: a degenerate val set the model overfits past quickly.
    rng = np.random.default_rng(0)
    X_train = rng.normal(0.0, 1.0, size=(128, 8)).astype(np.float32)
    y_train = (rng.random(128) > 0.5).astype(np.float32)
    # Val with reversed pattern so val_loss climbs as we fit train.
    X_val = rng.normal(0.0, 1.0, size=(64, 8)).astype(np.float32)
    y_val = (rng.random(64) > 0.5).astype(np.float32)

    model = ChurnMLP(n_features=8)
    result = train_mlp(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=50,
        patience=3,
    )
    assert result.stopped_early
    assert len(result.history) < 50
    # Final history length = best_epoch + patience (approximately).
    assert len(result.history) <= result.best_epoch + 3


def test_train_mlp_is_deterministic_under_fixed_seed() -> None:
    """Two runs with the same seed give identical histories and weights."""
    X_train, y_train, X_val, y_val = _make_separable_data(seed=1)

    torch.manual_seed(SEED)
    model_a = ChurnMLP(n_features=X_train.shape[1])
    torch.manual_seed(SEED)
    model_b = ChurnMLP(n_features=X_train.shape[1])

    result_a = train_mlp(
        model_a,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=5,
        patience=10,
    )
    result_b = train_mlp(
        model_b,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=5,
        patience=10,
    )

    losses_a = [e.val_loss for e in result_a.history]
    losses_b = [e.val_loss for e in result_b.history]
    np.testing.assert_allclose(losses_a, losses_b, rtol=1e-6)

    for k in result_a.model.state_dict():
        torch.testing.assert_close(
            result_a.model.state_dict()[k], result_b.model.state_dict()[k]
        )


def test_best_state_is_restored_after_training() -> None:
    """After training, ``model`` has the weights from ``best_epoch``."""
    X_train, y_train, X_val, y_val = _make_separable_data()
    model = ChurnMLP(n_features=X_train.shape[1])
    result = train_mlp(
        model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=10,
        patience=15,  # disable early stopping in this test
    )
    # Verify the restored weights produce ``best_val_loss`` on val.
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_val, dtype=torch.float32))
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(_compute_pos_weight(y_train))
        )
        val_loss = loss_fn(logits, torch.as_tensor(y_val, dtype=torch.float32)).item()
    assert val_loss == pytest.approx(result.best_val_loss, rel=1e-5)
