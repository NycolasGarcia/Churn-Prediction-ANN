"""PyTorch training loop for the churn MLP.

Mini-batch SGD with Adam, ``BCEWithLogitsLoss`` (numerical-stable pair
of the logit-returning :class:`~churn.models.mlp.ChurnMLP`), automatic
``pos_weight`` from the training labels, and early stopping monitoring
val loss with weight restoration. The trainer is data-agnostic — it
operates on already-preprocessed numpy arrays so that the upstream
``ColumnTransformer`` can be fit per CV fold without leakage.

CLAUDE.md §6 Fase 3 spec:

- Optimizer: Adam, ``lr = 1e-3``.
- Loss: ``BCEWithLogitsLoss`` with ``pos_weight`` (BCE-side analogue of
  ``class_weight='balanced'``).
- Batch size: 32 (reduced from 64 for implicit regularisation benefit).
- Max epochs: 100, early stopping patience 10 on ``val_loss``.
- Seed: ``churn.config.SEED`` (the trainer sets the torch seed before
  any random op).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.config import SEED
from churn.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)


# Default hyperparameters (CLAUDE.md §6 Fase 3).
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_MAX_EPOCHS: int = 100
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_PATIENCE: int = 10

# LR scheduler defaults — ReduceLROnPlateau reduces lr by factor when
# val_loss shows no improvement for scheduler_patience epochs. When the LR
# is actually reduced, the early-stopping counter is reset so the model gets
# a fresh patience window at the new (lower) learning rate.
DEFAULT_SCHEDULER_FACTOR: float = 0.5
DEFAULT_SCHEDULER_PATIENCE: int = 5


@dataclass(frozen=True)
class EpochMetrics:
    """One row of training history."""

    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float


@dataclass
class TrainingResult:
    """Output of :func:`train_mlp`.

    The model carries the **best** weights observed during training
    (restored after early stopping or after the final epoch, whichever
    came first). ``history`` covers every epoch executed, including the
    ones after the best epoch but before the patience window expired.
    """

    model: ChurnMLP
    history: list[EpochMetrics]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Binary focal loss with pos_weight.

    Focal loss down-weights easy examples (high-confidence predictions)
    and concentrates gradient on the hard-to-classify minority class.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t), with pos_weight rebalancing.

    Args:
        logits: Raw model outputs (not sigmoided).
        targets: Binary ground-truth labels.
        pos_weight: Per-sample weight for the positive class.
        gamma: Focusing parameter. ``0`` recovers standard BCE.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p_t = torch.exp(-bce)
    return ((1.0 - p_t) ** gamma * bce).mean()


def _compute_pos_weight(y_train: np.ndarray) -> float:
    """Return ``num_negative / num_positive`` for ``BCEWithLogitsLoss``.

    This is the pos_weight that, multiplied into the positive-class
    loss term, makes the effective per-class contribution balanced —
    equivalent in spirit to ``class_weight='balanced'`` in sklearn.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError("y_train has no positive samples; cannot fit MLP")
    return float(n_neg) / float(n_pos)


def _evaluate(
    model: ChurnMLP,
    loss_fn: nn.BCEWithLogitsLoss,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    """Compute ``(val_loss, val_auc)`` on the full validation slice."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        proba = torch.sigmoid(logits).cpu().numpy()
    auc = float(roc_auc_score(y.cpu().numpy(), proba))
    return float(loss), auc


def train_mlp(
    model: ChurnMLP,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    pos_weight: float | None = None,
    seed: int = SEED,
    use_lr_scheduler: bool = True,
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR,
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
    focal_gamma: float = 0.0,
    max_grad_norm: float = 1.0,
) -> TrainingResult:
    """Train ``model`` with early stopping on ``val_loss``.

    The trainer expects already-preprocessed inputs (numpy float arrays
    coming out of the project's ``ColumnTransformer``). It does **not**
    fit a preprocessor — that responsibility lives in the calling layer
    so that CV folds can refit the preprocessor on the train split only.

    Args:
        model: A fresh :class:`~churn.models.mlp.ChurnMLP`. The trainer
            will mutate its parameters in-place. Pass a new instance per
            fold when running CV.
        X_train, y_train: Training inputs and binary labels.
        X_val, y_val: Validation inputs and binary labels (used only for
            early stopping and metric logging — never for fitting).
        batch_size: Mini-batch size for Adam updates.
        max_epochs: Hard cap on epochs; early stopping usually fires
            earlier.
        learning_rate: Adam learning rate.
        patience: Stop after this many epochs without ``val_loss``
            improvement.
        pos_weight: Override for ``BCEWithLogitsLoss``'s ``pos_weight``.
            ``None`` (default) auto-computes ``num_neg / num_pos`` from
            ``y_train`` — this is the BCE-side analogue of
            ``class_weight='balanced'`` and is what the project uses.
        seed: Torch RNG seed; set before any data shuffling so the loop
            is deterministic given the inputs.
        use_lr_scheduler: When ``True`` (default), wraps the optimizer with
            :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`. When the
            scheduler actually reduces the LR, the early-stopping patience
            counter is reset, giving the model a fresh window at the new LR
            before the run is terminated.
        scheduler_factor: Multiplicative factor applied to the LR on each
            reduction (default ``0.5`` → halves the LR).
        scheduler_patience: Number of epochs with no ``val_loss`` improvement
            before the LR is reduced (default ``5``).
        max_grad_norm: Max L2 norm for gradient clipping (default ``1.0``).
            Set to ``0.0`` to disable clipping.

    Returns:
        A :class:`TrainingResult` whose ``model`` carries the best
        weights observed during training.
    """
    torch.manual_seed(seed)

    if pos_weight is None:
        pos_weight = _compute_pos_weight(y_train)

    x_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    # ``drop_last=True`` so the last partial batch never falls below the
    # 2-sample minimum that BatchNorm1d needs in train mode. With
    # ~4000 train samples per CV fold and batch_size=64 we drop at most
    # ~30 samples per epoch — negligible relative to the train pool.
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pw_tensor = torch.tensor(pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )
        if use_lr_scheduler
        else None
    )

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = (
                _focal_loss(logits, y_batch, pw_tensor, focal_gamma)
                if focal_gamma > 0.0
                else loss_fn(logits, y_batch)
            )
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
            optimizer.step()
            running_loss += loss.item() * x_batch.shape[0]
            n_seen += x_batch.shape[0]
        train_loss = running_loss / max(n_seen, 1)

        val_loss, val_auc = _evaluate(model, loss_fn, x_val_t, y_val_t)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=val_loss,
                val_auc=val_auc,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_loss=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                # LR was reduced — reset patience so the model gets a fresh
                # window to improve at the new (lower) learning rate.
                epochs_without_improvement = 0
                logger.info(
                    "LR reduced %.2e → %.2e at epoch %d; patience reset",
                    prev_lr,
                    new_lr,
                    epoch,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_early=stopped_early,
    )
