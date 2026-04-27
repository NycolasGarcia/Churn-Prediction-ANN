"""Multilayer perceptron for the churn classification task.

Architecture (CLAUDE.md §6 Fase 3)::

    Input -> BatchNorm1d
          -> Linear(n_features -> 64) -> ReLU -> Dropout(0.3)
          -> Linear(64 -> 32)         -> ReLU -> Dropout(0.2)
          -> Linear(32 -> 1)

The module returns **logits** (no final ``Sigmoid``). The training loop
pairs them with :class:`torch.nn.BCEWithLogitsLoss`, which is numerically
more stable than ``Sigmoid + BCELoss`` (it combines log-sigmoid into a
single fused op) and accepts ``pos_weight`` directly to compensate the
~26% / 74% class imbalance — the BCE-side analogue of
``class_weight="balanced"`` in the LogReg baseline.

The first ``BatchNorm1d`` is intentional even though the upstream
``StandardScaler`` (in :func:`churn.data.preprocessing.build_preprocessing_pipeline`)
already centres / scales the inputs. The two normalisations have
different roles:

- ``StandardScaler`` uses **train-set statistics** (fit once, applied at
  inference) and is part of the deterministic preprocessing pipeline.
- ``BatchNorm1d`` uses **per-batch statistics during training** and a
  running mean/var at eval time. It acts as a mild regulariser and
  reduces internal covariate shift between layers — orthogonal to what
  the scaler does upstream.

Reproducibility is the trainer's responsibility: callers must invoke
``torch.manual_seed(SEED)`` *before* instantiating the module so the
default Kaiming-uniform init of each ``Linear`` becomes deterministic.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

# Architecture constants — single source of truth for MLP shape.
HIDDEN_DIMS: Final[tuple[int, ...]] = (64, 32)
DROPOUT_RATES: Final[tuple[float, ...]] = (0.3, 0.2)


class ChurnMLP(nn.Module):
    """MLP for binary churn classification.

    Args:
        n_features: Width of the post-preprocessing input vector. With the
            current pipeline (27 features) this is ``27``; explicit so the
            module stays reusable when the feature set changes.
        hidden_dims: Two-tuple of hidden layer widths. Defaults to
            :data:`HIDDEN_DIMS` ``(64, 32)``.
        dropout_rates: Two-tuple of dropout probabilities applied after
            each hidden layer. Defaults to :data:`DROPOUT_RATES`
            ``(0.3, 0.2)``.

    Forward output:
        Tensor of shape ``(batch,)`` — raw logits, no activation. Pair
        with ``BCEWithLogitsLoss`` for training, ``torch.sigmoid`` for
        probabilities at inference.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
        dropout_rates: tuple[float, ...] = DROPOUT_RATES,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropout_rates):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dropout_rates "
                f"({len(dropout_rates)}) must have the same length"
            )

        self.input_norm = nn.BatchNorm1d(n_features)

        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden, dropout in zip(hidden_dims, dropout_rates, strict=True):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.hidden = nn.Sequential(*layers)

        self.output = nn.Linear(in_dim, 1)

        self.n_features: Final[int] = n_features
        self.hidden_dims: Final[tuple[int, ...]] = tuple(hidden_dims)
        self.dropout_rates: Final[tuple[float, ...]] = tuple(dropout_rates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.hidden(x)
        x = self.output(x)
        return x.squeeze(-1)
