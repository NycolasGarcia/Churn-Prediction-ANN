"""Tests for ``src/churn/models/mlp.py`` — smoke, schema, and API.

Three categories (same convention as ``test_baseline.py``, see ADR-006):

- **Smoke** — module instantiates and forwards a batch without errors.
- **Schema** — output shape and dtype are what downstream code expects.
- **API** — public contract (architecture wiring, parameter count,
  determinism given the project SEED) is preserved.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from churn.config import SEED
from churn.models.mlp import DROPOUT_RATES, HIDDEN_DIMS, ChurnMLP

# ---------------------------------------------------------------------------
# Smoke — instantiate + forward
# ---------------------------------------------------------------------------


def test_mlp_instantiates_and_forwards() -> None:
    """Forward of a small batch produces a 1-D tensor of finite floats."""
    torch.manual_seed(SEED)
    model = ChurnMLP(n_features=27)
    model.eval()
    x = torch.randn(4, 27)
    out = model(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Schema — shape and dtype
# ---------------------------------------------------------------------------


def test_mlp_output_shape_matches_batch_dim() -> None:
    """Output is shape ``(batch,)`` regardless of batch size."""
    torch.manual_seed(SEED)
    model = ChurnMLP(n_features=27)
    model.eval()
    for batch in (1, 4, 64, 256):
        out = model(torch.randn(batch, 27))
        assert out.shape == (batch,), f"batch={batch}: got {out.shape}"


def test_mlp_output_dtype_is_float32() -> None:
    """Logits are float32 by default — required for BCEWithLogitsLoss."""
    torch.manual_seed(SEED)
    model = ChurnMLP(n_features=27)
    model.eval()
    out = model(torch.randn(8, 27))
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# API — architecture wiring and contract
# ---------------------------------------------------------------------------


def test_mlp_default_architecture_matches_spec() -> None:
    """Default hidden dims / dropout rates match the CLAUDE.md spec."""
    assert HIDDEN_DIMS == (64, 32)
    assert DROPOUT_RATES == (0.3, 0.2)
    model = ChurnMLP(n_features=27)
    assert model.hidden_dims == (64, 32)
    assert model.dropout_rates == (0.3, 0.2)
    assert model.n_features == 27


def test_mlp_first_module_is_batchnorm() -> None:
    """The very first transformation must be BatchNorm1d (CLAUDE.md spec)."""
    model = ChurnMLP(n_features=27)
    assert isinstance(model.input_norm, nn.BatchNorm1d)
    assert model.input_norm.num_features == 27


def test_mlp_hidden_block_order_is_linear_relu_dropout() -> None:
    """Each hidden layer triplet is ordered ``Linear -> ReLU -> Dropout``."""
    model = ChurnMLP(n_features=27)
    children = list(model.hidden.children())
    # 2 hidden layers x 3 modules each = 6 modules total.
    assert len(children) == 6
    expected = [
        (nn.Linear, 27, 64),
        (nn.ReLU, None, None),
        (nn.Dropout, 0.3, None),
        (nn.Linear, 64, 32),
        (nn.ReLU, None, None),
        (nn.Dropout, 0.2, None),
    ]
    for module, (cls, in_dim, out_dim) in zip(children, expected, strict=True):
        assert isinstance(module, cls), f"expected {cls}, got {type(module)}"
        if cls is nn.Linear:
            assert module.in_features == in_dim
            assert module.out_features == out_dim
        elif cls is nn.Dropout:
            assert module.p == in_dim


def test_mlp_output_layer_is_single_neuron() -> None:
    """Final layer projects to one logit."""
    model = ChurnMLP(n_features=27)
    assert isinstance(model.output, nn.Linear)
    assert model.output.in_features == 32
    assert model.output.out_features == 1


def test_mlp_is_deterministic_under_fixed_seed() -> None:
    """Two builds under the same seed produce identical forward outputs."""
    x = torch.randn(8, 27)

    torch.manual_seed(SEED)
    model_a = ChurnMLP(n_features=27)
    model_a.eval()
    out_a = model_a(x)

    torch.manual_seed(SEED)
    model_b = ChurnMLP(n_features=27)
    model_b.eval()
    out_b = model_b(x)

    torch.testing.assert_close(out_a, out_b)


def test_mlp_rejects_mismatched_hidden_dropout_lengths() -> None:
    """Constructor validates that hidden_dims and dropout_rates align."""
    with pytest.raises(ValueError, match="same length"):
        ChurnMLP(n_features=27, hidden_dims=(64, 32, 16), dropout_rates=(0.3, 0.2))
