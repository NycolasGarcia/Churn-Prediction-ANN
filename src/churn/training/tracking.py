"""MLflow tracking helpers for the churn project.

Standardises how runs are configured, what metrics are computed, and how
artefacts are logged so the notebook in sub-checkpoint 2.4 — and the MLP
training in Phase 3 — do not rebuild the same boilerplate.

Conventions are documented in ``docs/ml_flow_tracking.md``. The decision
to log CV folds as a single aggregated run (as opposed to nested runs)
is justified in ADR-008 (``docs/architecture.md``).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import (
    AUTHOR,
    DATASET_VERSION,
    MLFLOW_EXPERIMENT_NAME,
    SEED,
)

logger = logging.getLogger(__name__)


# Canonical metric set logged on every run. Centralised so callers (tests,
# notebook, future MLP code) iterate over a single source of truth.
METRIC_KEYS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_mlflow(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> None:
    """Configure the MLflow tracking URI and set the active experiment.

    Idempotent: calling twice with the same arguments is safe.
    ``tracking_uri`` is left ``None`` in production so MLflow uses the
    default ``./mlruns`` (gitignored) — the argument exists primarily so
    tests can point at a temp directory without touching the project's
    real run history.

    Args:
        experiment_name: Name of the MLflow experiment to set as active.
            Defaults to :data:`churn.config.MLFLOW_EXPERIMENT_NAME`.
        tracking_uri: Optional override for the MLflow tracking URI.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow ready - experiment=%r tracking_uri=%s",
        experiment_name,
        mlflow.get_tracking_uri(),
    )


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute the project's standard classification metric set.

    Pure function — no MLflow side effects. Keys match :data:`METRIC_KEYS`
    so callers can prefix and log uniformly (``<metric>_mean``,
    ``holdout_val_<metric>``, etc.).

    ``precision_score`` is called with ``zero_division=0`` because the
    Dummy baseline predicts the majority class only — precision is
    undefined there and we want a numeric ``0.0`` rather than a warning
    + ``UndefinedMetricWarning``.

    Args:
        y_true: Ground-truth binary labels (1D).
        y_pred: Hard predictions, ``0``/``1`` (1D).
        y_proba: Probability of the positive class (1D).

    Returns:
        Dict with keys ``roc_auc``, ``pr_auc``, ``f1``, ``precision``,
        ``recall`` — values are ``float`` in ``[0, 1]``.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def log_confusion_matrix_artifact(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion matrix (val holdout)",
    artifact_name: str = "confusion_matrix.png",
) -> None:
    """Render and log a confusion matrix as an MLflow artefact.

    Must be called **inside** an active ``mlflow.start_run`` context — it
    relies on the implicit active run for ``mlflow.log_artifact``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Hard predictions.
        title: Title rendered on top of the figure.
        artifact_name: Filename used inside the MLflow artefact store.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / artifact_name
        fig.savefig(out, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CV + holdout run (the workhorse for sub-checkpoint 2.4)
# ---------------------------------------------------------------------------


def _infer_n_features(pipeline: Pipeline, x_fallback: pd.DataFrame) -> int:
    """Return the post-transform feature count, falling back to input width."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None:
        try:
            return int(len(preprocessor.get_feature_names_out()))
        except Exception:  # pragma: no cover - defensive
            pass
    return int(x_fallback.shape[1])


def log_baseline_cv_run(
    *,
    model_name: str,
    build_pipeline: Callable[[], Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    cv_folds: int = 5,
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Run stratified CV + holdout evaluation, log everything to one run.

    Mechanics (per ADR-008):

    1. Stratified ``cv_folds``-fold CV on ``(X_train, y_train)``. Each
       fold uses a fresh pipeline built by ``build_pipeline()``, so the
       preprocessor (StandardScaler statistics, OneHotEncoder categories)
       is fit only on that fold's train slice — no leakage.
    2. For each fold the metric set is computed on the fold's val slice
       and logged as ``<metric>_fold_<i>``; aggregates are logged as
       ``<metric>_mean`` and ``<metric>_std``.
    3. A fresh pipeline is refit on the **full** ``(X_train, y_train)``
       and evaluated on the holdout ``(X_val, y_val)``. Those numbers are
       logged as ``holdout_val_<metric>`` and used downstream for
       threshold / cost calibration (see :data:`churn.config.COST_*`).
    4. The refitted pipeline is logged as the run's model artefact and a
       confusion matrix from the holdout predictions is attached.

    Args:
        model_name: Run name (e.g. ``"logreg_baseline"``). Doubles as the
            handle used in the comparison notebook.
        build_pipeline: Zero-arg factory returning a fresh
            ``sklearn.pipeline.Pipeline``. Called once per fold and once
            for the holdout refit.
        X_train, y_train: Training portion of the project split.
        X_val, y_val: Holdout val portion of the project split.
        params: Hyperparameters to log. Canonical params
            (``model_type``, ``class_weight``, ``seed``,
            ``dataset_version``, ``n_features``, ``cv_folds``,
            ``cv_strategy``) are filled automatically and cannot be
            overridden here — use ``extra_tags`` for those.
        cv_folds: Number of stratified folds (default ``5``).
        extra_tags: Tags merged on top of the mandatory
            ``model_type`` / ``dataset_version`` / ``author`` set.

    Returns:
        The MLflow ``run_id`` of the logged run.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    fold_metrics: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}

    for train_idx, val_idx in skf.split(X_train, y_train):
        x_fit, x_eval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fit, y_eval = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe = build_pipeline()
        pipe.fit(x_fit, y_fit)
        y_pred = pipe.predict(x_eval)
        y_proba = pipe.predict_proba(x_eval)[:, 1]

        metrics = compute_classification_metrics(y_eval, y_pred, y_proba)
        for key, value in metrics.items():
            fold_metrics[key].append(value)

    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train, y_train)
    y_val_pred = final_pipeline.predict(X_val)
    y_val_proba = final_pipeline.predict_proba(X_val)[:, 1]
    holdout_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)

    classifier = final_pipeline.named_steps.get("classifier")
    model_type = type(classifier).__name__ if classifier is not None else "unknown"
    canonical_params: dict[str, Any] = {
        "model_type": model_type,
        "class_weight": getattr(classifier, "class_weight", None),
        "seed": SEED,
        "dataset_version": DATASET_VERSION,
        "n_features": _infer_n_features(final_pipeline, X_train),
        "cv_folds": cv_folds,
        "cv_strategy": "stratified",
    }
    # Caller-supplied params take precedence over canonical defaults but the
    # canonical *keys* are always present — even when ``params`` is None.
    merged_params = {**canonical_params, **(params or {})}

    tags = {
        "model_type": model_type,
        "dataset_version": DATASET_VERSION,
        "author": AUTHOR,
    }
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(merged_params)
        mlflow.set_tags(tags)

        for key, values in fold_metrics.items():
            arr = np.asarray(values, dtype=float)
            mlflow.log_metric(f"{key}_mean", float(arr.mean()))
            # ddof=0 (population std) is the convention for CV variance —
            # we are summarising a fixed set of folds, not estimating a
            # super-population. Switch to ddof=1 only if folds become a
            # random subsample, which is not the case here.
            mlflow.log_metric(f"{key}_std", float(arr.std(ddof=0)))
            for fold_idx, val in enumerate(values, start=1):
                mlflow.log_metric(f"{key}_fold_{fold_idx}", float(val))

        for key, value in holdout_metrics.items():
            mlflow.log_metric(f"holdout_val_{key}", float(value))

        log_confusion_matrix_artifact(y_val, y_val_pred)
        # Positional ``"model"`` works across the mlflow 2.x / 3.x split
        # where the keyword name changed from ``artifact_path`` to ``name``.
        mlflow.sklearn.log_model(final_pipeline, "model")

        logger.info(
            "Logged run %r - roc_auc_mean=%.4f holdout_val_roc_auc=%.4f",
            model_name,
            float(np.mean(fold_metrics["roc_auc"])),
            holdout_metrics["roc_auc"],
        )
        return run.info.run_id
