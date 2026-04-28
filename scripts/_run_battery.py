#!/usr/bin/env python
"""Systematic 48-run training battery — PLANO-DE-TREINAMENTO.md.

Covers:
- 30 baseline runs  (5 models x 2 splits x 3 tenure variants)
- 18 MLP runs       (3 batch sizes x 2 splits x 3 tenure variants)

Checks MLflow for existing run names before training (skip-if-exists),
so the script is safe to re-run after a partial failure.

Usage:
    python scripts/_run_battery.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from churn.config import MLFLOW_EXPERIMENT_NAME, SEED
from churn.data.loader import load_raw_data
from churn.data.preprocessing import (
    build_preprocessing_pipeline,
    clean_raw,
    split_features_target,
)
from churn.models.baseline import build_dummy_baseline, build_logreg_baseline
from churn.training.tracking import (
    log_baseline_cv_run,
    log_mlp_cv_run,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("battery")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def existing_run_names() -> set[str]:
    """Return set of run names already present in the active experiment."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string="",
            max_results=10_000,
        )
        if runs.empty:
            return set()
        col = "tags.mlflow.runName"
        if col not in runs.columns:
            return set()
        return set(runs[col].dropna().tolist())
    except Exception:
        return set()


def make_split(raw_df, *, test_size: float, val_size: float):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test) for given sizes."""
    cleaned = clean_raw(raw_df)
    X, y = split_features_target(cleaned)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, stratify=y_temp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

SPLIT_CONFIGS: dict[str, dict] = {
    "7015": {"test_size": 0.15, "val_size": 0.15},
    "8010": {"test_size": 0.10, "val_size": 0.10},
}

TENURE_VARIANTS: list[str] = ["orig", "le", "ohe"]
BATCH_SIZES: list[int] = [64, 32, 16]


def run_baseline_experiments(
    *,
    split_tag: str,
    tenure: str,
    X_train,
    X_val,
    y_train,
    y_val,
    done: set[str],
) -> None:
    """Run all 5 baseline variants for one (split, tenure) combination."""

    def skip(name: str) -> bool:
        if name in done:
            logger.info("SKIP (exists): %s", name)
            return True
        return False

    # 1. Dummy
    name = f"dummy_{split_tag}_{tenure}"
    if not skip(name):
        logger.info("START: %s", name)
        log_baseline_cv_run(
            model_name=name,
            build_pipeline=lambda: build_dummy_baseline(tenure_variant=tenure),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"tenure_variant": tenure, "split": split_tag},
        )

    # 2. LogReg full
    name = f"logreg_{split_tag}_{tenure}"
    if not skip(name):
        logger.info("START: %s", name)
        log_baseline_cv_run(
            model_name=name,
            build_pipeline=lambda t=tenure: build_logreg_baseline(tenure_variant=t),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"tenure_variant": tenure, "split": split_tag},
        )

    # 3. LogReg without Multiple Lines
    name = f"logreg_noml_{split_tag}_{tenure}"
    if not skip(name):
        logger.info("START: %s", name)
        log_baseline_cv_run(
            model_name=name,
            build_pipeline=lambda t=tenure: build_logreg_baseline(
                exclude_columns=("Multiple Lines",), tenure_variant=t
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={
                "tenure_variant": tenure,
                "split": split_tag,
                "exclude_columns": "Multiple Lines",
            },
        )

    # 4. LogReg without Phone Service
    name = f"logreg_nophone_{split_tag}_{tenure}"
    if not skip(name):
        logger.info("START: %s", name)
        log_baseline_cv_run(
            model_name=name,
            build_pipeline=lambda t=tenure: build_logreg_baseline(
                exclude_columns=("Phone Service",), tenure_variant=t
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={
                "tenure_variant": tenure,
                "split": split_tag,
                "exclude_columns": "Phone Service",
            },
        )

    # 5. LogReg without Phone Service + Multiple Lines
    name = f"logreg_nophone_noml_{split_tag}_{tenure}"
    if not skip(name):
        logger.info("START: %s", name)
        log_baseline_cv_run(
            model_name=name,
            build_pipeline=lambda t=tenure: build_logreg_baseline(
                exclude_columns=("Phone Service", "Multiple Lines"), tenure_variant=t
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={
                "tenure_variant": tenure,
                "split": split_tag,
                "exclude_columns": "Phone Service,Multiple Lines",
            },
        )


def run_mlp_experiments(
    *,
    split_tag: str,
    tenure: str,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    done: set[str],
) -> None:
    """Run MLP with 3 batch sizes for one (split, tenure) combination."""
    is_8010 = split_tag == "8010"

    for batch in BATCH_SIZES:
        name = f"mlp_{split_tag}_{tenure}_b{batch}"
        if name in done:
            logger.info("SKIP (exists): %s", name)
            continue

        logger.info("START: %s", name)
        # For batch=16 activate gradient clipping (noisy small batches benefit)
        grad_norm = 1.0 if batch <= 16 else 0.0
        log_mlp_cv_run(
            model_name=name,
            build_preprocessor=lambda t=tenure: build_preprocessing_pipeline(
                tenure_variant=t
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test if is_8010 else None,
            y_test=y_test if is_8010 else None,
            train_kwargs={
                "batch_size": batch,
                "max_grad_norm": grad_norm,
            },
            extra_tags={
                "tenure_variant": tenure,
                "split": split_tag,
                "batch_size": str(batch),
            },
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_mlflow()
    raw_df = load_raw_data()
    logger.info("Raw data loaded: %d rows", len(raw_df))

    for split_tag, split_cfg in SPLIT_CONFIGS.items():
        logger.info("=== Split: %s ===", split_tag)
        X_train, X_val, X_test, y_train, y_val, y_test = make_split(
            raw_df, **split_cfg
        )
        logger.info(
            "Split sizes — train=%d val=%d test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        for tenure in TENURE_VARIANTS:
            logger.info("--- Tenure variant: %s ---", tenure)

            # Refresh done-set each iteration so it picks up runs completed
            # in the current session (in case of partial previous failures).
            done = existing_run_names()

            run_baseline_experiments(
                split_tag=split_tag,
                tenure=tenure,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                done=done,
            )

            done = existing_run_names()

            run_mlp_experiments(
                split_tag=split_tag,
                tenure=tenure,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                done=done,
            )

    logger.info("Battery complete.")


if __name__ == "__main__":
    main()
