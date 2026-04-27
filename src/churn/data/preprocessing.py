"""Preprocessing utilities: cleaning, stratified split and sklearn pipeline.

The flow mirrors the production inference path so that the notebook, the
training scripts and the API all apply the exact same transformations:

    raw_df -> clean_raw -> split_features_target -> ColumnTransformer

``clean_raw`` does dataset-level decisions (drop, coercion, imputation,
collapsing redundant categories). The ``ColumnTransformer`` does the
sklearn-native vectorisation (scaling + one-hot).
"""

from __future__ import annotations

import logging
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import LEAKAGE_COLUMNS, SEED, TARGET_COLUMN, TEST_SIZE, VAL_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column groups (decisions justified in 01_eda.ipynb section 11.7)
# ---------------------------------------------------------------------------

# Dropped during cleaning, in addition to LEAKAGE_COLUMNS from config.
# Identifiers, constants, and geo features with no signal (|corr| ~ 0).
DROP_COLUMNS: Final[tuple[str, ...]] = (
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
)

# Service-related columns where a third value ("No internet service" or
# "No phone service") duplicates information already encoded in
# ``Internet Service`` / ``Phone Service``. Collapsing into "No" keeps the
# semantics ("does the customer have this?") and avoids 6+ colinear dummies.
NO_INTERNET_SERVICE_COLUMNS: Final[tuple[str, ...]] = (
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
)

# Continuous numerics; scaled with StandardScaler.
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
)

# Binary categoricals (2 levels after the cleaning step). OneHotEncoder
# with ``drop="if_binary"`` collapses each into a single 0/1 column.
BINARY_COLUMNS: Final[tuple[str, ...]] = (
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Multiple Lines",
)

# True multi-class categoricals. All categories are kept (no drop_first):
# regularisation handles colinearity in the linear baseline; the MLP is
# unaffected; interpretability is cleaner with every level represented.
MULTICLASS_COLUMNS: Final[tuple[str, ...]] = (
    "Contract",
    "Internet Service",
    "Payment Method",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SplitData(NamedTuple):
    """Tuple holding the 70/15/15 stratified split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-level cleaning before the sklearn pipeline.

    Steps (all justified in the EDA notebook):

    1. Drop leakage columns (``Churn Label``, ``Churn Score``, ``Churn Reason``).
    2. Drop identifiers, constants and geo features
       (``CustomerID``, ``Count``, ``Country``, ``State``, ``City``,
       ``Zip Code``, ``Lat Long``, ``Latitude``, ``Longitude``).
    3. Coerce ``Total Charges`` (object) to float, imputing ``0.0`` for the
       11 blank rows — all of which have ``Tenure Months == 0``.
    4. Collapse ``"No internet service"`` (6 columns) and
       ``"No phone service"`` (1 column) into ``"No"`` to remove redundant
       dummies — that information is already in ``Internet Service`` and
       ``Phone Service``.

    Args:
        df: Raw DataFrame as returned by :func:`churn.data.loader.load_raw_data`.

    Returns:
        A new cleaned DataFrame; ``df`` is not modified in place.
    """
    cleaned = df.copy()

    to_drop = list(LEAKAGE_COLUMNS) + list(DROP_COLUMNS)
    cleaned = cleaned.drop(columns=to_drop)
    logger.info("Dropped %d columns (leakage + identifiers + geo)", len(to_drop))

    tc_numeric = pd.to_numeric(cleaned["Total Charges"], errors="coerce")
    n_imputed = int(tc_numeric.isna().sum())
    cleaned["Total Charges"] = tc_numeric.fillna(0.0).astype(float)
    logger.info(
        "Coerced 'Total Charges' to float; imputed %d blank rows with 0.0",
        n_imputed,
    )

    for col in NO_INTERNET_SERVICE_COLUMNS:
        cleaned[col] = cleaned[col].replace("No internet service", "No")
    cleaned["Multiple Lines"] = cleaned["Multiple Lines"].replace(
        "No phone service", "No"
    )
    logger.info(
        "Collapsed 'No internet service' on %d cols and 'No phone service' on 1 col",
        len(NO_INTERNET_SERVICE_COLUMNS),
    )

    return cleaned


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (``X``) from target (``y``).

    Raises:
        KeyError: If :data:`churn.config.TARGET_COLUMN` is not present.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column {TARGET_COLUMN!r} not found in DataFrame")
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
) -> SplitData:
    """70/15/15 stratified split, deterministic given ``seed``.

    The validation slice is taken from the train portion using
    ``val_size / (1 - test_size)`` so the absolute proportions are
    ``(1 - test_size - val_size) / val_size / test_size``.

    Args:
        X: Feature DataFrame.
        y: Binary target Series (0/1).
        test_size: Fraction of the full data reserved for test.
        val_size: Fraction of the full data reserved for validation.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SplitData` named tuple.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_size_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_relative,
        stratify=y_temp,
        random_state=seed,
    )
    logger.info(
        "Stratified split: train=%d val=%d test=%d (target rates: %.4f / %.4f / %.4f)",
        len(X_train),
        len(X_val),
        len(X_test),
        y_train.mean(),
        y_val.mean(),
        y_test.mean(),
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(
    *, exclude_columns: tuple[str, ...] = (),
) -> ColumnTransformer:
    """Build the :class:`~sklearn.compose.ColumnTransformer` used to
    vectorise the cleaned data.

    - ``StandardScaler`` on continuous numerics (:data:`NUMERIC_COLUMNS`).
    - ``OneHotEncoder(drop="if_binary")`` on the union of binary and
      multi-class categoricals: binaries collapse to a single 0/1 dummy,
      multi-class columns keep all levels.
    - ``remainder="drop"`` — defensive: anything not declared is removed,
      so a future schema change becomes a loud error rather than silent leak.
    - ``handle_unknown="ignore"`` — at inference time, unseen categories
      become an all-zero one-hot row instead of an error.

    Args:
        exclude_columns: Optional tuple of feature names to drop from the
            pipeline before the transformers are wired up. Used for
            ablation studies (notably the 2x2 Phone/Multiple-Lines grid
            justified by ADR-005). Names that do not appear in any of the
            three column groups are silently ignored — callers should
            verify their spelling against
            :data:`NUMERIC_COLUMNS` / :data:`BINARY_COLUMNS` /
            :data:`MULTICLASS_COLUMNS`.
    """
    excluded = set(exclude_columns)
    numeric_cols = [c for c in NUMERIC_COLUMNS if c not in excluded]
    binary_cols = [c for c in BINARY_COLUMNS if c not in excluded]
    multiclass_cols = [c for c in MULTICLASS_COLUMNS if c not in excluded]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            (
                "cat",
                categorical_transformer,
                binary_cols + multiclass_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
