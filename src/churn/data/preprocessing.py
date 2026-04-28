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

# ---------------------------------------------------------------------------
# Feature-engineered column specs
# ---------------------------------------------------------------------------

# Tenure-bin cutoffs in months: (-1,12] | (12,24] | (24,48] | (48,inf]
# Labels use explicit cutoff strings so MLflow params are self-documenting.
_TENURE_BINS: Final[list[float]] = [-1, 12, 24, 48, float("inf")]
_TENURE_OHE_LABELS: Final[list[str]] = ["0-12m", "13-24m", "25-48m", "49+m"]
_TENURE_LE_LABELS: Final[list[int]] = [0, 1, 2, 3]

# Ordinal contract-risk: Month-to-month=2 (highest), One year=1, Two year=0.
_RISCO_MAP: Final[dict[str, int]] = {
    "Month-to-month": 2,
    "One year": 1,
    "Two year": 0,
}

# Columns created by clean_raw and consumed by the pipeline.
# COMMON: available in both "le" and "ohe" tenure variants.
# - risco_contrato:      ordinal int 0–2 (Contract type risk), StandardScaler
# - service_count:       int 0–7 (active add-on services), StandardScaler
# - is_new:             binary 0/1 (Tenure ≤ 3 months), StandardScaler
# - charges_per_tenure: float (Monthly Charges / (Tenure + 1)), StandardScaler
# LE_ONLY: added only when tenure_variant="le".
# - tenure_bin_le: ordinal int 0–3 (bins: 0-12m/13-24m/25-48m/49+m), StandardScaler
# OHE: added only when tenure_variant="ohe" (via dedicated OHE transformer).
# - tenure_bin_ohe: string label "0-12m"…"49+m", OneHotEncoder → 4 binary cols
ENGINEERED_NUMERIC_COMMON: Final[tuple[str, ...]] = (
    "risco_contrato",
    "service_count",
    "is_new",
    "charges_per_tenure",
)
ENGINEERED_NUMERIC_LE_ONLY: Final[tuple[str, ...]] = ("tenure_bin_le",)
ENGINEERED_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    *ENGINEERED_NUMERIC_COMMON,
    *ENGINEERED_NUMERIC_LE_ONLY,
)
ENGINEERED_CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("tenure_bin_ohe",)

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

    # Engineered features — always created; the ColumnTransformer drops them
    # silently (remainder="drop") when tenure_variant="orig".
    cleaned["tenure_bin_le"] = pd.cut(
        cleaned["Tenure Months"],
        bins=_TENURE_BINS,
        labels=_TENURE_LE_LABELS,
        right=True,
    ).astype(int)
    cleaned["tenure_bin_ohe"] = pd.cut(
        cleaned["Tenure Months"],
        bins=_TENURE_BINS,
        labels=_TENURE_OHE_LABELS,
        right=True,
    ).astype(str)
    cleaned["risco_contrato"] = cleaned["Contract"].map(_RISCO_MAP).astype(int)

    # service_count: number of active add-on services (0–7).
    # Proxy for switching cost — customers with more services are harder to churn.
    service_cols = list(NO_INTERNET_SERVICE_COLUMNS) + ["Multiple Lines"]
    cleaned["service_count"] = (cleaned[service_cols] == "Yes").sum(axis=1).astype(int)

    # is_new: Tenure ≤ 3 months — highest churn period identified in EDA.
    cleaned["is_new"] = (cleaned["Tenure Months"] <= 3).astype(int)

    # charges_per_tenure: price pressure relative to tenure.
    # Dividing by (Tenure + 1) avoids division by zero for new customers.
    cleaned["charges_per_tenure"] = (
        cleaned["Monthly Charges"] / (cleaned["Tenure Months"] + 1)
    ).astype(float)
    logger.info(
        "Engineered tenure_bin_le, tenure_bin_ohe and risco_contrato added"
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
    *,
    exclude_columns: tuple[str, ...] = (),
    tenure_variant: str = "orig",
) -> ColumnTransformer:
    """Build the :class:`~sklearn.compose.ColumnTransformer` used to
    vectorise the cleaned data.

    - ``StandardScaler`` on continuous numerics (:data:`NUMERIC_COLUMNS`).
    - ``OneHotEncoder(drop="if_binary")`` on binary and multi-class categoricals.
    - ``remainder="drop"`` — defensive: anything not declared is removed.
    - ``handle_unknown="ignore"`` — unseen categories become all-zero at inference.

    Args:
        exclude_columns: Feature names to drop before wiring transformers.
            Used for ablation studies (ADR-005). Names absent from any column
            group are silently ignored.
        tenure_variant: Controls which tenure feature engineering is applied.

            - ``"orig"`` (default): no binning — raw ``Tenure Months`` numeric.
            - ``"le"``: adds ``tenure_bin_le`` (ordinal 0–3, bins: 0–12 / 13–24
              / 25–48 / 49+ months) and ``risco_contrato`` (ordinal 0–2 from
              Contract type) as extra numeric inputs through ``StandardScaler``.
            - ``"ohe"``: adds ``risco_contrato`` (numeric) and ``tenure_bin_ohe``
              (string labels ``"0-12m"``, ``"13-24m"``, ``"25-48m"``, ``"49+m"``)
              through a dedicated ``OneHotEncoder`` → 4 binary indicator columns
              with explicit categories so output shape is always consistent.

            All engineered columns are created unconditionally by
            :func:`clean_raw`; the ColumnTransformer drops unused ones via
            ``remainder="drop"``.
    """
    if tenure_variant not in {"orig", "le", "ohe"}:
        raise ValueError(
            f"tenure_variant must be 'orig', 'le', or 'ohe'; got {tenure_variant!r}"
        )

    excluded = set(exclude_columns)
    numeric_cols = [c for c in NUMERIC_COLUMNS if c not in excluded]
    binary_cols = [c for c in BINARY_COLUMNS if c not in excluded]
    multiclass_cols = [c for c in MULTICLASS_COLUMNS if c not in excluded]

    if tenure_variant in {"le", "ohe"}:
        # Common engineered numerics (risco_contrato, service_count, is_new,
        # charges_per_tenure) are added for both "le" and "ohe".
        numeric_cols += [c for c in ENGINEERED_NUMERIC_COMMON if c not in excluded]
        if tenure_variant == "le":
            # "le" also adds tenure_bin_le (ordinal 0–3) through StandardScaler.
            numeric_cols += [
                c for c in ENGINEERED_NUMERIC_LE_ONLY if c not in excluded
            ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    transformers: list = [
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, binary_cols + multiclass_cols),
    ]

    if tenure_variant == "ohe":
        # Dedicated OHE with fixed categories so output shape is stable.
        tenure_ohe = OneHotEncoder(
            categories=[_TENURE_OHE_LABELS],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        transformers.append(("tenure_ohe", tenure_ohe, ["tenure_bin_ohe"]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
