"""Raw dataset loader with manual schema validation.

The loader is intentionally read-only: it returns the raw DataFrame as-is
(no type coercion, no cleaning). All transformations live downstream in
the preprocessing pipeline so that the EDA notebook always sees the same
view that the production pipeline starts from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churn.config import (
    EXPECTED_RAW_COLUMNS,
    EXPECTED_RAW_SHAPE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class RawDataValidationError(ValueError):
    """Raised when the raw dataset fails schema validation."""


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the raw Telco churn dataset.

    Args:
        path: Optional override for the dataset location. When ``None``,
            falls back to :data:`churn.config.RAW_DATA_PATH`.

    Returns:
        The raw DataFrame, unmodified, with 33 columns and 7043 rows.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        RawDataValidationError: If the loaded frame fails schema validation.
    """
    resolved_path = Path(path) if path is not None else RAW_DATA_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {resolved_path}")

    logger.info("Loading raw data from %s", resolved_path)
    df = pd.read_excel(resolved_path, engine="openpyxl")
    logger.info("Loaded raw data: %d rows x %d columns", df.shape[0], df.shape[1])

    validate_raw_data(df)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate that ``df`` matches the expected raw schema.

    Performs three checks, fail-fast in this order:

    1. Shape equals :data:`churn.config.EXPECTED_RAW_SHAPE`.
    2. Column names equal :data:`churn.config.EXPECTED_RAW_COLUMNS` (order
       included — keeps drift in upstream exports visible).
    3. Target column contains only the binary values ``{0, 1}``.

    Args:
        df: DataFrame to validate.

    Raises:
        RawDataValidationError: If any of the checks fails.
    """
    actual_shape = df.shape
    if actual_shape != EXPECTED_RAW_SHAPE:
        raise RawDataValidationError(
            f"Unexpected shape {actual_shape}, expected {EXPECTED_RAW_SHAPE}"
        )

    actual_cols = tuple(df.columns)
    if actual_cols != EXPECTED_RAW_COLUMNS:
        missing = sorted(set(EXPECTED_RAW_COLUMNS) - set(actual_cols))
        unexpected = sorted(set(actual_cols) - set(EXPECTED_RAW_COLUMNS))
        raise RawDataValidationError(
            f"Column mismatch — missing={missing}, unexpected={unexpected}"
        )

    # ``.tolist()`` casts numpy scalars to native Python types so error
    # messages don't leak ``np.int64(...)`` noise.
    target_values = set(df[TARGET_COLUMN].dropna().unique().tolist())
    invalid = target_values - {0, 1}
    if invalid:
        raise RawDataValidationError(
            f"Target column {TARGET_COLUMN!r} has unexpected values: {sorted(invalid)}"
        )

    logger.info("Raw data schema validated successfully")
