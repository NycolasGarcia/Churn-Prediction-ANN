"""Shared pytest fixtures.

Builds train/val/test DataFrames from the raw Excel on demand. Cached at
session scope so the heavy load + clean + split path runs once per pytest
invocation, keeping the test suite fast enough to stay under a few seconds.
"""

from __future__ import annotations

import pytest

from churn.data.loader import load_raw_data
from churn.data.preprocessing import (
    SplitData,
    clean_raw,
    split_features_target,
    stratified_split,
)


@pytest.fixture(scope="session")
def split_data() -> SplitData:
    """Full 70/15/15 stratified split, computed once per test session."""
    df_raw = load_raw_data()
    df_clean = clean_raw(df_raw)
    X, y = split_features_target(df_clean)
    return stratified_split(X, y)
