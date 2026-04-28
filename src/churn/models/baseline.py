"""Baseline models — sklearn `Pipeline`s of (preprocessor + classifier).

Two baselines are provided:

- :func:`build_dummy_baseline` — ``DummyClassifier(strategy="most_frequent")``,
  the absolute floor any real model must beat.
- :func:`build_logreg_baseline` — ``LogisticRegression`` with
  ``class_weight="balanced"`` to handle the ~26% / 74% imbalance.

Each builder returns a fresh :class:`~sklearn.pipeline.Pipeline` whose
first step is a fresh ``ColumnTransformer`` from
:func:`churn.data.preprocessing.build_preprocessing_pipeline`. Building
the pipeline this way is what makes :func:`~sklearn.model_selection.cross_val_score`
correct — the preprocessor is refit on the train folds only,
preventing leakage from val into the StandardScaler statistics or the
OneHotEncoder categories.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import SEED
from churn.data.preprocessing import build_preprocessing_pipeline


def build_dummy_baseline(*, tenure_variant: str = "orig") -> Pipeline:
    """Build the most-frequent baseline.

    Always predicts the majority class (``Churn Value == 0``, ~73.5% of
    the dataset). Serves as a sanity floor.

    Args:
        tenure_variant: Forwarded to :func:`build_preprocessing_pipeline`.

    Returns:
        ``Pipeline([("preprocessor", ...), ("classifier", DummyClassifier(...))])``.
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessing_pipeline(tenure_variant=tenure_variant),
            ),
            (
                "classifier",
                DummyClassifier(strategy="most_frequent", random_state=SEED),
            ),
        ]
    )


def build_logreg_baseline(
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    exclude_columns: tuple[str, ...] = (),
    tenure_variant: str = "orig",
) -> Pipeline:
    """Build the regularised logistic regression baseline.

    L2 regularisation with ``class_weight="balanced"`` to compensate the
    ~26% / 74% class imbalance.

    Args:
        C: Inverse regularisation strength (default 1.0).
        max_iter: Maximum LBFGS iterations (default 1000 avoids convergence
            warnings on this dataset).
        exclude_columns: Forwarded to :func:`build_preprocessing_pipeline`.
            Used by the 2x2 Phone/Multiple-Lines ablation grid (ADR-005).
        tenure_variant: Forwarded to :func:`build_preprocessing_pipeline`.
            Controls which tenure feature engineering is applied.

    Returns:
        ``Pipeline([("preprocessor", ...), ("classifier", LogisticRegression(...))])``.
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessing_pipeline(
                    exclude_columns=exclude_columns,
                    tenure_variant=tenure_variant,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
