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


def build_dummy_baseline() -> Pipeline:
    """Build the most-frequent baseline.

    Always predicts the majority class (``Churn Value == 0``, ~73,5% of
    the dataset). It serves as a sanity floor: any real model must beat
    it on every metric, otherwise something is wrong.

    Returns:
        ``Pipeline([("preprocessor", ...), ("classifier", DummyClassifier(...))])``.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
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
) -> Pipeline:
    """Build the regularised logistic regression baseline.

    L2 regularisation with ``class_weight="balanced"`` to compensate the
    ~26% / 74% class imbalance — equivalent to inversely weighting each
    sample by its class frequency, so the model does not collapse to
    predicting the majority class.

    Args:
        C: Inverse regularisation strength. ``C = 1.0`` is sklearn's
            default. Lower values increase regularisation.
        max_iter: Maximum LBFGS iterations. The default of 100 sometimes
            issues convergence warnings on this dataset; 1000 is enough
            for the LogReg to converge cleanly.
        exclude_columns: Forwarded to
            :func:`churn.data.preprocessing.build_preprocessing_pipeline`.
            Used by the 2x2 Phone Service / Multiple Lines ablation grid
            (ADR-005) to drop one or both features without touching the
            shared preprocessing module.

    Returns:
        ``Pipeline([("preprocessor", ...), ("classifier", LogisticRegression(...))])``.
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessing_pipeline(exclude_columns=exclude_columns),
            ),
            (
                "classifier",
                # ``penalty`` is left at its default ("l2") — explicitly
                # passing ``penalty="l2"`` raises a FutureWarning on
                # scikit-learn 1.8+ and will be removed in 1.10.
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
