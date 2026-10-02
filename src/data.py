"""Dataset loaders plus the config registry that drives the pipeline.

Every loader returns the same six-element tuple, so the modelling, fairness
and mitigation steps never need to know which dataset they are running on:

    X_train, X_test, y_train, y_test, A_train, A_test
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
TARGET_COL = "class"
SENSITIVE_COL = "sex"
# Label and attributes that must never leak into the Adult feature matrix.
DROP_COLS = [TARGET_COL, SENSITIVE_COL, "race", "fnlwgt"]

# Project rule: everything downstream must be reproducible from a fixed seed.
DEFAULT_SEED = 42


def _frame(bunch) -> pd.DataFrame:
    """Return the full frame (features + target) from a fetch_openml Bunch."""
    frame = getattr(bunch, "frame", None)
    if frame is not None:
        return frame
    return pd.concat([bunch.data, bunch.target], axis=1)


def _sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Rewrite one-hot names so every estimator accepts them.

    German Credit categories such as ``<0`` or ``0<=X<200`` leave a ``<`` in
    the column name, and XGBoost rejects feature names containing ``<``, ``[``
    or ``]``. The rewrite is a no-op for Adult.
    """
    X = X.copy()
    X.columns = [
        str(c)
        .replace("<=", "_le_")
        .replace(">=", "_ge_")
        .replace("<", "_lt_")
        .replace(">", "_gt_")
        .replace("[", "_")
        .replace("]", "_")
        for c in X.columns
    ]
    X.columns = [re.sub(r"__+", "_", c) for c in X.columns]
    if X.columns.duplicated().any():
        raise ValueError(f"column names collide after sanitising: {list(X.columns[X.columns.duplicated()])}")
    return X


def load_adult(seed: int = SEED):
    """Load UCI Adult (OpenML v2) and return a stratified 70/30 split.

    Missing rows are dropped, the label is binarised (>50K -> 1), the sex
    attribute is kept as plain strings, and the remaining features are
    one-hot encoded with ``drop_first=True``.

    Returns:
        X_train, X_test, y_train, y_test, A_train, A_test
    """
    df = _frame(fetch_openml("adult", version=2, as_frame=True)).copy()

    # OpenML marks gaps with "?" in several columns.
    df = df.replace("?", np.nan)
    df = df.dropna()

    # y = 1 when earning more than 50K, else 0.
    label = df[TARGET_COL].astype(str).str.strip().str.rstrip(".")
    y = (label == ">50K").astype(int)

    # Sensitive attribute kept as strings.
    A = df[SENSITIVE_COL].astype(str).str.strip()

    # Features: drop label, sensitive, race and the sampling weight.
    X = pd.get_dummies(df.drop(columns=DROP_COLS), drop_first=True, dtype=float)
    X = _sanitize_columns(X.astype(float))

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.3, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, A_train, A_test


# WARNING: German Credit is only 1000 rows. The 70/30 split leaves roughly
# 700 train and 300 test rows, so every per-group fairness number, per-group
# accuracy and codecarbon measurement is noisy - a handful of rows can move a
# selection rate by several points, and the smaller age group may hold only a
# few dozen test rows. This comment is mirrored by SMALL_SAMPLE_WARNING below
# so the caveat also appears in the program output.
def load_german_credit(seed: int = SEED):
    """Load German Credit (OpenML ``credit-g``) and return a 70/30 split.

    The target is 1 when ``class == "good"``, the sensitive attribute is age
    bucketed at 25 (``"25+"`` vs ``"under25"``), ``age`` itself is dropped
    from the features, and the rest are one-hot encoded with
    ``drop_first=True`` for consistency with :func:`load_adult`.

    Note:
        Only 1000 rows in total (see the WARNING comment above).

    Returns:
        X_train, X_test, y_train, y_test, A_train, A_test
    """
    df = _frame(fetch_openml("credit-g", version=1, as_frame=True)).copy()

    # y = 1 for a good credit risk, else 0.
    y = (df[TARGET_COL].astype(str).str.strip() == "good").astype(int)

    # Sensitive attribute: age bucketed at 25; age is dropped from X below.
    age = pd.to_numeric(df["age"])
    A = pd.Series(
        np.where(age >= 25, "25+", "under25"), index=df.index, name="age_group"
    )

    # Features: label and age removed, everything else one-hot encoded.
    # personal_status also encodes sex; it stays in X because only the audited
    # attribute (age) is dropped here. Disclosed in paper/paper.md.
    X = pd.get_dummies(df.drop(columns=[TARGET_COL, "age"]), drop_first=True, dtype=float)
    X = _sanitize_columns(X.astype(float))

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.3, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, A_train, A_test


SMALL_SAMPLE_WARNING = (
    "German Credit has only 1000 rows. The 70/30 split leaves ~700 train and "
    "~300 test rows, so per-group fairness metrics, per-group accuracy and CO2 "
    "readings are all noisy: a handful of rows can shift a selection rate by "
    "several points, and the 'under25' group may hold only a few dozen rows."
)


@dataclass(frozen=True)
class DatasetConfig:
    """How the pipeline should load and report one dataset."""

    name: str
    load: Callable
    warn: str = ""

    def announce(self) -> None:
        """Print the sample-size caveat, if the dataset has one."""
        if self.warn:
            print("!" * 72)
            print(f"WARNING [{self.name}]: {self.warn}")
            print("!" * 72)


# Add a dataset here and every step (train, fairness, mitigation) picks it up.
DATASETS: dict[str, DatasetConfig] = {
    "adult": DatasetConfig(name="adult", load=load_adult),
    "german_credit": DatasetConfig(
        name="german_credit", load=load_german_credit, warn=SMALL_SAMPLE_WARNING
    ),
}


def get_datasets(*names: str) -> list[DatasetConfig]:
    """Return the requested configs in order; no names means every dataset."""
    if not names:
        return list(DATASETS.values())
    missing = [n for n in names if n not in DATASETS]
    if missing:
        raise SystemExit(f"Unknown dataset(s) {missing}. Available: {list(DATASETS)}")
    return [DATASETS[n] for n in names]
