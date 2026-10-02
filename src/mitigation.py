"""Mitigation study: pre-, in- and post-processing, run per dataset.

Datasets come from the registry in ``data``. Three fairlearn strategies are
compared against their unmitigated baselines:

* ``threshold_optimizer`` -- post-processing, equalized odds, on XGBoost
* ``exponentiated_gradient`` -- in-processing, demographic parity, on LogReg
* ``reweighing`` -- pre-processing, Kamiran-Calders weights, on XGBoost

Each method runs for seeds 0-4 on the fixed split (data seed 42). Before and
after metrics are written to ``results/mitigation_summary.csv`` as mean/std,
with the raw per-seed rows in ``results/mitigation_runs.csv``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import DatasetConfig, get_datasets
from train import DATA_SEED, SEEDS, summarize, write_table

RESULTS = Path(__file__).resolve().parents[1] / "results"
GROUP_COLS = ["dataset", "method"]

METRICS = [
    "accuracy",
    "f1",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "demographic_parity_ratio",
]


class Float64Wrapper:
    """Cast ``predict_proba`` to float64.

    fairlearn's ThresholdOptimizer assigns into a pandas Series that starts
    from the estimator's probabilities; XGBoost emits float32 and pandas 3.0
    refuses the implicit downcast.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)


def score(y_true, y_pred, A) -> dict:
    """Compute accuracy, F1 and the three fairness metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=A
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=A
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true, y_pred, sensitive_features=A
        ),
    }


def reweighing_weights(A, y) -> np.ndarray:
    """Kamiran-Calders weights making the group attribute and label independent.

    ``w(a, y) = P(a) * P(y) / P(a, y)``, so the weighted joint factorises.
    """
    joint = pd.crosstab(pd.Series(A), pd.Series(y), normalize=True)
    P_a = pd.Series(A).value_counts(normalize=True)
    P_y = pd.Series(y).value_counts(normalize=True)
    weights = joint.copy().astype(float)
    for a in joint.index:
        for c in joint.columns:
            weights.loc[a, c] = P_a[a] * P_y[c] / joint.loc[a, c]
    return np.array([weights.loc[a, c] for a, c in zip(A, y)])


def fit_xgb(seed: int, X, y, sample_weight=None) -> XGBClassifier:
    """Fit the shared XGBoost configuration used across methods."""
    model = XGBClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1,
        eval_metric="logloss", verbosity=0,
    )
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def run_dataset(cfg: DatasetConfig) -> list[dict]:
    """Run all three mitigations for every seed on one dataset."""
    cfg.announce()
    X_train, X_test, y_train, y_test, A_train, A_test = cfg.load(seed=DATA_SEED)

    # fairlearn performs numpy-style masked assignment, so use plain arrays.
    Xtr, Xte = X_train.to_numpy(), X_test.to_numpy()
    ytr, yte = y_train.to_numpy(), y_test.to_numpy()
    Atr, Ate = A_train.to_numpy(), A_test.to_numpy()

    # Scaling lives outside the estimator: sklearn's Pipeline.fit rejects the
    # sample_weight that ExponentiatedGradient passes to its oracle.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    weights = reweighing_weights(Atr, ytr)
    print(
        f"[{cfg.name}] train={len(ytr)}  test={len(yte)}  "
        f"weight mean={weights.mean():.4f}"
    )

    rows: list[dict] = []

    for seed in SEEDS:
        # --- unmitigated XGBoost baseline (shared by methods 1 and 3) ---
        base_pred = fit_xgb(seed, Xtr, ytr).predict(Xte)
        baseline = score(yte, base_pred, Ate)

        # --- 1. post-processing: ThresholdOptimizer, equalized odds ---
        to = ThresholdOptimizer(
            estimator=Float64Wrapper(fit_xgb(seed, Xtr, ytr)),
            constraints="equalized_odds",
            prefit=True,
            predict_method="predict_proba",
        )
        to.fit(Xtr, ytr, sensitive_features=Atr)
        # predict() is randomised by fairlearn, so pin it to the loop seed.
        to_pred = to.predict(Xte, sensitive_features=Ate, random_state=seed)

        # --- 2. in-processing: ExponentiatedGradient, demographic parity ---
        lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr_s, ytr)
        lr_pred = lr.predict(Xte_s)
        eg = ExponentiatedGradient(
            LogisticRegression(max_iter=1000, random_state=seed), DemographicParity()
        )
        eg.fit(Xtr_s, ytr, sensitive_features=Atr)
        eg_pred = eg.predict(Xte_s)

        # --- 3. pre-processing: sample reweighing on XGBoost ---
        rw_pred = fit_xgb(seed, Xtr, ytr, sample_weight=weights).predict(Xte)

        for method, before, after in [
            ("threshold_optimizer", baseline, score(yte, to_pred, Ate)),
            ("exponentiated_gradient", score(yte, lr_pred, Ate), score(yte, eg_pred, Ate)),
            ("reweighing", baseline, score(yte, rw_pred, Ate)),
        ]:
            row = {"dataset": cfg.name, "method": method, "seed": seed}
            for metric in METRICS:
                row[f"{metric}_before"] = before[metric]
                row[f"{metric}_after"] = after[metric]
            rows.append(row)

        # rows are appended in order TO, EG, RW
        to_row, eg_row, rw_row = rows[-3], rows[-2], rows[-1]
        print(
            f"[{cfg.name}] seed={seed}  TO acc {to_row['accuracy_before']:.4f}"
            f"->{to_row['accuracy_after']:.4f}"
            f" eo {to_row['equalized_odds_difference_before']:.4f}"
            f"->{to_row['equalized_odds_difference_after']:.4f}"
            f" | EG acc {eg_row['accuracy_before']:.4f}->{eg_row['accuracy_after']:.4f}"
            f" dp {eg_row['demographic_parity_difference_before']:.4f}"
            f"->{eg_row['demographic_parity_difference_after']:.4f}"
            f" | RW acc {rw_row['accuracy_before']:.4f}->{rw_row['accuracy_after']:.4f}"
            f" dp {rw_row['demographic_parity_difference_before']:.4f}"
            f"->{rw_row['demographic_parity_difference_after']:.4f}"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Run every requested dataset and write the mitigation tables.

    Figures (including fig3) are built from these tables by ``figures.py``.
    """
    RESULTS.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    names = sys.argv[1:] if argv is None else argv
    t0 = time.perf_counter()

    rows: list[dict] = []
    for cfg in get_datasets(*names):
        rows.extend(run_dataset(cfg))

    runs = pd.DataFrame(rows)
    write_table(runs, "mitigation_runs.csv")

    summary = summarize(runs, GROUP_COLS)
    write_table(summary, "mitigation_summary.csv")

    print(f"\nelapsed {time.perf_counter() - t0:.0f}s")
    print_summary(summary)


def print_summary(summary: pd.DataFrame) -> None:
    """Print the before/after table, transposed for readability."""
    print("=== mitigation_summary.csv (mean over 5 seeds, rows transposed) ===")
    means = {c: summary[c] for c in GROUP_COLS}
    stds = {c: summary[c] for c in GROUP_COLS}
    for metric in METRICS:
        means[f"{metric}|before"] = summary[f"{metric}_before_mean"].values
        means[f"{metric}|after"] = summary[f"{metric}_after_mean"].values
        stds[f"{metric}|before"] = summary[f"{metric}_before_std"].values
        stds[f"{metric}|after"] = summary[f"{metric}_after_std"].values
    print(pd.DataFrame(means).set_index(GROUP_COLS).T.round(4).to_string())
    print("\n--- std over seeds ---")
    print(pd.DataFrame(stds).set_index(GROUP_COLS).T.round(4).to_string())


if __name__ == "__main__":
    main()
