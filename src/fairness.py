"""Fairness audit for every model and seed, merged into the run tables.

Runs over every dataset in the registry in ``data``. Predictions are never
written to disk, so this module retrains each model with the same seeds as
``train.py`` and writes ``results/model_runs.csv``,
``results/model_summary.csv`` and ``results/per_group_accuracy.csv``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import DatasetConfig, get_datasets
from train import (
    DATA_SEED,
    GROUP_COLS,
    RESULTS,
    SEEDS,
    build_models,
    measure,
    summarize,
    write_table,
)

FAIR_COLS = [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "demographic_parity_ratio",
]
# Fairness columns to show in the printed summary.
SHOW_COLS = FAIR_COLS + [
    "accuracy_female",
    "accuracy_male",
    "selection_rate_female",
    "selection_rate_male",
]


def fairness_block(y_true, y_pred, A) -> tuple:
    """Compute group fairness metrics and per-group accuracy/selection rate.

    Returns:
        ``(flat_metrics, per_group_rows)`` where ``flat_metrics`` holds the
        scalar fairness scores plus one accuracy/selection column per group.
    """
    flat = {
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=A
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=A
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true, y_pred, sensitive_features=A,
        ),
    }

    frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=A,
    )

    rows = []
    for group, values in frame.by_group.iterrows():
        # Column suffix must be safe for CSV and readable for any group label.
        key = str(group).lower().replace("+", "").replace(" ", "_")
        flat[f"accuracy_{key}"] = values["accuracy"]
        flat[f"selection_rate_{key}"] = values["selection_rate"]
        rows.append(
            {
                "group": str(group),
                "accuracy": values["accuracy"],
                "selection_rate": values["selection_rate"],
            }
        )
    return flat, rows


def run_dataset(cfg: DatasetConfig) -> tuple:
    """Retrain every model on one dataset and compute its fairness metrics.

    Returns:
        ``(run_rows, group_rows)``
    """
    cfg.announce()
    X_train, X_test, y_train, y_test, _A_tr, A_te = cfg.load(seed=DATA_SEED)
    print(
        f"[{cfg.name}] train={X_train.shape[0]}  test={X_test.shape[0]}  "
        f"features={X_train.shape[1]}"
    )

    runs: list[dict] = []
    groups: list[dict] = []

    for seed in SEEDS:
        for name, model in build_models(seed).items():
            _, train_g, train_s = measure(lambda m=model: m.fit(X_train, y_train))
            (y_pred, y_proba), inf_g, inf_s = measure(
                lambda m=model: (m.predict(X_test), m.predict_proba(X_test)[:, 1])
            )
            flat, rows = fairness_block(y_test, y_pred, A_te)

            runs.append(
                {
                    "dataset": cfg.name,
                    "model": name,
                    "seed": seed,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "train_co2_g": train_g,
                    "inference_co2_g": inf_g,
                    "train_seconds": train_s,
                    "inference_seconds": inf_s,
                    **flat,
                }
            )
            groups.extend(
                {"dataset": cfg.name, "model": name, "seed": seed, **r} for r in rows
            )
            print(
                f"[{cfg.name}] {name:>14} seed={seed}  acc={runs[-1]['accuracy']:.4f}"
                f"  dp_diff={flat['demographic_parity_difference']:.4f}"
                f"  eo_diff={flat['equalized_odds_difference']:.4f}"
                f"  dp_ratio={flat['demographic_parity_ratio']:.4f}"
            )
    return runs, groups


def main(argv: list[str] | None = None) -> None:
    """Run every requested dataset and rewrite the three fairness tables."""
    RESULTS.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    names = sys.argv[1:] if argv is None else argv
    t0 = time.perf_counter()

    run_rows: list[dict] = []
    group_rows: list[dict] = []
    for cfg in get_datasets(*names):
        r, g = run_dataset(cfg)
        run_rows.extend(r)
        group_rows.extend(g)

    runs = pd.DataFrame(run_rows)
    write_table(runs, "model_runs.csv")

    summary = summarize(runs, GROUP_COLS)
    write_table(summary, "model_summary.csv")

    write_table(pd.DataFrame(group_rows), "per_group_accuracy.csv")

    print(f"\nelapsed {time.perf_counter() - t0:.0f}s")
    print("=== Fairness summary (mean over", SEEDS, "seeds, rows transposed) ===")
    # Per-group columns are dataset-specific (Adult: female/male,
    # German: 25/under25), so they are NaN where they do not apply.
    metric_cols = [c for c in summary.columns if c.endswith("_mean")]
    view = summary[GROUP_COLS + metric_cols].set_index(GROUP_COLS).T.round(4)
    view = view.dropna(how="all")
    view = view.replace({float("nan"): "-"})
    print(view.to_string())
    print("\nwrote model_runs.csv, model_summary.csv, per_group_accuracy.csv")


if __name__ == "__main__":
    main()
