"""Write the class and audit-group composition of each test split.

Accuracy and selection rates are only readable against these baselines, and
the per-group counts show how noisy the per-group fairness metrics are.
"""

from __future__ import annotations

import sys

import pandas as pd

from data import get_datasets
from train import DATA_SEED, RESULTS, write_table


def run_dataset(cfg) -> list[dict]:
    """Class shares, majority baseline and audit-group sizes for one dataset."""
    X_train, X_test, y_train, y_test, A_train, A_test = cfg.load(seed=DATA_SEED)

    majority = int(y_test.value_counts().idxmax())
    shared = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate": float(y_test.mean()),
        "majority_class": majority,
        # Accuracy of a model that always predicts the test-set majority class.
        "majority_accuracy": float((y_test == majority).mean()),
    }
    rows = []
    for group, n in A_test.value_counts().sort_index().items():
        rows.append(
            {
                "dataset": cfg.name,
                "group": group,
                **shared,
                "group_n_test": int(n),
                "group_share_of_test": float(n) / len(y_test),
                "group_positive_rate": float(
                    (y_test[A_test == group] == 1).mean()
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Write results/baselines.csv for the requested datasets."""
    RESULTS.mkdir(exist_ok=True)
    names = sys.argv[1:] if argv is None else argv

    rows: list[dict] = []
    for cfg in get_datasets(*names):
        cfg.announce()
        rows.extend(run_dataset(cfg))

    table = pd.DataFrame(rows)
    write_table(table, "baselines.csv")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
