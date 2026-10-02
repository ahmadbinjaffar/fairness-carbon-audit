"""Exploratory table and figure: sex distribution and >50K share by sex."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import SEED, load_adult

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TARGET_LABEL = "share_earning_gt_50k"


def build_table() -> pd.DataFrame:
    """Compute the share of each sex and its >50K earnings rate."""
    _, _, y_train, y_test, A_train, A_test = load_adult(seed=SEED)

    # The split partitions the dataset, so rejoining recovers every row.
    A = pd.concat([A_train, A_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    counts = A.value_counts()
    table = pd.DataFrame(
        {
            "sex": counts.index,
            "n": counts.values,
            "share_of_data": counts.values / counts.sum(),
            TARGET_LABEL: y.groupby(A).mean().reindex(counts.index).values,
        }
    ).sort_values("sex", ignore_index=True)
    table.insert(0, "dataset", "adult")  # project convention: leading column
    return table


def plot(table: pd.DataFrame) -> None:
    """Bar chart of the >50K share for each sex."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    bars = ax.bar(table["sex"], table[TARGET_LABEL], color="#4C72B0", width=0.55)
    for bar, share, n in zip(bars, table[TARGET_LABEL], table["n"]):
        ax.annotate(
            f"{share:.1%}\n(n={n})",
            (bar.get_x() + bar.get_width() / 2, share),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Sex")
    ax.set_ylabel("Share earning >50K")
    ax.set_title("Earning >50K by sex — UCI Adult")
    ax.set_ylim(0, max(table[TARGET_LABEL]) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    sns.despine(fig=fig)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_outcome_by_sex.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Write the table and figure, then print the numbers."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    table = build_table()
    table.to_csv(RESULTS / "eda_sex_income.csv", index=False)
    plot(table)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
