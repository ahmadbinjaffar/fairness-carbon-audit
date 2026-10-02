"""Publication-quality figures built only from ``results/*.csv``.

Reading the saved tables keeps the figures reproducible without re-running any
experiment: every number drawn here already passed through the pipeline steps.
fig1 (outcome by sex) and fig5 (SHAP) are produced by ``eda.py`` and
``explain.py`` and are left untouched here.

Shared style for the whole paper: one model palette, one marker per dataset,
readable font sizes, std-based error bars, 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FOUR_FIFTHS = 0.8

# One palette for the paper: colour = model, marker = dataset.
MODEL_COLORS = {
    "logreg": "#4C72B0",
    "random_forest": "#55A868",
    "xgboost": "#C44E52",
    "mlp": "#8172B3",
}
METHOD_COLORS = {
    "threshold_optimizer": "#4C72B0",
    "exponentiated_gradient": "#55A868",
    "reweighing": "#C44E52",
}
DATASET_MARKERS = {"adult": "o", "german_credit": "s"}
DATASET_LABELS = {"adult": "Adult", "german_credit": "German Credit"}
# Bars stay coloured by model, so datasets are told apart by hatching.
DATASET_HATCH = {"adult": "", "german_credit": "///"}
MODEL_LABELS = {
    "logreg": "LogReg",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}
SHORT = {"logreg": "LogReg", "random_forest": "RF", "xgboost": "XGB", "mlp": "MLP"}
METHOD_LABELS = {
    "threshold_optimizer": "ThresholdOptimizer",
    "exponentiated_gradient": "ExponentiatedGradient",
    "reweighing": "Reweighing",
}

# Per-point label offsets so annotations never sit on top of a marker.
FIG2_OFFSETS = {
    ("adult", "logreg"): (8, -13),
    ("adult", "random_forest"): (-9, 6),
    ("adult", "xgboost"): (8, 7),
    ("adult", "mlp"): (8, 7),
    ("german_credit", "logreg"): (-9, -7),
    ("german_credit", "random_forest"): (8, 7),
    ("german_credit", "xgboost"): (-9, -7),
    ("german_credit", "mlp"): (-9, -7),
}
FIG6_OFFSETS = {
    ("adult", "logreg"): (8, -13),
    ("adult", "random_forest"): (8, -13),
    ("adult", "xgboost"): (8, 7),
    ("adult", "mlp"): (-9, 7),
    ("german_credit", "logreg"): (8, 7),
    ("german_credit", "random_forest"): (8, 8),
    ("german_credit", "xgboost"): (-9, -7),
    ("german_credit", "mlp"): (8, 7),
}
# The German mitigation points sit close together, so its threshold label goes
# above the star instead of grazing the baseline circle.
FIG3_OFFSETS = {
    ("german_credit", "threshold_optimizer"): (10, 12),
}


def style() -> None:
    """Apply the shared figure style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.dpi": 300,
            "figure.dpi": 100,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load(name: str) -> pd.DataFrame:
    """Read one results table, failing loudly if the step was never run."""
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} - run the pipeline step that writes it")
    return pd.read_csv(path)


def save(fig, name: str) -> None:
    """Write a figure at 300 dpi and release it."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def dataset_handles() -> list:
    """Legend artists for the dataset markers."""
    return [
        Line2D([], [], color="black", marker=DATASET_MARKERS[d], ls="", ms=8,
               label=DATASET_LABELS[d])
        for d in DATASET_MARKERS
    ]


def _label(ax, xy, text, offset, **kwargs):
    """Place a label on the far side of a marker so it never covers it.

    ``annotate`` anchors the text baseline, so a two-line label pointed
    "downwards" would otherwise slide back over the marker.
    """
    dx, dy = offset
    ax.annotate(
        text, xy, xytext=offset, textcoords="offset points",
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        **kwargs,
    )


def fig2_accuracy_vs_fairness(models: pd.DataFrame) -> None:
    """One point per model and dataset: accuracy against the parity ratio."""
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    ax.axhspan(0, FOUR_FIFTHS, color="#C44E52", alpha=0.07, zorder=0)
    ax.axhline(FOUR_FIFTHS, color="0.35", ls="--", lw=1.2, zorder=1)
    ax.annotate(
        "four-fifths rule (0.8)",
        xy=(models["accuracy_mean"].max() + 0.028, FOUR_FIFTHS + 0.008),
        ha="right", fontsize=9, color="0.30", va="bottom",
    )

    for _, row in models.iterrows():
        key = (row["dataset"], row["model"])
        ax.errorbar(
            row["accuracy_mean"], row["demographic_parity_ratio_mean"],
            xerr=row["accuracy_std"], yerr=row["demographic_parity_ratio_std"],
            fmt=DATASET_MARKERS[row["dataset"]], color=MODEL_COLORS[row["model"]],
            ms=9, mec="black", mew=1.1, ecolor="0.25", elinewidth=1.1, capsize=3,
            lw=0, zorder=3,
        )
        _label(
            ax,
            (row["accuracy_mean"], row["demographic_parity_ratio_mean"]),
            SHORT[row["model"]], FIG2_OFFSETS[key],
            fontsize=9, color=MODEL_COLORS[row["model"]],
        )

    ax.set_xlim(models["accuracy_mean"].min() - 0.03, models["accuracy_mean"].max() + 0.03)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("Test accuracy (mean over 5 seeds)")
    ax.set_ylabel("Demographic parity ratio (higher is fairer)")
    ax.set_title("Accuracy vs fairness by model")
    ax.grid(True, alpha=0.3)
    handles = [
        Line2D([], [], color=c, marker="o", ls="", ms=8, label=MODEL_LABELS[m])
        for m, c in MODEL_COLORS.items()
    ] + dataset_handles()
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True, framealpha=0.95)
    save(fig, "fig2_accuracy_vs_fairness.png")


def fig3_mitigation_tradeoff(mit: pd.DataFrame) -> None:
    """Accuracy against the parity gap, before and after each mitigation."""
    datasets = list(pd.unique(mit["dataset"]))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.6 * len(datasets), 5.4), squeeze=False
    )

    for ax, dataset in zip(axes[0], datasets):
        ax.grid(True, alpha=0.3)
        for _, row in mit[mit["dataset"] == dataset].iterrows():
            color = METHOD_COLORS[row["method"]]
            x0, x1 = row["demographic_parity_difference_before_mean"], row["demographic_parity_difference_after_mean"]
            y0, y1 = row["accuracy_before_mean"], row["accuracy_after_mean"]

            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.9, alpha=0.9,
                                shrinkA=7, shrinkB=8),
            )
            ax.errorbar(
                x0, y0,
                xerr=row["demographic_parity_difference_before_std"],
                yerr=row["accuracy_before_std"],
                fmt="o", ms=9, mfc="white", mec=color, mew=1.8, ecolor=color,
                elinewidth=1.1, capsize=3, lw=0, zorder=3,
            )
            ax.errorbar(
                x1, y1,
                xerr=row["demographic_parity_difference_after_std"],
                yerr=row["accuracy_after_std"],
                fmt="*", ms=17, mfc=color, mec="black", mew=0.7, ecolor=color,
                elinewidth=1.1, capsize=3, lw=0, zorder=4,
            )
            _label(
                ax, (x1, y1), METHOD_LABELS[row["method"]],
                FIG3_OFFSETS.get((dataset, row["method"]), (10, -5)),
                fontsize=9, color=color, fontweight="bold",
            )

        panel = mit[mit["dataset"] == dataset]
        gap = panel["demographic_parity_difference_before_mean"].max()
        acc_lo = panel[["accuracy_before_mean", "accuracy_after_mean"]].min().min()
        acc_hi = panel[["accuracy_before_mean", "accuracy_after_mean"]].max().max()
        pad = max(0.015, (acc_hi - acc_lo) * 0.4)
        ax.set_xlim(-0.02, gap * 1.45)
        ax.set_ylim(acc_lo - pad, acc_hi + pad)
        ax.set_xlabel("Demographic parity difference (lower is fairer)")
        ax.set_ylabel("Test accuracy (higher is better)")
        ax.set_title(DATASET_LABELS[dataset])

    handles = [
        Line2D([], [], color="black", marker="o", ls="", ms=8, mfc="white",
               label="before"),
        Line2D([], [], color="black", marker="*", ls="", ms=13, label="after"),
    ]
    axes[0][-1].legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)
    fig.suptitle("Mitigation trade-off: accuracy vs fairness gap (mean over 5 seeds)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig3_mitigation_tradeoff.png")


def fig4_carbon_per_model(models: pd.DataFrame) -> None:
    """Training and inference CO2eq per model, on a log scale."""
    order = list(MODEL_COLORS)
    datasets = list(pd.unique(models["dataset"]))
    width = 0.38
    ticks = range(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    for ax, phase, title in zip(axes, ["train", "inference"], ["Training", "Inference"]):
        for i, dataset in enumerate(datasets):
            sub = models[models["dataset"] == dataset].set_index("model").reindex(order)
            mean = sub[f"{phase}_co2_g_mean"].to_numpy()
            std = sub[f"{phase}_co2_g_std"].fillna(0).to_numpy()
            offset = (i - (len(datasets) - 1) / 2) * width
            bars = ax.bar(
                [t + offset for t in ticks], mean, width * 0.94,
                yerr=_yerr(mean, std),
                color=[MODEL_COLORS[m] for m in order],
                hatch=DATASET_HATCH[dataset],
                edgecolor="black", linewidth=0.6, capsize=3,
                error_kw=dict(ecolor="0.25", elinewidth=1.0, capthick=1.0),
            )
            for bar, value, spread in zip(bars, mean, std):
                ax.annotate(
                    _fmt(value),
                    (bar.get_x() + bar.get_width() / 2, value + spread),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

        ax.set_yscale("log")
        ax.set_xticks(list(ticks))
        ax.set_xticklabels([MODEL_LABELS[m] for m in order])
        ax.set_ylabel("CO2eq (grams)")
        ax.set_title(f"{title} carbon")
        ax.grid(True, axis="y", alpha=0.3)
        lo = models[f"{phase}_co2_g_mean"].min()
        hi = models[f"{phase}_co2_g_mean"].max()
        ax.set_ylim(lo / 4, hi * 12)

    # One shared legend above the panels: it can never sit on a bar or a value.
    handles = [
        Patch(facecolor=MODEL_COLORS[m], edgecolor="black", lw=0.6,
              label=MODEL_LABELS[m])
        for m in order
    ] + [
        Patch(facecolor="white", edgecolor="black", lw=0.6, hatch=hatch,
              label=DATASET_LABELS[dataset])
        for dataset, hatch in DATASET_HATCH.items()
    ]
    fig.suptitle("Carbon cost per model (mean over 5 seeds)", fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.93), columnspacing=1.0,
               handletextpad=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, "fig4_carbon_per_model.png")


def fig6_accuracy_per_gram_co2(models: pd.DataFrame) -> None:
    """Accuracy achieved per gram of training CO2eq (top-left is best)."""
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.grid(True, alpha=0.3)

    ratio_lo = ratio_hi = None
    for _, row in models.iterrows():
        key = (row["dataset"], row["model"])
        ratio = row["accuracy_mean"] / row["train_co2_g_mean"]
        ratio_lo = ratio if ratio_lo is None else min(ratio_lo, ratio)
        ratio_hi = ratio if ratio_hi is None else max(ratio_hi, ratio)

        ax.errorbar(
            row["train_co2_g_mean"], row["accuracy_mean"],
            xerr=_yerr(row["train_co2_g_mean"], row["train_co2_g_std"]),
            yerr=row["accuracy_std"],
            fmt=DATASET_MARKERS[row["dataset"]], color=MODEL_COLORS[row["model"]],
            ms=9, mec="black", mew=1.1, ecolor="0.25", elinewidth=1.1, capsize=3,
            lw=0, zorder=3,
        )
        _label(
            ax,
            (row["train_co2_g_mean"], row["accuracy_mean"]),
            f"{SHORT[row['model']]}\n{ratio:.1e} acc/g", FIG6_OFFSETS[key],
            fontsize=7.5, color=MODEL_COLORS[row["model"]],
        )

    ax.set_xscale("log")
    ax.set_xlim(models["train_co2_g_mean"].min() * 0.3,
                models["train_co2_g_mean"].max() * 5)
    ax.set_ylim(models["accuracy_mean"].min() - 0.035,
                models["accuracy_mean"].max() + 0.035)
    ax.set_xlabel("Training CO2eq (grams, log scale)")
    ax.set_ylabel("Test accuracy (mean over 5 seeds)")
    ax.set_title("Accuracy per gram of CO2eq")
    handles = [
        Line2D([], [], color=c, marker="o", ls="", ms=8, label=MODEL_LABELS[m])
        for m, c in MODEL_COLORS.items()
    ] + dataset_handles()
    # One row under the axes: nothing in the plot area is ever covered.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=6, frameon=False, fontsize=9, columnspacing=1.0,
              handletextpad=0.4)
    save(fig, "fig6_accuracy_per_gram_co2.png")


def _fmt(value: float) -> str:
    """Compact bar label: readable across five orders of magnitude."""
    return f"{value:.4f}" if value >= 1e-2 else f"{value:.2e}"


def _yerr(mean, std, floor: float = 0.02):
    """Symmetric error bars whose lower cap stays inside a log axis.

    A bar whose std exceeds its mean would fall below zero; clamping it to a
    small fraction of the mean keeps the whisker long instead of misleading.
    """
    mean, std = pd.Series(mean).astype(float), pd.Series(std).astype(float)
    low = (mean - std).clip(lower=mean * floor)
    return [mean - low, std]


def main() -> None:
    """Build every figure this module owns from the saved tables."""
    style()
    models = load("model_summary")
    mit = load("mitigation_summary")

    print("building figures from results/*.csv")
    fig2_accuracy_vs_fairness(models)
    fig3_mitigation_tradeoff(mit)
    fig4_carbon_per_model(models)
    fig6_accuracy_per_gram_co2(models)


if __name__ == "__main__":
    main()
