"""Train four classifiers and record accuracy, F1, ROC-AUC and CO2eq.

Datasets come from the registry in ``data``, so any registered dataset can be
run without touching this module: ``python src/train.py`` runs everything, and
``python src/train.py adult`` restricts the run.

The data split is loaded once with the project default seed (42) so every
model is compared on identical rows; the loop seed only varies each model's
``random_state``. CO2eq is tracked separately for training and inference and
is reported in grams. Per-seed predictions stay in memory and are never
written to disk.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import DatasetConfig, get_datasets

SEEDS = [0, 1, 2, 3, 4]
DATA_SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GROUP_COLS = ["dataset", "model"]

# Held in memory only: "dataset/model" -> seed -> (y_pred, y_proba).
PREDICTIONS: dict[str, dict[int, tuple]] = {}


def build_models(seed: int) -> dict:
    """Return the four estimators for one seed."""
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed
            ),
        ),
    }


def measure(fn: Callable[[], object]) -> tuple:
    """Run ``fn`` under a CO2 tracker.

    Returns:
        ``(result, emissions_grams, seconds)``
    """
    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    start = time.perf_counter()
    try:
        value = fn()
        elapsed = time.perf_counter() - start
    finally:
        kg = tracker.stop()

    if kg is None:  # phase too short to report
        kg = tracker.final_emissions
    grams = 0.0 if kg is None else float(kg) * 1000.0  # kg -> g CO2eq
    return value, grams, elapsed


def summarize(
    runs: pd.DataFrame, group_cols: list[str], id_cols: set | None = None
) -> pd.DataFrame:
    """Collapse per-seed rows into ``<metric>_mean`` / ``<metric>_std``.

    ``id_cols`` defaults to the grouping columns plus ``seed``; anything else
    is treated as a metric to aggregate.
    """
    if id_cols is None:
        id_cols = set(group_cols) | {"seed"}
    metrics = [c for c in runs.columns if c not in id_cols]
    out = runs.groupby(group_cols, sort=False)[metrics].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


def write_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Write ``results/<name>``, replacing only the datasets just processed.

    Stages run one dataset at a time (``python src/fairness.py adult``), so
    rows for datasets that were not in this invocation are carried over
    instead of silently dropped. Tables without a ``dataset`` column are
    simply overwritten.
    """
    path = RESULTS / name
    if "dataset" in frame.columns and path.exists():
        done = set(frame["dataset"].unique())
        old = pd.read_csv(path)
        frame = pd.concat([old[~old["dataset"].isin(done)], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def run_dataset(cfg: DatasetConfig) -> list[dict]:
    """Fit every model for every seed on one dataset and return the rows."""
    cfg.announce()
    X_train, X_test, y_train, y_test, _A_tr, _A_te = cfg.load(seed=DATA_SEED)
    print(
        f"[{cfg.name}] train={X_train.shape[0]}  test={X_test.shape[0]}  "
        f"features={X_train.shape[1]}"
    )

    rows: list[dict] = []
    for seed in SEEDS:
        for name, model in build_models(seed).items():
            _, train_g, train_s = measure(lambda m=model: m.fit(X_train, y_train))
            (y_pred, y_proba), inf_g, inf_s = measure(
                lambda m=model: (m.predict(X_test), m.predict_proba(X_test)[:, 1])
            )
            PREDICTIONS.setdefault(f"{cfg.name}/{name}", {})[seed] = (y_pred, y_proba)
            rows.append(
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
                }
            )
            last = rows[-1]
            print(
                f"[{cfg.name}] {name:>14} seed={seed}  acc={last['accuracy']:.4f}"
                f"  train={train_g:.4f}g/{train_s:.1f}s  infer={inf_g:.4f}g/{inf_s:.3f}s"
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Run the requested datasets and write the raw and summary tables."""
    RESULTS.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    names = sys.argv[1:] if argv is None else argv
    rows: list[dict] = []
    for cfg in get_datasets(*names):
        rows.extend(run_dataset(cfg))

    runs = pd.DataFrame(rows)
    write_table(runs, "model_runs.csv")

    summary = summarize(runs, GROUP_COLS)
    write_table(summary, "model_summary.csv")

    print("\n=== model_summary.csv: mean over seeds", SEEDS, "(rows transposed) ===")
    print(summary.set_index(GROUP_COLS).T.round(4).to_string())
    print(f"\nin-memory prediction sets: {len(PREDICTIONS)}")


if __name__ == "__main__":
    main()
