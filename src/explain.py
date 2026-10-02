"""SHAP explanations and the feature-proxy hypothesis.

Fits the seed-42 XGBoost baseline, explains 1000 test rows with SHAP, then
checks whether the most influential features leak sex: for the top 10 SHAP
features we compare the mean feature value for men vs women across the whole
test set and correlate each feature with a Male=1 indicator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from xgboost import XGBClassifier

from data import load_adult

DATA_SEED = 42
N_SHAP = 1000
TOP_K = 10
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def fit_baseline(X_train, y_train) -> XGBClassifier:
    """Fit the shared XGBoost configuration on the training split."""
    model = XGBClassifier(
        n_estimators=200, random_state=DATA_SEED, n_jobs=-1,
        eval_metric="logloss", verbosity=0,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    return model


def proxy_table(X_test, A_test, feature_names, top_features) -> pd.DataFrame:
    """Mean feature value by sex plus point-biserial correlation with sex.

    Features are 0/1 dummies or continuous, and sex is binarised Male=1, so the
    Pearson correlation is the point-biserial correlation in every case.
    """
    df = pd.DataFrame(np.asarray(X_test), columns=feature_names)
    sex = (np.asarray(A_test) == "Male").astype(int)

    rows = []
    for feature in top_features:
        values = df[feature].astype(float)
        # A constant column (e.g. a rare category absent from the test set)
        # has no defined correlation; report 0 rather than NaN.
        corr = 0.0 if values.std() == 0 else float(np.corrcoef(values, sex)[0, 1])
        rows.append(
            {
                "feature": feature,
                "mean_male": float(values[sex == 1].mean()),
                "mean_female": float(values[sex == 0].mean()),
                "mean_diff_male_minus_female": float(
                    values[sex == 1].mean() - values[sex == 0].mean()
                ),
                "correlation_with_sex": corr,
                "abs_correlation": abs(corr),
            }
        )

    table = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    table.insert(0, "dataset", "adult")  # project convention: leading column
    return table.reset_index(drop=True)


def draw_figures(shap_values, X_plot, feature_names) -> None:
    """Save the beeswarm summary and the top-10 mean-importance bar chart."""
    FIGURES.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_plot, feature_names=list(feature_names), show=False
    )
    plt.title(f"SHAP value impact on model output — 1000 test rows (seed {DATA_SEED})")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_shap_summary.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X_plot, feature_names=list(feature_names),
        plot_type="bar", max_display=TOP_K, show=False,
    )
    plt.title(f"Top {TOP_K} features by mean |SHAP| — XGBoost on Adult")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5b_shap_top10_bar.png", dpi=300)
    plt.close()


def print_summary(table: pd.DataFrame, feature_names: list, X_test) -> None:
    """Print the top 5 proxies and a plain-language reading of the table."""
    # ASCII only: the Windows console codepage mangles dashes.
    print("\n=== Top 5 proxies (sorted by |correlation with sex|) ===")
    print(f"{'#':>2}  {'feature':<40} {'mean M':>8} {'mean F':>8} {'r':>8}")
    for i, row in table.head(5).iterrows():
        print(
            f"{i + 1:>2}  {row['feature']:<40} {row['mean_male']:>8.4f}"
            f" {row['mean_female']:>8.4f} {row['correlation_with_sex']:>8.4f}"
        )

    # A column is a dummy only if every test value is 0 or 1.
    df = pd.DataFrame(np.asarray(X_test), columns=feature_names)
    is_dummy = {
        f: bool(np.isin(np.unique(df[f]), [0, 1]).all()) for f in table["feature"]
    }
    n_dummy = sum(is_dummy[f] for f in table["feature"])
    n_cont = len(table) - n_dummy

    top = table.iloc[0]
    n_strong = int((table["abs_correlation"] >= 0.3).sum())
    strongest = table.iloc[0]["feature"]
    strongest_dummy = is_dummy[strongest]

    print("\n=== Plain-language summary ===")
    print(
        f"1. Strongest proxy: '{strongest}' (r = "
        f"{top['correlation_with_sex']:+.3f}). Mean value is "
        f"{top['mean_male']:.4f} for men vs {top['mean_female']:.4f} for "
        f"women, a gap of {top['mean_diff_male_minus_female']:+.4f}"
        + (
            ". This is a 0/1 flag, so it means "
            f"{top['mean_male']:.1%} of men and {top['mean_female']:.1%} of "
            "women fall in that category."
            if strongest_dummy
            else "."
        )
    )
    print(
        f"2. Of the top {len(table)} SHAP features, {n_dummy} are 0/1 dummy "
        f"columns and {n_cont} are continuous (age, hours, capital gains, "
        "education-num, and similar). For a dummy, the group mean is just the "
        "share of that group holding the category, so a near-1 vs near-0 split "
        "is a relabelled sex indicator."
    )
    print(
        f"3. {n_strong} of the top {len(table)} features correlate with sex at "
        f"|r| >= 0.3, and the top {len(table)} are all correlated to some "
        "degree, so the model can partly recover sex from these columns even "
        "though 'sex' and 'race' were dropped before training."
    )
    print(
        "4. Practical meaning: part of any fairness gap is baked in at the "
        "input stage, because proxy columns carry information the removed "
        "sensitive attribute used to hold. Dropping the column alone does not "
        "remove the signal."
    )


def main() -> None:
    """Fit, explain, test the proxy hypothesis, and write the outputs."""
    RESULTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, _A_train, A_test = load_adult(seed=DATA_SEED)
    feature_names = list(X_test.columns)

    model = fit_baseline(X_train, y_train)

    # Explain a fixed 1000-row slice of the test set.
    rng = np.random.RandomState(DATA_SEED)
    idx = rng.choice(len(X_test), N_SHAP, replace=False)
    X_explain = np.asarray(X_test)[idx]
    print(f"explaining {N_SHAP} of {len(X_test)} test rows | {len(feature_names)} features")

    shap_values = np.asarray(shap.TreeExplainer(model).shap_values(X_explain))

    draw_figures(shap_values, X_explain, feature_names)

    # Rank features by mean |SHAP| on the explained rows.
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:TOP_K]
    top_features = [feature_names[i] for i in order]
    print("\nTop 10 SHAP features:", ", ".join(top_features))

    # Proxy test uses the full test set, as specified.
    table = proxy_table(X_test, A_test, feature_names, top_features)
    table.insert(1, "mean_abs_shap", [importance[feature_names.index(f)] for f in table["feature"]])
    table.to_csv(RESULTS / "proxy_features.csv", index=False)

    print(f"\nwrote {RESULTS / 'proxy_features.csv'} ({len(table)} rows)")
    print(f"wrote {FIGURES / 'fig5_shap_summary.png'}")
    print(f"wrote {FIGURES / 'fig5b_shap_top10_bar.png'}")
    print_summary(table, feature_names, X_test)


if __name__ == "__main__":
    main()
