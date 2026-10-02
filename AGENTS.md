# Project Rules

Reproducible research project: **"Accuracy, Fairness, and Carbon Cost: Auditing Machine Learning Models on Loan and Income Data"**

## Rules (apply to the whole project)

1. **Python 3.10+** and a virtual environment in `.venv`.
2. **All random seeds fixed** (default `42`).
3. **Layout**
   - code → `src/`
   - CSV outputs → `results/`
   - figures (PNG, 300 dpi) → `figures/`
   - paper → `paper/`
   - notebooks → `notebooks/`
   - raw data → `data/` (git-ignored, never committed)
4. **Never invent results.** Every number in the paper must come from files in `results/`.
5. **Short comments and docstrings** only — no long explanatory blocks.

## Conventions

- Generate figures with `matplotlib`/`seaborn` at `dpi=300`.
- The paper reads numbers from `results/*.csv`; it does not hard-code them.
- Record the environment with `python src/check_env.py`.
- Datasets are registered in `src/data.py` (`DATASETS`); every loader returns
  `X_train, X_test, y_train, y_test, A_train, A_test` from a stratified 70/30
  split. Pipeline steps take dataset names as CLI args and default to all
  (e.g. `python src/train.py german_credit`). Summary tables carry a leading
  `dataset` column so several datasets coexist in one file.
- `python run_all.py` runs the whole study (eda → baselines → train →
  fairness → mitigation → explain → german_credit → figures), one interpreter
  per stage, and stops on the first non-zero exit. Stage table writers replace
  only the datasets they processed, so Adult and German Credit accumulate in
  the same CSVs. `notebooks/fairness_carbon_audit.ipynb` walks the study
  through from `results/` and `figures/` only.
