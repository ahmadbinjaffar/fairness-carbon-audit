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
