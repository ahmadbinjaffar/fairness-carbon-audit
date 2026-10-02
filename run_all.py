"""Run the whole study end to end, stage by stage, in order.

    python run_all.py              # every stage, in order
    python run_all.py figures      # only the named stages
    python run_all.py --list       # show the stages

Each stage runs in its own interpreter, so a crash cannot leave half-imported
state behind; a non-zero exit stops the run immediately with the failing
command. The dataset stages run one dataset at a time and the table writers in
``src/train.py`` merge by dataset, so the final ``results/*.csv`` cover both
Adult and German Credit.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Stage name -> commands, executed in this order.
STAGES: list[tuple[str, list[list[str]]]] = [
    ("eda", [["eda.py"]]),
    ("baselines", [["baselines.py"]]),
    ("train", [["train.py", "adult"]]),
    ("fairness", [["fairness.py", "adult"]]),
    ("mitigation", [["mitigation.py", "adult"]]),
    ("explain", [["explain.py"]]),
    (
        "german_credit",
        [
            ["train.py", "german_credit"],
            ["fairness.py", "german_credit"],
            ["mitigation.py", "german_credit"],
        ],
    ),
    ("figures", [["figures.py"]]),
]

# What a complete run must leave behind (checked after a full run).
EXPECTED = [
    "results/eda_sex_income.csv",
    "results/baselines.csv",
    "results/model_runs.csv",
    "results/model_summary.csv",
    "results/per_group_accuracy.csv",
    "results/mitigation_runs.csv",
    "results/mitigation_summary.csv",
    "results/proxy_features.csv",
    "figures/fig1_outcome_by_sex.png",
    "figures/fig2_accuracy_vs_fairness.png",
    "figures/fig3_mitigation_tradeoff.png",
    "figures/fig4_carbon_per_model.png",
    "figures/fig5_shap_summary.png",
    "figures/fig5b_shap_top10_bar.png",
    "figures/fig6_accuracy_per_gram_co2.png",
]


def stage_python() -> str:
    """Interpreter for the stages: the project virtual environment."""
    for rel in ("Scripts/python.exe", "bin/python"):
        exe = ROOT / ".venv" / rel
        if exe.exists():
            return str(exe)
    raise SystemExit(f"run_all: no virtual environment at {ROOT / '.venv'}")


def run_command(python: str, stage: str, script: str, args: list[str]) -> None:
    """Run one stage command and stop the whole run if it fails."""
    display = f"python src/{script}{' ' + ' '.join(args) if args else ''}"
    print(f"  $ {display}", flush=True)
    code = subprocess.run([python, str(SRC / script), *args], cwd=ROOT).returncode
    if code != 0:
        raise SystemExit(
            f"\nrun_all: stage '{stage}' FAILED - '{display}' exited with {code}"
        )


def verify_outputs() -> None:
    """Fail loudly if a stage exited 0 but produced no file."""
    missing = [rel for rel in EXPECTED if not (ROOT / rel).exists()]
    if missing:
        raise SystemExit(f"run_all: missing outputs: {missing}")
    print(f"\nall {len(EXPECTED)} expected outputs are present")


def main(argv: list[str] | None = None) -> None:
    """Run the requested stages (all of them by default)."""
    argv = sys.argv[1:] if argv is None else argv

    if "--list" in argv:
        for name, commands in STAGES:
            line = " && ".join(
                f"src/{c[0]}{' ' + ' '.join(c[1:])}" for c in commands
            )
            print(f"{name:14} {line}")
        return

    known = [name for name, _ in STAGES]
    unknown = [a for a in argv if a not in known]
    if unknown:
        raise SystemExit(f"run_all: unknown stage(s) {unknown}; known: {known}")

    chosen = [(n, c) for n, c in STAGES if not argv or n in argv]
    python = stage_python()
    print(f"run_all: {len(chosen)} stage(s), interpreter {python}", flush=True)

    start = time.perf_counter()
    for index, (name, commands) in enumerate(chosen, 1):
        print(f"\n=== [{index}/{len(chosen)}] {name} ===", flush=True)
        stage_start = time.perf_counter()
        for command in commands:
            run_command(python, name, command[0], command[1:])
        print(f"--- {name} ok ({time.perf_counter() - stage_start:.0f}s)",
              flush=True)

    if not argv:  # only a complete run is checked end to end
        verify_outputs()
    print(f"run_all: finished in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
