#!/usr/bin/env python3
"""Run PerFeCT-VAR experiments, rebuild real-data inputs, and reproduce figures."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SIMULATIONS = {
    "vary_TN": ("simulation1_varyTN/vary_TN.py", 22567),
    "vary_m": ("simulation2_vary_m_het/vary_m_uniform_same_sign.py", 22567),
    "vary_het": ("simulation2_vary_m_het/vary_het.py", 82567),
    "vary_lq": ("simulation3_lq/vary_m_lq.py", 251001),
    "reduced_comm": ("simulation4_reduced_comm/reduced_comm.py", 22567),
}
SIMULATION_FIGURES = {
    "simulation1": ("simulation1_varyTN", "plot_vary_TN.py"),
    "simulation2": ("simulation2_vary_m_het", "plot_vary_m_het.py"),
    "simulation3": ("simulation3_lq", "plot_vary_m_lq.py"),
    "simulation4": ("simulation4_reduced_comm", "plot_reduced_comm.py"),
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run(script: Path, *arguments: object) -> None:
    command = [sys.executable, "-u", str(script), *map(str, arguments)]
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONHASHSEED"] = "0"
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)

    simulations = commands.add_parser("simulations", help="Generate data and fit simulations")
    simulations.add_argument("--experiments", choices=tuple(SIMULATIONS), nargs="+",
                             default=list(SIMULATIONS))
    simulations.add_argument("--replications", type=positive_int, default=200,
                             help="Seeds per experiment, starting at its archived first seed (default: 200)")

    preprocessing = commands.add_parser("preprocess", help="Rebuild real-data inputs from extracted movement CSVs")
    preprocessing.add_argument("--raw-dir", type=Path, required=True)

    realdata = commands.add_parser("realdata", help="Run all 11 methods, CV, test, and full-sample refit")
    realdata.add_argument("--workers", type=positive_int, default=4)
    realdata.add_argument("--stage", dest="realdata_stage",
                          choices=("all", "cv", "test", "refit", "check"), default="all")
    realdata.add_argument("--output-dir", type=Path,
                         default=ROOT / "reproduced" / "realdata_results")

    figures = commands.add_parser("figures", help="Plot all figures from complete results")
    figures.add_argument("--source", choices=("reference", "reproduced"), default="reference")
    figures.add_argument("--scope", choices=("all", "simulations", "realdata", *SIMULATION_FIGURES), default="all")
    args = parser.parse_args()

    if args.stage == "simulations":
        for experiment in dict.fromkeys(args.experiments):
            relative, seed = SIMULATIONS[experiment]
            if experiment == "vary_lq":
                # Keep the original Lq runner and its single-seed interface intact.
                output = ROOT / "reproduced" / "simulation" / "simulation3_lq" / "vary_m_lq_outputs"
                seeds = range(seed, seed + args.replications)
                existing = [output / f"seed_{value}.csv" for value in seeds
                            if (output / f"seed_{value}.csv").exists()]
                if existing:
                    parser.error(f"output already exists: {existing[0]}; choose a new output directory using the direct runner")
                for value in seeds:
                    run(ROOT / "simulation" / relative, value, "--output-dir", output)
            else:
                run(ROOT / "simulation" / relative, seed, "--replications", args.replications)
    elif args.stage == "preprocess":
        output = ROOT / "reproduced" / "realdata_data"
        run(ROOT / "realdata" / "00a_build_panel.py", "--raw-dir", args.raw_dir.resolve(),
            "--output", output / "raw_panel.npz")
        run(ROOT / "realdata" / "00b_preprocess_panel.py", "--input", output / "raw_panel.npz",
            "--output", output / "processed.npz")
    elif args.stage == "realdata":
        run(ROOT / "realdata" / "01_run_experiment.py", "--stage", args.realdata_stage,
            "--output-dir", args.output_dir.resolve(), "--workers", args.workers)
    elif args.stage == "figures":
        if args.source == "reference":
            simulation_root = ROOT / "simulation"
            realdata_results = ROOT / "realdata" / "results"
        else:
            simulation_root = ROOT / "reproduced" / "simulation"
            realdata_results = ROOT / "reproduced" / "realdata_results"
        for scope, (folder, script) in SIMULATION_FIGURES.items():
            if args.scope in ("all", "simulations", scope):
                run(ROOT / "simulation" / folder / script,
                    "--root", simulation_root / folder)
        if args.scope in ("all", "realdata"):
            run(ROOT / "realdata" / "figures" / "plot_formal_figures.py",
                "--results-dir", realdata_results,
                "--output-dir", ROOT / "reproduced" / "figures" / "realdata")


if __name__ == "__main__":
    main()
