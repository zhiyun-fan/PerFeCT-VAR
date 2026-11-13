#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import tempfile

# Writable cache for local and HPC execution; does not depend on the shell cwd.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "perfect_var_mplcache"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "reproduced" / "figures" / "simulation1"
SEEDS = set(range(22567, 22767))
T_METHODS = {"pfedvar", "local_lrs", "pooled_lrs", "ditto"}
N_METHODS = {"pfedvar"}
COMPONENTS = ("A_relative_error", "B_relative_error", "S0_relative_error", "S_local_relative_error")


def read_outputs(root: Path) -> pd.DataFrame:
    folder = "vary_TN_outputs"
    files = sorted((root / folder).glob("*.csv"))
    expected_files = 400
    if len(files) != expected_files:
        raise ValueError(f"{folder}: expected {expected_files} CSV files, found {len(files)}")
    frames = []
    for path in files:
        match = re.search(r"seed_(\d+)\.csv$", path.name)
        if match is None:
            raise ValueError(f"Unexpected input filename: {path}")
        data = pd.read_csv(path)
        if set(data["seed"]) != {int(match.group(1))}:
            raise ValueError(f"Seed column disagrees with filename: {path}")
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    if set(data["seed"]) != SEEDS:
        raise ValueError(f"{folder}: expected seeds {min(SEEDS)}--{max(SEEDS)}")
    if set(data["experiment"]) != {"vary_T", "vary_N"}:
        raise ValueError("Unexpected T/N experiment names")
    specifications = [
        ("vary_T", data[data["experiment"] == "vary_T"], "T", (100,200,400,800,1200,1600,2000), T_METHODS),
        ("vary_N", data[data["experiment"] == "vary_N"], "N", (10,20,30,40,50,60), N_METHODS),
    ]
    for name, frame, x, levels, methods in specifications:
        if set(frame[x]) != set(levels) or set(frame["method"]) != methods:
            raise ValueError(f"{name}: incomplete or unexpected grid")
        if frame.duplicated(["seed", "method", x]).any():
            raise ValueError(f"{name}: duplicate seed/method/condition")
        counts = frame.groupby(["method", x])["seed"].nunique()
        if not (counts == 200).all() or len(counts) != len(levels) * len(methods):
            raise ValueError(f"{name}: each plotted point must contain 200 replicates")
        plotted = frame[frame["method"] == "pfedvar"]
        metrics = list(COMPONENTS)
        if name == "vary_T":
            metrics.append("initial_A_relative_error")
            comparison = frame[frame["method"].isin(["pfedvar","local_lrs","pooled_lrs","ditto"])]
            if not np.isfinite(comparison["A_relative_error"]).all():
                raise ValueError("Non-finite errors in the T comparison")
        if not np.isfinite(plotted[metrics].to_numpy()).all():
            raise ValueError(f"{name}: non-finite plotted values")
    return data


def save_figure(fig, output: Path, stem: str) -> None:
    # Retain vector paths and text in PDF.
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    FIGURE_DIR = args.output.resolve() if args.output else DEFAULT_OUTPUT
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tn_data = read_outputs(ROOT)
    METHOD_LABELS = {
        "pfedvar": "PerFeCT-VAR",
        "local_lrs": "Local LRS",
        "pooled_lrs": "Pooled LRS",
        "ditto": "Ditto",
    }
    COMPONENT_LABELS = {
        "A_relative_error": r"$A_i$",
        "B_relative_error": r"$LR'$",
        "S0_relative_error": r"$S_0$",
        "S_local_relative_error": r"$S_i$",
    }
    COLORS = plt.get_cmap("tab10").colors
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
    })
    vary_t = tn_data[tn_data["experiment"] == "vary_T"]
    vary_n = tn_data[tn_data["experiment"] == "vary_N"]

    pfedvar_n = vary_n[vary_n["method"] == "pfedvar"]
    component_colors = dict(zip(COMPONENT_LABELS, COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for index, (method, label) in enumerate(METHOD_LABELS.items()):
        means_t = vary_t[vary_t["method"] == method].groupby("T")["A_relative_error"].mean().sort_index()
        axes[0].plot(means_t.index, means_t, marker="s", markersize=4, linewidth=1.8, color=COLORS[index], label=label)
    initial_t = vary_t[vary_t["method"] == "pfedvar"].groupby("T")["initial_A_relative_error"].mean().sort_index()
    axes[0].plot(initial_t.index, initial_t, marker="s", markersize=4, linewidth=1.8, color=COLORS[7], label="Initial")

    for component, label in COMPONENT_LABELS.items():
        means = pfedvar_n.groupby("N")[component].mean().sort_index()
        axes[1].plot(means.index, means, linestyle="none", marker="s", markersize=6, color=component_colors[component], label=label, zorder=3)
        if component in ("A_relative_error", "S_local_relative_error"):
            axes[1].plot(means.index, means, linewidth=1.8, color=component_colors[component], label="_nolegend_", zorder=2)
        if component in ("B_relative_error", "S0_relative_error"):
            inverse_n = 1.0 / means.index.to_numpy(dtype=float)
            c = np.linalg.lstsq(inverse_n[:, None], means.to_numpy() ** 2, rcond=None)[0][0]
            n_grid = np.linspace(means.index.min(), means.index.max(), 200)
            axes[1].plot(n_grid, np.sqrt(c / n_grid), linestyle="--", linewidth=1.8, color=component_colors[component])
    axes[0].set_xlabel(r"$T$")
    axes[0].set_xticks(sorted(vary_t["T"].unique()))
    axes[0].tick_params(axis="x", labelrotation=0)
    t_labels = axes[0].get_xticklabels()
    t_labels[0].set_transform(t_labels[0].get_transform() + ScaledTranslation(-11 / 72, 0, fig.dpi_scale_trans))
    t_labels[1].set_transform(t_labels[1].get_transform() + ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans))
    axes[1].set(xlabel=r"$N$", ylabel="Relative error")
    axes[1].set_xticks(sorted(vary_n["N"].unique()))
    axes[1].legend(
        frameon=True,
        loc="lower left",
        ncol=2,
        fontsize=12,
        handlelength=0.8,
        handletextpad=0.3,
        labelspacing=0.3,
        columnspacing=0.8,
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Relative error of $\{A_i\}$")
    axes[0].legend(frameon=True, loc="upper right")
    # Keep the reference legend location while avoiding the new pooled curve.
    t_panel_max = max(float(np.max(line.get_ydata())) for line in axes[0].lines)
    axes[0].set_ylim(top=1.3 * t_panel_max)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR, "vary_TN")

    print(FIGURE_DIR / "vary_TN.pdf")

if __name__ == "__main__":
    main()
