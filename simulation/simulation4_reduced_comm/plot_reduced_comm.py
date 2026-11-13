#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _plot_common import read_outputs, save_figure, plt
DEFAULT_OUTPUT = HERE.parents[1] / "reproduced" / "figures" / "simulation4"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    FIGURE_DIR = args.output.resolve()
    communication = read_outputs(args.root.resolve(), "reduced_comm_outputs")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    COMMUNICATION_LABELS = {
        "pfedvar_exact": "Full communication",
        "pfedvar_k40_selective": r"$\tilde{s}_0=40$, selective",
        "pfedvar_k160_selective": r"$\tilde{s}_0=160$, selective",
        "pfedvar_k40_every10": r"$\tilde{s}_0=40$, every 10",
        "pfedvar_k160_every10": r"$\tilde{s}_0=160$, every 10",
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
        "pdf.fonttype": 42,
    })
    communication = communication[communication["iteration"] <= 200]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for index, (method, label) in enumerate(COMMUNICATION_LABELS.items()):
        by_iteration = communication[communication["method"] == method].groupby("iteration")["relative_error"].mean().sort_index()
        by_communication = communication[(communication["method"] == method) & (communication["iteration"] > 0)].groupby("iteration")[["cumulative_uplink_mib", "relative_error"]].mean().sort_index()
        axes[0].plot(by_iteration.index, by_iteration, linewidth=1.8, color=COLORS[index], label=label)
        axes[1].plot(by_communication["cumulative_uplink_mib"], by_communication["relative_error"], linewidth=1.8, color=COLORS[index], label=label)
    axes[0].set(xlabel="Iteration", ylabel=r"Mean relative error of all $A_i$")
    axes[0].set_xticks([0, 40, 80, 120, 160, 200])
    axes[1].set_xlabel("Cumulative uplink (MiB, log scale)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR, "communication")

    print(FIGURE_DIR / "communication.pdf")

if __name__ == "__main__":
    main()
