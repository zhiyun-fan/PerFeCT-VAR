#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _plot_common import (COMPONENTS, HETEROGENEITY_FOLDER,
                          read_outputs, save_figure, np, pd, plt)
DEFAULT_OUTPUT = HERE.parents[1] / "reproduced" / "figures" / "simulation2"

def plot_heterogeneity_figures(vary_m: pd.DataFrame, vary_het: pd.DataFrame, FIGURE_DIR: Path) -> None:
    """Plot the combined vary-m and heterogeneity figure."""
    COMPONENT_LABELS = {
        "A_relative_error": r"$A_i$", "B_relative_error": r"$LR'$",
        "S0_relative_error": r"$S_0$", "S_local_relative_error": r"$S_i$",
    }
    component_colors = dict(zip(COMPONENT_LABELS, plt.get_cmap("tab10").colors))
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.titleweight": "bold", "figure.dpi": 120, "pdf.fonttype": 42,
    })
    plotted = {}

    def capture_lines(fig):
        return [[{"label": line.get_label(), "x": np.asarray(line.get_xdata()).tolist(),
                  "y": np.asarray(line.get_ydata()).tolist()} for line in ax.lines]
                for ax in fig.axes]

    pfedvar_m = vary_m[vary_m["method"] == "pfedvar"]

    pfedvar_het = vary_het[vary_het["method"] == "pfedvar"]
    levels = ["zero", "low", "medium", "high"]
    positions = np.arange(len(levels))
    panels = (("B_relative_error", "S0_relative_error"), ("A_relative_error", "S_local_relative_error"))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex="row")
    for column, components in enumerate(panels):
        for component in components:
            means_m = pfedvar_m.groupby("m_star")[component].mean().sort_index()
            means_het = pfedvar_het.groupby("heterogeneity_level")[component].mean().reindex(levels)
            axes[0, column].plot(means_m.index, means_m, marker="s", markersize=5, linewidth=1.8, color=component_colors[component], label=COMPONENT_LABELS[component])
            axes[1, column].plot(positions, means_het, marker="s", markersize=5, linewidth=1.8, color=component_colors[component], label=COMPONENT_LABELS[component])
        axes[0, column].set_xlabel(r"$m_*$")
        axes[0, column].set_xticks(sorted(pfedvar_m["m_star"].unique()))
        axes[0, column].legend(frameon=True, loc="lower left")
        axes[1, column].set_xlabel("covariance discrepancy level")
        axes[1, column].set_xticks(positions, levels)
        axes[1, column].legend(frameon=True, loc="upper left" if column == 0 else "center right")
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].margins(y=0.10)
        for ax in axes[:, column]:
            ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Relative error")
    axes[1, 0].set_ylabel("Relative error")
    fig.tight_layout()
    plotted["vary_m_het"] = capture_lines(fig)
    save_figure(fig, FIGURE_DIR, "vary_m_het")

    means = pfedvar_het.groupby("heterogeneity_level")[list(COMPONENTS)].mean().reindex(levels)
    for panel_index, components in enumerate(panels):
        for line_index, component in enumerate(components):
            line = plotted["vary_m_het"][panel_index + 2][line_index]
            if not np.array_equal(line["x"], positions) or not np.allclose(line["y"], means[component], atol=1e-14, rtol=0):
                raise AssertionError(f"Plotted data mismatch: vary_m_het/{component}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    vary_m = read_outputs(args.root.resolve(), "vary_m_uniform_same_sign_outputs")
    vary_het = read_outputs(args.root.resolve(), HETEROGENEITY_FOLDER)
    output.mkdir(parents=True, exist_ok=True)
    plot_heterogeneity_figures(vary_m, vary_het, output)
    print(output / "vary_m_het.pdf")

if __name__ == "__main__":
    main()
