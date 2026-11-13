#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _plot_common import np, pd, plt

SEEDS = set(range(251001, 251201))
M_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14)
COMPONENTS = ("A_relative_error", "B_relative_error", "S0_relative_error", "S_local_relative_error")
DEFAULT_OUTPUT = HERE.parents[1] / "reproduced" / "figures" / "simulation3"


def read_outputs(root: Path) -> pd.DataFrame:
    folder = root / "vary_m_lq_outputs"
    files = sorted(folder.glob("*.csv"))
    expected_names = {f"seed_{seed}.csv" for seed in SEEDS}
    if {p.name for p in files} != expected_names:
        raise ValueError("Lq: expected exactly seeds 251001--251200")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if set(frame["seed"]) != {int(path.stem.removeprefix("seed_"))}:
            raise ValueError(f"Seed column disagrees with filename: {path}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    fixed = {
        "experiment": "vary_m_lq", "method": "pfedvar", "N": 30, "T": 400,
        "p": 40, "rank": 1, "q": 0.25, "B_singular_value": 0.6,
        "personal_support_pool": "full_block_15x15", "m_star": 14,
        "s0_true": 12, "s_local_true": 60, "s0_fit": 16, "s_local_fit": 75,
        "initializer_rounds": 100, "FedVAR_rounds": 200,
        "eta_B": 0.1, "eta_0": 0.1, "eta_local": 0.3,
    }
    for key, value in fixed.items():
        if not data[key].eq(value).all():
            raise ValueError(f"Lq: unexpected {key}")
    if set(data["m_fit"]) != set(M_GRID) or not data["actual_frequency_cap"].eq(data["m_fit"]).all():
        raise ValueError("Lq: unexpected working-cap grid")
    if data.duplicated(["seed", "m_fit"]).any():
        raise ValueError("Lq: duplicate seed/working-cap pair")
    counts = data.groupby("m_fit")["seed"].nunique()
    if len(data) != 2000 or not counts.eq(200).all():
        raise ValueError("Lq: each working cap must have 200 replications")
    if not np.isfinite(data[list(COMPONENTS)].to_numpy()).all():
        raise ValueError("Lq: non-finite component errors")
    if not data.groupby("seed")["data_sha256"].nunique().eq(1).all():
        raise ValueError("Lq: working caps did not share one input per seed")
    return data


def plot_results(data: pd.DataFrame, output: Path) -> None:
    means = data.groupby("m_fit")[list(COMPONENTS)].mean().reindex(M_GRID)
    labels = {"A_relative_error": r"$A_i$", "B_relative_error": r"$B$",
              "S0_relative_error": r"$S_0$", "S_local_relative_error": r"$S_i$"}
    colors = dict(zip(COMPONENTS, plt.get_cmap("tab10").colors))
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "figure.dpi": 120, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for ax, components in zip(axes, (("B_relative_error", "S0_relative_error"),
                                    ("A_relative_error", "S_local_relative_error"))):
        for component in components:
            ax.plot(means.index, means[component], marker="s", markersize=5,
                    linewidth=1.8, color=colors[component], label=labels[component])
        ax.set_xlabel(r"Working cap $m$")
        ax.set_xticks(M_GRID)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, loc="lower left")
    axes[0].set_ylabel("Relative error")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "vary_m_lq.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_results(read_outputs(args.root.resolve()), args.output.resolve())
    print(args.output.resolve() / "vary_m_lq.pdf")


if __name__ == "__main__":
    main()
