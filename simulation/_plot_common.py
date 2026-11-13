from __future__ import annotations
import json
import os
from pathlib import Path
import re
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "perfect_var_mplcache"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEEDS = set(range(22567, 22767))
HETEROGENEITY_FOLDER = "vary_het_rank1_outputs"
HETEROGENEITY_SEEDS = set(range(82567, 82767))
HETEROGENEITY_METHODS = {"pfedvar"}
COMM_METHODS = {
    "pfedvar_exact", "pfedvar_k40_selective", "pfedvar_k160_selective",
    "pfedvar_k40_every10", "pfedvar_k160_every10",
}
COMPONENTS = ("A_relative_error", "B_relative_error", "S0_relative_error", "S_local_relative_error")


def read_outputs(root: Path, folder: str) -> pd.DataFrame:
    files = sorted((root / folder).glob("*.csv"))
    expected_files = 200
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
    expected_seeds = HETEROGENEITY_SEEDS if folder == HETEROGENEITY_FOLDER else SEEDS
    if set(data["seed"]) != expected_seeds:
        raise ValueError(f"{folder}: expected seeds {min(expected_seeds)}--{max(expected_seeds)}")
    if folder == "vary_m_uniform_same_sign_outputs":
        if set(data["sign_mode"]) != {"same"} or set(data["magnitude_mode"]) != {"uniform"}:
            raise ValueError("The vary-m manuscript figure requires uniform same sign")
        specifications = [("vary_m", data, "m_star", (1,2,3,4,6,9,12), {"pfedvar"})]
    elif folder == HETEROGENEITY_FOLDER:
        for key, value in {"N": 30, "T": 800, "p": 40, "rank": 1,
                           "S0_magnitude_min": 0.03, "S0_magnitude_max": 0.07,
                           "local_singular_value": 0.4}.items():
            if not data[key].eq(value).all():
                raise ValueError(f"{folder}: unexpected {key}")
        if not data["B_singular_values"].map(json.loads).apply(lambda x: x == [0.3]).all():
            raise ValueError(f"{folder}: expected B singular value 0.3")
        if not np.isfinite(data["A_relative_error"]).all():
            raise ValueError(f"{folder}: non-finite method error")
        specifications = [("vary_het", data, "heterogeneity_level", ("zero","low","medium","high"), HETEROGENEITY_METHODS)]
    elif folder == "reduced_comm_outputs":
        specifications = [("communication", data, "iteration", tuple(range(201)), COMM_METHODS)]
    else:
        raise ValueError(folder)
    for name, frame, x, levels, methods in specifications:
        if set(frame[x]) != set(levels) or set(frame["method"]) != methods:
            raise ValueError(f"{name}: incomplete or unexpected grid")
        if frame.duplicated(["seed", "method", x]).any():
            raise ValueError(f"{name}: duplicate seed/method/condition")
        counts = frame.groupby(["method", x])["seed"].nunique()
        if not (counts == 200).all() or len(counts) != len(levels) * len(methods):
            raise ValueError(f"{name}: each plotted point must contain 200 replicates")
        if name == "communication":
            plotted = frame
            metrics = ["relative_error", "cumulative_uplink_mib"]
        else:
            plotted = frame[frame["method"] == "pfedvar"]
            metrics = list(COMPONENTS)
        if not np.isfinite(plotted[metrics].to_numpy()).all():
            raise ValueError(f"{name}: non-finite plotted values")
    return data

def save_figure(fig, output: Path, stem: str) -> None:
    # Retain vector paths and text in PDF.
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
