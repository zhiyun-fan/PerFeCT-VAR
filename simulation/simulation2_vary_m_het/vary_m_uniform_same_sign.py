from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "methods"))

import pfedvar


REPLICATIONS = 4
M_STAR_GRID = (1, 2, 3, 4, 6, 9, 12)

N_CLIENTS = 30
T = 800
P = 40
RANK = 1
BURN_IN = 200
STABILITY_LIMIT = 0.8

B_SINGULAR_VALUE = 0.5
LOCAL_BLOCK_WIDTH = 15
S0_TRUE = 18
S_LOCAL_TRUE = 6
SHARED_DEGREE_TRUE = 2
S0_MAGNITUDE_RANGE = (0.1, 0.2)
S_LOCAL_ABS_VALUE = 0.15
S_LOCAL_MAGNITUDE_RANGE = (0.1, 0.2)

S0_FIT = 20
S_LOCAL_FIT = 8
GAMMA_FIT = 3 / P

BASU_PENALTY_MULTIPLIER = 0.5

INITIALIZER_ROUNDS = 100
PFEDVAR_ROUNDS = 200
STEP_SIZE = 0.1
PERSONAL_STEP_SIZE = 0.3

OUTPUT_DIR = HERE.parents[1] / "reproduced/simulation/simulation2_vary_m_het/vary_m_uniform_same_sign_outputs"


def allocate_local_support(
    rng: np.random.Generator,
    n_clients: int,
    sparsity: int,
    m_star: int,
    coordinate_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Random coordinates with bipartite Havel--Hakimi client allocation."""

    full_groups, residual = divmod(n_clients * sparsity, m_star)
    degrees = [m_star] * full_groups + ([residual] if residual else [])
    if m_star > n_clients or len(degrees) > len(coordinate_pool):
        raise ValueError("The requested local support is infeasible")

    coordinates = rng.choice(coordinate_pool, size=len(degrees), replace=False)
    incidence = np.zeros((n_clients, len(degrees)), dtype=bool)
    remaining = np.full(n_clients, sparsity)

    for coordinate_index, degree in enumerate(degrees):
        random_order = rng.permutation(n_clients)
        ranked = random_order[
            np.argsort(-remaining[random_order], kind="stable")
        ]
        selected = ranked[:degree]
        incidence[selected, coordinate_index] = True
        remaining[selected] -= 1

    if np.any(remaining):
        raise RuntimeError("Local support allocation did not fill every client")
    return coordinates, incidence


def generate_truth_families(
    rng: np.random.Generator,
    magnitude_mode: str = "fixed",
) -> tuple[
    dict[str, dict[int, dict[str, np.ndarray]]],
    dict[str, dict[int, dict[str, float | int | str]]],
    np.ndarray,
]:
    """Generate paired same-sign and independent-sign truth families."""

    if magnitude_mode not in {"fixed", "uniform"}:
        raise ValueError("magnitude_mode must be 'fixed' or 'uniform'")

    direction = np.zeros(P)
    direction[:LOCAL_BLOCK_WIDTH] = 1 / np.sqrt(LOCAL_BLOCK_WIDTH)
    B = B_SINGULAR_VALUE * np.outer(direction, direction)
    block_coordinates = np.array(
        [
            row * P + column
            for row in range(LOCAL_BLOCK_WIDTH)
            for column in range(LOCAL_BLOCK_WIDTH)
        ]
    )

    for _ in range(1_000):
        shared_support: set[tuple[int, int]] = set()
        row_counts = np.zeros(P, dtype=int)
        column_counts = np.zeros(P, dtype=int)
        while len(shared_support) < S0_TRUE:
            row, column = rng.integers(P, size=2)
            coordinate = (int(row), int(column))
            if (
                coordinate not in shared_support
                and row_counts[row] < SHARED_DEGREE_TRUE
                and column_counts[column] < SHARED_DEGREE_TRUE
            ):
                shared_support.add(coordinate)
                row_counts[row] += 1
                column_counts[column] += 1

        rows, columns = np.array(sorted(shared_support)).T
        S0 = np.zeros((P, P))
        S0[rows, columns] = rng.choice(
            (-1.0, 1.0), size=S0_TRUE
        ) * rng.uniform(*S0_MAGNITUDE_RANGE, size=S0_TRUE)

        block_order = rng.permutation(block_coordinates)
        shared_signs = rng.choice((-1.0, 1.0), size=P * P)
        independent_signs = rng.choice(
            (-1.0, 1.0), size=(N_CLIENTS, P * P)
        )
        if magnitude_mode == "fixed":
            local_magnitudes = np.full(
                (N_CLIENTS, S_LOCAL_TRUE), S_LOCAL_ABS_VALUE
            )
        else:
            local_magnitudes = rng.uniform(
                *S_LOCAL_MAGNITUDE_RANGE,
                size=(N_CLIENTS, S_LOCAL_TRUE),
            )
        truth_families: dict[str, dict[int, dict[str, np.ndarray]]] = {
            "same": {},
            "independent": {},
        }
        diagnostics: dict[
            str, dict[int, dict[str, float | int | str]]
        ] = {"same": {}, "independent": {}}

        for m_star in M_STAR_GRID:
            n_active_coordinates = N_CLIENTS * S_LOCAL_TRUE // m_star
            coordinates, incidence = allocate_local_support(
                rng,
                N_CLIENTS,
                S_LOCAL_TRUE,
                m_star,
                block_order[:n_active_coordinates],
            )
            frequency = incidence.sum(axis=0)
            if not np.all(frequency == m_star):
                raise RuntimeError("Local support did not attain m_star")

            S_local_same = np.zeros((N_CLIENTS, P, P))
            S_local_independent = np.zeros((N_CLIENTS, P, P))
            for client in range(N_CLIENTS):
                client_coordinates = coordinates[incidence[client]]
                S_local_same[client].ravel()[client_coordinates] = (
                    local_magnitudes[client]
                    * shared_signs[client_coordinates]
                )
                S_local_independent[client].ravel()[client_coordinates] = (
                    local_magnitudes[client]
                    * independent_signs[client, client_coordinates]
                )

            for sign_mode, S_local in (
                ("same", S_local_same),
                ("independent", S_local_independent),
            ):
                A_clients = B[None, :, :] + S0[None, :, :] + S_local
                max_operator_norm = float(
                    np.linalg.svd(A_clients, compute_uv=False)[:, 0].max()
                )
                if max_operator_norm >= STABILITY_LIMIT:
                    break

                truth_families[sign_mode][m_star] = {
                    "A_clients": A_clients,
                    "B": B,
                    "S0": S0,
                    "S_local": S_local,
                }
                mean_local = S_local.mean(axis=0)
                diagnostics[sign_mode][m_star] = {
                    "sign_mode": sign_mode,
                    "magnitude_mode": magnitude_mode,
                    "m_star": m_star,
                    "m_fit": m_star + 1,
                    "max_true_operator_norm": max_operator_norm,
                    "S_local_union_size": n_active_coordinates,
                    "S_local_frequency_min": int(frequency.min()),
                    "S_local_frequency_max": int(frequency.max()),
                    "mean_local_squared_norm": float(np.sum(mean_local**2)),
                    "mean_client_local_squared_norm": float(
                        np.mean(np.sum(S_local**2, axis=(1, 2)))
                    ),
                    "mean_local_B_inner_product": float(np.sum(mean_local * B)),
                }
            else:
                continue
            break

        if all(
            len(truth_families[sign_mode]) == len(M_STAR_GRID)
            for sign_mode in truth_families
        ):
            shocks = rng.normal(
                size=(N_CLIENTS, BURN_IN + T + 1, P)
            )
            return truth_families, diagnostics, shocks

    raise RuntimeError("Could not generate stable vary-m truth families")


def simulate_clients(
    A_clients: np.ndarray,
    shocks: np.ndarray,
) -> list[pfedvar.ClientData]:
    values = np.zeros_like(shocks)
    values[:, 0] = shocks[:, 0]
    for time_index in range(1, values.shape[1]):
        values[:, time_index] = (
            A_clients @ values[:, time_index - 1, :, None]
        )[:, :, 0] + shocks[:, time_index]
    return [
        pfedvar.ClientData(str(client), values[client, -(T + 1) :])
        for client in range(N_CLIENTS)
    ]


def basu_penalties(
    stats: pfedvar.SufficientStats,
) -> dict[str, float | np.ndarray]:
    """Clientwise spectral scales and penalties for initialization."""
    n_clients = stats.n_clients
    p = stats.Gxx.shape[-1]
    initializer_phi = np.array([np.linalg.norm(Gxx, 2) for Gxx in stats.Gxx])
    return {
        "initializer_phi_hat": initializer_phi,
        "initializer_lambda_B": BASU_PENALTY_MULTIPLIER
        * np.sqrt(initializer_phi * (p + np.log(n_clients)) / stats.sample_sizes),
        "initializer_lambda_C": BASU_PENALTY_MULTIPLIER
        * np.sqrt(initializer_phi * np.log(p * n_clients) / stats.sample_sizes),
    }


def fit_pfedvar(
    clients: list[pfedvar.ClientData],
    m_star: int,
    *,
    initializer_rounds: int = INITIALIZER_ROUNDS,
    pfedvar_rounds: int = PFEDVAR_ROUNDS,
) -> tuple[
    pfedvar.SufficientStats,
    pfedvar.FitResult,
    float,
]:
    stats = pfedvar.build_statistics(clients)
    penalties = basu_penalties(stats)
    initializer_eta = penalties["initializer_phi_hat"]
    initializer = pfedvar.local_basu_lrs(
        stats,
        pfedvar.BasuConfig(
            lambda_B=penalties["initializer_lambda_B"],
            lambda_C=penalties["initializer_lambda_C"],
            nu=P,
            max_iter=initializer_rounds,
            eta_init=initializer_eta,
            eta_min=initializer_eta / 10,
        ),
    )
    config = pfedvar.FedVARConfig(
        rank=RANK,
        s0=S0_FIT,
        s_local=[S_LOCAL_FIT] * N_CLIENTS,
        gamma=GAMMA_FIT,
        freq_cap=m_star + 1,
        rounds=pfedvar_rounds,
        eta_B=STEP_SIZE,
        eta_0=STEP_SIZE,
        eta_local=PERSONAL_STEP_SIZE,
    )
    started = perf_counter()
    state = pfedvar.initialize_fedvar(stats, config, initializer)
    fit = pfedvar.fedvar(stats, config, state)
    runtime = initializer.runtime + perf_counter() - started
    return stats, fit, runtime


def relative_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    squared_error = np.sum((estimate - truth) ** 2, axis=(-2, -1))
    squared_truth = np.sum(truth**2, axis=(-2, -1))
    if weights is not None:
        squared_error = weights @ squared_error
        squared_truth = weights @ squared_truth
    return float(np.sqrt(squared_error / squared_truth))


def support_f1(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_support = estimate != 0.0
    truth_support = truth != 0.0
    true_positive = np.count_nonzero(estimate_support & truth_support)
    denominator = np.count_nonzero(estimate_support) + np.count_nonzero(
        truth_support
    )
    return 2.0 * true_positive / denominator if denominator else 1.0


def metrics(
    fit: pfedvar.FitResult,
    truth: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float | str]:
    A_client_relative_errors = np.linalg.norm(
        fit.A_clients - truth["A_clients"], axis=(1, 2)
    ) / np.linalg.norm(truth["A_clients"], axis=(1, 2))
    B_hat = fit.components["B"]
    S0_hat = fit.components["S0"]
    S_local_hat = fit.components["S_local"]
    return {
        "A_relative_error": relative_error(
            fit.A_clients, truth["A_clients"], weights
        ),
        "A_client_relative_errors": json.dumps(
            np.round(A_client_relative_errors, 6).tolist()
        ),
        "B_relative_error": relative_error(B_hat, truth["B"]),
        "S0_relative_error": relative_error(S0_hat, truth["S0"]),
        "S_local_relative_error": relative_error(
            S_local_hat, truth["S_local"], weights
        ),
        "S0_support_f1": support_f1(S0_hat, truth["S0"]),
        "S_local_support_f1": support_f1(S_local_hat, truth["S_local"]),
        "estimated_local_frequency_max": float(
            np.count_nonzero(S_local_hat, axis=0).max()
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing results: {path}; use a new --output-dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_seed(
    seed: int,
    *,
    initializer_rounds: int = INITIALIZER_ROUNDS,
    pfedvar_rounds: int = PFEDVAR_ROUNDS,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    truth_families, diagnostics, shocks = generate_truth_families(rng, "uniform")
    rows: list[dict[str, object]] = []

    for m_star in M_STAR_GRID:
        started = perf_counter()
        truth = truth_families["same"][m_star]
        clients = simulate_clients(truth["A_clients"], shocks)
        stats, fit, runtime = fit_pfedvar(
            clients,
            m_star,
            initializer_rounds=initializer_rounds,
            pfedvar_rounds=pfedvar_rounds,
        )
        rows.append(
            {
                "seed": seed,
                "N": N_CLIENTS,
                "T": T,
                "p": P,
                "rank": RANK,
                "method": "pfedvar",
                "runtime_seconds": runtime,
                **diagnostics["same"][m_star],
                **metrics(fit, truth, stats.weights),
            }
        )
        print(
            f"seed={seed} sign=same m_star={m_star:2d} "
            f"pFedVAR={rows[-1]['A_relative_error']:.4f} "
            f"elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, help="first Monte Carlo seed")
    parser.add_argument("--replications", type=int, default=REPLICATIONS,
                        help="number of consecutive seeds (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="directory for new per-seed CSV files")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.replications < 1:
        parser.error("--replications must be positive")
    output_dir = args.output_dir.resolve()
    seeds = range(args.seed, args.seed + args.replications)
    targets = [output_dir / f"seed_{seed}.csv" for seed in seeds]
    existing = [path for path in targets if path.exists()]
    if existing:
        parser.error(f"output already exists: {existing[0]}; use a new --output-dir")

    for seed in seeds:
        rows = run_seed(seed)
        write_rows(output_dir / f"seed_{seed}.csv", rows)


if __name__ == "__main__":
    main()
