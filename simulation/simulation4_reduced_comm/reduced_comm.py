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
from reducedcom_pfedvar import reduced_communication_fedvar


OUTPUT_DIR = HERE.parents[1] / "reproduced/simulation/simulation4_reduced_comm/reduced_comm_outputs"
REPLICATIONS = 4

N_CLIENTS = 30
T = 800
P = 40
RANK = 2
BURN_IN = 200
SINGULAR_VALUES = np.array([1.0, 0.5])
MU_LIMIT = 5.0
STABILITY_LIMIT = 0.8

S0_TRUE = 36
S_LOCAL_TRUE = 18
SHARED_DEGREE_TRUE = 2
M_STAR_TRUE = 2
S0_MAGNITUDE_RANGE = (0.1, 0.2)
S_LOCAL_MAGNITUDE_RANGE = (0.2, 0.4)

S0_FIT = 40
S_LOCAL_FIT = 20
GAMMA_FIT = 3 / P
FREQUENCY_FIT = 3

INITIALIZER_C = 0.5
INITIALIZER_ROUNDS = 100
PFEDVAR_ROUNDS = 200
STEP_SIZE = 0.1
PERSONAL_STEP_SIZE = 0.3

FLOAT_BYTES = 8
INDEX_BYTES = max(1, ((P * P - 1).bit_length() + 7) // 8)
PAIR_BYTES = FLOAT_BYTES + INDEX_BYTES
CANDIDATE_BUDGET = min(
    P * P,
    S_LOCAL_FIT + (N_CLIENTS - 1) * S_LOCAL_FIT // FREQUENCY_FIT,
)

SETTINGS = (
    ("pfedvar_exact", None, "exact"),
    ("pfedvar_k40_selective", S0_FIT, "selective"),
    ("pfedvar_k160_selective", 4 * S0_FIT, "selective"),
    ("pfedvar_k40_every10", S0_FIT, "every10"),
    ("pfedvar_k160_every10", 4 * S0_FIT, "every10"),
)


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


def generate_truth(
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Generate the N=30 population used by the formal vary-T/N DGP."""

    for _ in range(1_000):
        while True:
            U, _ = np.linalg.qr(rng.normal(size=(P, RANK)))
            V, _ = np.linalg.qr(rng.normal(size=(P, RANK)))
            mu = max(
                P / RANK * np.max(np.sum(U**2, axis=1)),
                P / RANK * np.max(np.sum(V**2, axis=1)),
            )
            if mu <= MU_LIMIT:
                break
        roots = np.sqrt(SINGULAR_VALUES)
        L = U * roots
        R = V * roots
        B = L @ R.T

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

        coordinates, incidence = allocate_local_support(
            rng,
            N_CLIENTS,
            S_LOCAL_TRUE,
            M_STAR_TRUE,
            np.arange(P * P),
        )
        signs = rng.choice((-1.0, 1.0), size=(N_CLIENTS, S_LOCAL_TRUE))
        magnitudes = rng.uniform(
            *S_LOCAL_MAGNITUDE_RANGE,
            size=(N_CLIENTS, S_LOCAL_TRUE),
        )
        S_local = np.zeros((N_CLIENTS, P, P))
        for client in range(N_CLIENTS):
            client_coordinates = coordinates[incidence[client]]
            S_local[client].ravel()[client_coordinates] = (
                signs[client] * magnitudes[client]
            )

        A_clients = B[None, :, :] + S0[None, :, :] + S_local
        max_radius = float(pfedvar.spectral_radii(A_clients).max())
        if max_radius < STABILITY_LIMIT:
            return {
                "A_clients": A_clients,
                "B": B,
                "S0": S0,
                "S_local": S_local,
            }, float(mu), max_radius
    raise RuntimeError("Could not generate a stable N=30 truth")


def simulate_clients(
    truth: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[pfedvar.ClientData]:
    shocks = rng.normal(size=(N_CLIENTS, BURN_IN + T + 1, P))
    values = np.zeros_like(shocks)
    values[:, 0] = shocks[:, 0]
    for time_index in range(1, values.shape[1]):
        values[:, time_index] = (
            truth["A_clients"] @ values[:, time_index - 1, :, None]
        )[:, :, 0] + shocks[:, time_index]
    return [
        pfedvar.ClientData(str(client), values[client, -(T + 1) :])
        for client in range(N_CLIENTS)
    ]


def refresh_rounds(policy: str, rounds: int) -> tuple[int, ...]:
    if policy == "selective":
        selected = {
            *range(0, min(40, rounds)),
            *range(44, min(100, rounds), 5),
            *range(109, rounds, 10),
        }
    elif policy == "every10":
        selected = set(range(9, rounds, 10))
    else:
        return ()
    return tuple(sorted(selected))


def communication_counts(
    iteration: int,
    shared_upload_budget: int | None,
    refreshes: tuple[int, ...],
) -> dict[str, int]:
    """Aggregate uplink bytes after ``iteration`` completed updates."""

    if shared_upload_budget is None:
        dense_gradient = iteration * N_CLIENTS * P * P * FLOAT_BYTES
        dense_score = dense_gradient
        factor_gradient = 0
        shared_topk = 0
        fcht_score = 0
    else:
        dense_gradient = 0
        dense_score = 0
        factor_gradient = (
            iteration * N_CLIENTS * 2 * P * RANK * FLOAT_BYTES
        )
        shared_topk = (
            iteration * N_CLIENTS * shared_upload_budget * PAIR_BYTES
        )
        completed_refreshes = sum(
            round_index < iteration for round_index in refreshes
        )
        fcht_score = (
            completed_refreshes
            * N_CLIENTS
            * CANDIDATE_BUDGET
            * PAIR_BYTES
        )
    total = (
        dense_gradient
        + dense_score
        + factor_gradient
        + shared_topk
        + fcht_score
    )
    return {
        "cumulative_dense_gradient_bytes": dense_gradient,
        "cumulative_dense_score_bytes": dense_score,
        "cumulative_factor_gradient_bytes": factor_gradient,
        "cumulative_shared_topk_bytes": shared_topk,
        "cumulative_fcht_score_bytes": fcht_score,
        "cumulative_uplink_bytes": total,
    }


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


def final_metrics(
    fit: pfedvar.FitResult,
    truth: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float | str]:
    A_client_errors = np.linalg.norm(
        fit.A_clients - truth["A_clients"], axis=(1, 2)
    ) / np.linalg.norm(truth["A_clients"], axis=(1, 2))
    return {
        "final_A_relative_error": relative_error(
            fit.A_clients, truth["A_clients"], weights
        ),
        "final_A_client_relative_errors": json.dumps(
            np.round(A_client_errors, 6).tolist()
        ),
        "final_B_relative_error": relative_error(
            fit.components["B"], truth["B"]
        ),
        "final_S0_relative_error": relative_error(
            fit.components["S0"], truth["S0"]
        ),
        "final_S_local_relative_error": relative_error(
            fit.components["S_local"], truth["S_local"], weights
        ),
        "final_S0_support_f1": support_f1(
            fit.components["S0"], truth["S0"]
        ),
        "final_S_local_support_f1": support_f1(
            fit.components["S_local"], truth["S_local"]
        ),
    }


def run_seed(seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    truth, mu, max_radius = generate_truth(rng)
    stats = pfedvar.build_statistics(simulate_clients(truth, rng))

    initializer_phi = np.array([np.linalg.norm(Gxx, 2) for Gxx in stats.Gxx])
    initializer = pfedvar.local_basu_lrs(
        stats,
        pfedvar.BasuConfig(
            lambda_B=INITIALIZER_C
            * np.sqrt(initializer_phi * (P + np.log(N_CLIENTS)) / stats.sample_sizes),
            lambda_C=INITIALIZER_C
            * np.sqrt(initializer_phi * np.log(P * N_CLIENTS) / stats.sample_sizes),
            nu=P,
            max_iter=INITIALIZER_ROUNDS,
            eta_init=initializer_phi,
            eta_min=initializer_phi / 10,
        ),
    )
    config = pfedvar.FedVARConfig(
        rank=RANK,
        s0=S0_FIT,
        s_local=[S_LOCAL_FIT] * N_CLIENTS,
        gamma=GAMMA_FIT,
        freq_cap=FREQUENCY_FIT,
        rounds=PFEDVAR_ROUNDS,
        eta_B=STEP_SIZE,
        eta_0=STEP_SIZE,
        eta_local=PERSONAL_STEP_SIZE,
    )
    started = perf_counter()
    initial_state = pfedvar.initialize_fedvar(stats, config, initializer)
    initialization_runtime = initializer.runtime + perf_counter() - started

    fits: dict[str, pfedvar.FitResult] = {}
    schedules: dict[str, tuple[int, ...]] = {}
    for method, shared_budget, policy in SETTINGS:
        schedule = refresh_rounds(policy, PFEDVAR_ROUNDS)
        if shared_budget is None:
            fit = pfedvar.fedvar(
                stats, config, initial_state, truth["A_clients"]
            )
        else:
            fit = reduced_communication_fedvar(
                stats,
                config,
                initial_state,
                shared_upload_budget=shared_budget,
                refresh_rounds=schedule,
                truth_A=truth["A_clients"],
            )
        fits[method] = fit
        schedules[method] = schedule
        final_bytes = communication_counts(
            PFEDVAR_ROUNDS, shared_budget, schedule
        )["cumulative_uplink_bytes"]
        print(
            f"seed={seed} {method} "
            f"error={fit.history[-1]['relative_error']:.4f} "
            f"uplink_MiB={final_bytes / 2**20:.2f}",
            flush=True,
        )

    exact_final_bytes = communication_counts(
        PFEDVAR_ROUNDS, None, ()
    )["cumulative_uplink_bytes"]
    rows: list[dict[str, object]] = []
    for method, shared_budget, policy in SETTINGS:
        fit = fits[method]
        schedule = schedules[method]
        metrics = final_metrics(fit, truth, stats.weights)
        method_final_bytes = communication_counts(
            PFEDVAR_ROUNDS, shared_budget, schedule
        )["cumulative_uplink_bytes"]
        for history_row in fit.history:
            iteration = int(history_row["iteration"])
            communication = communication_counts(
                iteration, shared_budget, schedule
            )
            rows.append(
                {
                    "seed": seed,
                    "N": N_CLIENTS,
                    "T": T,
                    "p": P,
                    "rank": RANK,
                    "method": method,
                    "iteration": iteration,
                    "relative_error": history_row["relative_error"],
                    "data_loss": history_row["data_loss"],
                    "relative_change": history_row["relative_change"],
                    "L_relative_change": history_row["L_relative_change"],
                    "R_relative_change": history_row["R_relative_change"],
                    "S0_relative_change": history_row["S0_relative_change"],
                    "S_local_relative_change": history_row[
                        "S_local_relative_change"
                    ],
                    "optimizer_elapsed_seconds": history_row["elapsed_time"],
                    "optimizer_runtime_seconds": fit.runtime,
                    "initializer_runtime_seconds": initialization_runtime,
                    "total_runtime_seconds": initialization_runtime + fit.runtime,
                    "shared_upload_budget": (
                        0 if shared_budget is None else shared_budget
                    ),
                    "refresh_policy": policy,
                    "refresh_count": len(schedule),
                    "refresh_this_update": (
                        iteration > 0 and iteration - 1 in schedule
                    ),
                    "candidate_budget_per_client": CANDIDATE_BUDGET,
                    **communication,
                    "cumulative_uplink_mib": (
                        communication["cumulative_uplink_bytes"] / 2**20
                    ),
                    "final_uplink_reduction_vs_exact": (
                        exact_final_bytes / method_final_bytes
                    ),
                    "max_true_spectral_radius": max_radius,
                    "truth_incoherence_mu": mu,
                    **(
                        metrics
                        if iteration == PFEDVAR_ROUNDS
                        else {
                            key: "" if isinstance(value, str) else np.nan
                            for key, value in metrics.items()
                        }
                    ),
                }
            )
    return rows


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
        path = output_dir / f"seed_{seed}.csv"
        write_rows(path, rows)
        print(f"seed={seed}: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
