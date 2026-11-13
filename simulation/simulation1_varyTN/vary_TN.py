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

import baselines
import pfedvar


OUTPUT_DIR = HERE.parents[1] / "reproduced/simulation/simulation1_varyTN/vary_TN_outputs"
REPLICATIONS = 4
T_GRID = (100, 200, 400, 800, 1200, 1600, 2000)
N_GRID = (10, 20, 30, 40, 50, 60)
N_FIXED = 30
T_FIXED = 800

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

BASU_PENALTY_MULTIPLIER = 0.5
DITTO_LAMBDA = 1.5

INITIALIZER_ROUNDS = 100
LOCAL_LRS_ROUNDS = 200
POOLED_LRS_ROUNDS = 600
PFEDVAR_ROUNDS = 200
STEP_SIZE = 0.1
PERSONAL_STEP_SIZE = 0.3

METHODS_BY_EXPERIMENT = {
    "T": ("pfedvar", "local_lrs", "pooled_lrs", "ditto"),
    "N": ("pfedvar",),
}


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


def generate_truths(
    rng: np.random.Generator,
    client_counts: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Generate L, R, S0 and Si, keeping L, R and S0 common across N."""

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
        S0[rows, columns] = rng.choice((-1.0, 1.0), size=S0_TRUE) * rng.uniform(
            *S0_MAGNITUDE_RANGE, size=S0_TRUE
        )

        truths: dict[int, dict[str, np.ndarray]] = {}
        # generate Si
        for n_clients in client_counts:
            coordinates, incidence = allocate_local_support(
                rng,
                n_clients,
                S_LOCAL_TRUE,
                M_STAR_TRUE,
                np.arange(P * P),
            )
            signs = rng.choice((-1.0, 1.0), size=(n_clients, S_LOCAL_TRUE))
            magnitudes = rng.uniform(*S_LOCAL_MAGNITUDE_RANGE, size=(n_clients, S_LOCAL_TRUE))

            S_local = np.zeros((n_clients, P, P))
            for client in range(n_clients):
                client_coordinates = coordinates[incidence[client]]
                S_local[client].ravel()[client_coordinates] = (
                    signs[client] * magnitudes[client]
                )

            A_clients = B[None, :, :] + S0[None, :, :] + S_local
            if pfedvar.spectral_radii(A_clients).max() >= STABILITY_LIMIT:
                break
            truths[n_clients] = {
                "A_clients": A_clients,
                "L": L,
                "R": R,
                "B": B,
                "S0": S0,
                "S_local": S_local,
            }

        if len(truths) == len(client_counts):
            return truths
    raise RuntimeError("Could not generate a stable truth family")


def simulate_values(
    truth: dict[str, np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    A_clients = truth["A_clients"]
    n_clients = A_clients.shape[0]
    shocks = rng.normal(size=(n_clients, BURN_IN + sample_size + 1, P))
    values = np.zeros_like(shocks)
    values[:, 0] = shocks[:, 0]
    for time_index in range(1, values.shape[1]):
        values[:, time_index] = (
            (A_clients @ values[:, time_index - 1, :, None])[:, :, 0]
            + shocks[:, time_index]
        )
    return values[:, -(sample_size + 1) :]


def basu_penalties(
    stats: pfedvar.SufficientStats,
) -> dict[str, float | np.ndarray]:
    """Initialization and Local use clientwise spectra; Pooled uses its own.

    Local/initialization denominators are T_i; the pooled denominator is
    sum_i T_i. Initialization uses p + log(N) and log(p * N);
    Local and Pooled use p and log(p).
    """
    n_clients = stats.n_clients
    p = stats.Gxx.shape[-1]
    local_phi = np.array([np.linalg.norm(Gxx, 2) for Gxx in stats.Gxx])
    pooled_Gxx, _, _ = pfedvar.pooled_statistics(stats)
    pooled_phi = float(np.linalg.norm(pooled_Gxx, 2))
    total_sample_size = float(stats.sample_sizes.sum())
    initializer_nuclear_factor = p + np.log(n_clients)
    initializer_sparse_factor = np.log(p * n_clients)
    baseline_nuclear_factor = p
    baseline_sparse_factor = np.log(p)
    return {
        "initializer_phi_hat": local_phi,
        "local_phi_max_hat": local_phi,
        "pooled_phi_max_hat": pooled_phi,
        "initializer_lambda_B": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * initializer_nuclear_factor / stats.sample_sizes),
        "initializer_lambda_C": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * initializer_sparse_factor / stats.sample_sizes),
        "local_lambda_B": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * baseline_nuclear_factor / stats.sample_sizes),
        "local_lambda_C": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * baseline_sparse_factor / stats.sample_sizes),
        "pooled_lambda_B": float(
            BASU_PENALTY_MULTIPLIER
            * np.sqrt(pooled_phi * baseline_nuclear_factor / total_sample_size)
        ),
        "pooled_lambda_C": float(
            BASU_PENALTY_MULTIPLIER
            * np.sqrt(pooled_phi * baseline_sparse_factor / total_sample_size)
        ),
    }


def fit_methods(
    clients: list[pfedvar.ClientData],
    truth_A: np.ndarray,
    *,
    methods: tuple[str, ...],
    initializer_rounds: int,
    local_lrs_rounds: int,
    pooled_lrs_rounds: int,
    pfedvar_rounds: int,
) -> tuple[
    pfedvar.SufficientStats,
    dict[str, pfedvar.FitResult],
    dict[str, float],
]:
    stats = pfedvar.build_statistics(clients)
    n_clients = stats.n_clients
    penalties = basu_penalties(stats)
    initializer_eta = penalties["initializer_phi_hat"]
    local_eta = penalties["local_phi_max_hat"]
    pooled_eta = penalties["pooled_phi_max_hat"]

    # This initializer has different penalties from the Local LRS comparison.
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
    fits: dict[str, pfedvar.FitResult] = {}
    runtimes: dict[str, float] = {}
    if "local_lrs" in methods:
        local_fit = pfedvar.local_basu_lrs(
            stats,
            pfedvar.BasuConfig(
                lambda_B=penalties["local_lambda_B"],
                lambda_C=penalties["local_lambda_C"],
                nu=P,
                max_iter=local_lrs_rounds,
                eta_init=local_eta,
                eta_min=local_eta / 10,
            ),
        )
        fits["local_lrs"] = local_fit
        runtimes["local_lrs"] = local_fit.runtime
    if "pooled_lrs" in methods:
        pooled_fit = pfedvar.pooled_basu_lrs(
            stats,
            pfedvar.BasuConfig(
                lambda_B=penalties["pooled_lambda_B"],
                lambda_C=penalties["pooled_lambda_C"],
                nu=P,
                max_iter=pooled_lrs_rounds,
                eta_init=pooled_eta,
                eta_min=pooled_eta / 10,
            ),
        )
        fits["pooled_lrs"] = pooled_fit
        runtimes["pooled_lrs"] = pooled_fit.runtime
    if "ditto" in methods:
        ditto_fit = baselines.ditto(
            stats,
            lambda_personalization=DITTO_LAMBDA,
        )
        fits["ditto"] = ditto_fit
        runtimes["ditto"] = ditto_fit.runtime

    config = pfedvar.FedVARConfig(
        rank=RANK,
        s0=S0_FIT,
        s_local=[S_LOCAL_FIT] * n_clients,
        gamma=GAMMA_FIT,
        freq_cap=FREQUENCY_FIT,
        rounds=pfedvar_rounds,
        eta_B=STEP_SIZE,
        eta_0=STEP_SIZE,
        eta_local=PERSONAL_STEP_SIZE,
    )
    started = perf_counter()
    state = pfedvar.initialize_fedvar(stats, config, initializer)
    fit = pfedvar.fedvar(stats, config, state, truth_A)
    fits["pfedvar"] = fit
    runtimes["pfedvar"] = initializer.runtime + perf_counter() - started
    return stats, fits, runtimes


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
    denominator = np.count_nonzero(estimate_support) + np.count_nonzero(truth_support)
    return 2.0 * true_positive / denominator if denominator else 1.0


def metrics(
    method: str,
    fit: pfedvar.FitResult,
    truth: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float | str]:
    A_client_relative_errors = np.linalg.norm(
        fit.A_clients - truth["A_clients"], axis=(1, 2)
    ) / np.linalg.norm(truth["A_clients"], axis=(1, 2))
    output = {
        "initial_A_relative_error": (
            fit.history[0]["relative_error"]
            if method == "pfedvar"
            else np.nan
        ),
        "A_relative_error": relative_error(
            fit.A_clients, truth["A_clients"], weights
        ),
        "A_client_relative_errors": json.dumps(
            np.round(A_client_relative_errors, 6).tolist()
        ),
        "B_relative_error": np.nan,
        "S0_relative_error": np.nan,
        "S_local_relative_error": np.nan,
        "S0_support_f1": np.nan,
        "S_local_support_f1": np.nan,
        "estimated_local_frequency_max": np.nan,
    }

    if method == "pfedvar":
        B_hat = fit.components["B"]
        S0_hat = fit.components["S0"]
        S_local_hat = fit.components["S_local"]
        output["B_relative_error"] = relative_error(B_hat, truth["B"])
        output["S0_relative_error"] = relative_error(S0_hat, truth["S0"])
        output["S_local_relative_error"] = relative_error(
            S_local_hat, truth["S_local"], weights
        )
        output["S0_support_f1"] = support_f1(S0_hat, truth["S0"])
        output["S_local_support_f1"] = support_f1(
            S_local_hat, truth["S_local"]
        )
        output["estimated_local_frequency_max"] = float(
            np.count_nonzero(S_local_hat, axis=0).max()
        )
    return output


def evaluate(
    experiment: str,
    seed: int,
    rng: np.random.Generator,
    levels: tuple[int, ...],
    truths: dict[int, dict[str, np.ndarray]],
    *,
    initializer_rounds: int,
    local_lrs_rounds: int,
    pooled_lrs_rounds: int,
    pfedvar_rounds: int,
) -> list[dict[str, object]]:
    methods = METHODS_BY_EXPERIMENT[experiment]
    rows: list[dict[str, object]] = []
    if experiment == "T":
        truth = truths[N_FIXED]
        full_values = simulate_values(truth, max(levels), rng)

    for level in levels:
        started = perf_counter()
        if experiment == "T":
            n_clients, sample_size = N_FIXED, level
            truth = truths[n_clients]
            values = full_values
        else:
            n_clients, sample_size = level, T_FIXED
            truth = truths[n_clients]
            values = simulate_values(truth, sample_size, rng)
        clients = [
            pfedvar.ClientData(
                str(index), values[index, -(sample_size + 1) :]
            )
            for index in range(n_clients)
        ]

        stats, fits, runtimes = fit_methods(
            clients,
            truth["A_clients"],
            methods=methods,
            initializer_rounds=initializer_rounds,
            local_lrs_rounds=local_lrs_rounds,
            pooled_lrs_rounds=pooled_lrs_rounds,
            pfedvar_rounds=pfedvar_rounds,
        )
        max_radius = float(pfedvar.spectral_radii(truth["A_clients"]).max())
        method_metrics = {
            method: metrics(method, fits[method], truth, stats.weights)
            for method in methods
        }
        for method in methods:
            rows.append(
                {
                    "experiment": f"vary_{experiment}",
                    "seed": seed,
                    "N": n_clients,
                    "T": sample_size,
                    "p": P,
                    "rank": RANK,
                    "method": method,
                    "runtime_seconds": runtimes[method],
                    "max_true_spectral_radius": max_radius,
                    **method_metrics[method],
                }
            )
        print(
            f"seed={seed} {experiment}={level} "
            f"pFedVAR={method_metrics['pfedvar']['A_relative_error']:.4f} "
            f"elapsed={perf_counter() - started:.1f}s",
            flush=True,
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
    targets = [output_dir / f"vary_{experiment}_seed_{seed}.csv"
               for seed in seeds for experiment in ("T", "N")]
    existing = [path for path in targets if path.exists()]
    if existing:
        parser.error(f"output already exists: {existing[0]}; use a new --output-dir")

    required_clients = tuple(sorted(set(N_GRID) | {N_FIXED}))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        truths = generate_truths(rng, required_clients)

        T_rows = evaluate(
            "T",
            seed,
            rng,
            T_GRID,
            truths,
            initializer_rounds=INITIALIZER_ROUNDS,
            local_lrs_rounds=LOCAL_LRS_ROUNDS,
            pooled_lrs_rounds=POOLED_LRS_ROUNDS,
            pfedvar_rounds=PFEDVAR_ROUNDS,
        )
        write_rows(output_dir / f"vary_T_seed_{seed}.csv", T_rows)

        N_rows = evaluate(
            "N",
            seed,
            rng,
            N_GRID,
            truths,
            initializer_rounds=INITIALIZER_ROUNDS,
            local_lrs_rounds=LOCAL_LRS_ROUNDS,
            pooled_lrs_rounds=POOLED_LRS_ROUNDS,
            pfedvar_rounds=PFEDVAR_ROUNDS,
        )
        write_rows(output_dir / f"vary_N_seed_{seed}.csv", N_rows)


if __name__ == "__main__":
    main()
