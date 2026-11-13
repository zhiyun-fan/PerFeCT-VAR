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


OUTPUT_DIR = HERE.parents[1] / "reproduced/simulation/simulation2_vary_m_het/vary_het_rank1_outputs"
REPLICATIONS = 4
LEVELS = ("zero", "low", "medium", "high")
MULTIPLICITY = {"zero": 0, "low": 5, "medium": 10, "high": 15}

N_CLIENTS = 30
PAIR_COUNT = N_CLIENTS // 2
T = 800
P = 40
RANK = 1

M_STAR_TRUE = 2
M_FIT = 3
M_DAGGER = 2 * M_FIT + M_STAR_TRUE

SINGULAR_VALUES = np.array([0.3])
S0_TRUE = 18
S_LOCAL_TRUE = 18
SHARED_DEGREE_TRUE = 2
S0_MAGNITUDE_RANGE = (0.03, 0.07)
LOCAL_SINGULAR_VALUE = 0.4
A_OPERATOR_CAP = 0.6

COVARIANCE_CENTER = 1.0
HETEROGENEITY_AMPLITUDE = 0.45
PHI_MIN = COVARIANCE_CENTER - HETEROGENEITY_AMPLITUDE
PHI_MAX = COVARIANCE_CENTER + HETEROGENEITY_AMPLITUDE
TAU_ANCHOR = P - 1
LOW_CURVATURE_ANCHOR = P - 2

S0_FIT = 20
S_LOCAL_FIT = 20
GAMMA_FIT = 3 / P

BASU_PENALTY_MULTIPLIER = 0.5

INITIALIZER_ROUNDS = 100
PFEDVAR_ROUNDS = 200
STEP_SIZE = 0.1
PERSONAL_STEP_SIZE = 0.3



def generate_truth(
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], float]:
    for _ in range(1_000):
        predictor_order = rng.permutation(P - 2)
        v_support = predictor_order[:8]
        leaves = predictor_order[8 : 8 + PAIR_COUNT]
        u_support = rng.choice(P - 1, size=S_LOCAL_TRUE, replace=False)

        U = np.zeros((P, RANK))
        U[u_support, 0] = 1 / np.sqrt(len(u_support))
        V = np.zeros((P, RANK))
        V[v_support, 0] = 1 / np.sqrt(len(v_support))
        u1, v1 = U[:, 0], V[:, 0]
        B = U @ np.diag(SINGULAR_VALUES) @ V.T

        shared_support: set[tuple[int, int]] = set()
        row_counts = np.zeros(P, dtype=int)
        column_counts = np.zeros(P, dtype=int)
        while len(shared_support) < S0_TRUE:
            row = int(rng.integers(P - 1))
            column = int(rng.integers(P))
            coordinate = (row, column)
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

        paired_clients = rng.permutation(N_CLIENTS).reshape(PAIR_COUNT, 2)
        S_local = np.zeros((N_CLIENTS, P, P))
        for pair_index, (plus_client, minus_client) in enumerate(
            paired_clients
        ):
            local_direction = np.outer(u1, np.eye(P)[leaves[pair_index]])
            S_local[plus_client] = LOCAL_SINGULAR_VALUE * local_direction
            S_local[minus_client] = -LOCAL_SINGULAR_VALUE * local_direction

        A_clients = B[None, :, :] + S0[None, :, :] + S_local
        max_operator_norm = float(
            np.linalg.svd(A_clients, compute_uv=False)[:, 0].max()
        )
        if max_operator_norm > A_OPERATOR_CAP:
            continue

        frequency = np.count_nonzero(S_local, axis=0)
        if frequency.max() != M_STAR_TRUE:
            raise RuntimeError("Local support frequency is not m_star=2")
        return {
            "A_clients": A_clients,
            "B": B,
            "S0": S0,
            "S_local": S_local,
            "u1": u1,
            "v1": v1,
            "leaves": leaves,
            "paired_clients": paired_clients,
        }, max_operator_norm

    raise RuntimeError("Could not generate a truth below the operator cap")


def covariance_directions(
    level: str,
    v1: np.ndarray,
    leaves: np.ndarray,
) -> np.ndarray:
    directions = np.zeros((PAIR_COUNT, P, P))
    multiplicity = MULTIPLICITY[level]
    if multiplicity == 0:
        return directions

    leaf_vectors = np.eye(P)[leaves]
    for pair_index in range(PAIR_COUNT):
        z = leaf_vectors[pair_index]
        if pair_index < multiplicity:
            b = v1
        else:
            remainder_size = PAIR_COUNT - multiplicity
            previous = multiplicity + (
                (pair_index - multiplicity - 1) % remainder_size
            )
            b = leaf_vectors[previous]
        directions[pair_index] = np.outer(b, z) + np.outer(z, b)
    return directions


def generate_experiment(
    rng: np.random.Generator,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, float | int | str]],
]:
    truth, max_operator_norm = generate_truth(rng)
    base_gamma = np.eye(P)
    base_gamma[TAU_ANCHOR, TAU_ANCHOR] = PHI_MAX
    base_gamma[LOW_CURVATURE_ANCHOR, LOW_CURVATURE_ANCHOR] = PHI_MIN

    initial_slots = rng.normal(size=(N_CLIENTS, P))
    innovation_slots = rng.normal(size=(N_CLIENTS, T, P))
    values_by_level: dict[str, np.ndarray] = {}
    gammas_by_level: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float | int | str]] = {}

    for level in LEVELS:
        directions = covariance_directions(
            level, truth["v1"], truth["leaves"]
        )
        gammas = np.broadcast_to(base_gamma, (N_CLIENTS, P, P)).copy()
        for direction, (plus_client, minus_client) in zip(
            directions, truth["paired_clients"]
        ):
            gammas[plus_client] += HETEROGENEITY_AMPLITUDE * direction
            gammas[minus_client] -= HETEROGENEITY_AMPLITUDE * direction

        innovations = np.array(
            [
                gamma - A @ gamma @ A.T
                for gamma, A in zip(gammas, truth["A_clients"])
            ]
        )
        innovations = 0.5 * (
            innovations + np.swapaxes(innovations, 1, 2)
        )
        innovation_eigenvalues = np.linalg.eigvalsh(innovations)
        if innovation_eigenvalues.min() <= 0.0:
            raise RuntimeError(f"Innovation covariance is not SPD at {level}")

        gamma_cholesky = np.linalg.cholesky(gammas)
        innovation_cholesky = np.linalg.cholesky(innovations)
        values = np.empty((N_CLIENTS, T + 1, P))
        values[:, 0] = (gamma_cholesky @ initial_slots[:, :, None])[:, :, 0]
        shocks = (
            innovation_cholesky[:, None, :, :]
            @ innovation_slots[:, :, :, None]
        )
        shocks = shocks[:, :, :, 0]
        for time_index in range(1, T + 1):
            values[:, time_index] = (
                truth["A_clients"]
                @ values[:, time_index - 1, :, None]
            )[:, :, 0] + shocks[:, time_index - 1]

        gamma_bar = gammas.mean(axis=0)
        centered = gammas - gamma_bar
        stack = sum(E.T @ E for E in centered)
        lambda_exact = float(
            np.sqrt(np.linalg.eigvalsh(stack).max() / M_DAGGER)
        )
        expected_lambda = HETEROGENEITY_AMPLITUDE * np.sqrt(
            2.0 * MULTIPLICITY[level] / M_DAGGER
        )
        if not np.isclose(lambda_exact, expected_lambda, atol=1e-10):
            raise RuntimeError(f"Lambda audit failed at {level}")

        truth_leakage = np.mean(truth["S_local"] @ centered, axis=0)
        B_direction = np.outer(truth["u1"], truth["v1"])
        leakage_projection = float(np.sum(truth_leakage * B_direction))
        leakage_norm = float(np.linalg.norm(truth_leakage))
        gamma_eigenvalues = np.linalg.eigvalsh(gammas)
        lyapunov_residual = max(
            np.linalg.norm(gamma - A @ gamma @ A.T - Q)
            for gamma, A, Q in zip(
                gammas, truth["A_clients"], innovations
            )
        )

        values_by_level[level] = values
        gammas_by_level[level] = gammas
        diagnostics[level] = {
            "heterogeneity_level": level,
            "aligned_pair_count": MULTIPLICITY[level],
            "m_star_true": M_STAR_TRUE,
            "m_fit": M_FIT,
            "lambda_LR_exact": lambda_exact,
            "phi_min": float(gamma_eigenvalues.min()),
            "phi_max": float(gamma_eigenvalues.max()),
            "E_operator_norm_max": float(
                max(np.linalg.norm(E, 2) for E in centered)
            ),
            "E_frobenius_rms": float(
                np.sqrt(np.mean(np.sum(centered**2, axis=(1, 2))))
            ),
            "innovation_min_eigenvalue": float(
                innovation_eigenvalues.min()
            ),
            "innovation_max_eigenvalue": float(
                innovation_eigenvalues.max()
            ),
            "lyapunov_residual": float(lyapunov_residual),
            "max_true_operator_norm": max_operator_norm,
            "truth_heterogeneity_leakage_fro": leakage_norm,
            "truth_heterogeneity_B_projection": leakage_projection,
            "truth_heterogeneity_B_projection_abs": abs(
                leakage_projection
            ),
            "truth_heterogeneity_B_alignment_cosine": (
                abs(leakage_projection) / leakage_norm
                if leakage_norm > 0.0
                else 0.0
            ),
        }

    return truth, values_by_level, gammas_by_level, diagnostics


def basu_penalties(
    stats: pfedvar.SufficientStats,
) -> dict[str, float | np.ndarray]:
    """Compute initializer penalties from this level's clientwise spectra."""
    n_clients = stats.n_clients
    p = stats.Gxx.shape[-1]
    local_phi = np.array([np.linalg.norm(Gxx, 2) for Gxx in stats.Gxx])
    initializer_nuclear_factor = p + np.log(n_clients)
    initializer_sparse_factor = np.log(p * n_clients)
    return {
        "initializer_phi_hat": local_phi,
        "initializer_lambda_B": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * initializer_nuclear_factor / stats.sample_sizes),
        "initializer_lambda_C": BASU_PENALTY_MULTIPLIER
        * np.sqrt(local_phi * initializer_sparse_factor / stats.sample_sizes),
    }


def fit_methods(
    stats: pfedvar.SufficientStats,
    *,
    initializer_rounds: int = INITIALIZER_ROUNDS,
    pfedvar_rounds: int = PFEDVAR_ROUNDS,
) -> tuple[dict[str, pfedvar.FitResult], dict[str, float]]:
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
        freq_cap=M_FIT,
        rounds=pfedvar_rounds,
        eta_B=STEP_SIZE,
        eta_0=STEP_SIZE,
        eta_local=PERSONAL_STEP_SIZE,
    )
    started = perf_counter()
    state = pfedvar.initialize_fedvar(stats, config, initializer)
    fit = pfedvar.fedvar(stats, config, state)
    runtime = initializer.runtime + perf_counter() - started
    return {"pfedvar": fit}, {"pfedvar": runtime}


def squared_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    client_errors = np.sum((estimate - truth) ** 2, axis=(-2, -1))
    return float(
        weights @ client_errors if weights is not None else np.sum(client_errors)
    )


def relative_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    return float(
        np.sqrt(
            squared_error(estimate, truth, weights)
            / squared_error(np.zeros_like(truth), truth, weights)
        )
    )


def support_f1(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_support = estimate != 0.0
    truth_support = truth != 0.0
    true_positive = np.count_nonzero(estimate_support & truth_support)
    denominator = np.count_nonzero(estimate_support) + np.count_nonzero(
        truth_support
    )
    return 2.0 * true_positive / denominator if denominator else 1.0


def metrics(
    method: str,
    fit: pfedvar.FitResult,
    truth: dict[str, np.ndarray],
    gammas: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | str]:
    A_client_relative_errors = np.linalg.norm(
        fit.A_clients - truth["A_clients"], axis=(1, 2)
    ) / np.linalg.norm(truth["A_clients"], axis=(1, 2))
    output: dict[str, float | str] = {
        "A_squared_error": squared_error(
            fit.A_clients, truth["A_clients"], weights
        ),
        "A_relative_error": relative_error(
            fit.A_clients, truth["A_clients"], weights
        ),
        "A_client_relative_errors": json.dumps(
            np.round(A_client_relative_errors, 6).tolist()
        ),
        "B_squared_error": np.nan,
        "S0_squared_error": np.nan,
        "S_local_squared_error": np.nan,
        "B_relative_error": np.nan,
        "S0_relative_error": np.nan,
        "S_local_relative_error": np.nan,
        "S0_support_f1": np.nan,
        "S_local_support_f1": np.nan,
        "estimated_local_frequency_max": np.nan,
        "B_error_leading_direction_projection": np.nan,
        "B_error_leading_direction_projection_abs": np.nan,
        "realized_heterogeneity_B_projection": np.nan,
        "realized_heterogeneity_B_projection_abs": np.nan,
    }

    if method == "pfedvar":
        B_hat = fit.components["B"]
        S0_hat = fit.components["S0"]
        S_local_hat = fit.components["S_local"]
        output.update(
            {
                "B_squared_error": squared_error(B_hat, truth["B"]),
                "S0_squared_error": squared_error(S0_hat, truth["S0"]),
                "S_local_squared_error": squared_error(
                    S_local_hat, truth["S_local"], weights
                ),
                "B_relative_error": relative_error(B_hat, truth["B"]),
                "S0_relative_error": relative_error(S0_hat, truth["S0"]),
                "S_local_relative_error": relative_error(
                    S_local_hat, truth["S_local"], weights
                ),
                "S0_support_f1": support_f1(S0_hat, truth["S0"]),
                "S_local_support_f1": support_f1(
                    S_local_hat, truth["S_local"]
                ),
                "estimated_local_frequency_max": float(
                    np.count_nonzero(S_local_hat, axis=0).max()
                ),
            }
        )
        B_direction = np.outer(truth["u1"], truth["v1"])
        B_error = B_hat - truth["B"]
        centered = gammas - gammas.mean(axis=0)
        local_error = S_local_hat - truth["S_local"]
        leakage = np.sum(
            weights[:, None, None] * (local_error @ centered), axis=0
        )
        output["B_error_leading_direction_projection"] = float(
            np.sum(B_error * B_direction)
        )
        output["realized_heterogeneity_B_projection"] = float(
            np.sum(leakage * B_direction)
        )
        output["B_error_leading_direction_projection_abs"] = abs(
            output["B_error_leading_direction_projection"]
        )
        output["realized_heterogeneity_B_projection_abs"] = abs(
            output["realized_heterogeneity_B_projection"]
        )
    return output


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
    truth, values, gammas, diagnostics = generate_experiment(rng)
    stats_by_level = {
        level: pfedvar.build_statistics(
            [
                pfedvar.ClientData(str(client), values[level][client])
                for client in range(N_CLIENTS)
            ]
        )
        for level in LEVELS
    }
    rows: list[dict[str, object]] = []

    for level in LEVELS:
        started = perf_counter()
        stats = stats_by_level[level]
        fits, runtimes = fit_methods(
            stats,
            initializer_rounds=initializer_rounds,
            pfedvar_rounds=pfedvar_rounds,
        )
        pfedvar_metrics = metrics(
            "pfedvar", fits["pfedvar"], truth, gammas[level], stats.weights
        )
        rows.append(
            {
                "seed": seed,
                "N": N_CLIENTS,
                "T": T,
                "p": P,
                "rank": RANK,
                "B_singular_values": json.dumps(SINGULAR_VALUES.tolist()),
                "S0_magnitude_min": S0_MAGNITUDE_RANGE[0],
                "S0_magnitude_max": S0_MAGNITUDE_RANGE[1],
                "local_singular_value": LOCAL_SINGULAR_VALUE,
                "method": "pfedvar",
                "runtime_seconds": runtimes["pfedvar"],
                **diagnostics[level],
                **pfedvar_metrics,
            }
        )
        print(
            f"seed={seed} level={level:6s} "
            f"lambda={diagnostics[level]['lambda_LR_exact']:.3f} "
            f"pFedVAR={pfedvar_metrics['A_relative_error']:.4f} "
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
