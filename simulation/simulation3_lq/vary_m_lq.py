from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "methods"))
import pfedvar

OUTPUT_DIR = HERE / "vary_m_lq_outputs"
M_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14)
P = 40
T = 400
T_STREAM = 1600  # Preserve the innovation layout of the selected exploration.
RANK = 1
BURN_IN = 200
STABILITY_LIMIT = 0.8
B_SINGULAR_VALUE = 0.6
B_BLOCK_WIDTH = 15
S0_TRUE = 12
S_LOCAL_TRUE = 60
SHARED_DEGREE_TRUE = 2
S0_MAGNITUDE_RANGE = (0.1, 0.2)
Q = 0.25
A_MAX = 0.4
DECAY_OFFSET = 2.5

S0_FIT = 16
S_LOCAL_FIT = 75
GAMMA_FIT = 3 / P
BASU_PENALTY_MULTIPLIER = 0.5
INITIALIZER_ROUNDS = 100
PFEDVAR_ROUNDS = 200
STEP_SIZE = 0.1
PERSONAL_STEP_SIZE = 0.3


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


def magnitude_profile(degree: int) -> np.ndarray:
    ranks = np.arange(1, degree + 1, dtype=float)
    return A_MAX / (1 + ((ranks - 1) / DECAY_OFFSET) ** (1 / Q))


def generate_truths(seed: int) -> tuple[dict, dict]:
    """Sample Si from all 225 block coordinates, including diagonal entries."""
    direction = np.zeros(P)
    direction[:B_BLOCK_WIDTH] = 1 / np.sqrt(B_BLOCK_WIDTH)
    B = B_SINGULAR_VALUE * np.outer(direction, direction)
    rng0 = np.random.default_rng([seed, 101, 20261109])
    support = set()
    row_counts = np.zeros(P, dtype=int)
    column_counts = np.zeros(P, dtype=int)
    while len(support) < S0_TRUE:
        row, column = rng0.integers(P, size=2)
        if ((row, column) not in support and row_counts[row] < SHARED_DEGREE_TRUE
                and column_counts[column] < SHARED_DEGREE_TRUE):
            support.add((row, column))
            row_counts[row] += 1
            column_counts[column] += 1
    rows, columns = np.array(sorted(support)).T
    S0 = np.zeros((P, P))
    S0[rows, columns] = rng0.choice((-1., 1.), S0_TRUE) * rng0.uniform(
        *S0_MAGNITUDE_RANGE, size=S0_TRUE)
    rng = np.random.default_rng([seed, 103, 20261109])
    pool = np.array([row*P+column for row in range(B_BLOCK_WIDTH)
                     for column in range(B_BLOCK_WIDTH)])
    n_clients, m_star = 30, 14
    for attempt in range(1, 10001):
        coordinates, incidence = allocate_local_support(rng, n_clients, S_LOCAL_TRUE, m_star, pool)
        # Retain this unused draw to reproduce the earlier exploration bitwise.
        rng.choice((-1., 1.), P)
        S_local = np.zeros((n_clients, P, P))
        for index, coordinate in enumerate(coordinates):
            clients = rng.permutation(np.flatnonzero(incidence[:, index]))
            sign = rng.choice((-1., 1.))
            S_local.reshape(n_clients, -1)[clients, coordinate] = sign * magnitude_profile(len(clients))
        A_clients = B + S0 + S_local
        max_radius = float(pfedvar.spectral_radii(A_clients).max())
        if max_radius < STABILITY_LIMIT:
            break
    else:
        raise RuntimeError('No stable truth in 10000 attempts')
    frequency = np.count_nonzero(S_local, axis=0)
    assert np.all(np.count_nonzero(S_local, axis=(1, 2)) == S_LOCAL_TRUE)
    assert sorted(frequency[frequency > 0]) == [8] + [14] * 128
    assert np.all(np.abs(np.sign(S_local).sum(axis=0)) == frequency)
    magnitudes = np.sort(np.abs(S_local), axis=0)[::-1]
    truth = dict(B=B, S0=S0, S_local=S_local, A_clients=A_clients)
    diagnostic = dict(m_star=m_star, q=Q, a_max=A_MAX,
        m_q_star=float((np.abs(S_local)**Q).sum(axis=0).max()),
        truth_attempts=attempt, max_true_spectral_radius=max_radius,
        mean_client_local_squared_norm=float(np.sum(S_local**2)/n_clients),
        personal_union_size=int(np.count_nonzero(frequency)),
        top_m_tail_energies={m:float(np.sum(magnitudes[m:]**2)/n_clients) for m in range(m_star+1)})
    return {n_clients:truth}, {n_clients:diagnostic}


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


def fit_initializer(
    clients: list[pfedvar.ClientData],
    *,
    initializer_rounds: int,
) -> tuple[
    pfedvar.SufficientStats,
    pfedvar.FitResult,
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
    return stats, initializer


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


def data_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_seed(seed, m_grid, rounds, output_dir):
    truths, diagnostics = generate_truths(seed)
    truth, diagnostic = truths[30], diagnostics[30]
    rng = np.random.default_rng([seed, 30, 20261110])
    values = simulate_values(truth, T_STREAM, rng)[:, :T + 1].copy()
    data_hash = data_digest(truth['A_clients'], values)
    stats, initializer = fit_initializer(
        [pfedvar.ClientData(str(i), values[i]) for i in range(30)],
        initializer_rounds=INITIALIZER_ROUNDS,
    )
    print(f'seed={seed} required Basu initializer completed', flush=True)
    rows = []
    for m in m_grid:
        config = pfedvar.FedVARConfig(
            rank=RANK, s0=S0_FIT, s_local=[S_LOCAL_FIT]*30,
            gamma=GAMMA_FIT, freq_cap=m, rounds=rounds,
            eta_B=STEP_SIZE, eta_0=STEP_SIZE, eta_local=PERSONAL_STEP_SIZE,
        )
        started = perf_counter()
        state = pfedvar.initialize_fedvar(stats, config, initializer)
        fit = pfedvar.fedvar(stats, config, state, truth['A_clients'])
        if not np.isfinite(fit.A_clients).all():
            raise RuntimeError(f'Nonfinite estimate: seed={seed}, m={m}')
        row = dict(
            experiment='vary_m_lq', seed=seed, N=30, T=T, p=P, rank=RANK,
            method='pfedvar', m_fit=m, actual_frequency_cap=m,
            runtime_seconds=initializer.runtime+perf_counter()-started,
            shared_initializer_runtime_seconds=initializer.runtime,
            data_sha256=data_hash,
            q=Q, B_singular_value=B_SINGULAR_VALUE,
            personal_support_pool='full_block_15x15',
            s0_true=S0_TRUE, s_local_true=S_LOCAL_TRUE,
            s0_fit=S0_FIT, s_local_fit=S_LOCAL_FIT, gamma=GAMMA_FIT,
            initializer_rounds=INITIALIZER_ROUNDS, FedVAR_rounds=rounds,
            eta_B=STEP_SIZE, eta_0=STEP_SIZE, eta_local=PERSONAL_STEP_SIZE,
            m_star=diagnostic['m_star'], m_q_star=diagnostic['m_q_star'],
            max_true_spectral_radius=diagnostic['max_true_spectral_radius'],
            truth_attempts=diagnostic['truth_attempts'],
            initial_A_relative_error=fit.history[0]['relative_error'],
            A_relative_error=relative_error(fit.A_clients, truth['A_clients'], stats.weights),
            top_m_tail_energy=diagnostic['top_m_tail_energies'][m],
            S0_support_f1=support_f1(fit.components['S0'], truth['S0']),
            S_local_support_f1=support_f1(fit.components['S_local'], truth['S_local']),
            estimated_local_frequency_max=int(np.count_nonzero(fit.components['S_local'], axis=0).max()),
        )
        for key, initial in [('B',state.L@state.R.T),('S0',state.S0),('S_local',state.S_local)]:
            weights = stats.weights if key == 'S_local' else None
            row[key+'_relative_error'] = relative_error(fit.components[key], truth[key], weights)
            row['initial_'+key+'_relative_error'] = relative_error(initial, truth[key], weights)
        rows.append(row)
        # Keep completed caps if a later cap is interrupted; one CSV per seed.
        write_rows(output_dir/f'seed_{seed}.csv', rows)
        print(f"seed={seed} m={m:2d} FedVAR={row['A_relative_error']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int)
    parser.add_argument('--m-grid', type=int, nargs='+', default=M_GRID)
    parser.add_argument('--pfedvar-rounds', type=int, default=PFEDVAR_ROUNDS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.seed < 0 or args.pfedvar_rounds < 1:
        parser.error('seed must be nonnegative and rounds must be positive')
    if (not args.m_grid or len(set(args.m_grid)) != len(args.m_grid)
            or any(m < 1 or m >= 15 for m in args.m_grid)):
        parser.error('m values must be distinct integers between 1 and 14')
    target = args.output_dir/f'seed_{args.seed}.csv'
    if target.exists() and not args.overwrite:
        parser.error(f'Existing output: {target}; use --overwrite to rerun this seed')
    run_seed(args.seed, args.m_grid, args.pfedvar_rounds, args.output_dir)


if __name__ == '__main__':
    main()
