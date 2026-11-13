from __future__ import annotations
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Mapping, Sequence
from warnings import warn
import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


# ---------------------------------------------------------------------------
# Data, configuration, state, and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientData:
    """One client's ordered VAR(1) trajectory."""

    client_id: str
    values: np.ndarray

    @property
    def Ti(self) -> int:
        return self.values.shape[0] - 1


@dataclass(frozen=True, slots=True, kw_only=True)
class BasuConfig:
    """AFNSL configuration for the low-rank-plus-sparse program.

    ``lambda_B`` and ``lambda_C`` must be explicit length-N sequences for the
    local estimator and scalars for the pooled estimator. ``nu`` is the spikiness radius.
    For local fits, eta_init and eta_min may be scalars or length-N
    sequences; pooled fits require scalars. ``tol`` stops on the combined
    aggregate B/C change divided by max(stop_norm_floor, the previous
    aggregate norm). The default floor 1.0 retains the historical rule;
    use 0.0 for pure relative change. At zero previous norm, a nonzero
    change is undefined and cannot trigger stopping; zero change can stop.
    Set tol to None to run the full outer iteration budget.
    """

    lambda_B: float | Sequence[float]
    lambda_C: float | Sequence[float]
    nu: float
    max_iter: int = 100
    tol: float | None = 2e-4
    stop_norm_floor: float = 1.0
    eta_min: float | Sequence[float] = 1e-7
    eta_init: float | Sequence[float] = 1.0
    sigma: float = 2.0
    line_search_C: float = 0.0
    max_backtracking: int = 50
    prox_max_iter: int = 5_000
    prox_tol: float = 1e-5


@dataclass(frozen=True, slots=True, kw_only=True)
class FedVARConfig:
    """Hyperparameters used only by FedVAR and its initializer.

    ``lossless_screening`` enables top-L_i candidate screening for FCHT in
    both ``initialize_fedvar`` and ``fedvar``. It is disabled by default.
    """

    rank: int
    s0: int
    s_local: Sequence[int]
    gamma: float
    freq_cap: int
    rounds: int
    eta_B: float
    eta_0: float
    eta_local: float
    lossless_screening: bool = False


@dataclass(frozen=True, slots=True)
class SufficientStats:
    """Clientwise full-batch sufficient statistics."""

    Gxx: np.ndarray
    Gyx: np.ndarray
    gyy: np.ndarray
    sample_sizes: np.ndarray
    weights: np.ndarray
    client_ids: tuple[str, ...]

    @property
    def n_clients(self) -> int:
        return self.Gxx.shape[0]


@dataclass(slots=True)
class FedVARState:
    L: np.ndarray
    R: np.ndarray
    S0: np.ndarray
    S_local: np.ndarray
    iteration: int = 0

    def A_clients(self) -> np.ndarray:
        return (self.L @ self.R.T)[None, :, :] + self.S0[None, :, :] + self.S_local


@dataclass(slots=True)
class FitResult:
    method: str
    client_ids: tuple[str, ...]
    A_clients: np.ndarray
    components: dict[str, np.ndarray]
    history: list[dict[str, Any]]
    runtime: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentResult:
    results: dict[str, FitResult]
    initial_state: FedVARState
    metrics: dict[str, Any]
    sample_sizes: np.ndarray
    weights: np.ndarray


@dataclass(slots=True)
class _BasuSolution:
    B: np.ndarray
    C: np.ndarray
    history: list[dict[str, Any]]
    n_iter: int
    backtracks: int
    converged: bool


# ---------------------------------------------------------------------------
# Data conversion, loss, gradients, and configuration expansion
# ---------------------------------------------------------------------------


def build_statistics(clients: Sequence[ClientData]) -> SufficientStats:
    """Construct clientwise sufficient statistics from ordered trajectories."""

    client_ids = tuple(client.client_id for client in clients)
    sample_sizes = np.array([client.Ti for client in clients])
    weights = sample_sizes / sample_sizes.sum()
    Gxx = np.stack(
        [
            client.values[:-1].T @ client.values[:-1] / client.Ti
            for client in clients
        ]
    )
    Gyx = np.stack(
        [
            client.values[1:].T @ client.values[:-1] / client.Ti
            for client in clients
        ]
    )
    gyy = np.asarray(
        [
            np.sum(client.values[1:] * client.values[1:]) / client.Ti
            for client in clients
        ],
        dtype=np.float64,
    )
    return SufficientStats(
        Gxx=Gxx,
        Gyx=Gyx,
        gyy=gyy,
        sample_sizes=sample_sizes,
        weights=weights,
        client_ids=client_ids,
    )


def pooled_statistics(stats: SufficientStats) -> tuple[np.ndarray, np.ndarray, float]:
    Gxx = np.tensordot(stats.weights, stats.Gxx, axes=(0, 0))
    Gyx = np.tensordot(stats.weights, stats.Gyx, axes=(0, 0))
    gyy = float(stats.weights @ stats.gyy)
    return Gxx, Gyx, gyy


def loss_from_statistics(
    A: np.ndarray,
    Gxx: np.ndarray,
    Gyx: np.ndarray,
    gyy: float,
) -> float:
    quadratic = np.sum((A @ Gxx) * A)
    linear = np.sum(A * Gyx)
    return float(0.5 * (gyy - 2.0 * linear + quadratic))


def batch_loss(A_clients: np.ndarray, stats: SufficientStats) -> np.ndarray:
    quadratic = np.sum(
        (A_clients @ stats.Gxx) * A_clients,
        axis=(1, 2),
    )
    linear = np.sum(A_clients * stats.Gyx, axis=(1, 2))
    return 0.5 * (stats.gyy - 2.0 * linear + quadratic)


def global_loss(A_clients: np.ndarray, stats: SufficientStats) -> float:
    return float(stats.weights @ batch_loss(A_clients, stats))


def batch_gradients(A_clients: np.ndarray, stats: SufficientStats) -> np.ndarray:
    return A_clients @ stats.Gxx - stats.Gyx


def _history_row(
    *,
    iteration: int,
    data_loss: float,
    elapsed: float,
    client_id: str | None = None,
    penalized_objective: float | None = None,
    relative_change: float | None = 0.0,
    relative_error: float | None = None,
) -> dict[str, Any]:
    row = {
        "client_id": client_id,
        "iteration": int(iteration),
        "data_loss": float(data_loss),
        "penalized_objective": (
            None if penalized_objective is None else float(penalized_objective)
        ),
        "relative_change": (
            None if relative_change is None else float(relative_change)
        ),
        "elapsed_time": float(elapsed),
    }
    if relative_error is not None:
        row["relative_error"] = float(relative_error)
    return row


def _relative_change(new: Sequence[np.ndarray], old: Sequence[np.ndarray]) -> float:
    numerator = np.sqrt(
        sum(float(np.sum((new_value - old_value) ** 2)) for new_value, old_value in zip(new, old)))
    new_norm = np.sqrt(sum(float(np.sum(new_value**2)) for new_value in new))
    old_norm = np.sqrt(sum(float(np.sum(old_value**2)) for old_value in old))
    denominator = max(1e-10,old_norm,)
    return float(numerator / denominator)


def _fedvar_block_relative_changes(
    new: FedVARState,
    old: FedVARState,
) -> dict[str, Any]:
    L_change = _relative_change((new.L,), (old.L,))
    R_change = _relative_change((new.R,), (old.R,))
    S0_change = _relative_change((new.S0,), (old.S0,))
    S_i_changes = [
        _relative_change(
            (new.S_local[index],),
            (old.S_local[index],),
        )
        for index in range(new.S_local.shape[0])
    ]
    S_local_change = max(S_i_changes, default=0.0)
    return {
        "L_relative_change": L_change,
        "R_relative_change": R_change,
        "S0_relative_change": S0_change,
        "S_local_relative_change": S_local_change,
        "S_i_relative_changes": S_i_changes,
        "relative_change": max(
            L_change,
            R_change,
            S0_change,
            S_local_change,
        ),
    }


def _right_solve(matrix: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """Return X satisfying X @ gram = matrix"""
    return np.linalg.solve(gram.T, matrix.T).T


# ---------------------------------------------------------------------------
# Structural and proximal operators
# ---------------------------------------------------------------------------


def singular_value_threshold(
    matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    kept = np.maximum(singular_values - threshold, 0.0)
    return (U * kept) @ Vt


def truncated_svd(matrix: np.ndarray, rank: int) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    return (U[:, :rank] * singular_values[:rank]) @ Vt[:rank]


def balanced_factors(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    roots = np.sqrt(singular_values[:rank])
    return U[:, :rank] * roots, Vt[:rank].T * roots


def capacity_threshold(
    matrix: np.ndarray,
    sparsity: int,
    gamma: float,
) -> np.ndarray:
    """Paper operator G_{s,gamma}, with flat-index tie breaking."""

    p = matrix.shape[0]
    row_column_cap = int(np.floor(gamma * p))
    output = np.zeros_like(matrix)
    flat = matrix.ravel()
    order = np.lexsort((np.arange(flat.size), -np.abs(flat)))
    row_counts = np.zeros(p, dtype=np.int64)
    column_counts = np.zeros(p, dtype=np.int64)
    accepted = 0
    for flat_index in order:
        if accepted == sparsity or flat[flat_index] == 0.0:
            break
        row, column = divmod(int(flat_index), p)
        if (
            row_counts[row] < row_column_cap
            and column_counts[column] < row_column_cap
        ):
            output[row, column] = flat[flat_index]
            row_counts[row] += 1
            column_counts[column] += 1
            accepted += 1
    return output


def prox_nuclear_max_norm(
    matrix: np.ndarray,
    nuclear_threshold: float,
    max_entry: float,
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Prox of nuclear norm plus a max-entry ball via Dykstra iteration."""

    if max_entry == 0.0:
        return np.zeros_like(matrix)
    if nuclear_threshold == 0.0:
        return np.clip(matrix, -max_entry, max_entry)

    value = matrix.copy()
    nuclear_correction = np.zeros_like(matrix)
    box_correction = np.zeros_like(matrix)
    for _ in range(max_iter):
        nuclear_input = value + nuclear_correction
        after_nuclear = singular_value_threshold(nuclear_input, nuclear_threshold)
        nuclear_correction = nuclear_input - after_nuclear

        box_input = after_nuclear + box_correction
        value = np.clip(box_input, -max_entry, max_entry)
        box_correction = box_input - value

        dual_sum = nuclear_correction + box_correction
        primal = 0.5 * float(np.sum((value - matrix) ** 2)) + nuclear_threshold * float(
            np.linalg.svd(value, compute_uv=False).sum()
        )
        dual = (
            -0.5 * float(np.sum(dual_sum**2))
            + float(np.sum(matrix * dual_sum))
            - max_entry * float(np.abs(box_correction).sum())
        )
        if primal - dual <= tol * max(1.0, abs(primal), abs(dual)):
            return value
    relative_gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    warn(
        f"Dykstra proximal iteration reached max_iter={max_iter}; returning the "
        f"last iterate (relative_gap={relative_gap:.3e}, tol={tol:.3e}).",
        RuntimeWarning,
        stacklevel=2,
    )
    return value


def _top_k_entries(matrix: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest magnitudes, breaking ties by flat index."""

    output = np.zeros_like(matrix)
    flat = matrix.ravel()
    order = np.lexsort((np.arange(flat.size), -np.abs(flat)))
    selected = order[:k]
    output.ravel()[selected] = flat[selected]
    return output


def _candidate_budgets(
    s_local: np.ndarray,
    p: int,
    freq_cap: int,
) -> np.ndarray:
    """Compute the paper's lossless FCHT screening budgets L_i."""

    return np.minimum(
        p * p,
        s_local + (s_local.sum() - s_local) // freq_cap,
    )


def weighted_fcht(
    proposals: np.ndarray,
    weights: np.ndarray,
    s_local: Sequence[int],
    freq_cap: int,
    *,
    lossless_screening: bool = False,
) -> np.ndarray:
    """Exact weighted FCHT through the integral bipartite b-matching LP.

    Optional lossless screening keeps each client's top L_i magnitudes,
    where L_i = min(p**2, s_i + floor(sum_{h != i} s_h / freq_cap)).
    The restricted graph retains a full-problem optimizer; tied optima
    can have different supports. Screening is defined for freq_cap >= 1.
    """

    n_clients, p, _ = proposals.shape
    q = p * p
    if lossless_screening and freq_cap > 0:
        candidate_budgets = _candidate_budgets(
            np.asarray(s_local, dtype=int), p, freq_cap
        )
        proposals = np.stack(
            [
                _top_k_entries(proposal, int(budget))
                for proposal, budget in zip(proposals, candidate_budgets)
            ]
        )
    if freq_cap >= n_clients:
        values = np.zeros_like(proposals)
        for client_index in range(n_clients):
            flat = proposals[client_index].ravel()
            order = np.lexsort((np.arange(flat.size), -np.abs(flat)))
            selected = order[: s_local[client_index]]
            values[client_index].ravel()[selected] = flat[selected]
        return values

    # The independent clientwise top-s solution is optimal for the relaxation
    # without frequency constraints.  If it is already frequency-feasible, it
    # is therefore also an exact optimum of the full FCHT problem.
    relaxed_values = np.zeros_like(proposals)
    relaxed_frequency = np.zeros(q, dtype=np.int64)
    for client_index in range(n_clients):
        flat = proposals[client_index].ravel()
        coordinates = np.flatnonzero(flat != 0.0)
        order = np.lexsort((coordinates, -np.abs(flat[coordinates])))
        selected = coordinates[order[: s_local[client_index]]]
        relaxed_values[client_index].ravel()[selected] = flat[selected]
        np.add.at(relaxed_frequency, selected, 1)
    if relaxed_frequency.max(initial=0) <= freq_cap:
        return relaxed_values

    edge_clients: list[int] = []
    edge_coordinates: list[int] = []
    edge_scores: list[float] = []

    for client_index in range(n_clients):
        if s_local[client_index] == 0:
            continue
        flat = proposals[client_index].ravel()
        coordinates = np.flatnonzero(flat != 0.0)
        edge_clients.extend([client_index] * len(coordinates))
        edge_coordinates.extend(coordinates.tolist())
        edge_scores.extend(
            (weights[client_index] * flat[coordinates] ** 2).tolist()
        )

    values = np.zeros_like(proposals)

    clients_array = np.asarray(edge_clients, dtype=np.int64)
    coordinates_array = np.asarray(edge_coordinates, dtype=np.int64)
    scores = np.asarray(edge_scores, dtype=np.float64)
    n_edges = scores.size
    columns = np.arange(n_edges, dtype=np.int64)
    rows = np.concatenate([clients_array, n_clients + coordinates_array])
    cols = np.concatenate([columns, columns])
    incidence = coo_matrix(
        (np.ones(2 * n_edges), (rows, cols)),
        shape=(n_clients + q, n_edges),
    ).tocsr()
    capacities = np.concatenate(
        [s_local, np.full(q, float(freq_cap))]
    )

    result = linprog(
        c=-scores / float(scores.max()),
        A_ub=incidence,
        b_ub=capacities,
        bounds=(0.0, 1.0),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        first_message = result.message
        warn(
            "Exact FCHT: initial LP solve failed; retrying the same LP with "
            f"presolve=False. Original message: {first_message}",
            RuntimeWarning,
            stacklevel=2,
        )
        result = linprog(
            c=-scores / float(scores.max()),
            A_ub=incidence,
            b_ub=capacities,
            bounds=(0.0, 1.0),
            method="highs-ds",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "presolve": False,
            },
        )
        if not result.success:
            raise RuntimeError(
                f"Exact FCHT failed: initial solve: {first_message}; "
                f"presolve=False retry: {result.message}"
            )

    rounded = np.rint(result.x)
    if np.max(np.abs(result.x - rounded)) > 1e-6:
        raise RuntimeError("Exact FCHT LP returned a non-integral solution.")
    if np.any(incidence @ rounded > capacities + 1e-9):
        raise RuntimeError("Exact FCHT LP returned an infeasible support.")
    selected_edges = np.flatnonzero(rounded > 0.5)
    for edge in selected_edges:
        client_index = int(clients_array[edge])
        coordinate = int(coordinates_array[edge])
        values[client_index].ravel()[coordinate] = proposals[client_index].ravel()[
            coordinate
        ]
    return values


# ---------------------------------------------------------------------------
# Basu penalized low-rank-plus-sparse estimators via AFNSL
# ---------------------------------------------------------------------------


def _basu_objective(
    B: np.ndarray,
    C: np.ndarray,
    Gxx: np.ndarray,
    Gyx: np.ndarray,
    gyy: float,
    lambda_B: float,
    lambda_C: float,
) -> float:
    return (
        loss_from_statistics(B + C, Gxx, Gyx, gyy)
        + lambda_B * float(np.linalg.svd(B, compute_uv=False).sum())
        + lambda_C * float(np.abs(C).sum())
    )


def _basu_relative_change(
    new: Sequence[np.ndarray], old: Sequence[np.ndarray], *,
    stop_norm_floor: float = 1.0,
) -> float | None:
    """Combined B/C change; None denotes a nonzero change from zero norm."""

    if not np.isfinite(stop_norm_floor) or stop_norm_floor < 0:
        raise ValueError("stop_norm_floor must be finite and nonnegative.")

    numerator = np.sqrt(
        sum(float(np.sum((new_value - old_value) ** 2))
            for new_value, old_value in zip(new, old))
    )
    old_norm = np.sqrt(sum(float(np.sum(old_value ** 2)) for old_value in old))
    denominator = max(stop_norm_floor, old_norm)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    return float(numerator / denominator)


def _solve_afnsl(
    Gxx: np.ndarray,
    Gyx: np.ndarray,
    gyy: float,
    *,
    lambda_B: float,
    lambda_C: float,
    config: BasuConfig,
    client_id: str | None,
    clock_start: float,
    truth_A: np.ndarray | None,
    truth_weights: np.ndarray | None,
) -> _BasuSolution:
    p = Gxx.shape[0]
    max_entry = config.nu / p
    B = np.zeros((p, p), dtype=np.float64)
    C = np.zeros((p, p), dtype=np.float64)
    B_aggregate = B.copy()
    C_aggregate = C.copy()
    Q = 0.0
    previous_A: np.ndarray | None = None
    previous_alpha = 1.0
    previous_eta = config.eta_init
    total_backtracks = 0
    converged = False

    initial_row = _history_row(
        client_id=client_id,
        iteration=0,
        data_loss=loss_from_statistics(B, Gxx, Gyx, gyy),
        penalized_objective=_basu_objective(
            B,
            C,
            Gxx,
            Gyx,
            gyy,
            lambda_B,
            lambda_C,
        ),
        relative_error=(
            None
            if truth_A is None
            else _weighted_relative_frobenius(
                (B + C)[None, :, :], truth_A, truth_weights
            )
        ),
        elapsed=perf_counter() - clock_start,
    )
    initial_row.update(
        {
            "alpha": None,
            "eta0": None,
            "eta": None,
            "gamma": None,
            "Q": Q,
            "backtracks": 0,
        }
    )
    history = [initial_row]

    for iteration in range(1, config.max_iter + 1):
        A = B + C
        if iteration == 1:
            eta0 = config.eta_init
        else:
            bb_delta = A - previous_A
            bb_norm = float(np.sum(bb_delta**2))
            eta0 = (
                config.eta_min
                if bb_norm == 0.0
                else max(
                    config.eta_min,
                    float(
                        np.einsum(
                            "ab,bc,ac->",
                            bb_delta,
                            Gxx,
                            bb_delta,
                            optimize=True,
                        )
                    )
                    / bb_norm,
                )
            )

        accepted = False
        for backtracks in range(config.max_backtracking + 1):
            if iteration == 1:
                alpha = 1.0
            else:
                ratio = eta0 / (previous_alpha * previous_eta)
                alpha = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * ratio))
            eta = alpha * eta0

            B_middle = (1.0 - alpha) * B_aggregate + alpha * B
            C_middle = (1.0 - alpha) * C_aggregate + alpha * C
            gradient = (B_middle + C_middle) @ Gxx - Gyx
            B_new = prox_nuclear_max_norm(
                B - gradient / eta,
                lambda_B / eta,
                max_entry,
                max_iter=config.prox_max_iter,
                tol=config.prox_tol,
            )
            C_input = C - gradient / eta
            C_new = np.sign(C_input) * np.maximum(
                np.abs(C_input) - lambda_C / eta,
                0.0,
            )
            delta = B_new + C_new - A
            gamma = float(np.sum(delta**2)) - (
                alpha / eta
            ) * float(
                np.einsum(
                    "ab,bc,ac->",
                    delta,
                    Gxx,
                    delta,
                    optimize=True,
                )
            )
            beta = (1.0 - 1.0 / iteration) ** 2
            Q_new = beta * Q + gamma
            if Q_new >= -config.line_search_C / iteration**2:
                accepted = True
                break
            eta0 *= config.sigma

        if not accepted:
            raise RuntimeError(
                "AFNSL line search failed after "
                f"{config.max_backtracking} backtracking steps."
            )

        B_aggregate_new = (1.0 - alpha) * B_aggregate + alpha * B_new
        C_aggregate_new = (1.0 - alpha) * C_aggregate + alpha * C_new
        relative = _basu_relative_change(
            (B_aggregate_new, C_aggregate_new),
            (B_aggregate, C_aggregate),
            stop_norm_floor=config.stop_norm_floor,
        )
        row = _history_row(
            client_id=client_id,
            iteration=iteration,
            data_loss=loss_from_statistics(
                B_aggregate_new + C_aggregate_new,
                Gxx,
                Gyx,
                gyy,
            ),
            penalized_objective=_basu_objective(
                B_aggregate_new,
                C_aggregate_new,
                Gxx,
                Gyx,
                gyy,
                lambda_B,
                lambda_C,
            ),
            relative_change=relative,
            relative_error=(
                None
                if truth_A is None
                else _weighted_relative_frobenius(
                    (B_aggregate_new + C_aggregate_new)[None, :, :],
                    truth_A,
                    truth_weights,
                )
            ),
            elapsed=perf_counter() - clock_start,
        )
        row.update(
            {
                "relative_change_defined": relative is not None,
                "alpha": float(alpha),
                "eta0": float(eta0),
                "eta": float(eta),
                "gamma": float(gamma),
                "Q": float(Q_new),
                "backtracks": backtracks,
            }
        )
        history.append(row)

        previous_A = A
        previous_alpha = alpha
        previous_eta = eta
        B, C = B_new, C_new
        B_aggregate, C_aggregate = B_aggregate_new, C_aggregate_new
        Q = Q_new
        total_backtracks += backtracks
        if config.tol is not None and relative is not None and relative <= config.tol:
            converged = True
            break

    return _BasuSolution(
        B_aggregate,
        C_aggregate,
        history,
        len(history) - 1,
        total_backtracks,
        converged,
    )


def local_basu_lrs(
    stats: SufficientStats,
    config: BasuConfig,
    truth_A: np.ndarray | None = None,
) -> FitResult:
    """Fit one independent penalized LR+S VAR per client with AFNSL."""

    lambda_B = config.lambda_B
    lambda_C = config.lambda_C
    eta_init = np.broadcast_to(config.eta_init, (stats.n_clients,))
    eta_min = np.broadcast_to(config.eta_min, (stats.n_clients,))
    start = perf_counter()
    solutions: list[_BasuSolution] = []
    for index, client_id in enumerate(stats.client_ids):
        client_config = replace(
            config,
            eta_init=float(eta_init[index]),
            eta_min=float(eta_min[index]),
        )
        solutions.append(
            _solve_afnsl(
                stats.Gxx[index],
                stats.Gyx[index],
                stats.gyy[index],
                lambda_B=lambda_B[index],
                lambda_C=lambda_C[index],
                config=client_config,
                client_id=client_id,
                clock_start=start,
                truth_A=(
                    None if truth_A is None else truth_A[index : index + 1]
                ),
                truth_weights=(
                    None
                    if truth_A is None
                    else stats.weights[index : index + 1]
                ),
            )
        )
    B_clients = np.stack([solution.B for solution in solutions])
    C_clients = np.stack([solution.C for solution in solutions])
    return FitResult(
        method="local_basu_lrs",
        client_ids=stats.client_ids,
        A_clients=B_clients + C_clients,
        components={"B_clients": B_clients, "C_clients": C_clients},
        history=[row for solution in solutions for row in solution.history],
        runtime=perf_counter() - start,
        diagnostics={
            "solver": "AFNSL",
            "client_iterations": [solution.n_iter for solution in solutions],
            "tol": config.tol,
            "stop_norm_floor": config.stop_norm_floor,
            "client_converged": [solution.converged for solution in solutions],
            "client_stop_reasons": [
                "relative_change" if solution.converged else "max_iter"
                for solution in solutions
            ],
            "client_backtracks": [solution.backtracks for solution in solutions],
        },
    )


def pooled_basu_lrs(
    stats: SufficientStats,
    config: BasuConfig,
    truth_A: np.ndarray | None = None,
) -> FitResult:
    """Fit one penalized LR+S VAR to pooled statistics with AFNSL."""

    lambda_B = config.lambda_B
    lambda_C = config.lambda_C
    Gxx, Gyx, gyy = pooled_statistics(stats)
    start = perf_counter()
    solution = _solve_afnsl(
        Gxx,
        Gyx,
        gyy,
        lambda_B=lambda_B,
        lambda_C=lambda_C,
        config=config,
        client_id=None,
        clock_start=start,
        truth_A=truth_A,
        truth_weights=None if truth_A is None else stats.weights,
    )
    A_global = solution.B + solution.C
    return FitResult(
        method="pooled_basu_lrs",
        client_ids=stats.client_ids,
        A_clients=np.repeat(A_global[None, :, :], stats.n_clients, axis=0),
        components={
            "A_global": A_global,
            "B_global": solution.B,
            "C_global": solution.C,
        },
        history=solution.history,
        runtime=perf_counter() - start,
        diagnostics={
            "solver": "AFNSL",
            "iterations": solution.n_iter,
            "tol": config.tol,
            "stop_norm_floor": config.stop_norm_floor,
            "converged": solution.converged,
            "stop_reason": "relative_change" if solution.converged else "max_iter",
            "backtracks": solution.backtracks,
        },
    )


# ---------------------------------------------------------------------------
# FedVAR initializer and Algorithm 1
# ---------------------------------------------------------------------------


def initialize_fedvar(
    stats: SufficientStats,
    config: FedVARConfig,
    local_basu_fit: FitResult,
) -> FedVARState:
    """Construct the paper initializer from clientwise Basu estimates."""

    rank = config.rank
    local_B = local_basu_fit.components["B_clients"]
    local_C = local_basu_fit.components["C_clients"]

    compressed_B = np.stack(
        [truncated_svd(local_B[index], rank) for index in range(stats.n_clients)]
    )
    average_B = np.tensordot(stats.weights, compressed_B, axes=(0, 0))
    B0 = truncated_svd(average_B, rank)
    L0, R0 = balanced_factors(B0, rank)

    sparse_median = np.median(local_C, axis=0)
    S0 = capacity_threshold(sparse_median, config.s0, config.gamma)
    local_proposals = local_C - S0[None, :, :]
    S_local = weighted_fcht(
        local_proposals,
        stats.weights,
        config.s_local,
        config.freq_cap,
        lossless_screening=config.lossless_screening,
    )
    return FedVARState(
        L=L0,
        R=R0,
        S0=S0,
        S_local=S_local,
        iteration=0,
    )


def fedvar(
    stats: SufficientStats,
    config: FedVARConfig,
    initial_state: FedVARState,
    truth_A: np.ndarray | None = None,
) -> FitResult:
    """Run the exact synchronous FedVAR updates for exactly K rounds."""

    start = perf_counter()
    state = FedVARState(
        L=initial_state.L.copy(),
        R=initial_state.R.copy(),
        S0=initial_state.S0.copy(),
        S_local=initial_state.S_local.copy(),
        iteration=0,
    )
    A_old = state.A_clients()
    initial_history_row = _history_row(
        iteration=0,
        data_loss=global_loss(A_old, stats),
        relative_error=(
            None
            if truth_A is None
            else _weighted_relative_frobenius(A_old, truth_A, stats.weights)
        ),
        elapsed=perf_counter() - start,
    )
    initial_history_row.update(_fedvar_block_relative_changes(state, state))
    history = [initial_history_row]

    for iteration in range(1, config.rounds + 1):
        old_L = state.L
        old_R = state.R
        old_S0 = state.S0
        old_S_local = state.S_local
        gradients = batch_gradients(A_old, stats)
        global_gradient = np.tensordot(stats.weights, gradients, axes=(0, 0))

        L_new = old_L - config.eta_B * _right_solve(
            global_gradient @ old_R,
            old_R.T @ old_R,
        )
        R_new = old_R - config.eta_B * _right_solve(
            global_gradient.T @ old_L,
            old_L.T @ old_L,
        )
        S0_new = capacity_threshold(
            old_S0 - config.eta_0 * global_gradient,
            config.s0,
            config.gamma,
        )
        local_proposals = old_S_local - config.eta_local * gradients
        S_local_new = weighted_fcht(
            local_proposals,
            stats.weights,
            config.s_local,
            config.freq_cap,
            lossless_screening=config.lossless_screening,
        )

        new_state = FedVARState(
            L=L_new,
            R=R_new,
            S0=S0_new,
            S_local=S_local_new,
            iteration=iteration,
        )
        A_new = new_state.A_clients()
        history_row = _history_row(
            iteration=iteration,
            data_loss=global_loss(A_new, stats),
            relative_error=(
                None
                if truth_A is None
                else _weighted_relative_frobenius(
                    A_new, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )
        history_row.update(_fedvar_block_relative_changes(new_state, state))
        history.append(history_row)
        state = new_state
        A_old = A_new

    return FitResult(
        method="fedvar",
        client_ids=stats.client_ids,
        A_clients=state.A_clients(),
        components={
            "L": state.L,
            "R": state.R,
            "B": state.L @ state.R.T,
            "S0": state.S0,
            "S_local": state.S_local,
        },
        history=history,
        runtime=perf_counter() - start,
        diagnostics={
            "rounds": config.rounds,
            "lossless_screening": config.lossless_screening,
        },
    )


# ---------------------------------------------------------------------------
# Metrics and experiment orchestration
# ---------------------------------------------------------------------------


def spectral_radii(A_clients: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.max(np.abs(np.linalg.eigvals(A))) for A in A_clients],
        dtype=np.float64,
    )


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    error = float(np.linalg.norm(estimate - truth))
    scale = float(np.linalg.norm(truth))
    return error / scale


def _weighted_relative_frobenius(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> float:
    errors = estimate - truth
    squared_errors = np.sum(errors**2, axis=(1, 2))
    squared_truth = np.sum(truth**2, axis=(1, 2))
    numerator = weights @ squared_errors
    denominator = weights @ squared_truth
    return float(np.sqrt(numerator / denominator))


def _support_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimated_support = estimate != 0.0
    true_support = truth != 0.0
    true_positive = int(np.count_nonzero(estimated_support & true_support))
    predicted = int(np.count_nonzero(estimated_support))
    actual = int(np.count_nonzero(true_support))
    precision = true_positive / predicted if predicted else float(actual == 0)
    recall = true_positive / actual if actual else float(predicted == 0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _truth_metrics(
    result: FitResult,
    truth: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if "A_clients" in truth:
        A_truth = truth["A_clients"]
        output["A_relative_error"] = _weighted_relative_frobenius(
            result.A_clients,
            A_truth,
            weights,
        )

    has_decomposition = all(key in truth for key in ("B", "S0", "S_local"))
    if not has_decomposition:
        return output
    B_truth = truth["B"]
    S0_truth = truth["S0"]
    S_local_truth = truth["S_local"]

    if result.method == "fedvar":
        output.update(
            {
                "B_relative_error": _relative_frobenius(
                    result.components["B"], B_truth
                ),
                "S0_relative_error": _relative_frobenius(
                    result.components["S0"], S0_truth
                ),
                "S_local_relative_error": _weighted_relative_frobenius(
                    result.components["S_local"], S_local_truth, weights
                ),
                "S0_support": _support_metrics(result.components["S0"], S0_truth),
                "S_local_support": _support_metrics(
                    result.components["S_local"], S_local_truth
                ),
            }
        )
    elif result.method.startswith("local_"):
        combined_sparse_truth = S0_truth[None, :, :] + S_local_truth
        output.update(
            {
                "B_relative_error": _weighted_relative_frobenius(
                    result.components["B_clients"],
                    np.broadcast_to(B_truth, result.components["B_clients"].shape),
                    weights,
                ),
                "C_relative_error": _weighted_relative_frobenius(
                    result.components["C_clients"], combined_sparse_truth, weights
                ),
                "C_support": _support_metrics(
                    result.components["C_clients"], combined_sparse_truth
                ),
            }
        )
    else:
        output.update(
            {
                "B_relative_error": _relative_frobenius(
                    result.components["B_global"], B_truth
                ),
                "C_relative_error": _relative_frobenius(
                    result.components["C_global"], S0_truth
                ),
                "C_support": _support_metrics(
                    result.components["C_global"], S0_truth
                ),
            }
        )
    return output


def compute_metrics(
    stats: SufficientStats,
    results: Mapping[str, FitResult],
    truth: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return in-window estimation metrics for every fitted method."""

    metrics: dict[str, Any] = {}
    for method, result in results.items():
        losses = batch_loss(result.A_clients, stats)
        per_coordinate_mse = 2.0 * losses / stats.Gxx.shape[1]
        radii = spectral_radii(result.A_clients)
        method_metrics: dict[str, Any] = {
            "clients": [
                {
                    "client_id": client_id,
                    "loss": float(losses[index]),
                    "mse": float(per_coordinate_mse[index]),
                    "spectral_radius": float(radii[index]),
                }
                for index, client_id in enumerate(stats.client_ids)
            ],
            "sample_weighted": {
                "loss": float(stats.weights @ losses),
                "mse": float(stats.weights @ per_coordinate_mse),
            },
        }
        if truth is not None:
            method_metrics["truth"] = _truth_metrics(result, truth, stats.weights)
        metrics[method] = method_metrics
    return metrics


__all__ = [
    "BasuConfig",
    "ClientData",
    "ExperimentResult",
    "FedVARConfig",
    "FedVARState",
    "FitResult",
    "SufficientStats",
    "balanced_factors",
    "batch_gradients",
    "batch_loss",
    "build_statistics",
    "capacity_threshold",
    "compute_metrics",
    "fedvar",
    "global_loss",
    "initialize_fedvar",
    "local_basu_lrs",
    "loss_from_statistics",
    "pooled_basu_lrs",
    "pooled_statistics",
    "prox_nuclear_max_norm",
    "spectral_radii",
    "truncated_svd",
    "weighted_fcht",
]
