"""Shared VAR baselines for simulation and real-data callers.

LASSO and personalized methods consume SufficientStats. OLS consumes
ClientData trajectories to solve the original lag design with lstsq.
All fits return pfedvar.FitResult; preprocessing and CV belong to callers.
"""

from time import perf_counter
from typing import Sequence

import numpy as np

import pfedvar


def ditto(
    stats: pfedvar.SufficientStats,
    *,
    lambda_personalization: float,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Solve the Ditto global and personalized quadratic programs exactly."""

    start = perf_counter()
    Gxx_global, Gyx_global, _ = pfedvar.pooled_statistics(stats)
    A_global = pfedvar._right_solve(Gyx_global, Gxx_global)
    identity = np.eye(Gxx_global.shape[0])
    A_clients = np.stack(
        [
            pfedvar._right_solve(
                stats.Gyx[index] + lambda_personalization * A_global,
                stats.Gxx[index] + lambda_personalization * identity,
            )
            for index in range(stats.n_clients)
        ]
    )
    personalization = A_clients - A_global
    data_loss = pfedvar.global_loss(A_clients, stats)
    penalty = stats.weights @ np.sum(personalization**2, axis=(1, 2))
    objective = data_loss + 0.5 * lambda_personalization * penalty
    history = [
        pfedvar._history_row(
            iteration=0,
            data_loss=data_loss,
            penalized_objective=objective,
            relative_error=(
                None
                if truth_A is None
                else pfedvar._weighted_relative_frobenius(
                    A_clients, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )
    ]

    return pfedvar.FitResult(
        method="ditto",
        client_ids=stats.client_ids,
        A_clients=A_clients,
        components={
            "A_global": A_global,
            "personalization": personalization,
        },
        history=history,
        runtime=perf_counter() - start,
    )


def pfedme(
    stats: pfedvar.SufficientStats,
    *,
    lambda_personalization: float,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Exactly solve the sample-size-weighted pFedMe Moreau objective.

    Jointly minimize ``sum_i q_i [loss_i(A_i) + lambda/2 * ||A_i-W||_F^2]``
    over the personalized matrices and their common center. The quadratic
    problem is solved directly, without extra ridge or a pseudoinverse fallback.
    """

    start = perf_counter()
    lam = float(lambda_personalization)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("pFedMe lambda must be positive and finite.")

    identity = np.eye(stats.Gxx.shape[1])
    inverse_grams = np.stack(
        [
            pfedvar._right_solve(identity, gram + lam * identity)
            for gram in stats.Gxx
        ]
    )
    weighted_inverse = np.tensordot(stats.weights, inverse_grams, axes=(0, 0))
    right_hand_side = np.einsum(
        "n,nab,nbc->ac", stats.weights, stats.Gyx, inverse_grams, optimize=True
    )
    A_global = pfedvar._right_solve(
        right_hand_side, identity - lam * weighted_inverse
    )
    A_clients = np.stack(
        [
            pfedvar._right_solve(
                stats.Gyx[index] + lam * A_global,
                stats.Gxx[index] + lam * identity,
            )
            for index in range(stats.n_clients)
        ]
    )
    personalization = A_clients - A_global
    data_loss = pfedvar.global_loss(A_clients, stats)
    penalty = stats.weights @ np.sum(personalization**2, axis=(1, 2))
    objective = data_loss + 0.5 * lam * penalty
    history = [
        pfedvar._history_row(
            iteration=0,
            data_loss=data_loss,
            penalized_objective=objective,
            relative_error=(
                None
                if truth_A is None
                else pfedvar._weighted_relative_frobenius(
                    A_clients, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )
    ]

    return pfedvar.FitResult(
        method="pfedme",
        client_ids=stats.client_ids,
        A_clients=A_clients,
        components={
            "A_global": A_global,
            "personalization": personalization,
        },
        history=history,
        runtime=perf_counter() - start,
        diagnostics={
            "solver": "exact_full_batch_moreau",
            "lambda_personalization": lam,
            "fixed_point_residual": float(
                np.linalg.norm(
                    A_global
                    - np.tensordot(stats.weights, A_clients, axes=(0, 0))
                )
            ),
        },
    )


def _validate_lasso_options(
    alpha: np.ndarray, max_iter: int, tol: float
) -> None:
    if not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("LASSO alpha must be nonnegative and finite.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("LASSO max_iter must be a nonnegative integer.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("LASSO tol must be nonnegative and finite.")


def _solve_lasso(
    Gxx: np.ndarray,
    Gyx: np.ndarray,
    gyy: float,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    restart: bool,
    client_id: str | None,
    clock_start: float,
    truth_A: np.ndarray | None,
    truth_weights: np.ndarray,
) -> tuple[np.ndarray, list[dict], bool, int]:
    """Matrix FISTA with the historical objective restart and stopping rule."""

    def objective(A: np.ndarray) -> float:
        return pfedvar.loss_from_statistics(A, Gxx, Gyx, gyy) + alpha * float(
            np.abs(A).sum()
        )

    def history_row(
        A: np.ndarray, iteration: int, relative: float = 0.0,
        objective_value: float | None = None,
    ) -> dict:
        if objective_value is None:
            objective_value = objective(A)
        return pfedvar._history_row(
            client_id=client_id,
            iteration=iteration,
            data_loss=objective_value - alpha * float(np.abs(A).sum()),
            penalized_objective=objective_value,
            relative_change=relative,
            relative_error=(
                None if truth_A is None else pfedvar._weighted_relative_frobenius(
                    A[None, :, :], truth_A, truth_weights
                )
            ),
            elapsed=perf_counter() - clock_start,
        )

    A = np.zeros_like(Gyx)
    history = [history_row(A, 0)]
    if alpha == 0.0:
        A = pfedvar._right_solve(Gyx, Gxx)
        history.append(history_row(A, 1))
        return A, history, True, 1
    lipschitz = float(np.linalg.eigvalsh(Gxx).max())
    if lipschitz <= 0.0:
        return A, history, True, 0
    step = 1.0 / lipschitz
    middle = A.copy()
    momentum = 1.0
    for iteration in range(1, max_iter + 1):
        gradient = middle @ Gxx - Gyx
        proposal = middle - step * gradient
        candidate = np.sign(proposal) * np.maximum(np.abs(proposal) - step * alpha, 0.0)
        candidate_objective = objective(candidate)
        if restart and candidate_objective > history[-1]["penalized_objective"] + 1e-12:
            middle = A
            momentum = 1.0
            gradient = middle @ Gxx - Gyx
            proposal = middle - step * gradient
            candidate = np.sign(proposal) * np.maximum(np.abs(proposal) - step * alpha, 0.0)
            candidate_objective = objective(candidate)
        # Keep the original LASSO criterion; pfedvar's logging helper uses
        # a different normalization and is not this solver's stopping rule.
        relative = float(np.linalg.norm(candidate - A) / max(1.0, np.linalg.norm(A)))
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        middle = candidate + ((momentum - 1.0) / next_momentum) * (candidate - A)
        A, momentum = candidate, next_momentum
        history.append(history_row(A, iteration, relative, candidate_objective))
        if relative <= tol:
            return A, history, True, iteration
    return A, history, False, max_iter


def local_lasso(
    stats: pfedvar.SufficientStats,
    *,
    alpha: float | Sequence[float],
    max_iter: int = 20_000,
    tol: float = 1e-6,
    restart: bool = True,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Minimize loss_i(A_i) + alpha_i * sum(abs(A_i)) independently.

    Each loss is ||Y_i - X_i A_i.T||_F^2 / (2 T_i), without an intercept.
    ``alpha`` is a scalar or one value per client in ``stats.client_ids``
    order. It is the actual L1 penalty, not a grid multiplier c. Fits start
    at zero and use matrix FISTA with objective restart by default.
    """

    penalties = np.asarray(alpha, dtype=np.float64)
    if penalties.ndim == 0:
        penalties = np.full(stats.n_clients, float(penalties))
    if penalties.shape != (stats.n_clients,):
        raise ValueError("Local LASSO alpha must be a scalar or a length-N sequence.")
    _validate_lasso_options(penalties, max_iter, tol)
    start = perf_counter()
    solutions = [
        _solve_lasso(
            stats.Gxx[index], stats.Gyx[index], float(stats.gyy[index]),
            alpha=float(penalties[index]), max_iter=max_iter, tol=tol,
            restart=restart, client_id=client_id, clock_start=start,
            truth_A=None if truth_A is None else truth_A[index : index + 1],
            truth_weights=stats.weights[index : index + 1],
        )
        for index, client_id in enumerate(stats.client_ids)
    ]
    return pfedvar.FitResult(
        method="local_lasso",
        client_ids=stats.client_ids,
        A_clients=np.stack([solution[0] for solution in solutions]),
        components={},
        history=[row for solution in solutions for row in solution[1]],
        runtime=perf_counter() - start,
        diagnostics={
            "solver": "matrix_fista", "alpha": penalties.tolist(),
            "max_iter": max_iter, "tol": tol, "restart": restart,
            "initialization": "zeros",
            "client_converged": [solution[2] for solution in solutions],
            "client_iterations": [solution[3] for solution in solutions],
        },
    )


def pooled_lasso(
    stats: pfedvar.SufficientStats,
    *,
    alpha: float,
    max_iter: int = 20_000,
    tol: float = 1e-6,
    restart: bool = True,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Fit a shared no-intercept matrix LASSO using sample-size weights.

    The objective is sum_i T_i/sum(T) * loss_i(A) + alpha * sum(abs(A)).
    ``alpha`` is the actual scalar penalty. Each fit starts at zero.
    """

    penalty = np.asarray(alpha, dtype=np.float64)
    if penalty.ndim != 0:
        raise ValueError("Pooled LASSO alpha must be a scalar.")
    _validate_lasso_options(penalty, max_iter, tol)
    start = perf_counter()
    Gxx, Gyx, gyy = pfedvar.pooled_statistics(stats)
    A_global, history, converged, iterations = _solve_lasso(
        Gxx, Gyx, gyy, alpha=float(penalty), max_iter=max_iter, tol=tol,
        restart=restart, client_id=None, clock_start=start,
        truth_A=truth_A, truth_weights=stats.weights,
    )
    return pfedvar.FitResult(
        method="pooled_lasso",
        client_ids=stats.client_ids,
        A_clients=np.repeat(A_global[None, :, :], stats.n_clients, axis=0),
        components={"A_global": A_global},
        history=history,
        runtime=perf_counter() - start,
        diagnostics={
            "solver": "matrix_fista", "alpha": float(penalty),
            "max_iter": max_iter, "tol": tol, "restart": restart,
            "initialization": "zeros", "converged": converged,
            "iterations": iterations,
        },
    )


def _ols_inputs(
    clients: Sequence[pfedvar.ClientData],
) -> tuple[list[pfedvar.ClientData], pfedvar.SufficientStats]:
    """Retain raw lag matrices for least squares, including rank-deficient fits."""

    trajectories = []
    for client in clients:
        values = np.asarray(client.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("OLS clients need a (time, variables) array with at least two rows.")
        if not np.isfinite(values).all():
            raise ValueError("OLS client values must be finite.")
        if trajectories and values.shape[1] != trajectories[0].values.shape[1]:
            raise ValueError("OLS clients must have the same number of variables.")
        trajectories.append(pfedvar.ClientData(client_id=client.client_id, values=values))
    if not trajectories:
        raise ValueError("OLS requires at least one client.")
    return trajectories, pfedvar.build_statistics(trajectories)


def local_ols(
    clients: Sequence[pfedvar.ClientData],
    *,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Fit each no-intercept VAR by np.linalg.lstsq(X_i, Y_i, rcond=None).

    Raw trajectories preserve the original least-squares rank decision.
    No ridge is added; rank-deficient designs use lstsq's minimum-norm fit.
    Coefficient rows are responses and columns are lagged predictors.
    """

    start = perf_counter()
    trajectories, stats = _ols_inputs(clients)
    solutions = [
        np.linalg.lstsq(client.values[:-1], client.values[1:], rcond=None)
        for client in trajectories
    ]
    A_clients = np.stack([solution[0].T for solution in solutions])
    return pfedvar.FitResult(
        method="local_ols", client_ids=stats.client_ids,
        A_clients=A_clients, components={},
        history=[pfedvar._history_row(
            iteration=0, data_loss=pfedvar.global_loss(A_clients, stats),
            relative_error=(
                None if truth_A is None else pfedvar._weighted_relative_frobenius(
                    A_clients, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )],
        runtime=perf_counter() - start,
        diagnostics={"solver": "lstsq", "rcond": None,
                     "client_ranks": [int(solution[2]) for solution in solutions]},
    )


def pooled_ols(
    clients: Sequence[pfedvar.ClientData],
    *,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Fit a shared no-intercept VAR by raw pooled least squares, without ridge.

    Form X_i/Y_i within each trajectory before concatenation so no lag pair
    crosses a client boundary. Uses np.linalg.lstsq(..., rcond=None).
    """

    start = perf_counter()
    trajectories, stats = _ols_inputs(clients)
    X = np.concatenate([client.values[:-1] for client in trajectories], axis=0)
    Y = np.concatenate([client.values[1:] for client in trajectories], axis=0)
    coefficients, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    A_global = coefficients.T
    A_clients = np.repeat(A_global[None, :, :], stats.n_clients, axis=0)
    return pfedvar.FitResult(
        method="pooled_ols", client_ids=stats.client_ids,
        A_clients=A_clients, components={"A_global": A_global},
        history=[pfedvar._history_row(
            iteration=0, data_loss=pfedvar.global_loss(A_clients, stats),
            relative_error=(
                None if truth_A is None else pfedvar._weighted_relative_frobenius(
                    A_clients, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )],
        runtime=perf_counter() - start,
        diagnostics={"solver": "lstsq", "rcond": None, "rank": int(rank)},
    )


__all__ = [
    "ditto", "pfedme",
    "local_lasso", "pooled_lasso", "local_ols", "pooled_ols",
]
