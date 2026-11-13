from __future__ import annotations
from collections.abc import Iterable
from time import perf_counter
import numpy as np
import pfedvar
from pfedvar import _candidate_budgets, _top_k_entries


def reduced_communication_fedvar(
    stats: pfedvar.SufficientStats,
    config: pfedvar.FedVARConfig,
    initial_state: pfedvar.FedVARState,
    *,
    shared_upload_budget: int,
    refresh_rounds: Iterable[int] | None = None,
    truth_A: np.ndarray | None = None,
) -> pfedvar.FitResult:
    """Run the paper's reduced-communication FedVAR for exactly K rounds.

    Parameters
    ----------
    shared_upload_budget:
        The paper's ``tilde{s}_0``: the number of error-compensated gradient
        entries uploaded by each client in every round.
    refresh_rounds:
        The zero-based refresh set ``R``.  ``None`` refreshes FCHT in every
        round, which retains exact FCHT support updates while still screening
        each client's uploaded scores to its lossless top-``L_i`` set.
        Screening at refresh rounds is always enabled in this routine,
        independently of ``config.lossless_screening``.
    """

    p = stats.Gxx.shape[1]
    s_local = np.asarray(config.s_local, dtype=int)
    shared_upload_budget = int(shared_upload_budget)
    refresh_set = frozenset(
        range(config.rounds) if refresh_rounds is None else refresh_rounds
    )

    candidate_budgets = _candidate_budgets(s_local, p, config.freq_cap)
    state = pfedvar.FedVARState(
        L=initial_state.L.copy(),
        R=initial_state.R.copy(),
        S0=initial_state.S0.copy(),
        S_local=initial_state.S_local.copy(),
        iteration=0,
    )
    error_feedback = np.zeros_like(state.S_local)
    supports = state.S_local != 0.0
    start = perf_counter()
    A_old = state.A_clients()
    initial_row = pfedvar._history_row(
        iteration=0,
        data_loss=pfedvar.global_loss(A_old, stats),
        relative_error=(
            None
            if truth_A is None
            else pfedvar._weighted_relative_frobenius(
                A_old, truth_A, stats.weights
            )
        ),
        elapsed=perf_counter() - start,
    )
    initial_row.update(pfedvar._fedvar_block_relative_changes(state, state))
    history = [initial_row]

    for round_index in range(config.rounds):
        old_L, old_R = state.L, state.R
        gradients = pfedvar.batch_gradients(A_old, stats)

        # Clients upload only the two p-by-r factor gradients.
        factor_L_messages = gradients @ old_R
        factor_R_messages = np.swapaxes(gradients, 1, 2) @ old_L
        aggregate_L = np.tensordot(
            stats.weights, factor_L_messages, axes=(0, 0)
        )
        aggregate_R = np.tensordot(
            stats.weights, factor_R_messages, axes=(0, 0)
        )

        # Top-tilde{s}_0 shared-gradient compression with error feedback.
        compensated = gradients + error_feedback
        compressed = np.stack(
            [
                _top_k_entries(message, shared_upload_budget)
                for message in compensated
            ]
        )
        error_feedback = compensated - compressed
        aggregate_sparse = np.tensordot(
            stats.weights, compressed, axes=(0, 0)
        )

        L_new = old_L - config.eta_B * pfedvar._right_solve(
            aggregate_L, old_R.T @ old_R
        )
        R_new = old_R - config.eta_B * pfedvar._right_solve(
            aggregate_R, old_L.T @ old_L
        )
        S0_new = pfedvar.capacity_threshold(
            state.S0 - config.eta_0 * aggregate_sparse,
            config.s0,
            config.gamma,
        )

        local_proposals = state.S_local - config.eta_local * gradients
        if round_index in refresh_set:
            S_local_new = pfedvar.weighted_fcht(
                local_proposals,
                stats.weights,
                s_local,
                config.freq_cap,
                lossless_screening=True,
            )
            supports = S_local_new != 0.0
        else:
            S_local_new = np.where(supports, local_proposals, 0.0)

        new_state = pfedvar.FedVARState(
            L=L_new,
            R=R_new,
            S0=S0_new,
            S_local=S_local_new,
            iteration=round_index + 1,
        )
        A_new = new_state.A_clients()
        row = pfedvar._history_row(
            iteration=round_index + 1,
            data_loss=pfedvar.global_loss(A_new, stats),
            relative_error=(
                None
                if truth_A is None
                else pfedvar._weighted_relative_frobenius(
                    A_new, truth_A, stats.weights
                )
            ),
            elapsed=perf_counter() - start,
        )
        row.update(pfedvar._fedvar_block_relative_changes(new_state, state))
        history.append(row)
        state = new_state
        A_old = A_new

    return pfedvar.FitResult(
        method="reduced_communication_fedvar",
        client_ids=stats.client_ids,
        A_clients=A_old,
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
            "shared_upload_budget": shared_upload_budget,
            "refresh_rounds": tuple(sorted(refresh_set)),
            "candidate_budgets": candidate_budgets,
        },
    )


__all__ = ["reduced_communication_fedvar"]
