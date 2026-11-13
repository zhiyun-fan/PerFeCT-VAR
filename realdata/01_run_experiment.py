#!/usr/bin/env python3
"""N25P17: CV, rolling one-step forecasts, and descriptive full-sample refit.

Read data/processed.npz as supplied; this script never estimates preprocessing.
All substantive estimators are imported from the archive's shared methods directory.
AR(1) and zero are the two elementary reference forecasts retained from the
original calling layer. Fixed training settings are listed below; no external
settings file is read. Candidate fits use temporary storage; only CV summaries,
selection, test predictions/metrics and full-sample coefficients are retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter
import traceback
import warnings

# Each candidate is a worker; avoid BLAS multiplying the requested concurrency.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent

# Fixed paper experiment. Coordinates and their order come from processed.npz.
SPLITS = {
    'validation': {'window_weeks': 82, 'origins': 20, 'scored_weeks': 17},
    'test': {'window_weeks': 102, 'origins': 20, 'scored_weeks': 20},
    'full': {'window_weeks': 122, 'origins': 1},
}
INITIALIZER_C = 0.5
BASU_MAX_ENTRY = 0.5
BASU_SOLVER = dict(max_iter=100, tol=2e-4, stop_norm_floor=0.0,
                   eta_min=1e-7, sigma=2.0, line_search_C=0.0,
                   max_backtracking=50, prox_max_iter=5000, prox_tol=1e-5)
LASSO_SOLVER = dict(max_iter=20000, tol=1e-6, restart=True)
PFEDVAR_CANDIDATES = [
    dict(method='pfedvar', rank=2, gamma_p=2, s0=s0, s_local=s_local,
         freq_cap=freq_cap, global_step=0.05, local_step=0.001,
         first_iterations=100, warm_iterations=50)
    for s0 in (17, 25, 34)
    for s_local in (0, 10, 20, 30, 40)
    for freq_cap in (3, 5, 10)
]
REGULARIZATION_GRID = (0.1, 0.2, 0.4, 0.7, 1.0, 2.0)
PERSONALIZATION_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
BASELINE_GRIDS = {
    'local_basu': REGULARIZATION_GRID, 'pooled_basu': REGULARIZATION_GRID,
    'local_lasso': REGULARIZATION_GRID, 'pooled_lasso': REGULARIZATION_GRID,
    'ditto': PERSONALIZATION_GRID, 'pfedme': PERSONALIZATION_GRID,
    'ar1': (None,), 'local_ols': (None,), 'pooled_ols': (None,), 'zero': (None,),
}
BASELINE_CANDIDATES = [dict(method=method, parameter=value)
                       for method, grid in BASELINE_GRIDS.items() for value in grid]

# Per-process input cache and the paths needed by worker processes.
ENV = {}
ARRAYS = None
CONTEXTS = {}
fm = None
baselines = None


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
def fingerprint(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def locate_methods(explicit):
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        candidates = [HERE.parent / 'methods']
    for path in candidates:
        if all((path / f).is_file() for f in ('pfedvar.py', 'baselines.py')):
            return path
    raise FileNotFoundError('Shared methods not found; supply --methods-dir /path/to/archive/methods')


def initialize_worker(env):
    global ENV, ARRAYS, CONTEXTS, fm, baselines
    ENV = env
    ARRAYS = None
    CONTEXTS = {}
    methods = Path(env['methods_dir'])
    sys.path.insert(0, str(methods))
    fm = importlib.import_module('pfedvar')
    baselines = importlib.import_module('baselines')
    for module in (fm, baselines):
        if Path(module.__file__).resolve() != (methods / (module.__name__ + '.py')).resolve():
            raise RuntimeError('Imported an unexpected methods module')


def arrays():
    global ARRAYS
    if ARRAYS is None:
        with np.load(ENV['input_path'], allow_pickle=False) as z:
            ARRAYS = {k: z[k] for k in z.files}
    return ARRAYS


def contexts(split):
    if split not in CONTEXTS:
        values = arrays()
        data = values[split + '_standardized']
        policy = SPLITS[split]
        width, count = policy['window_weeks'], policy['origins']
        work = []
        for offset in range(count):
            clients = [fm.ClientData(str(cid), np.ascontiguousarray(data[i, offset:offset + width]))
                       for i, cid in enumerate(values['client_ids'])]
            stats = fm.build_statistics(clients)
            if not np.all(stats.sample_sizes == width - 1): raise ValueError('Lag count mismatch')
            work.append((clients, stats))
        CONTEXTS[split] = work
    return CONTEXTS[split]


def metrics(prediction, actual, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite forecast')
    if not np.isfinite(actual[mask]).all(): raise FloatingPointError('Nonfinite scored target')
    counts = mask.sum(axis=(1, 2))
    if np.any(counts == 0): raise ValueError('Empty client score mask')
    errors = prediction - actual
    client_mse = np.where(mask, errors**2, 0.0).sum(axis=(1, 2)) / counts
    return dict(mse=float(client_mse.mean()), median_client_mse=float(np.median(client_mse)),
                mae=float(np.where(mask, abs(errors), 0.0).sum() / mask.sum()),
                client_mse=client_mse, scored_cells=int(mask.sum()),
                scored_weeks=int(mask.any(axis=(0, 2)).sum()))


def basu_config(first, current, c, pooled=False):
    p = current.Gxx.shape[1]
    if pooled:
        Gxx, _, _ = fm.pooled_statistics(first)
        current_Gxx, _, _ = fm.pooled_statistics(current)
        size = float(first.sample_sizes.sum())
        nuclear = float(c) * np.sqrt(float(np.linalg.eigvalsh(Gxx)[-1]) * p / size)
        sparse = float(c) * np.sqrt(float(np.diag(Gxx).max()) * np.log(p * p) / size)
        eta = float(np.linalg.eigvalsh(current_Gxx)[-1])
    else:
        c = np.broadcast_to(np.asarray(c, dtype=float), (first.n_clients,))
        nuclear = c * np.sqrt(np.linalg.eigvalsh(first.Gxx)[:, -1] * p / first.sample_sizes)
        sparse = c * np.sqrt(np.diagonal(first.Gxx, axis1=1, axis2=2).max(axis=1)
                             * np.log(p * p) / first.sample_sizes)
        eta = np.linalg.eigvalsh(current.Gxx)[:, -1]
    if np.any(np.asarray(eta) <= 0): raise ValueError('Nonpositive Gram curvature')
    return fm.BasuConfig(lambda_B=nuclear, lambda_C=sparse, nu=BASU_MAX_ENTRY * p,
                         eta_init=eta, **BASU_SOLVER)


def fedvar_config(stats, spec, rounds):
    return fm.FedVARConfig(rank=spec['rank'], s0=spec['s0'],
                          s_local=[spec['s_local']] * stats.n_clients,
                          gamma=spec['gamma_p'] / stats.Gxx.shape[1], freq_cap=spec['freq_cap'],
                          rounds=rounds, eta_B=spec['global_step'], eta_0=spec['global_step'],
                          eta_local=spec['local_step'], lossless_screening=True)


def fit_key(spec): return spec['method'] + '_' + fingerprint(spec)[:16]


def fit_paths(split, spec):
    if spec['method'] == 'initializer':
        stem = Path(ENV['work_dir']) / 'initializers' / split
    else:
        stem = Path(ENV['work_dir']) / 'fits' / split / fit_key(spec)
    return stem.with_suffix('.json'), stem.with_suffix('.npz')




def fit_record(split, spec):
    js, _ = fit_paths(split, spec)
    return read(js)


def initial_fit(split):
    spec = {'method': 'initializer'}
    _, path = fit_paths(split, spec)
    with np.load(path, allow_pickle=False) as z:
        return fm.FitResult(method='local_basu_lrs', client_ids=tuple(str(x) for x in arrays()['client_ids']),
                            A_clients=z['A_clients'].copy(),
                            components={k: z[k].copy() for k in ('B_clients', 'C_clients')},
                            history=[], runtime=0.0)


def fit_job(split, spec):
    started = perf_counter()
    js, nz = fit_paths(split, spec)
    values = arrays()
    work = contexts(split)
    first = work[0][1]
    method, parameter = spec['method'], spec.get('parameter')
    forecasts, matrices, components, warning_rows = [], [], [], []
    fit = None
    if method == 'initializer':
        config = basu_config(first, first, INITIALIZER_C)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            fit = fm.local_basu_lrs(first, config)
        warning_rows = [str(w.message) for w in observed]
        save_npz(nz, **fit.components, A_clients=fit.A_clients, client_ids=values['client_ids'])
        result = {}
    else:
        initial = initial_fit(split) if method == 'pfedvar' else None
        width = SPLITS[split]['window_weeks']
        for offset, (clients, stats) in enumerate(work):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                if method == 'pfedvar':
                    rounds = spec['first_iterations'] if offset == 0 else spec['warm_iterations']
                    config = fedvar_config(stats, spec, rounds)
                    state = (fm.initialize_fedvar(stats, config, initial) if offset == 0 else
                             fm.FedVARState(**{k: fit.components[k].copy() for k in ('L', 'R', 'S0', 'S_local')}))
                    fit = fm.fedvar(stats, config, state)
                elif method in ('local_basu', 'pooled_basu'):
                    config = basu_config(first, stats, parameter, pooled=method == 'pooled_basu')
                    fit = (fm.local_basu_lrs if method == 'local_basu' else fm.pooled_basu_lrs)(stats, config)
                elif method in ('local_lasso', 'pooled_lasso'):
                    size = first.sample_sizes if method == 'local_lasso' else first.sample_sizes.sum()
                    alpha = np.asarray(parameter) * np.sqrt(np.log(stats.Gxx.shape[1]) / size)
                    if method == 'pooled_lasso': alpha = float(alpha)
                    fit_config = dict(alpha=alpha, **LASSO_SOLVER)
                    fit = getattr(baselines, method)(stats, **fit_config)
                elif method in ('ditto', 'pfedme'):
                    fit_config = dict(lambda_personalization=float(parameter))
                    fit = getattr(baselines, method)(stats, **fit_config)
                elif method in ('local_ols', 'pooled_ols'):
                    fit = getattr(baselines, method)(clients)
                elif method in ('ar1', 'zero'):
                    # Elementary reference forecasts; no substantive estimator is duplicated.
                    A = np.zeros_like(stats.Gxx)
                    if method == 'ar1':
                        for i, client in enumerate(clients):
                            X, Y = client.values[:-1], client.values[1:]
                            den = np.sum(X * X, axis=0)
                            phi = np.divide(np.sum(X * Y, axis=0), den,
                                            out=np.zeros_like(den), where=den > 0)
                            A[i] = np.diag(phi)
                    fit = fm.FitResult(method=method, client_ids=stats.client_ids, A_clients=A,
                                       components={}, history=[], runtime=0.0)
                else:
                    raise ValueError(f'Unknown method: {method}')
            messages = [str(w.message) for w in observed]
            warning_rows.extend(messages)
            if not np.isfinite(fit.A_clients).all(): raise FloatingPointError(f'{method}, origin {offset}')
            matrices.append(fit.A_clients.copy())
            if method == 'pfedvar':
                components.append({k: fit.components[k].copy() for k in ('L', 'R', 'B', 'S0', 'S_local')})
            if split != 'full':
                lagged = values[split + '_standardized'][:, width + offset - 1]
                forecasts.append(np.einsum('nij,nj->ni', fit.A_clients, lagged, optimize=True))
        output = dict(A_clients=np.stack(matrices), client_ids=values['client_ids'], family_ids=values['family_ids'])
        if components:
            output.update({k: np.stack([x[k] for x in components]) for k in components[0]})
        result = {}
        if split != 'full':
            prediction = np.stack(forecasts, axis=1)
            actual, mask = values[split + '_actual'], values[split + '_score_mask']
            output.update(forecast=prediction, actual=actual, score_mask=mask,
                          errors=prediction - actual,
                          squared_errors_scored=np.where(mask, (prediction - actual)**2, np.nan))
            for name in ('target_panel_weeks', 'target_week_start_dates', 'original_mask'):
                if split + '_' + name in values: output[name] = values[split + '_' + name]
            result['metrics'] = metrics(prediction, actual, mask)
        else:
            output = {k: (v[0] if k in ('A_clients', 'L', 'R', 'B', 'S0', 'S_local') else v) for k, v in output.items()}
            for key in ('fixed_effect_coefficients', 'client_means', 'variable_stds', 'fixed_effect_names', 'week_start_dates'):
                if key in values: output[key] = values[key]
            output.update({k: v for k, v in fit.components.items() if k not in output})
            result['training_loss'] = fm.global_loss(fit.A_clients, first)
        save_npz(nz, **output)
    result.update(spec=spec, split=split,
                  warnings_count=len(warning_rows), warnings=warning_rows[:50],
                  seconds=perf_counter() - started)
    write(js, result)
    return str(js)


def run_jobs(jobs, workers):
    pending = list(jobs)
    print(f'JOBS total={len(pending)} workers={workers}', flush=True)
    if not pending: return
    errors = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(ENV,)) as executor:
        futures = {executor.submit(fit_job, split, spec): (split, spec) for split, spec in pending}
        for number, future in enumerate(as_completed(futures), 1):
            split, spec = futures[future]
            try:
                rec = read(future.result())
                print(f'DONE {number}/{len(pending)} {split} {fit_key(spec)} seconds={rec["seconds"]:.1f} warnings={rec["warnings_count"]}', flush=True)
            except Exception:
                err = dict(split=split, spec=spec, traceback=traceback.format_exc())
                errors.append(err)
                print(f'FAILED {split} {fit_key(spec)} {err["traceback"].splitlines()[-1]}', flush=True)
    if errors: raise RuntimeError(f'{len(errors)} fits failed; see the errors above')


def select_models():
    output = Path(ENV['output_dir'])
    ranked = [dict(spec=s, metrics=fit_record('validation', s)['metrics']) for s in PFEDVAR_CANDIDATES]
    ranked.sort(key=lambda r: (r['metrics']['mse'], r['metrics']['median_client_mse'],
                               r['spec']['s0'], r['spec']['s_local'], r['spec']['freq_cap']))
    selected = {'pfedvar': ranked[0]['spec']}
    cv_metrics = {'pfedvar': ranked[0]['metrics']}
    baseline_ranking = {}
    for method in BASELINE_GRIDS:
        candidates = [s for s in BASELINE_CANDIDATES if s['method'] == method]
        rows = [dict(spec=s, metrics=fit_record('validation', s)['metrics']) for s in candidates]
        baseline_ranking[method] = rows
        if method in ('local_basu', 'local_lasso'):
            choices = np.argmin([r['metrics']['client_mse'] for r in rows], axis=0)
            selected[method] = dict(method=method, parameter=[candidates[int(i)]['parameter'] for i in choices])
            prediction = np.empty_like(arrays()['validation_actual'])
            for choice in np.unique(choices):
                _, path = fit_paths('validation', candidates[int(choice)])
                with np.load(path, allow_pickle=False) as z: prediction[choices == choice] = z['forecast'][choices == choice]
            cv_metrics[method] = metrics(prediction, arrays()['validation_actual'], arrays()['validation_score_mask'])
        else:
            winner = rows[int(np.argmin([r['metrics']['mse'] for r in rows]))]
            selected[method], cv_metrics[method] = winner['spec'], winner['metrics']
    result = dict(run_id=ENV['run_id'], selected=selected, validation_metrics=cv_metrics,
                  selection_rule='client-mean validation MSE; local Basu/LASSO per-client; pFedVAR ties by median MSE/s0/s_local/freq_cap',
                  scored_validation_weeks=int(arrays()['validation_score_mask'].any(axis=(0, 2)).sum()))
    write(output / 'cv' / 'pfedvar_ranking.json', ranked)
    write(output / 'cv' / 'baseline_ranking.json', baseline_ranking)
    write(output / 'selection.json', result)
    print('SELECTED', canonical(selected['pfedvar']), flush=True)
    return result


def checked_selection():
    output = Path(ENV['output_dir'])
    selection = read(output / 'selection.json')
    if selection['run_id'] != ENV['run_id']: raise RuntimeError('Selection identity mismatch; run --stage cv')
    return selection


def export_test(selection):
    output, values = Path(ENV['output_dir']), arrays()
    predictions = dict(client_ids=values['client_ids'], family_ids=values['family_ids'],
                       actual=values['test_actual'], score_mask=values['test_score_mask'])
    for key in ('test_target_panel_weeks', 'test_target_week_start_dates', 'test_original_mask'):
        if key in values: predictions[key.removeprefix('test_')] = values[key]
    all_metrics = {}
    for method, spec in selection['selected'].items():
        rec = fit_record('test', spec)
        _, path = fit_paths('test', spec)
        with np.load(path, allow_pickle=False) as z:
            predictions[method + '_forecast'] = z['forecast']
            predictions[method + '_errors'] = z['errors']
        all_metrics[method] = rec['metrics']
    pf = np.asarray(all_metrics['pfedvar']['client_mse'])
    comparisons = {}
    for method, met in all_metrics.items():
        if method == 'pfedvar': continue
        diff = np.asarray(met['client_mse']) - pf
        comparisons[method] = dict(pfedvar_wins=int((diff > 0).sum()), ties=int((diff == 0).sum()),
                                   pfedvar_losses=int((diff < 0).sum()),
                                   baseline_minus_pfedvar_client_mse=diff,
                                   relative_mse_reduction=(met['mse'] - all_metrics['pfedvar']['mse']) / met['mse'])
    save_npz(output / 'test' / 'predictions.npz', **predictions)
    write(output / 'test' / 'metrics.json', dict(run_id=ENV['run_id'], score_scale='supplied standardized residuals',
                                               methods=all_metrics, client_comparisons=comparisons))


def export_refit(selection):
    spec = selection['selected']['pfedvar']
    js, nz = fit_paths('full', spec)
    dest = Path(ENV['output_dir']) / 'full_sample'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(nz, dest / 'coefficients.npz')



def validate_input(input_path):
    with np.load(input_path, allow_pickle=False) as z:
        ids, families = z['client_ids'], z['family_ids']
        n, p = len(ids), len(families)
        if (n, p) != (25, 17) or len(set(ids.tolist())) != n or len(set(families.tolist())) != p:
            raise ValueError('Expected a 25-client, 17-variable panel with distinct coordinates')
        for split, policy in SPLITS.items():
            x = z[split + '_standardized']
            if x.shape != (n, 122, p) or not np.isfinite(x).all():
                raise ValueError(f'Invalid {split} standardized panel')
            if int(z[split + '_fit_end']) != policy['window_weeks']:
                raise ValueError(f'Prepared {split} fit end differs from fixed window')
            if split != 'full':
                actual, mask = z[split + '_actual'], z[split + '_score_mask']
                if actual.shape != (n, policy['origins'], p) or mask.shape != actual.shape or mask.dtype != bool:
                    raise ValueError(f'Invalid {split} targets or score mask')
                start = policy['window_weeks']
                if not np.array_equal(actual, x[:, start:start + policy['origins']]):
                    raise ValueError(f'{split} targets differ from standardized panel')
                if int(mask.any(axis=(0, 2)).sum()) != policy['scored_weeks']:
                    raise ValueError(f'{split} scoring weeks differ from the fixed stage')
        return dict(clients=n, variables=p, weeks=122,
                    validation_scored_weeks=int(z['validation_score_mask'].any(axis=(0, 2)).sum()),
                    test_scored_weeks=int(z['test_score_mask'].any(axis=(0, 2)).sum()),
                    pfedvar_candidates=len(PFEDVAR_CANDIDATES),
                    baseline_candidates=len(BASELINE_CANDIDATES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'cv', 'test', 'refit', 'check'], default='all')
    parser.add_argument('--methods-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    input_path = HERE / 'data' / 'processed.npz'
    methods = locate_methods(args.methods_dir)
    summary = validate_input(input_path)
    method_hashes = {name: digest(methods / name) for name in ('pfedvar.py', 'baselines.py')}
    identity = dict(schema_version=3, input_sha256=digest(input_path),
                    runner_sha256=digest(Path(__file__)), shared_methods_sha256=method_hashes,
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    run_id = fingerprint(identity)
    env = dict(input_path=str(input_path), methods_dir=str(methods),
               output_dir=str(args.output_dir.resolve()), run_id=run_id)
    initialize_worker(env)
    # Check is deliberately read-only, including when a previous result exists.
    if args.stage == 'check':
        session = args.output_dir / 'run_identity.json'
        print(json.dumps(dict(status='input_check_passed', run_id=run_id, methods_dir=str(methods),
                              existing_results_match=(read(session).get('run_id') == run_id) if session.exists() else None,
                              **summary), ensure_ascii=False, indent=2), flush=True)
        return
    output = args.output_dir.resolve()
    session = output / 'run_identity.json'
    if session.exists() and read(session) != dict(run_id=run_id, identity=identity):
        raise RuntimeError('Data/script/method/runtime identity changed. Use a new --output-dir; do not mix runs.')
    if output.exists() and any(output.iterdir()) and not session.exists():
        raise RuntimeError('Nonempty output has no identity; use an empty --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    write(session, dict(run_id=run_id, identity=identity))
    # A fresh temporary workspace serves this invocation only, with no checkpoint cache.
    with tempfile.TemporaryDirectory(prefix='dominick_fits_') as workspace:
        ENV['work_dir'] = workspace
        if args.stage in ('all', 'cv'):
            run_jobs([('validation', {'method': 'initializer'})], args.workers)
            run_jobs([('validation', s) for s in PFEDVAR_CANDIDATES + BASELINE_CANDIDATES], args.workers)
            select_models()
        if args.stage in ('all', 'test', 'refit'):
            selection = checked_selection()
            if args.stage in ('all', 'test'):
                run_jobs([('test', {'method': 'initializer'})], args.workers)
                run_jobs([('test', s) for s in selection['selected'].values()], args.workers)
                export_test(selection)
            if args.stage in ('all', 'refit'):
                run_jobs([('full', {'method': 'initializer'})], args.workers)
                run_jobs([('full', selection['selected']['pfedvar'])], args.workers)
                export_refit(selection)
    print('COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
