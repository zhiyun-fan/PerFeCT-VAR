# PerFeCT-VAR

Code and data for the article *Personalized Federated Vector Autoregression with Personalization Diversity*.

**Authors:** Zhiyun Fan, Xiaoyu Zhang, Guodong Li, and Di Wang.

All commands below are run from the repository root.

## 1. File structure

```text
methods/
  pfedvar.py                   PerFeCT-VAR, initialization, and LRS estimators
  baselines.py                 Comparison methods
  reducedcom_pfedvar.py         Communication-efficient updates
simulation/
  _plot_common.py              Shared plotting utilities
  simulation1_varyTN/          Experiment 1: T and N
  simulation2_vary_m_het/      Experiment 2: diversity and covariance discrepancy
  simulation3_lq/              Experiment 3: working-cap tradeoff
  simulation4_reduced_comm/   Experiment 4: communication
realdata/
  00a_build_panel.py           Revenue-panel construction
  00b_preprocess_panel.py      Model-input construction
  01_run_experiment.py         Model selection, forecasting, and full-sample fitting
  data/                       Input arrays and DATA_CONTRACT.md
  results/                    Model selections, predictions, metrics, and coefficients
  figures/                    Plotting script and reference figures
reproduce.py                  Entry point for experiments and figures
```

Each simulation directory contains experiment scripts, `run_*.sh` launchers, seed-level CSVs in `*_outputs/`, and reference figures in `figures/`. New runs write to `reproduced/`, which is created when needed.

### Real-data inputs

The processed data used in the real-data analysis are stored in `realdata/data/processed.npz`, which is read directly by `realdata/01_run_experiment.py`.

The raw records are weekly UPC-level movement data from **Dominick's Finer Foods**, provided by the **Kilts Center for Marketing at the University of Chicago Booth School of Business**. To rebuild the processed inputs, download the **Movement .csv File** datasets from the [official dataset page](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) for the 17 categories listed in [realdata/data/DATA_CONTRACT.md](realdata/data/DATA_CONTRACT.md). Place the downloaded `w<category>.csv` files in `realdata/data/dominick_raw/`.

Run [00a_build_panel.py](realdata/00a_build_panel.py), followed by [00b_preprocess_panel.py](realdata/00b_preprocess_panel.py):

```bash
python realdata/00a_build_panel.py --raw-dir realdata/data/dominick_raw
python realdata/00b_preprocess_panel.py
```

The first script regenerates `realdata/data/raw_panel.npz`; the second regenerates `realdata/data/processed.npz`. The data-construction and preprocessing rules are documented in [realdata/data/DATA_CONTRACT.md](realdata/data/DATA_CONTRACT.md).

## 2. Fit PerFeCT-VAR to your own data

Prepare `client_series`, a list of NumPy arrays with one array per client. Each array has shape `(T_i + 1, p)`: rows are consecutive observations and columns are variables in the same order across clients. Client sample sizes may differ. Inputs must be finite; the estimator fits a VAR(1) without an intercept, so center or otherwise preprocess the series as appropriate.

The following function builds sufficient statistics, computes the initializer, and fits the model using [methods/pfedvar.py](methods/pfedvar.py). Its default initialization penalties and update step sizes follow the simulation scripts.

```python
import sys
import numpy as np

sys.path.insert(0, "methods")
import pfedvar as fm


def fit_perfect_var(client_series, *, rank, s0, s_local, gamma, m, rounds=200):
    clients = [
        fm.ClientData(str(i), np.asarray(values, dtype=float))
        for i, values in enumerate(client_series)
    ]
    stats = fm.build_statistics(clients)
    n, p = stats.n_clients, stats.Gxx.shape[-1]
    phi = np.array([np.linalg.norm(g, 2) for g in stats.Gxx])

    initializer = fm.local_basu_lrs(
        stats,
        fm.BasuConfig(
            lambda_B=0.5 * np.sqrt(phi * (p + np.log(n)) / stats.sample_sizes),
            lambda_C=0.5 * np.sqrt(phi * np.log(p * n) / stats.sample_sizes),
            nu=p,
            max_iter=100,
            eta_init=phi,
            eta_min=phi / 10,
        ),
    )
    config = fm.FedVARConfig(
        rank=rank, s0=s0, s_local=s_local, gamma=gamma, freq_cap=m,
        rounds=rounds, eta_B=0.1, eta_0=0.1, eta_local=0.3,
    )
    initial_state = fm.initialize_fedvar(stats, config, initializer)
    return fm.fedvar(stats, config, initial_state)
```

Supply the following fitting parameters:

| Parameter | Meaning |
| --- | --- |
| `rank` | Rank of the shared low-rank component |
| `s0` | Total sparsity budget for the shared sparse component |
| `s_local` | A list of personalized sparsity budgets, one per client |
| `gamma` | Shared sparse row/column cap as a fraction of `p` |
| `m` | Maximum client frequency of a personalized coordinate |
| `rounds` | Number of update rounds |

For example, with `p = 40` and at least three clients:

```python
fit = fit_perfect_var(
    client_series,
    rank=2, s0=40, s_local=[20] * len(client_series),
    gamma=3 / 40, m=3, rounds=200,
)
A_hat = fit.A_clients
B_hat = fit.components["B"]
S0_hat = fit.components["S0"]
Si_hat = fit.components["S_local"]
```

Choose fitting parameters for your data; the values above are an example. `A_hat` has shape `(N, p, p)`, `B_hat` and `S0_hat` have shape `(p, p)`, and `Si_hat` has shape `(N, p, p)`. Iteration diagnostics are in `fit.history`. A one-step prediction for client `i` is `A_hat[i] @ client_series[i][-1]`.

## 3. Rerun the experiments and reproduce the figures and table

### Rerun the experiments

```bash
# Run all four simulation experiments, with 200 replications per condition.
python reproduce.py simulations

# Run a selected experiment; this example runs one replication.
python reproduce.py simulations --experiments vary_lq --replications 1

# Run real-data model selection, test forecasting, and full-sample fitting.
python reproduce.py realdata --workers 4
```

The simulation selectors are `vary_TN`, `vary_m`, `vary_het`, `vary_lq`, and `reduced_comm`. Experiment 2 consists of `vary_m` and `vary_het`; multiple selectors may be passed together. Simulation results go to `reproduced/simulation/` under the corresponding experiment directories.

The real-data runner reads `realdata/data/processed.npz` and writes to `reproduced/realdata_results/`. Its `--stage` options are `all` (default), `check`, `cv`, `test`, and `refit`. Run `cv` before separate `test` or `refit` stages, using the same output directory. `--output-dir` selects a different result directory; use a fresh one when changing data, code, or the environment.

For a custom starting seed or output directory, invoke a simulation script directly:

```bash
python simulation/simulation1_varyTN/vary_TN.py 22567 --replications 4 --output-dir reproduced/custom_TN
python simulation/simulation3_lq/vary_m_lq.py 251001 --output-dir reproduced/custom_lq
```

The Lq script runs one seed per invocation; the other simulation scripts default to four consecutive seeds. Pass `--output-dir` when running Lq directly, since its default is the reference result directory. Existing seed CSVs are protected against overwriting by default. Formal plots require all 200 seeds at each condition.

For Slurm, submit the relevant `run_*.sh` from its experiment directory after creating `out/`. Set `PYTHON_BIN` to select the Python executable and `PERFECT_VAR_OUTPUT_DIR` to select the output directory; specify your partition when submitting with `sbatch`.

### Reproduce figures

```bash
# Draw all six experimental figures from the supplied reference results.
python reproduce.py figures --source reference

# Draw figures after completing the simulation and real-data reruns above.
python reproduce.py figures --source reproduced

# Draw one experiment only.
python reproduce.py figures --scope simulation3 --source reference
```

Scopes are `all`, `simulations`, `simulation1`, `simulation2`, `simulation3`, `simulation4`, and `realdata`. New figures are written under `reproduced/figures/`, in the experiment-specific folders listed below.

### Figure index

The table maps the main-text figures to their input files, plotting scripts, and reference PDFs. All paths are relative to the repository root. New PDFs use the same filenames under `reproduced/figures/` as indicated by the scope.

| Figure / LaTeX label | Input results or data | Plot script | Reference PDF / scope |
| --- | --- | --- | --- |
| Fig. 4 / `fig:sim_vary_TN` | `simulation/simulation1_varyTN/vary_TN_outputs/vary_[TN]_seed_*.csv` | [plot_vary_TN.py](simulation/simulation1_varyTN/plot_vary_TN.py) | [vary_TN.pdf](simulation/simulation1_varyTN/figures/vary_TN.pdf) / `simulation1` |
| Fig. 5 / `fig:sim_diversity_heterogeneity` | `simulation/simulation2_vary_m_het/vary_m_uniform_same_sign_outputs/seed_*.csv`<br>`simulation/simulation2_vary_m_het/vary_het_rank1_outputs/seed_*.csv` | [plot_vary_m_het.py](simulation/simulation2_vary_m_het/plot_vary_m_het.py) | [vary_m_het.pdf](simulation/simulation2_vary_m_het/figures/vary_m_het.pdf) / `simulation2` |
| Fig. 6 / `fig:sim_working_cap` | `simulation/simulation3_lq/vary_m_lq_outputs/seed_*.csv` | [plot_vary_m_lq.py](simulation/simulation3_lq/plot_vary_m_lq.py) | [vary_m_lq.pdf](simulation/simulation3_lq/figures/vary_m_lq.pdf) / `simulation3` |
| Fig. 7 / `fig:sim_communication` | `simulation/simulation4_reduced_comm/reduced_comm_outputs/seed_*.csv` | [plot_reduced_comm.py](simulation/simulation4_reduced_comm/plot_reduced_comm.py) | [communication.pdf](simulation/simulation4_reduced_comm/figures/communication.pdf) / `simulation4` |
| Fig. 8 / `fig:real_decomposition` | `realdata/results/full_sample/coefficients.npz`; `realdata/data/panel_metadata.json` | [plot_formal_figures.py](realdata/figures/plot_formal_figures.py) | [01_decomposition_varimax.pdf](realdata/figures/01_decomposition_varimax.pdf) / `realdata` |
| Fig. 9 / `fig:real_personalization` | `realdata/results/full_sample/coefficients.npz`; `realdata/data/panel_metadata.json` | [plot_formal_figures.py](realdata/figures/plot_formal_figures.py) | [03_breakfast_and_coo_personalization.pdf](realdata/figures/03_breakfast_and_coo_personalization.pdf) / `realdata` |

Figures 1–3 are manuscript illustrations of the model (`fig:perfect_var`), personalization diversity (`fig:personalization_diversity`), and estimation protocol (`fig:fed_protocol`). The real-data plotting script generates Figures 8 and 9 together.

For custom result locations, call the individual plotter with `--root` and `--output` for simulations, or `--results-dir` and `--output-dir` for real data. A simulation `--root` is the directory containing its `*_outputs/` folder.

### Reproduce the forecasting table (Table 3)

The table entries are the `mse` and `median_client_mse` fields in `realdata/results/test/metrics.json`. Print the paper's entries to three decimals:

```python
import json
from pathlib import Path

path = Path("realdata/results/test/metrics.json")
metrics = json.loads(path.read_text())["methods"]
methods = [
    ("Pooled OLS", "pooled_ols"), ("Local OLS", "local_ols"),
    ("Pooled LASSO", "pooled_lasso"), ("Local LASSO", "local_lasso"),
    ("Pooled LRS", "pooled_basu"), ("Local LRS", "local_basu"),
    ("Ditto", "ditto"), ("Zero", "zero"),
    ("PerFeCT-VAR", "pfedvar"), ("iAR(1)", "ar1"),
]
print(f"{'Method':<15} {'Mean MSE':>10} {'Median MSE':>12}")
for label, key in methods:
    row = metrics[key]
    print(f"{label:<15} {row['mse']:>10.3f} {row['median_client_mse']:>12.3f}")
```

For newly fitted results, change `path` to `reproduced/realdata_results/test/metrics.json`.
