#!/bin/bash

#SBATCH --job-name=sim_m_uniform_same
#SBATCH --array=0-49
#SBATCH --output=out/vary_m_uniform_same_sign_%A_%a.out
#SBATCH --error=out/vary_m_uniform_same_sign_%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# 50 array tasks x 4 consecutive seeds = 200 Monte Carlo replications.
# Activate the Python environment before submission. From this experiment
# directory, create out/ and submit with the cluster partition, for example:
#   mkdir -p out
#   sbatch --partition=<partition> run_vary_m_uniform_same_sign.sh
# PYTHON_BIN may specify the Python interpreter (default: python).
# PERFECT_VAR_OUTPUT_DIR may override the runner output directory.
# Otherwise, CSVs go under reproduced/simulation/ at the package root.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the experiment directory}"

TASK_ID=${SLURM_ARRAY_TASK_ID:?This launcher requires a Slurm array job}
SCRIPT=vary_m_uniform_same_sign.py
SEED_BASE=22567
FIRST_SEED=$((SEED_BASE + 4 * TASK_ID))
LAST_SEED=$((FIRST_SEED + 3))

echo "Task starts at: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Array task ID: $TASK_ID"
echo "Experiment: $SCRIPT"
echo "Seeds: $FIRST_SEED to $LAST_SEED"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

if [[ ! -f "$SCRIPT" ]]; then
    echo "Submit from the experiment directory containing $SCRIPT" >&2
    exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_ARGS=(-u "$SCRIPT" "$FIRST_SEED" --replications 4)
if [[ -n "${PERFECT_VAR_OUTPUT_DIR:-}" ]]; then
    PYTHON_ARGS+=(--output-dir "$PERFECT_VAR_OUTPUT_DIR")
fi
"$PYTHON_BIN" "${PYTHON_ARGS[@]}"

echo "Task finished at: $(date)"
