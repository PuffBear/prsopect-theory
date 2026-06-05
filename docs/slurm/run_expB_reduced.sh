#!/bin/bash
#SBATCH --job-name=exp_b_reduced
#SBATCH --array=0-299
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/expB_%A_%a.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs docs/expB_reduced_figures

# activate virtualenv (adjust path if yours differs)
if [ -f "$HOME/envs/marl/bin/activate" ]; then
    source "$HOME/envs/marl/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "ERROR: no virtualenv found at ~/envs/marl or ./venv" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python experiments/exp_b_reduced.py --run_idx "$SLURM_ARRAY_TASK_ID"
