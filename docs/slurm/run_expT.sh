#!/bin/bash
#SBATCH --job-name=exp_T_horizon
#SBATCH --array=0-139
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=logs/expT_%A_%a.out

set -euo pipefail

# --- project root (two levels up from docs/slurm/) ---
PROJECT_ROOT="$HOME/prospect-theory"
cd "$PROJECT_ROOT"
mkdir -p logs docs/expT_figures

# --- activate virtualenv ---
if [ -f "$HOME/envs/marl/bin/activate" ]; then
    source "$HOME/envs/marl/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "ERROR: no virtualenv found at ~/envs/marl or ./venv" >&2
    exit 1
fi

# single-threaded per task to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python experiments/exp_T_horizon.py --run_idx "$SLURM_ARRAY_TASK_ID"
