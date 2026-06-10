#!/bin/bash
#SBATCH --job-name=exp_null
#SBATCH --array=0-179
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/expN_%A_%a.out

set -euo pipefail

# --- project root (directory two levels up from this script: docs/slurm/ -> repo) ---
PROJECT_ROOT="$HOME/prospect-theory"
cd "$PROJECT_ROOT"
mkdir -p logs docs/expN_figures

# --- activate virtualenv (adjust path if yours differs) ---
# Try the requested env first, then fall back to the repo-local venv/.
if [ -f "$HOME/envs/marl/bin/activate" ]; then
    source "$HOME/envs/marl/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "ERROR: no virtualenv found at ~/envs/marl or ./venv" >&2
    exit 1
fi

# single-threaded per task (cpus-per-task=1) to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python experiments/exp_null_model.py --run_idx "$SLURM_ARRAY_TASK_ID"
