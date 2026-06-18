#!/bin/bash
# Submit all PENDING experiments for the AAAI revision, in priority order.
# Run from the project root on the HPC login node (10.1.4.95).
# Each line prints the job id; note the smoke-test step before the big arrays.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "== STEP 1: validate the centralized critic is REALLY wired (no training) =="
echo "  python3 experiments/exp_mappo_orgym.py --wiring_check"
echo "  (must print 'WIRING CHECK PASS' before spending cluster time)"
echo
echo "== STEP 2: smoke test on ONE run =="
echo "  python3 experiments/exp_mappo_orgym.py --run_idx 0 --timesteps 20000 --out /tmp/smoke_orgym.csv"
echo

# --- P1: second algorithm on the primary env (loss-aversion replication) -----
echo ">> MAPPO on or-gym supply chain (700 runs)"
qsub run_mappo_orgym.pbs

# --- Already queued by you (listed for completeness; uncomment to (re)submit) -
# qsub run_mappo_mpe_highlambda.pbs   # MPE cross-env (you queued the full MPE sweep already)
# qsub run_shapley.pbs                # Point 2  (80 runs)
# qsub run_heterogeneous.pbs          # Point 3  (400 runs)
# qsub run_entropy.pbs                # Point 4  (400 runs)

echo
echo "Check status:  qstat -tu \$USER     (or: qstat -t)"
echo "When a job finishes, pull results with:  ./sync_results.sh pull"
echo "Then analyze:"
echo "  python3 experiments/analyze_mpe_decomposition.py        # MPE 5-cond decomposition"
echo "  python3 experiments/analyze_reviewer_response.py        # shapley/het/entropy"
echo "  python3 experiments/analyze_firth_lambda_sweep.py       # supply-chain lambda law (done)"
