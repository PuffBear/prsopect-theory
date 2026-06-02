import sys, os
from experiments.phase3_9_leadtime import _run_single_config
args = (42, 8, 1.0)
print("Starting L=8, alpha=1.0, seed=42")
res = _run_single_config(args)
print(res)
