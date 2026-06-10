"""
fix_and_smoke_test.py  -- run from project root:  python fix_and_smoke_test.py

PURPOSE
  Catch HPC failures locally BEFORE submitting, so you stop debugging on the cluster.

WHY THIS DIFFERS FROM THE ORIGINAL DRAFT
  The original plan renamed self.observation_spaces/action_spaces -> _private in
  every env/*.py to dodge a BaseParallelWrapper @property conflict. That conflict
  does NOT exist on the installed PettingZoo (1.26.1): BaseParallelWrapper has no
  such property, the assignments are plain attributes, and the full stack already
  trains end-to-end. Worse, renaming would break things: PettingZoo's ParallelEnv
  relies on the .action_spaces/.observation_spaces attributes, and
  BaseParallelWrapper.__getattr__ forwards unknown attrs to the WRAPPED env -- so a
  rename makes wrapper.action_spaces silently return the unwrapped env's spaces
  (wrong for ScaledBaseStockWrapper, which remaps actions to [-1,1]).

  The actual "works locally / fails on HPC" risk is a VERSION MISMATCH. So step 1
  here is a dependency-parity manifest (pin these on HPC), not a destructive rewrite.
  Step 3 exercises the real job entry point (run_one), which is exactly what PBS runs.
"""
import os, sys, ast, tempfile, traceback, importlib, re

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
ENV_DIR = "env"
EXPERIMENT_FILES = [
    "experiments/exp1_null_600k.py",
    "experiments/exp_ppo_lambda_sweep.py",
    "experiments/exp_ppo_mpe.py",
]
EXPECTED = {"pettingzoo": "1.26.1"}   # version the env wrappers were validated on

ok = True

# -- STEP 1: dependency-parity manifest + assumption check -------------------
print("=" * 64)
print("STEP 1 - dependency manifest (pin these in the HPC conda env)")
print("=" * 64)
manifest = {}
for pkg in ["pettingzoo", "stable_baselines3", "supersuit", "gymnasium",
            "numpy", "pandas", "torch", "filelock", "mpe2"]:
    try:
        v = getattr(importlib.import_module(pkg), "__version__", "?")
    except Exception as e:
        v = f"NOT INSTALLED ({e.__class__.__name__})"
    manifest[pkg] = v
    flag = ""
    if pkg in EXPECTED and not str(v).startswith(EXPECTED[pkg]):
        flag = f"  <-- expected {EXPECTED[pkg]} (HPC mismatch risk!)"; ok = False
    print(f"  {pkg:18s} {v}{flag}")
with open("env_manifest.txt", "w") as f:
    for k, v in manifest.items():
        f.write(f"{k}=={v}\n")
print("  (written to env_manifest.txt -- recreate this exact set on HPC)")

from pettingzoo.utils import BaseParallelWrapper
prop = isinstance(getattr(BaseParallelWrapper, "observation_spaces", None), property)
print(f"\n  BaseParallelWrapper.observation_spaces is a @property: {prop}")
if prop:
    print("  !! On THIS version it IS a property -> the wrappers' direct assignment will")
    print("     fail. Correct fix is property overrides per wrapper, NOT a private rename.")
    ok = False
else:
    print("  -> no conflict; the env wrappers' attribute assignments are correct as-is.")

# -- STEP 2: AST parse check -------------------------------------------------
print("\n" + "=" * 64)
print("STEP 2 - AST parse (env/ + experiment runners)")
print("=" * 64)
for path in [os.path.join(ENV_DIR, f) for f in os.listdir(ENV_DIR) if f.endswith(".py")] + EXPERIMENT_FILES:
    try:
        ast.parse(open(path).read()); print(f"  [ok]   {path}")
    except SyntaxError as e:
        print(f"  [FAIL] {path} -> {e}"); ok = False

# -- STEP 3: real job-path smoke test (run_one, tiny horizon) ----------------
print("\n" + "=" * 64)
print("STEP 3 - smoke test the ACTUAL HPC entry point (run_one, 2000 steps)")
print("=" * 64)

def smoke_runner(module_name, run_idx, label):
    global ok
    print(f"\n  [{label}]")
    try:
        mod = importlib.import_module(module_name)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        for lf in (tmp, tmp + ".lock"):
            if os.path.exists(lf): os.remove(lf)
        row = mod.run_one(run_idx, 2000, tmp)   # build + SuperSuit + PPO.learn + eval + label + CSV append
        import pandas as pd
        n = len(pd.read_csv(tmp))
        print(f"    run_one OK: w={row['w']} collapsed={row['collapsed']}; CSV rows={n}")
        for lf in (tmp, tmp + ".lock"):
            if os.path.exists(lf): os.remove(lf)
        print(f"  PASS: {label}")
    except Exception:
        print(f"  FAIL: {label}"); traceback.print_exc(); ok = False

smoke_runner("exp1_null_600k", 60, "exp1 stack (alpha=beta=1, lambda=1, w=0.60)")               # idx 60 -> w=0.60
smoke_runner("exp_ppo_lambda_sweep", 0, "lambda_sweep stack (alpha=beta=0.88, lambda=2, w=0.50)")  # idx 0  -> w=0.50, lam=2
smoke_runner("exp_ppo_mpe", 0, "MPE stack (w=0.30, lambda=1, seed=42 -- checks local reward wrapper)")

# -- STEP 4: non-destructive diagnostic scan ---------------------------------
print("\n" + "=" * 64)
print("STEP 4 - diagnostic scan: env-space attribute assignments (informational)")
print("=" * 64)
for root, dirs, files in os.walk(ENV_DIR):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fn in files:
        if fn.endswith(".py"):
            p = os.path.join(root, fn)
            for i, line in enumerate(open(p), 1):
                if re.search(r'self\.(observation_spaces|action_spaces)\s*=', line):
                    print(f"  {p}:{i}: {line.strip()}")
print("  (informational only -- correct as plain attributes on PettingZoo 1.26.1)")

# -- SUMMARY -----------------------------------------------------------------
print("\n" + "=" * 64)
print("ALL CHECKS PASSED - safe to SCP and requeue." if ok
      else "FAILURES / VERSION MISMATCH - resolve before requeuing.")
print("=" * 64)
sys.exit(0 if ok else 1)
