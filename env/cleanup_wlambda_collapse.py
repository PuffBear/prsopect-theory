"""
Pre-registered collapse label for the custom-Cleanup w x lambda sweep (Experiment B,
in-repo `env/cleanup_env.py` substrate -- used because the real DeepMind Melting Pot
dmlab2d engine requires glibc >= 2.31 and the cluster is CentOS 7 / glibc 2.17; see
RESULTS_B.md).

Single boolean function of one episode summary statistic:

    collapsed  <=>  mean_collective_return  <  THRESHOLD

where collective_return = mean over eval episodes of the total TRUE apples harvested by
the team. THRESHOLD is read from a JSON committed to git BEFORE the main analysis
(pre-registration); set from a small pilot (low-w free-riding vs high-w coordination),
not tuned to a clean boundary.

NOTE: env/cleanup_env.py already ships an `is_collapsed_cleanup` with a separate default
threshold (5.0 apples/episode) used by the older exp_ppo_cleanup.py sweep. This module
is a DISTINCT, independently-frozen label for the new w x lambda x 20-seed sweep so its
threshold is pre-registered from this sweep's own pilot and lives under this experiment's
figures dir. Keeping them separate avoids silently inheriting the old sweep's number.
"""
import json
import os

_THRESHOLD_JSON = os.path.join(
    os.path.dirname(__file__), "..",
    "docs", "expCleanup_wlambda_figures", "cleanup_wlambda_collapse_threshold.json")

_DEFAULT_THRESHOLD = None   # None => is_collapsed raises if unfrozen (fail loud)


def load_cleanup_wlambda_threshold(default=_DEFAULT_THRESHOLD):
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return default


def is_collapsed_cleanup_wlambda(mean_collective_return, threshold=None):
    """Pre-registered collapse label. Raises if no threshold is frozen and none passed,
    so a sweep can never silently score collapse against an un-frozen label."""
    theta = load_cleanup_wlambda_threshold() if threshold is None else threshold
    if theta is None:
        raise RuntimeError(
            "No frozen Cleanup w-lambda collapse threshold at "
            f"{_THRESHOLD_JSON}. Run exp_cleanup_wlambda.py --pilot and commit the "
            "threshold JSON BEFORE the main analysis (pre-registration).")
    return bool(mean_collective_return < theta)
