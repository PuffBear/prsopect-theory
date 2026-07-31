"""
Pre-registered collapse label for Melting Pot Cleanup (Experiment B).

The label is a single boolean function of one episode summary statistic -- the mean
per-episode TRUE collective return (total apples harvested by the team). This mirrors
the supply-chain and in-repo-Cleanup convention (a frozen scalar threshold).

    collapsed  <=>  mean_collective_return  <  THRESHOLD

THRESHOLD is read from a JSON file that is COMMITTED TO GIT BEFORE the main analysis
runs (pre-registration). The threshold is set from a small pilot that reads the scale
of the collective return under obvious free-riding (low w) vs obvious coordination
(high w) -- NOT tuned to produce a clean boundary. Once committed, it is frozen; the
configuration record is stored with the analysis outputs and in the driver header.

Rationale for "collective return below a frozen threshold": in Cleanup, free-riding
(everyone harvests, nobody cleans) drives waste up -> apple spawn -> 0 -> the commons
collapses and total harvest craters. A coordinated population sustains apple supply and
harvests steadily. So a low collective return is the direct, substrate-native signature
of commons collapse (the local-cost/global-benefit failure the paper studies).
"""
import json
import os

_THRESHOLD_JSON = os.path.join(
    os.path.dirname(__file__), "..",
    "docs", "expMeltingpot_cleanup_figures", "meltingpot_collapse_threshold.json")

# Fallback only if the frozen file is absent. A run should FREEZE the real value first;
# this default exists so imports don't crash, and is deliberately not a "nice" number.
_DEFAULT_THRESHOLD = None   # None => is_collapsed raises if no frozen file (fail loud)


def load_meltingpot_collapse_threshold(default=_DEFAULT_THRESHOLD):
    """Return the frozen collapse threshold, or `default` if the JSON is absent."""
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return default


def is_collapsed_meltingpot(mean_collective_return, threshold=None):
    """Pre-registered collapse label for Melting Pot Cleanup.

    Args:
        mean_collective_return: mean over eval episodes of the TRUE collective return
            (sum over the team of raw apples harvested per episode).
        threshold: override (analysis/tests only). Default reads the frozen JSON.

    Returns:
        bool -- True iff the team harvested fewer than `threshold` apples/episode.

    Raises:
        RuntimeError if no threshold is frozen and none is passed -- so a sweep can
        never silently score collapse against an un-frozen label.
    """
    theta = load_meltingpot_collapse_threshold() if threshold is None else threshold
    if theta is None:
        raise RuntimeError(
            "No frozen Melting Pot collapse threshold found at "
            f"{_THRESHOLD_JSON}. Run the pilot (exp_meltingpot_cleanup.py --pilot) and "
            "commit the threshold JSON BEFORE the main analysis (pre-registration).")
    return bool(mean_collective_return < theta)
