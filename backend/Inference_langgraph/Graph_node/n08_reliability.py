"""
backend/langgraph_pipeline/nodes/n08_reliability.py
=====================================================
LangGraph Node 8 — Reliability.

Computes live survival probability for the active incident using
pre-fitted Weibull parameters from weibull_params.json.

Design:
  - Weibull fitting is an offline batch job (run_weibull_fitter.py).
    This node READS the fitted parameters — it does NOT refit Weibull on
    every pipeline cycle (that would be too slow: O(seconds)).
  - On each cycle: S(t) = exp(-(elapsed_s / eta) ^ beta) * 100
  - The failure mode → Weibull group mapping is defined here.
  - When a new episode starts (episode_id changes), the previous episode's
    TTF observation is appended to life_data_extracted.csv for future
    reliability analysis.

Live data feed behaviour:
  - Every time run_simulator.py writes new rows, those rows flow through
    this node, keeping the survival curve updated in real-time.
  - When an episode ends (elapsed_s resets), the "failure event" is recorded.

Weibull groups (from design document):
  Group 1 "Immediate trigger"        : BAD_DEPLOY, CASCADING_FAILURE, ERROR_STORM
  Group 2 "Fast accumulation"        : CPU_SATURATION, DB_SLOWDOWN, DISK_IO_SATURATION
  Group 3 "Progressive degradation"  : MEMORY_LEAK, LATENCY_SPIKE, QUEUE_BACKUP
  Group 4 "Slow/latent degradation"  : RETRY_STORM, CACHE_STAMPEDE, DEPENDENCY_TIMEOUT

Returns:
    active_failure_group, survival_probability, weibull_beta, weibull_eta
"""
from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import NODES_DIR
from Inference_langgraph.state import AIOpsLangState

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIBULL_PARAMS_JSON  = NODES_DIR / "reliability" / "weibull_params.json"
LIFE_DATA_CSV        = NODES_DIR / "reliability" / "output" / "life_data_extracted.csv"

# ── Failure mode → Weibull group mapping ──────────────────────────────────────
# Keep in sync with weibull_fitter.py GROUP_MAP
_MODE_TO_GROUP: dict[str, str] = {
    "BAD_DEPLOY":          "Immediate trigger",
    "BAD_DEPLOYMENT":      "Immediate trigger",
    "CASCADING_FAILURE":   "Immediate trigger",
    "ERROR_STORM":         "Immediate trigger",
    "CPU_SATURATION":      "Fast accumulation",
    "DB_SLOWDOWN":         "Fast accumulation",
    "DISK_IO_SATURATION":  "Fast accumulation",
    "MEMORY_LEAK":         "Progressive resource degradation",
    "LATENCY_SPIKE":       "Progressive resource degradation",
    "QUEUE_BACKUP":        "Progressive resource degradation",
    "RETRY_STORM":         "Slow or latent degradation",
    "CACHE_STAMPEDE":      "Slow or latent degradation",
    "DEPENDENCY_TIMEOUT":  "Slow or latent degradation",
    "NONE":                None,
}

# ── Module-level state ────────────────────────────────────────────────────────
_lock              = threading.Lock()
_weibull_params:   dict | None = None   # loaded from weibull_params.json
_prev_episode_id:  str         = ""
_prev_elapsed_s:   float       = 0.0

# ── CSV ───────────────────────────────────────────────────────────────────────
_csv_fh     = None
_csv_writer = None
_csv_lock   = threading.Lock()


def _load_weibull_params() -> dict:
    """Load Weibull params from JSON sidecar. Falls back to defaults if missing."""
    if WEIBULL_PARAMS_JSON.exists():
        with WEIBULL_PARAMS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("groups", {})
    # Fallback: approximate parameters for all groups if not yet fitted
    print("[n08_reliability] WARNING: weibull_params.json not found. Using fallback params.")
    print("  Run: python backend/nodes/reliability/run_weibull_fitter.py")
    return {
        "Immediate trigger":                {"beta": 23.2, "eta": 10.3},
        "Fast accumulation":                {"beta": 4.3,  "eta": 15.2},
        "Progressive resource degradation": {"beta": 1.98, "eta": 46.5},
        "Slow or latent degradation":       {"beta": 7.5,  "eta": 373.2},
    }


def init() -> None:
    """Load Weibull parameters. Called once at pipeline startup."""
    global _weibull_params
    with _lock:
        if _weibull_params is None:
            print("[n08_reliability] Loading Weibull parameters...")
            _weibull_params = _load_weibull_params()


def _survival_probability(elapsed_s: float, beta: float, eta: float) -> float:
    """
    Compute two-parameter Weibull survival probability at time t.

    S(t) = exp(-(t / eta)^beta) × 100   → percentage 0–100

    Args:
        elapsed_s: Time elapsed in seconds since episode start.
        beta:      Shape parameter (Weibull β).
        eta:       Scale parameter / characteristic life (Weibull η), in seconds.

    Returns:
        Survival probability as a percentage (0.0–100.0).
        Returns 100.0 when elapsed_s ≤ 0.
    """
    if elapsed_s <= 0:
        return 100.0
    try:
        exponent = -(elapsed_s / eta) ** beta
        return float(math.exp(exponent) * 100.0)
    except (OverflowError, ZeroDivisionError):
        return 0.0


def _write_life_event(episode_id: str, failure_mode: str, duration_s: float, observed: bool) -> None:
    """Append a TTF observation to life_data_extracted.csv."""
    try:
        from Inference_langgraph.Graph_node.n01_collect import _LIVE_MODE
        if _LIVE_MODE:
            return
    except Exception:
        pass

    global _csv_writer, _csv_fh
    row = {
        "episode_id":   episode_id,
        "failure_mode": failure_mode,
        "duration_s":   round(duration_s, 2),
        "observed":     int(observed),   # 1=failure, 0=censored
    }
    with _csv_lock:
        if _csv_fh is None:
            LIFE_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = LIFE_DATA_CSV.open("a", newline="", encoding="utf-8")
        if _csv_writer is None:
            write_header = _csv_fh.tell() == 0
            _csv_writer  = csv.DictWriter(
                _csv_fh, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()



# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """
    Reliability node — computes survival probability using pre-fitted Weibull params.

    Also records episode TTF events to life_data_extracted.csv when episodes end,
    so these observations feed future Weibull re-fitting runs.
    """
    global _weibull_params, _prev_episode_id, _prev_elapsed_s

    if _weibull_params is None:
        init()

    episode_id   = state.get("episode_id", "")
    elapsed_s    = state.get("elapsed_s", 0.0)
    failure_mode = state.get("dominant_state") or state.get("failure_mode", "NONE")

    # ── Episode rollover detection ────────────────────────────────────────────
    # When episode_id changes, the previous episode ended — record its TTF.
    if _prev_episode_id and _prev_episode_id != episode_id and _prev_elapsed_s > 0:
        try:
            prev_mode = state.get("failure_mode", "NONE")
            # Record as a failure event (observed=True) if it was an active mode
            is_failure = prev_mode not in ("NONE", "")
            _write_life_event(
                episode_id   = _prev_episode_id,
                failure_mode = prev_mode,
                duration_s   = _prev_elapsed_s,
                observed     = is_failure,
            )
        except Exception as exc:
            print(f"[n08_reliability] Life-data write error: {exc}")

    _prev_episode_id = episode_id
    _prev_elapsed_s  = elapsed_s

    # ── Group lookup ──────────────────────────────────────────────────────────
    group = _MODE_TO_GROUP.get(failure_mode.upper() if failure_mode else "NONE")
    if group is None:
        # NONE mode — no active failure, max survival
        return {
            "active_failure_group": None,
            "survival_probability": 100.0,
            "weibull_beta":         None,
            "weibull_eta":          None,
        }

    params = _weibull_params.get(group, {})
    beta   = params.get("beta", 2.0)
    eta    = params.get("eta",  60.0)

    surv   = _survival_probability(elapsed_s, beta, eta)

    return {
        "active_failure_group": group,
        "survival_probability": round(surv, 2),
        "weibull_beta":         beta,
        "weibull_eta":          eta,
    }
