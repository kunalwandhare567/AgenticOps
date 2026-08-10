"""
backend/langgraph_pipeline/nodes/n02_feature_engineering.py
=============================================================
LangGraph Node 2 — Feature Engineering.

Wraps the existing run_feature_engineering_from_raw() orchestrator.
Holds Drain3 singleton state across cycles.

Writes one row to engineered_features.csv after each successful run.

Returns:
    classifier_input  — 32-feature dict for LightGBM
    evidence          — diagnostic metadata
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import (
    DRAIN_INI, DRAIN_STATE, KNOWN_TEMPLATES_JSON, ENGINEERED_FEAT_CSV,
)
from Inference_langgraph.nodes.feature_engineering.orchestrator import run_feature_engineering_from_raw
from Inference_langgraph.nodes.feature_engineering.log_features import (
    load_template_miner, load_known_template_ids,
)
from Inference_langgraph.state import AIOpsLangState

# ── Drain3 singletons ─────────────────────────────────────────────────────────
_lock            = threading.Lock()
_template_miner  = None
_known_ids       = set()
_initialized     = False

# ── CSV file handle singleton ──────────────────────────────────────────────────
_csv_fh          = None
_csv_writer      = None
_csv_lock        = threading.Lock()


def init() -> None:
    """
    Initialise Drain3 and CSV writer singletons.
    Called once at pipeline startup from run_langgraph.py.
    """
    global _template_miner, _known_ids, _csv_fh, _csv_writer, _initialized
    with _lock:
        if not _initialized:
            _initialized = True
            print("[n02_feature_engineering] Loading Drain3 artifacts...")
            _template_miner = load_template_miner(str(DRAIN_STATE), str(DRAIN_INI))
            if _template_miner is None:
                # In-memory fallback if pre-trained state file is absent
                try:
                    from drain3 import TemplateMiner
                    from drain3.template_miner_config import TemplateMinerConfig
                    cfg = TemplateMinerConfig()
                    if DRAIN_INI.exists():
                        cfg.load(str(DRAIN_INI))
                    _template_miner = TemplateMiner(config=cfg)
                    print("[n02_feature_engineering] Initialised in-memory Drain3 miner.")
                except Exception as e:
                    print(f"[n02_feature_engineering] Could not init Drain3: {e}")
            _known_ids = load_known_template_ids(str(KNOWN_TEMPLATES_JSON)) or set()

    with _csv_lock:
        if _csv_fh is None:
            ENGINEERED_FEAT_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = ENGINEERED_FEAT_CSV.open("a", newline="", encoding="utf-8")
            _csv_writer = None  # header written lazily on first write


def _write_csv(row: dict) -> None:
    """Thread-safe append to engineered_features.csv."""
    global _csv_writer
    with _csv_lock:
        if _csv_writer is None:
            fieldnames = list(row.keys())
            write_header = _csv_fh.tell() == 0
            _csv_writer = csv.DictWriter(
                _csv_fh, fieldnames=fieldnames, extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """Feature engineering node — derives 32 features from raw telemetry."""
    global _template_miner, _known_ids

    # Lazy init if startup init() was not called
    if not _initialized:
        init()

    metric   = state.get("raw_metric", {})
    log      = state.get("raw_log", {})
    cycle    = state.get("cycle", 0)

    try:
        fe_result = run_feature_engineering_from_raw(
            metric             = metric,
            log                = log,
            episode_id         = state.get("episode_id", ""),
            failure_mode       = state.get("failure_mode", "NONE"),
            timestamp          = state.get("timestamp", 0.0),
            elapsed_s          = state.get("elapsed_s", 0.0),
            template_miner     = _template_miner,
            known_template_ids = _known_ids,
        )
    except Exception as exc:
        print(f"[n02_feature_engineering] Error in FE: {exc}")
        return {"error": f"FE error: {exc}", "classifier_input": {}, "evidence": {}}

    classifier_input = fe_result.get("classifier_input", {})
    evidence         = fe_result.get("evidence", {})

    # Append to CSV
    if classifier_input:
        try:
            _write_csv({
                "cycle":        cycle,
                "episode_id":   state.get("episode_id", ""),
                "failure_mode": state.get("failure_mode", "NONE"),
                "timestamp":    state.get("timestamp", 0.0),
                "elapsed_s":    state.get("elapsed_s", 0.0),
                **classifier_input,
            })
        except Exception as exc:
            print(f"[n02_feature_engineering] CSV write error: {exc}")

    return {
        "classifier_input": classifier_input,
        "evidence":         evidence,
    }
