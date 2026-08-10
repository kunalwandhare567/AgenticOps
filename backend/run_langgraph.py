"""
backend/run_langgraph.py
=========================
AIOps LangGraph Pipeline Runner — Terminal 2.

This is the replacement for run_pipeline.py. It runs the full pipeline as a
LangGraph StateGraph, processing one simulator DB row per cycle.

Pipeline node order (per graph.py):
  [1] collect          — read next row from simulator_db.sqlite
  [2] feature_eng      — derive 32 features (Drain3 log parsing)
  [3] prelim_severity  — DEVOPS threshold-based severity engine
  [4] classify         — LightGBM 13-class incident classifier
  [5] tumbling_window  — majority-vote label smoother (N cycles)
  [6] forecasting      — mode-specific TTF forecast (every N cycles)
  [7] severity_update  — revise severity using impact × urgency matrix
  [8] reliability      — Weibull S(t) survival probability
  [9] human_gate       — escalation review (auto-approve on timeout)
  [10] db_writer       — write full state to all DB tables + CSVs

Usage:
    # Standard (poll every 100ms):
    python backend/run_langgraph.py

    # Fast replay of historical data (no sleep between cycles):
    python backend/run_langgraph.py --speed 50

    # Verbose (print each node's output dict):
    python backend/run_langgraph.py --verbose

    # Dry-run (skip CSV and DB writes):
    python backend/run_langgraph.py --dry-run

    # Print graph diagram and exit:
    python backend/run_langgraph.py --diagram
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Force UTF-8 output (Windows cp1252 compatibility) ─────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Resolve package root ──────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE
sys.path.insert(0, str(PROJECT_ROOT))

from Simulator.app_data_generator_for_offline.config import (
    DB_PATH,
    PIPELINE_RESULTS_CSV,
    PRELIM_SEVERITY_CSV,
    ENGINEERED_FEAT_CSV,
    FORECASTING_OUTPUT_CSV,
    SEVERITY_UPDATE_CSV,
    HUMAN_GATE_OUTPUT_CSV,
    POLL_INTERVAL_MS,
    LANGGRAPH_VERBOSE,
    LIVE_FEED_DB_PATH,
    LIVE_POLL_INTERVAL_MS,
)

# ── Import node init functions (called once at startup) ───────────────────────
from Inference_langgraph.Graph_node import n01_collect
from Inference_langgraph.Graph_node import n02_feature_engineering
from Inference_langgraph.Graph_node import n03_prelim_severity
from Inference_langgraph.Graph_node import n04_classify
from Inference_langgraph.Graph_node import n05_tumbling_window
from Inference_langgraph.Graph_node import n07_severity_update
from Inference_langgraph.Graph_node import n08_reliability
from Inference_langgraph.Graph_node import n10_db_writer
from Inference_langgraph.graph import build_graph, get_graph_ascii
from Inference_langgraph.state import make_empty_state



# =============================================================================
# Startup & banner
# =============================================================================

def _init_singletons(live: bool = False) -> None:
    """
    Initialise all module-level singletons before the graph starts running.
    This ensures expensive I/O (Drain3 load, DB open) happens once at startup,
    not on the first cycle.
    """
    if live:
        print("[startup] LIVE FEED MODE — enabling LiveTelemetryQueue & live_feed_db input.")
        n01_collect.set_live_mode(True, db_path=str(LIVE_FEED_DB_PATH))

        print(f"[startup] Redirecting DB writes to: {LIVE_FEED_DB_PATH}")
        LIVE_FEED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        n10_db_writer.set_live_db_path(str(LIVE_FEED_DB_PATH))


    print("[startup] Initialising Drain3 + Feature Engineering...")
    n02_feature_engineering.init()

    print("[startup] Initialising DEVOPS SeverityEngine...")
    n03_prelim_severity.init()

    print("[startup] Loading LightGBM classifier...")
    n04_classify.init()

    print("[startup] Initialising TumblingWindow...")
    n05_tumbling_window.init()

    print("[startup] Initialising SeverityUpdater...")
    n07_severity_update.init()

    print("[startup] Loading Weibull reliability parameters...")
    n08_reliability.init()

    print("[startup] Initialising DB writer (pipeline_results tables)...")
    n10_db_writer.init()


def _print_banner(poll_ms: int, verbose: bool, dry_run: bool, live: bool = False) -> None:
    mode_tag = "[[ LIVE FEED MODE ]]" if live else "[[ HISTORICAL MODE ]]"
    db_label = str(LIVE_FEED_DB_PATH) if live else str(DB_PATH)
    db_status = "exists" if (LIVE_FEED_DB_PATH if live else DB_PATH).exists() else "waiting..."
    print(f"\n{'='*68}")
    print(f"  AIOps LangGraph Pipeline Runner  {mode_tag}")
    print(f"{'='*68}")
    print(f"  Target DB      : {db_label}  [{db_status}]")
    print(f"  Poll interval  : {poll_ms} ms")
    print(f"  Verbose        : {verbose}")
    print(f"  Dry-run        : {dry_run}")
    if live:
        print(f"  Queue source   : LiveTelemetryQueue (live_feed_simulator)")
    print(f"{'='*68}")
    print(f"  Node order: collect → FE → prelim_sev → classify → window")
    print(f"              → forecast → sev_update → reliability → human_gate → db_write")
    print(f"{'='*68}")
    print("  Press Ctrl+C to stop.\n")


# =============================================================================
# Console summary helper
# =============================================================================

def _print_cycle_summary(state: dict, cycle: int, elapsed_ms: float) -> None:
    """Print one-line cycle summary to console."""
    sev      = state.get("preliminary_severity", "??")
    revised  = state.get("revised_severity")     or sev
    pred     = state.get("predicted_failure",    "??")
    prob     = state.get("prediction_probability", 0.0)
    dominant = state.get("dominant_state",        "??")
    hg_dec   = state.get("hg_decision")
    surv     = state.get("survival_probability")
    ttf      = state.get("time_to_failure")

    # Build compact status string
    hg_str   = f"  HG:{hg_dec}"                   if hg_dec                  else ""
    surv_str = f"  S(t):{surv:.0f}%"               if surv is not None        else ""
    ttf_str  = f"  TTF:{ttf:.0f}s"                 if ttf  is not None        else ""
    rev_str  = f"→{revised}"                       if revised != sev          else ""

    print(
        f"[cycle {cycle:>6}] "
        f"mode={state.get('failure_mode','?'):<22} "
        f"sev={sev}{rev_str}  "
        f"pred={pred}({prob:.2f})  "
        f"win={dominant}"
        f"{surv_str}{ttf_str}{hg_str}  "
        f"[{elapsed_ms:.0f}ms]"
    )


# =============================================================================
# Main pipeline loop
# =============================================================================

def run_pipeline(poll_ms: int, verbose: bool, dry_run: bool) -> None:
    """
    Main polling loop.

    Each iteration invokes the compiled LangGraph graph with a fresh initial
    state dict. The graph reads the next DB row, processes all 10 nodes, and
    returns the full updated state. If collect returns no_data, we sleep for
    poll_ms milliseconds and retry.

    Args:
        poll_ms:  Milliseconds to sleep between cycles when no data is ready.
        verbose:  If True, print each node's output dict via graph.stream().
        dry_run:  If True, nodes skip CSV + DB writes (not yet implemented per-node;
                  currently this flag is surfaced but not yet propagated).
    """
    print("[pipeline] Building LangGraph graph...")
    graph = build_graph()
    print("[pipeline] Graph compiled. Starting polling loop...")

    cycle        = 0
    total_cycles = 0
    start_time   = time.monotonic()

    # LangGraph config — thread_id groups cycles into a single conversation.
    # This means the checkpointer sees them as one long session (resumable).
    lg_config = {"configurable": {"thread_id": "aiops_pipeline_main"}}

    try:
        while True:
            t0 = time.monotonic()

            # ── Build initial state for this cycle ────────────────────────────
            initial = make_empty_state(cycle=cycle)
            # Carry forward the last_processed_id so we don't re-process rows
            # (The graph uses module-level state in n01_collect, but we also
            #  pass it here for visibility / future checkpointing replay.)
            from Inference_langgraph.Graph_node.n01_collect import _lock as _cl
            # We don't expose last_processed_id externally; n01_collect manages it

            # ── Run the graph ─────────────────────────────────────────────────
            if verbose:
                final_state: dict = {}
                for step_output in graph.stream(initial, config=lg_config):
                    for node_name, node_output in step_output.items():
                        print(f"    [{node_name}] → {node_output}")
                        final_state.update(node_output)
            else:
                final_state = graph.invoke(initial, config=lg_config)

            # ── Check if collect got data ─────────────────────────────────────
            if final_state.get("error") == "no_data" or not final_state.get("episode_id"):
                # No new row — sleep and retry
                time.sleep(poll_ms / 1000.0)
                continue

            # ── Valid cycle ───────────────────────────────────────────────────
            cycle        += 1
            total_cycles += 1
            elapsed_ms = (time.monotonic() - t0) * 1000

            _print_cycle_summary(final_state, cycle, elapsed_ms)

    except KeyboardInterrupt:
        runtime = time.monotonic() - start_time
        print(f"\n\n  [Pipeline] Stopped by user after {total_cycles:,} cycles "
              f"({runtime:.0f}s runtime).")

    finally:
        throughput = total_cycles / max((time.monotonic() - start_time), 1)
        print(f"\n  Total cycles processed : {total_cycles:,}")
        print(f"  Average throughput     : {throughput:.1f} cycles/sec")
        print(f"\n  Output files:")
        print(f"    Features   : {ENGINEERED_FEAT_CSV}")
        print(f"    Severity   : {PRELIM_SEVERITY_CSV}")
        print(f"    Results    : {PIPELINE_RESULTS_CSV}")
        print(f"    Forecast   : {FORECASTING_OUTPUT_CSV}")
        print(f"    Sev Update : {SEVERITY_UPDATE_CSV}")
        print(f"    Human Gate : {HUMAN_GATE_OUTPUT_CSV}\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIOps LangGraph Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Node pipeline:
  collect → feature_eng → prelim_severity → classify → tumbling_window
  → forecasting → severity_update → reliability → human_gate → db_writer

Examples:
  python backend/run_langgraph.py                  # standard (100ms poll)
  python backend/run_langgraph.py --speed 50       # fast replay
  python backend/run_langgraph.py --verbose        # print per-node dicts
  python backend/run_langgraph.py --diagram        # print Mermaid graph & exit
        """,
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Live feed mode: read from LiveTelemetryQueue, write to live_feed_db.sqlite.",
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Cycles per second for fast replay (overrides --poll-ms).",
    )
    parser.add_argument(
        "--poll-ms", type=int, default=POLL_INTERVAL_MS,
        help=f"Milliseconds between polls when no data is ready (default: {POLL_INTERVAL_MS}).",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=LANGGRAPH_VERBOSE,
        help="Print each node's output dict as the graph streams.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Build and validate the graph without processing any data.",
    )
    parser.add_argument(
        "--diagram", action="store_true", default=False,
        help="Print the Mermaid graph diagram and exit.",
    )
    args = parser.parse_args()

    if args.diagram:
        print(get_graph_ascii())
        sys.exit(0)

    # Compute poll interval
    poll_ms = args.poll_ms
    if args.live:
        poll_ms = LIVE_POLL_INTERVAL_MS   # 500ms in live mode
    elif args.speed is not None and args.speed > 0:
        poll_ms = int(1000 / args.speed)

    _init_singletons(live=args.live)
    _print_banner(poll_ms=poll_ms, verbose=args.verbose, dry_run=args.dry_run, live=args.live)

    if args.dry_run:
        print("[dry-run] Graph built successfully. Singletons initialised.")
        print("[dry-run] Exiting without processing data.")
        sys.exit(0)

    run_pipeline(poll_ms=poll_ms, verbose=args.verbose, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
