"""
backend/live_feed_simulator/run_live_feed.py
=============================================
AIOps Live Feed Simulator — Real-Time Telemetry Generator.

Generates an infinite stream of realistic failure episodes using a
Markov transition matrix. Each episode consists of 120 ticks at
0.5s intervals (~60 seconds per episode).

Usage:
    # Run indefinitely (Ctrl+C to stop):
    python live_feed_simulator/run_live_feed.py

    # Stop after N episodes:
    python live_feed_simulator/run_live_feed.py --max-episodes 5

    # Reproducible session (same seed = same mode sequence):
    python live_feed_simulator/run_live_feed.py --seed 42

    # Faster ticks for testing (0.1s per tick):
    python live_feed_simulator/run_live_feed.py --tick 0.1

Output:
    live_feed_simulator/output/live_feed_db.sqlite  (always APPENDS)
    live_feed_simulator/output/live_metrics.csv     (versioned)
    live_feed_simulator/output/live_logs.csv        (versioned)
    live_feed_simulator/output/live_traces.csv      (versioned)

The live feed also pushes each tick to LiveTelemetryQueue so
run_langgraph.py --live can process them in the same process.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── Resolve package root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent   # d:/AIOps_Incident_Management/backend
sys.path.insert(0, str(PROJECT_ROOT))

# ── Force UTF-8 output ────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from Simulator.app_data_generator_for_offline.config import (
    SERVICES, SERVICE_VERSIONS,
    STEPS_PER_EPISODE,
    LIVE_FEED_DB_PATH, LIVE_FEED_OUTPUT_DIR, LIVE_STEP_INTERVAL_S,
)
from Simulator.app_data_generator_for_offline.state import SimulatorState
from Simulator.app_data_generator_for_offline.physics import load_scenario
from Simulator.app_data_generator_for_offline.physics.distributions import Dist
from Simulator.app_data_generator_for_offline.generators.metrics_generator import MetricsGenerator
from Simulator.app_data_generator_for_offline.generators.log_generator import LogGenerator
from Simulator.app_data_generator_for_offline.generators.trace_generator import TraceGenerator
from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter
from Simulator.app_data_generator_for_offline.storage.csv_writer import CsvWriter

from Simulator.live_feed_simulator.world_scenario_engine import WorldScenarioEngine
from Simulator.live_feed_simulator.live_queue import LiveTelemetryQueue


# =============================================================================
# Banner + progress helpers
# =============================================================================

def _print_banner(tick_s: float, max_ep: int | None, seed: int | None) -> None:
    ep_str = str(max_ep) if max_ep else "∞ (until Ctrl+C)"
    seed_str = str(seed) if seed is not None else "random"
    print(f"\n{'=' * 68}")
    print(f"  AIOps Live Feed Simulator")
    print(f"{'=' * 68}")
    print(f"  Mode sequencing  : Markov probabilistic (real-world cascades)")
    print(f"  Episodes         : {ep_str}")
    print(f"  Steps/episode    : {STEPS_PER_EPISODE}  ({tick_s:.2f}s/tick = {STEPS_PER_EPISODE * tick_s:.0f}s/ep)")
    print(f"  Seed             : {seed_str}")
    print(f"  Live DB          : {LIVE_FEED_DB_PATH}  [APPEND MODE]")
    print(f"  CSV output       : {LIVE_FEED_OUTPUT_DIR}/")
    print(f"{'=' * 68}")
    print(f"  Make sure run_langgraph.py --live is running in Terminal 2.")
    print(f"  Press Ctrl+C to stop. All data written so far is already saved.\n")


def _print_episode_header(
    ep_idx: int, mode: str, prev_mode: str, engine_summary: str
) -> None:
    arrow = f"  [{prev_mode}] → [{mode}]" if prev_mode else f"  [START] → [{mode}]"
    print(f"\n  ── Episode {ep_idx + 1} ─────────────────────────────────────────────")
    print(f"  World: {arrow}")
    print(f"  Chain: {engine_summary}")
    print(f"  {'─' * 60}")


def _print_tick_progress(
    ep_idx: int, mode: str, step: int, total: int,
    tick_s: float, queue_sz: int, metric: dict,
) -> None:
    pct = int(step * 100 / total)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(
        f"\r  ep={ep_idx + 1} [{bar}] {pct:3d}%  "
        f"mode={mode:<22}  step={step:3d}/{total}  "
        f"cpu={metric.get('cpu_utilization', 0):5.1f}%  "
        f"err={metric.get('error_rate', 0) * 100:4.1f}%  "
        f"queue={queue_sz:4d}",
        end="",
        flush=True,
    )


# =============================================================================
# Main simulator
# =============================================================================

def run(
    tick_s: float = LIVE_STEP_INTERVAL_S,
    max_episodes: int | None = None,
    seed: int | None = None,
) -> None:
    """
    Main live feed loop.

    Args:
        tick_s:        Seconds between ticks (default 0.5).
        max_episodes:  Stop after this many episodes (None = infinite).
        seed:          numpy random seed for reproducibility.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    LIVE_FEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng  = np.random.default_rng(seed)
    dist = Dist(rng)

    # Use the live DB path — APPEND mode (no truncation)
    db  = DbWriter(LIVE_FEED_DB_PATH)
    csv = CsvWriter(LIVE_FEED_OUTPUT_DIR)
    db.setup()
    csv.setup()

    metrics_gen = MetricsGenerator()
    log_gen     = LogGenerator()
    trace_gen   = TraceGenerator()

    engine = WorldScenarioEngine(seed=seed)
    _print_banner(tick_s, max_episodes, seed)

    # Pre-load all physics scenarios (fast, avoid per-episode reload)
    from Simulator.app_data_generator_for_offline.config import ALL_MODES
    scenarios = {mode: load_scenario(mode) for mode in ALL_MODES}

    global_row   = 0
    session_start = time.monotonic()
    prev_mode    = ""
    ep_counter   = 0  # running global episode index (across all runs this session)

    # ── Session counter: find max existing episode index in DB for append numbering
    try:
        import sqlite3
        if LIVE_FEED_DB_PATH.exists():
            with sqlite3.connect(str(LIVE_FEED_DB_PATH)) as _conn:
                cur = _conn.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM metrics"
                    if "metrics" in [r[0] for r in _conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()] else "SELECT 0"
                )
                row_count = cur.fetchone()[0]
                ep_counter = row_count if row_count else 0
    except Exception:
        ep_counter = 0

    try:
        for mode, local_ep_idx in engine.generate_session(max_episodes=max_episodes):
            scenario = scenarios[mode]
            ep_id    = f"live_ep_{ep_counter:06d}_{mode}"
            service  = random.choice(SERVICES)
            svc_ver  = random.choice(SERVICE_VERSIONS)

            _print_episode_header(
                local_ep_idx, mode, prev_mode, engine.transition_summary()
            )

            state = SimulatorState(
                episode_id      = ep_id,
                failure_mode    = mode,
                service         = service,
                service_version = svc_ver,
                onset_delay_s   = scenario.onset_delay_s(dist.rng),
            )

            # ── Inner tick loop ───────────────────────────────────────────────
            for step in range(STEPS_PER_EPISODE):
                t0 = time.perf_counter()

                state.timestamp = time.time()
                state.elapsed_s = step * tick_s     # use live tick_s, not 2.0
                state.step      = step

                # Physics simulation
                state = scenario.apply(state, dist)

                # Generate telemetry
                metric = metrics_gen.generate(state)
                log    = log_gen.generate(state)
                spans  = trace_gen.generate(state, rng)

                # Persist to live_feed_db.sqlite (APPEND)
                db.write_tick(metric, log, spans)
                csv.write_tick(metric, log, spans)

                # Push to in-process queue for LangGraph pipeline
                LiveTelemetryQueue.push(metric, log, spans)

                global_row += 1

                # Live progress bar (update every tick)
                _print_tick_progress(
                    local_ep_idx, mode, step + 1, STEPS_PER_EPISODE,
                    tick_s, LiveTelemetryQueue.size(), metric,
                )

                # Pace to tick interval
                elapsed = time.perf_counter() - t0
                sleep_s = max(0.0, tick_s - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            print()  # newline after progress bar
            print(f"  Episode {ep_counter + 1} complete: {mode}  ({STEPS_PER_EPISODE} ticks, {STEPS_PER_EPISODE * tick_s:.0f}s)")
            prev_mode = mode
            ep_counter += 1

    except KeyboardInterrupt:
        print(f"\n\n  [LiveFeed] Stopped by user after {global_row:,} rows / {ep_counter} episodes.")

    finally:
        db.close()
        runtime = time.monotonic() - session_start
        print(f"\n{'=' * 68}")
        print(f"  LIVE FEED SESSION COMPLETE")
        print(f"{'=' * 68}")
        print(f"  Total ticks generated : {global_row:,}")
        print(f"  Total episodes        : {ep_counter}")
        print(f"  Session runtime       : {runtime:.1f}s")
        print(f"  Avg ticks/sec         : {global_row / max(runtime, 1):.1f}")
        print(f"  Transition chain      : {engine.transition_summary()}")
        print(f"\n  Data saved (APPENDED):")
        print(f"    SQLite : {LIVE_FEED_DB_PATH}")
        print(f"  Queue depth remaining : {LiveTelemetryQueue.size()} items")
        print(f"{'=' * 68}\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIOps Live Feed Simulator — real-world Markov failure cascades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_feed_simulator/run_live_feed.py                      # infinite
  python live_feed_simulator/run_live_feed.py --max-episodes 5     # 5 episodes
  python live_feed_simulator/run_live_feed.py --seed 42            # reproducible
  python live_feed_simulator/run_live_feed.py --tick 0.1           # 0.1s ticks
        """,
    )
    parser.add_argument(
        "--tick", type=float, default=LIVE_STEP_INTERVAL_S,
        help=f"Seconds per tick (default {LIVE_STEP_INTERVAL_S}s = 0.5s)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Stop after this many episodes (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible mode sequence",
    )
    args = parser.parse_args()
    run(tick_s=args.tick, max_episodes=args.max_episodes, seed=args.seed)


if __name__ == "__main__":
    main()
