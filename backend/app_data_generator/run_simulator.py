"""
app_simulator/run_simulator.py
================================
AIOps Live Telemetry Simulator — generates ALL 13 failure modes.

Generates data for ALL failure modes automatically:
  13 modes x EPISODES_PER_MODE episodes x STEPS_PER_EPISODE steps
  = 120,120 total metric rows (matches DEVOPS generate_full_dataset.py)

Usage:
    python app_simulator/run_simulator.py
    python app_simulator/run_simulator.py --speed 5.0
    python app_simulator/run_simulator.py --episodes 10    # quick test (10 eps/mode)
    python app_simulator/run_simulator.py --speed 50 --episodes 77

Press Ctrl+C at ANY time to stop. Data written so far is already saved.

Output:
    output/metrics_N.csv      (versioned, new N on each run)
    output/logs_N.csv
    output/traces_N.csv
    simulator_db.sqlite       (cumulative, appends across runs)

Behaviour:
  - Iterates ALL_MODES in sequence.
  - For each mode: runs EPISODES_PER_MODE episodes of STEPS_PER_EPISODE steps.
  - Each tick: writes to SQLite + versioned CSV + pushes to TelemetryQueue.
  - Prints progress every 1000 rows and at every episode boundary.
  - Prints final summary when all modes are done or Ctrl+C is pressed.
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np

# ── resolve package root ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app_data_generator.config import (
    ALL_MODES, DB_PATH, OUTPUT_DIR, SERVICES, SERVICE_VERSIONS,
    STEP_INTERVAL_SECONDS, STEPS_PER_EPISODE, EPISODES_PER_MODE, TARGET_ROWS,
)
from app_data_generator.state import SimulatorState
from app_data_generator.physics import load_scenario
from app_data_generator.physics.distributions import Dist
from app_data_generator.generators.metrics_generator import MetricsGenerator
from app_data_generator.generators.log_generator import LogGenerator
from app_data_generator.generators.trace_generator import TraceGenerator
from app_data_generator.storage.db_writer import DbWriter
from app_data_generator.storage.csv_writer import CsvWriter
from nodes.collect.queue_bridge import TelemetryQueue


# ─────────────────────────────────────────────────────────────────────────────
def _new_episode_id(failure_mode: str, ep_index: int) -> str:
    return f"ep_{ep_index:05d}_{failure_mode}"


def _print_banner(speed: float, interval: float, episodes_per_mode: int) -> None:
    total = len(ALL_MODES) * episodes_per_mode * STEPS_PER_EPISODE
    print(f"\n{'=' * 65}")
    print(f"  AIOps Live Simulator  --  ALL {len(ALL_MODES)} Failure Modes")
    print(f"  Episodes/mode: {episodes_per_mode}  |  Steps/episode: {STEPS_PER_EPISODE}")
    print(f"  Total rows target: {total:,}")
    print(f"  Speed: {speed:.1f}x  |  Tick interval: {interval:.2f}s")
    print(f"  DB   : {DB_PATH}")
    print(f"  CSV  : {OUTPUT_DIR}/")
    print(f"{'=' * 65}")
    print("  Press Ctrl+C to stop at any time. Data is already saved.\n")


def _print_mode_header(mode: str, mode_idx: int, total_modes: int, ep_count: int) -> None:
    print(f"\n  [{mode_idx:02d}/{total_modes}] Failure mode: {mode}  ({ep_count} episodes)")
    print(f"       {'-'*50}")


def _print_progress(global_row: int, total: int, mode: str, ep_id: str, step: int,
                    metric: dict) -> None:
    pct = min(100, global_row * 100 // total)
    print(
        f"  row={global_row:>7,}/{total:,} ({pct:>3}%)  "
        f"mode={mode:<22}  ep={ep_id[-10:]}  "
        f"step={step:>3}  "
        f"cpu={metric.get('cpu_utilization', 0):>5.1f}%  "
        f"p99={metric.get('p99_latency', 0):>6.0f}ms  "
        f"err={metric.get('error_rate', 0)*100:>4.1f}%"
    )


def _print_mode_summary(mode: str, rows: int, ep_count: int) -> None:
    print(f"\n       Done: {mode}  rows={rows:,}  episodes={ep_count}")


# ─────────────────────────────────────────────────────────────────────────────
def run(speed: float = 1.0, episodes_per_mode: int = 50) -> None:
    interval      = STEP_INTERVAL_SECONDS / speed
    total_target  = len(ALL_MODES) * episodes_per_mode * STEPS_PER_EPISODE

    # ── One-time setup ───────────────────────────────────────────────────────
    rng  = np.random.default_rng()   # fresh seed every run = realistic variation
    dist = Dist(rng)

    db  = DbWriter(DB_PATH)
    csv = CsvWriter(OUTPUT_DIR)
    db.setup()
    csv.setup()

    metrics_gen = MetricsGenerator()
    log_gen     = LogGenerator()
    trace_gen   = TraceGenerator()

    _print_banner(speed, interval, episodes_per_mode)

    # Cache all physics scenarios
    scenarios = {mode: load_scenario(mode) for mode in ALL_MODES}

    # Generate flat list of episodes and shuffle them randomly
    episodes_to_run = [(mode, ep_num) for mode in ALL_MODES for ep_num in range(episodes_per_mode)]
    random.shuffle(episodes_to_run)
    total_episodes = len(episodes_to_run)

    global_row   = 0
    mode_summary = {mode: 0 for mode in ALL_MODES}

    try:
        # ── Shuffled Episode Loop ────────────────────────────────────────────
        for global_ep, (failure_mode, ep_num) in enumerate(episodes_to_run, start=1):
            scenario = scenarios[failure_mode]
            ep_id   = _new_episode_id(failure_mode, global_ep)
            service = random.choice(SERVICES)
            svc_ver = random.choice(SERVICE_VERSIONS)

            print(f"\n  [Episode {global_ep}/{total_episodes}] failure_mode={failure_mode:<22} (id={ep_id})")
            print(f"       {'-'*65}")

            state = SimulatorState(
                episode_id      = ep_id,
                failure_mode    = failure_mode,
                service         = service,
                service_version = svc_ver,
            )

            # ── Inner loop: STEPS_PER_EPISODE steps per episode ──────────
            for step in range(STEPS_PER_EPISODE):
                t0 = time.perf_counter()

                # Synchronisation point — set timestamp ONCE per tick
                state.timestamp = time.time()
                state.elapsed_s = step * STEP_INTERVAL_SECONDS
                state.step      = step

                # Physics
                state = scenario.apply(state, dist)

                # Generate telemetry
                metric = metrics_gen.generate(state)
                log    = log_gen.generate(state)
                spans  = trace_gen.generate(state, rng)

                # Persist to SQLite + versioned CSV
                db.write_tick(metric, log, spans)
                csv.write_tick(metric, log, spans)

                # Push to in-process queue for pipeline consumer
                TelemetryQueue.push(metric, log, spans)

                global_row += 1
                mode_summary[failure_mode] += 1

                # Progress every 1000 rows
                if global_row % 1000 == 0:
                    _print_progress(global_row, total_target, failure_mode,
                                    ep_id, step, metric)

                # Pace to interval
                elapsed = time.perf_counter() - t0
                sleep_s = max(0.0, interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    except KeyboardInterrupt:
        print(f"\n\n  [Simulator] Stopped by user after {global_row:,} rows.")

    finally:
        db.close()

        # ── Final summary ────────────────────────────────────────────────────
        print(f"\n{'=' * 65}")
        print(f"  SIMULATION COMPLETE")
        print(f"  {'Mode':<25}  {'Rows':>8}")
        print(f"  {'-'*35}")
        for mode, cnt in mode_summary.items():
            print(f"  {mode:<25}  {cnt:>8,}")
        print(f"  {'-'*35}")
        print(f"  {'TOTAL':<25}  {global_row:>8,}")
        print(f"\n  Data saved:")
        print(f"    SQLite : {DB_PATH}")
        for name, path in csv.paths.items():
            print(f"    CSV    : {path}")
        print(f"  Queue  : {TelemetryQueue.size()} items waiting")
        print(f"{'=' * 65}\n")
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AIOps Live Simulator — generates ALL 13 failure modes.\n"
            "13 modes x EPISODES x 120 steps = up to 120,120 rows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="N",
        help="Speed multiplier (default 1.0=real-time, 50.0=50x faster)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_MODE,
        metavar="N",
        help=f"Episodes per failure mode (default {EPISODES_PER_MODE} = 120,120 total rows)",
    )
    args = parser.parse_args()
    run(speed=args.speed, episodes_per_mode=args.episodes)


if __name__ == "__main__":
    main()
