"""
backend/nodes/human_gate/run_human_gate.py
==========================================
Offline Batch Runner — Human Gate Node (Stage 8).

Pipeline position: runs AFTER run_severity_update.py has produced
severity_update_output.csv.

What this runner does
---------------------
1. Reads severity_update_output.csv
2. Filters rows where is_escalated == True
3. For each escalation row:
   a. Builds a HumanReviewRequest (review_builder)
   b. Posts it to the InterruptManager SQLite queue (interrupt_manager)
   c. Blocks for up to HUMAN_GATE_TIMEOUT_SECONDS polling for a decision
      (API process can write a decision; else auto-approve fires)
   d. Computes ApprovalResult via ApprovalEngine
   e. Records the decision in AuditLogger (SQLite + CSV)
4. Writes human_gate_output.csv with final_severity column
5. Prints a summary report

Inputs
------
    pipeline/output/severity_update_output.csv   (is_escalated column)

Outputs
-------
    nodes/human_gate/output/human_gate_output.csv
    nodes/human_gate/output/human_gate_audit.db   (SQLite)

Usage
-----
    python backend/nodes/human_gate/run_human_gate.py

    # Run with FastAPI open so operators can review live:
    # Terminal 1: uvicorn backend.api.main:app --port 8080
    # Terminal 2: python backend/nodes/human_gate/run_human_gate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, timezone

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from app_data_generator.config import (
    SEVERITY_UPDATE_CSV,
    HUMAN_GATE_OUTPUT_CSV,
    HUMAN_GATE_TIMEOUT_SECONDS,
    PIPELINE_OUTPUT_DIR,
)
from nodes.human_gate.escalation_detector import EscalationDetector
from nodes.human_gate.review_builder      import ReviewRequestBuilder
from nodes.human_gate.interrupt_manager   import InterruptManager
from nodes.human_gate.approval_engine     import ApprovalEngine
from nodes.human_gate.audit_logger        import AuditLogger
from nodes.human_gate.timeout_manager     import TimeoutManager


# ---------------------------------------------------------------------------
# Output CSV columns
# ---------------------------------------------------------------------------

_OUTPUT_COLS = [
    "episode_id", "incident_id", "failure_mode",
    "old_severity", "new_severity", "final_severity",
    "decision", "operator", "reason",
    "confidence", "ttf_seconds",
    "impact_band", "urgency_band", "is_large_jump",
    "escalation_summary",
    "response_ms", "timeout_seconds",
    "created_at", "decided_at",
]


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(total_episodes: int, escalation_count: int) -> None:
    print(f"\n{'='*65}")
    print(f"  AIOps Human Gate Runner")
    print(f"  Timeout        : {HUMAN_GATE_TIMEOUT_SECONDS}s per review (then auto-approve)")
    print(f"  Input          : {SEVERITY_UPDATE_CSV.name}")
    print(f"  Output         : {HUMAN_GATE_OUTPUT_CSV.name}")
    print(f"  Total Episodes : {total_episodes:,}")
    print(f"  Escalations    : {escalation_count:,} (require human review)")
    print(f"{'='*65}")
    print(f"  Tip: Start FastAPI server to review live:")
    print(f"       uvicorn backend.api.main:app --port 8080")
    print(f"  Then open: http://localhost:5173 — Human Gate panel appears.")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_human_gate() -> None:
    """
    Main batch runner entry point.

    Reads severity_update_output.csv, processes all escalation rows through
    the Human Gate state machine, and writes results to CSV + SQLite.
    """

    # ── 1. Load input ─────────────────────────────────────────────────────
    if not SEVERITY_UPDATE_CSV.exists():
        print(f"[HumanGate] ERROR: {SEVERITY_UPDATE_CSV} not found.")
        print("  Run: python backend/nodes/severity_update/run_severity_update.py")
        sys.exit(1)

    print("[HumanGate] Loading severity_update_output.csv ...")
    df = pd.read_csv(SEVERITY_UPDATE_CSV)
    df.columns = df.columns.str.strip()

    if df.empty:
        print("[HumanGate] No rows in severity_update_output.csv. Nothing to do.")
        return

    # ── 2. Filter escalation rows ────────────────────────────────────────
    # is_escalated is written by run_severity_update.py (True/False column)
    detector = EscalationDetector()

    if "is_escalated" in df.columns:
        # Primary: use the pre-computed flag from severity_update node
        esc_df = df[df["is_escalated"] == True].copy()
    else:
        # Fallback: recompute from severity columns
        print("[HumanGate] WARNING: 'is_escalated' column missing. Recomputing.")
        esc_df = df[
            df.apply(
                lambda r: detector.needs_review(
                    str(r.get("preliminary_severity", "P4")),
                    str(r.get("revised_severity",     "P4")),
                ),
                axis=1,
            )
        ].copy()

    _print_banner(total_episodes=len(df), escalation_count=len(esc_df))

    if esc_df.empty:
        print("[HumanGate] No escalation rows found. All episodes are stable or de-escalating.")
        print("[HumanGate] Writing empty output CSV ...")
        _write_empty_csv()
        return

    # ── 3. Initialise all subsystems ──────────────────────────────────────
    builder  = ReviewRequestBuilder()
    manager  = InterruptManager()
    engine   = ApprovalEngine()
    logger   = AuditLogger()
    timer    = TimeoutManager()

    output_rows: list[dict] = []

    # ── 4. Process each escalation ────────────────────────────────────────
    print(f"[HumanGate] Processing {len(esc_df):,} escalation(s) ...\n")

    for idx, (_, row) in enumerate(
        tqdm(esc_df.iterrows(), total=len(esc_df), desc="HumanGate"),
        start=1,
    ):
        row_dict = row.to_dict()
        ep_id    = str(row_dict.get("episode_id", f"unknown_{idx}"))

        # ── Step A: Build the review request ─────────────────────────
        request = builder.from_severity_update_row(row_dict)

        print(
            f"\n[HumanGate] Review #{idx}/{len(esc_df)}"
            f"  {request.incident_id}  {request.escalation_summary}"
            f"  (conf={request.confidence:.0%}, ttf={request.ttf_seconds:.1f}s)"
        )
        print(
            f"  Waiting up to {HUMAN_GATE_TIMEOUT_SECONDS}s for operator decision "
            f"| Expires: {request.expires_at}"
        )

        # ── Step B: Post to interrupt queue (write to SQLite) ─────────
        manager.post_review(request)

        # ── Step C: Block — poll for decision or auto-approve ─────────
        settled = manager.poll_for_decision(
            review_id = request.review_id,
            timeout   = HUMAN_GATE_TIMEOUT_SECONDS,
        )

        # ── Step D: Compute ApprovalResult via state machine ──────────
        result = engine.compute_result(settled)

        decision_str = result.decision
        sev_str = (
            f"{result.old_severity} → {result.final_severity}"
            if result.final_severity != result.old_severity
            else f"{result.final_severity} (unchanged)"
        )
        op_str = result.operator if result.operator != "system" else "AUTO"
        print(
            f"  Decision: [{decision_str}] by {op_str}"
            f"  | Final Severity: {sev_str}"
            f"  | Response: {result.response_ms}ms"
        )

        # ── Step E: Record in audit log ───────────────────────────────
        logger.record(result=result, request=request)

        # ── Accumulate output row ─────────────────────────────────────
        output_rows.append({
            "episode_id":          ep_id,
            "incident_id":         result.incident_id,
            "failure_mode":        request.failure_mode,
            "old_severity":        result.old_severity,
            "new_severity":        result.new_severity,
            "final_severity":      result.final_severity,
            "decision":            result.decision,
            "operator":            result.operator,
            "reason":              result.reason,
            "confidence":          request.confidence,
            "ttf_seconds":         request.ttf_seconds,
            "impact_band":         request.impact_band,
            "urgency_band":        request.urgency_band,
            "is_large_jump":       int(request.is_large_jump),
            "escalation_summary":  request.escalation_summary,
            "response_ms":         result.response_ms,
            "timeout_seconds":     request.timeout_seconds,
            "created_at":          request.created_at,
            "decided_at":          result.decided_at,
        })

    # ── 5. Write output CSV ───────────────────────────────────────────────
    _write_output_csv(output_rows)

    # ── 6. Print summary ─────────────────────────────────────────────────
    _print_summary(output_rows, logger)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_output_csv(rows: list[dict]) -> None:
    """Write human_gate_output.csv with all processed rows."""
    if not rows:
        _write_empty_csv()
        return

    out_df = pd.DataFrame(rows)
    cols   = [c for c in _OUTPUT_COLS if c in out_df.columns]
    out_df = out_df[cols]
    HUMAN_GATE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(HUMAN_GATE_OUTPUT_CSV, index=False)
    print(f"\n[HumanGate] Output written: {HUMAN_GATE_OUTPUT_CSV}")


def _write_empty_csv() -> None:
    """Create an empty output CSV with correct headers."""
    import csv as _csv
    HUMAN_GATE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HUMAN_GATE_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=_OUTPUT_COLS)
        writer.writeheader()


def _print_summary(rows: list[dict], logger: AuditLogger) -> None:
    """Print a formatted summary report after processing completes."""
    if not rows:
        print("\n[HumanGate] No escalations processed.")
        return

    total    = len(rows)
    approved = sum(1 for r in rows if r["decision"] == "APPROVED")
    rejected = sum(1 for r in rows if r["decision"] == "REJECTED")
    auto     = sum(1 for r in rows if r["decision"] == "AUTO_APPROVED")

    metrics = logger.get_metrics()
    avg_ms  = metrics.get("avg_response_ms", 0.0)

    print(f"\n{'='*65}")
    print(f"  Human Gate Summary")
    print(f"{'='*65}")
    print(f"  Total Escalations Reviewed  : {total:>6,}")
    print(f"  Approved by Human           : {approved:>6,}  ({approved/total*100:.1f}%)")
    print(f"  Rejected by Human           : {rejected:>6,}  ({rejected/total*100:.1f}%)")
    print(f"  Auto-Approved (Timeout)     : {auto:>6,}  ({auto/total*100:.1f}%)")
    print(f"  Avg Human Response Time     : {avg_ms:>6.0f} ms")
    print(f"  False Escalations (Rejected): {rejected:>6,}")
    print(f"{'='*65}")

    # Final severity distribution
    from collections import Counter
    dist = Counter(r["final_severity"] for r in rows)
    print(f"\n  Final Severity Distribution:")
    for sev in ["P1", "P2", "P3", "P4"]:
        cnt = dist.get(sev, 0)
        bar = "█" * int(cnt / max(dist.values(), 1) * 20)
        print(f"    {sev}  {bar:<20} {cnt:>5,}")

    print(f"\n  Output  : {HUMAN_GATE_OUTPUT_CSV}")
    print(f"  Audit DB: {logger._db_path}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_human_gate()
