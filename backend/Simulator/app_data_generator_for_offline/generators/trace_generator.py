"""
app_simulator/generators/trace_generator.py
============================================
Produces 3-4 span dicts per tick from SimulatorState.

Span topology is built from scratch here (ported from generate_full_dataset.py
build_trace_rows, lines 238-396). Each call generates one trace with:
  - 1 root span (HTTP request)
  - 1-2 DB spans (child of root)
  - 0-1 cache span (child of root)
  - 0-1 downstream span (child of root)

All spans share state.timestamp — SYNC FIELD set once at tick start.
All spans share the same trace_id (one trace per tick).

Span durations are driven by state metric fields so they reflect
the current failure mode (e.g., db_p99 in DB_SLOWDOWN, gc_pause in MEMORY_LEAK).
"""
from __future__ import annotations
import uuid
import numpy as np
from ..state import SimulatorState
from ..config import DOWNSTREAM_SERVICES


# Span name pools by service layer
_SPAN_NAMES_HTTP = [
    "GET /api/order", "POST /api/payment", "GET /api/auth",
    "PUT /api/inventory", "GET /api/health", "POST /api/checkout",
]
_SPAN_NAMES_DB = [
    "db.query.select", "db.query.insert", "db.query.update",
    "db.transaction", "db.query.delete",
]
_SPAN_NAMES_CACHE = ["cache.get", "cache.set", "cache.delete"]
_SPAN_NAMES_DOWNSTREAM = [
    "http.client.get", "http.client.post", "grpc.call",
]
_DB_OP_TYPES = ["READ", "WRITE", "READ", "READ", "WRITE"]  # weighted toward reads


class TraceGenerator:
    """Stateless — call generate(state, rng) every tick."""

    def generate(self, state: SimulatorState, rng: np.random.Generator) -> list[dict]:
        """
        Build one trace (3-4 spans) reflecting the current state.

        Returns a list of span dicts, each conforming to TRACE_FIELDS schema.
        All spans share: episode_id, failure_mode, service, elapsed_s, timestamp, trace_id.
        """
        trace_id  = uuid.uuid4().hex
        root_id   = uuid.uuid4().hex
        spans: list[dict] = []

        # ── Root span (HTTP handler) ──────────────────────────────────────────
        root_dur = float(np.clip(
            state.p99_latency + rng.normal(0, state.p99_latency * 0.1),
            max(1.0, state.p50_latency * 0.8),
            state.p99_latency * 2.0
        ))
        root_status = "ERROR" if rng.random() < state.error_rate else "OK"
        spans.append(self._span(
            state, trace_id,
            span_id        = root_id,
            parent_span_id = "",
            span_name      = _SPAN_NAMES_HTTP[rng.integers(len(_SPAN_NAMES_HTTP))],
            db_op          = "",
            duration_ms    = root_dur,
            status         = root_status,
            peer_service   = state.service,
            svc_version    = state.service_version,
        ))

        # ── DB span(s) (child of root) ────────────────────────────────────────
        n_db = 2 if rng.random() < 0.6 else 1
        for _ in range(n_db):
            db_dur = float(np.clip(
                state.db_p99 * rng.uniform(0.7, 1.3) + rng.normal(0, 5),
                1.0, state.db_p99 * 2.0
            ))
            db_op     = _DB_OP_TYPES[rng.integers(len(_DB_OP_TYPES))]
            db_status = "ERROR" if rng.random() < state.error_rate * 0.5 else "OK"
            spans.append(self._span(
                state, trace_id,
                span_id        = uuid.uuid4().hex,
                parent_span_id = root_id,
                span_name      = _SPAN_NAMES_DB[rng.integers(len(_SPAN_NAMES_DB))],
                db_op          = db_op,
                duration_ms    = db_dur,
                status         = db_status,
                peer_service   = "user-db",
                svc_version    = state.service_version,
            ))

        # ── Cache span (child of root, 70% chance) ────────────────────────────
        if rng.random() < 0.70:
            # Cache hit → fast (1-5ms); cache miss → slow (30-80ms)
            if rng.random() < state.cache_miss_rate:
                cache_dur = float(np.clip(rng.normal(50, 15), 10, 200))
            else:
                cache_dur = float(np.clip(rng.normal(2, 1), 0.5, 8))
            spans.append(self._span(
                state, trace_id,
                span_id        = uuid.uuid4().hex,
                parent_span_id = root_id,
                span_name      = _SPAN_NAMES_CACHE[rng.integers(len(_SPAN_NAMES_CACHE))],
                db_op          = "",
                duration_ms    = cache_dur,
                status         = "OK",
                peer_service   = "cache-cluster",
                svc_version    = state.service_version,
            ))

        # ── Downstream call (child of root, 40% chance) ───────────────────────
        if rng.random() < 0.40:
            timeout_fired = rng.random() < state.upstream_timeout_rate
            down_dur = float(np.clip(
                rng.normal(5000, 200) if timeout_fired else rng.normal(30, 10),
                1, 10000
            ))
            down_svc    = DOWNSTREAM_SERVICES[rng.integers(len(DOWNSTREAM_SERVICES))]
            down_status = "ERROR" if timeout_fired or rng.random() < state.error_rate else "OK"
            spans.append(self._span(
                state, trace_id,
                span_id        = uuid.uuid4().hex,
                parent_span_id = root_id,
                span_name      = _SPAN_NAMES_DOWNSTREAM[rng.integers(len(_SPAN_NAMES_DOWNSTREAM))],
                db_op          = "",
                duration_ms    = down_dur,
                status         = down_status,
                peer_service   = down_svc,
                svc_version    = state.service_version,
            ))

        return spans

    @staticmethod
    def _span(
        state: SimulatorState,
        trace_id: str,
        span_id: str,
        parent_span_id: str,
        span_name: str,
        db_op: str,
        duration_ms: float,
        status: str,
        peer_service: str,
        svc_version: str,
    ) -> dict:
        return {
            "episode_id":        state.episode_id,
            "failure_mode":      state.failure_mode,
            "service":           state.service,
            "elapsed_s":         round(state.elapsed_s, 1),
            "timestamp":         state.timestamp,       # ← SYNC FIELD
            "span_id":           span_id,
            "parent_span_id":    parent_span_id,
            "span_name":         span_name,
            "db_operation_type": db_op,
            "span_duration_ms":  round(duration_ms, 2),
            "span_status":       status,
            "peer_service":      peer_service,
            "service_version":   svc_version,
            "trace_id":          trace_id,
        }
