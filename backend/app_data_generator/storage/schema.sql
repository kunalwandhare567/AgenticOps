-- =============================================================================
-- schema.sql
-- AIOps Incident Management — SQLite Schema
-- =============================================================================
-- Tables:
--   Raw telemetry (from simulator):
--     metrics, logs, traces, severity
--   Node outputs (Option 2 — dedicated per-node tables):
--     node_feature_engineering, node_preliminary_severity,
--     node_classification, node_tumbling_window, node_forecasting,
--     node_severity_update, node_human_gate
--   Combined pipeline snapshot (Option 3):
--     pipeline_results
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

-- =============================================================================
-- RAW TELEMETRY TABLES (written by run_simulator.py)
-- =============================================================================

CREATE TABLE IF NOT EXISTS metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id               TEXT    NOT NULL,
    failure_mode             TEXT    NOT NULL,
    service                  TEXT,
    source                   TEXT,
    elapsed_s                REAL,
    timestamp                REAL,
    active_connections       REAL,
    cache_hit_rate           REAL,
    cache_miss_rate          REAL,
    circuit_breaker_state    TEXT,
    cpu_saturation           REAL,
    cpu_utilization          REAL,
    db_connection_pool       REAL,
    db_connection_wait       REAL,
    db_p99                   REAL,
    disk_read_latency        REAL,
    disk_write_latency       REAL,
    error_rate               REAL,
    gc_pause_p99             REAL,
    heap_mb                  REAL,
    http_4xx_rate            REAL,
    http_5xx_rate            REAL,
    iops_utilization         REAL,
    memory_utilization       REAL,
    network_errors           REAL,
    p50_latency              REAL,
    p95_latency              REAL,
    p99_latency              REAL,
    queue_lag                REAL,
    retry_count_per_request  REAL,
    rps                      REAL,
    thread_pool_queue        REAL,
    upstream_timeout_rate    REAL,
    UNIQUE(episode_id, timestamp)
);

CREATE TABLE IF NOT EXISTS logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      TEXT NOT NULL,
    failure_mode    TEXT,
    service         TEXT,
    elapsed_s       REAL,
    timestamp       REAL,
    log_level       TEXT,
    exception_type  TEXT,
    log_message     TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id          TEXT NOT NULL,
    failure_mode        TEXT,
    service             TEXT,
    elapsed_s           REAL,
    timestamp           REAL,
    span_id             TEXT,
    parent_span_id      TEXT,
    span_name           TEXT,
    db_operation_type   TEXT,
    span_duration_ms    REAL,
    span_status         TEXT,
    peer_service        TEXT,
    service_version     TEXT,
    trace_id            TEXT
);

CREATE TABLE IF NOT EXISTS severity (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id          TEXT NOT NULL,
    timestamp           REAL,
    elapsed_s           REAL,
    failure_mode        TEXT,
    Severity            TEXT,
    RawSeverity         TEXT,
    WeightedScore       REAL,
    CriticalCount       INTEGER,
    WarningCount        INTEGER,
    BlastSize           INTEGER,
    HighRiskMode        INTEGER,
    BlastRadiusGrowing  INTEGER,
    Reason              TEXT,
    RecommendedAction   TEXT,
    UNIQUE(episode_id, timestamp)
);

-- =============================================================================
-- NODE OUTPUT TABLE 1 — Feature Engineering
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_feature_engineering (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                   INTEGER NOT NULL,
    episode_id              TEXT    NOT NULL,
    failure_mode            TEXT,
    timestamp               REAL,
    elapsed_s               REAL,
    -- 27 raw metric features
    cpu_utilization         REAL,
    memory_utilization      REAL,
    heap_mb                 REAL,
    db_p99                  REAL,
    disk_read_latency       REAL,
    disk_write_latency      REAL,
    error_rate              REAL,
    gc_pause_p99            REAL,
    cache_hit_rate          REAL,
    cache_miss_rate         REAL,
    active_connections      REAL,
    network_errors          REAL,
    p50_latency             REAL,
    p95_latency             REAL,
    p99_latency             REAL,
    queue_lag               REAL,
    retry_count_per_request REAL,
    rps                     REAL,
    upstream_timeout_rate   REAL,
    circuit_breaker_state   REAL,
    http_4xx_rate           REAL,
    http_5xx_rate           REAL,
    iops_utilization        REAL,
    thread_pool_queue       REAL,
    cpu_saturation          REAL,
    db_connection_pool      REAL,
    db_connection_wait      REAL,
    -- 5 log features
    log_count               REAL,
    log_max_severity        REAL,
    log_critical_count      REAL,
    log_has_exception       INTEGER,
    log_has_novel_template  INTEGER
);

-- =============================================================================
-- NODE OUTPUT TABLE 2 — Preliminary Severity
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_preliminary_severity (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                INTEGER NOT NULL,
    episode_id           TEXT    NOT NULL,
    failure_mode         TEXT,
    timestamp            REAL,
    elapsed_s            REAL,
    preliminary_severity TEXT,
    severity_raw         TEXT,
    weighted_score       REAL,
    critical_count       INTEGER,
    warning_count        INTEGER,
    blast_size           INTEGER,
    high_risk_mode       INTEGER,
    blast_radius_growing INTEGER,
    reason               TEXT,
    recommended_action   TEXT
);

-- =============================================================================
-- NODE OUTPUT TABLE 3 — Classification
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_classification (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                   INTEGER NOT NULL,
    episode_id              TEXT    NOT NULL,
    failure_mode            TEXT,
    timestamp               REAL,
    elapsed_s               REAL,
    predicted_failure       TEXT,
    prediction_probability  REAL
);

-- =============================================================================
-- NODE OUTPUT TABLE 4 — Tumbling Window
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_tumbling_window (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle            INTEGER NOT NULL,
    episode_id       TEXT    NOT NULL,
    failure_mode     TEXT,
    timestamp        REAL,
    elapsed_s        REAL,
    dominant_state   TEXT,
    vote_distribution TEXT,   -- JSON string  e.g. '{"CPU_SATURATION": 8, "NONE": 2}'
    window_margin    REAL,
    window_full      INTEGER, -- 1 = full, 0 = warming up
    window_size      INTEGER
);

-- =============================================================================
-- NODE OUTPUT TABLE 5 — Forecasting + Convergence
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_forecasting (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                INTEGER NOT NULL,
    episode_id           TEXT    NOT NULL,
    failure_mode         TEXT,
    timestamp            REAL,
    elapsed_s            REAL,
    algorithm_used       TEXT,
    history_steps        INTEGER,
    forecast_horizon_s   REAL,
    time_to_failure      REAL,   -- seconds; NULL = no breach predicted
    earliest_ttf_feature TEXT,
    forecast_confidence  REAL,
    confidence_reason    TEXT,
    threshold_crossed    INTEGER, -- 1 = breach predicted, 0 = not
    feature_ttfs         TEXT,   -- JSON dict  {"heap_mb": 42.5, ...}
    feature_slopes       TEXT,   -- JSON dict  {"heap_mb": 0.003, ...}
    predictions          TEXT,   -- JSON dict  {"heap_mb": [510, 520, ...], ...}
    current_values       TEXT    -- JSON dict  {"heap_mb": 505.2, ...}
);

-- =============================================================================
-- NODE OUTPUT TABLE 6 — Severity Update
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_severity_update (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                INTEGER NOT NULL,
    episode_id           TEXT    NOT NULL,
    failure_mode         TEXT,
    timestamp            REAL,
    elapsed_s            REAL,
    preliminary_severity TEXT,
    forecast_confidence  REAL,
    time_to_failure      REAL,
    earliest_ttf_feature TEXT,
    impact_band          TEXT,   -- 'High' | 'Moderate' | 'None'
    urgency_band         TEXT,   -- 'Imminent' | 'Near' | 'Distant'
    gate_passed          INTEGER,
    candidate_severity   TEXT,
    revised_severity     TEXT,
    is_escalated         INTEGER,
    is_deescalated       INTEGER,
    dwell_count          INTEGER,
    reason               TEXT
);

-- =============================================================================
-- NODE OUTPUT TABLE 7 — Human Gate
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_human_gate (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id           TEXT UNIQUE NOT NULL,
    incident_id         TEXT,
    episode_id          TEXT NOT NULL,
    failure_mode        TEXT,
    failure_label       TEXT,
    old_severity        TEXT,
    new_severity        TEXT,
    final_severity      TEXT,
    decision            TEXT,   -- 'APPROVED' | 'REJECTED' | 'AUTO_APPROVED'
    operator            TEXT,
    reason              TEXT,
    confidence          REAL,
    ttf_seconds         REAL,
    impact_band         TEXT,
    urgency_band        TEXT,
    is_large_jump       INTEGER,
    escalation_summary  TEXT,
    response_ms         INTEGER,
    timeout_seconds     INTEGER,
    created_at          TEXT,
    decided_at          TEXT,
    recorded_at         TEXT NOT NULL
);

-- =============================================================================
-- OPTION 3 — Combined Pipeline Results (one flat row per cycle)
-- =============================================================================

CREATE TABLE IF NOT EXISTS pipeline_results (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle                    INTEGER NOT NULL,
    episode_id               TEXT    NOT NULL,
    failure_mode             TEXT,
    timestamp                REAL,
    elapsed_s                REAL,
    -- Key Feature Engineering metrics (representative subset)
    fe_cpu_utilization       REAL,
    fe_memory_utilization    REAL,
    fe_heap_mb               REAL,
    fe_error_rate            REAL,
    fe_p99_latency           REAL,
    fe_p95_latency           REAL,
    fe_db_p99                REAL,
    fe_queue_lag             REAL,
    fe_log_count             REAL,
    fe_log_critical_count    REAL,
    fe_log_has_exception     INTEGER,
    fe_log_has_novel_template INTEGER,
    -- Preliminary Severity node outputs
    preliminary_severity     TEXT,
    severity_weighted_score  REAL,
    severity_critical_count  INTEGER,
    severity_warning_count   INTEGER,
    severity_blast_size      INTEGER,
    severity_reason          TEXT,
    -- Classification node outputs
    predicted_failure        TEXT,
    prediction_probability   REAL,
    -- Tumbling Window node outputs
    dominant_state           TEXT,
    vote_distribution        TEXT,
    window_margin            REAL,
    window_full              INTEGER,
    window_size              INTEGER,
    -- Forecasting + Convergence node outputs
    forecast_algorithm       TEXT,
    time_to_failure          REAL,
    forecast_confidence      REAL,
    threshold_crossed        INTEGER,
    earliest_ttf_feature     TEXT,
    -- Severity Update node outputs
    revised_severity         TEXT,
    candidate_severity       TEXT,
    impact_band              TEXT,
    urgency_band             TEXT,
    gate_passed              INTEGER,
    is_escalated             INTEGER,
    is_deescalated           INTEGER,
    su_reason                TEXT,
    -- Human Gate outputs (NULL until a review is settled for this episode)
    hg_review_id             TEXT,
    hg_decision              TEXT,
    hg_final_severity        TEXT,
    hg_operator              TEXT,
    hg_response_ms           INTEGER
);
