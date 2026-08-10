# Simulator Module

The `Simulator` package generates realistic multi-signal failure telemetry (metrics, logs, traces) for offline training and real-time live feed demonstration.

---

## Subpackages

```
Simulator/
├── live_feed_simulator/
│   ├── run_live_feed.py           # Real-time incident telemetry injector
│   ├── live_queue.py              # In-memory LiveTelemetryQueue bridge
│   └── output/
│       └── live_feed_db.sqlite    # Active live feed SQLite database
│
└── app_data_generator_for_offline/
    ├── config.py                  # ALL platform constants & path tokens
    ├── state.py                   # SimulatorState dataclass
    ├── run_simulator.py           # Entry point for offline 120K-row dataset generation
    ├── generators/                # Signal generators (metrics, logs, traces)
    ├── physics/                   # Failure degradation physics models
    └── storage/
        ├── db_writer.py           # SQLite DbWriter storage class
        └── schema.sql             # Full DDL schema for SQLite tables
```

---

## 🚀 Execution

### Run Live Telemetry Simulator (Real-time injection)
```powershell
python backend/Simulator/live_feed_simulator/run_live_feed.py --speed 1
```

### Run Offline Dataset Generator (120,120 rows)
```powershell
python backend/Simulator/app_data_generator_for_offline/run_simulator.py --speed 50
```
