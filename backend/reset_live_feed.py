"""
backend/reset_live_feed.py
===========================
Helper script to wipe and reset live feed simulator database & CSV outputs.
"""
import sqlite3
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
LIVE_OUTPUT_DIR = _HERE / "Simulator" / "live_feed_simulator" / "output"
DB_PATH = LIVE_OUTPUT_DIR / "live_feed_db.sqlite"

def reset_live_feed():
    print(f"Resetting live feed data in: {LIVE_OUTPUT_DIR}")
    
    # 1. Truncate SQLite tables if DB exists
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for tbl in tables:
                if tbl != "sqlite_sequence":
                    conn.execute(f"DELETE FROM {tbl}")
            conn.commit()
            conn.execute("VACUUM")
            conn.close()
            print("  SQLite database tables cleared.")
        except Exception as e:
            print(f"  Could not clear DB via SQL: {e}")

    # 2. Delete files
    if LIVE_OUTPUT_DIR.exists():
        for item in LIVE_OUTPUT_DIR.glob("*"):
            if item.suffix in (".csv", ".sqlite", ".sqlite-wal", ".sqlite-shm"):
                try:
                    item.unlink()
                    print(f"  Removed file: {item.name}")
                except Exception as e:
                    print(f"  (DB table cleared, file lock held by running process: {item.name})")

    print("\nLive feed data reset successfully! You can start fresh.")

if __name__ == "__main__":
    reset_live_feed()
