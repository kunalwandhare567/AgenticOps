import sqlite3
db = r'D:\AIOps_Incident_Management\backend\Simulator\live_feed_simulator\output\live_feed_db.sqlite'
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    count = cur.fetchone()[0]
    print(f"{t}: {count} rows")
conn.close()
