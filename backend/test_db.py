import os, sys
from pathlib import Path

ROOT = Path('QueryLanggraph/QueryLangGraph02-main').resolve()
SIMULATOR_DB = Path('Simulator/app_data_generator_for_offline/output/simulator_db.sqlite').resolve()

os.environ['AIOPS_DB_PATH'] = str(SIMULATOR_DB)
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

# Test the DB connection directly
from persistence.database import DatabaseManager
from persistence.repository import AIOpsRepository

db = DatabaseManager(db_path=str(SIMULATOR_DB))
print(f"DB path: {db.db_path}")

repo = AIOpsRepository(db_manager=db)

# Fetch with no service filter
rows = repo.get_metrics(services=[], metrics=[], limit=5)
print(f"get_metrics (no filter): {len(rows)} rows")
if rows:
    print("Sample:", {k: rows[0][k] for k in list(rows[0].keys())[:6]})

# Fetch with auth-service filter
rows2 = repo.get_metrics(services=['auth-service'], metrics=[], limit=5)
print(f"get_metrics (auth-service): {len(rows2)} rows")

# Direct SQL test
result = db.execute_query("SELECT COUNT(*) as cnt FROM metrics", ())
print(f"Direct COUNT(*) from metrics: {result}")
