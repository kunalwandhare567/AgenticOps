import os, sys
from pathlib import Path

ROOT = Path('QueryLanggraph/QueryLangGraph02-main').resolve()
SIMULATOR_DB = Path('Simulator/app_data_generator_for_offline/output/simulator_db.sqlite').resolve()

os.environ['AIOPS_DB_PATH'] = str(SIMULATOR_DB)
os.environ['OPENROUTER_API_KEY'] = ''   # blank - will use rule-based fallback

sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'), override=True)

# Re-set DB path explicitly (override .env which might still have old path)
os.environ['AIOPS_DB_PATH'] = str(SIMULATOR_DB)
print(f"DB: {os.environ['AIOPS_DB_PATH']}")

from graph import query_graph_app, create_initial_state

state = create_initial_state('show me recent CPU metrics')
result = query_graph_app.invoke(state)
resp = result.get('final_response', {})
print('STATUS:', resp.get('status'))
print('RECORDS:', result.get('retrieval_metadata', {}).get('total_records_fetched', 0))
print('ANSWER:', str(resp.get('answer', ''))[:200])
