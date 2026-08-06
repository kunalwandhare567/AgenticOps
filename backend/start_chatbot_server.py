"""
backend/start_chatbot_server.py
================================
Launcher for the QueryLangGraph02 AIOps Chat API.

Starts a separate Uvicorn server on port 8001 so the chatbot
can run alongside the existing Sentinel Dashboard API (port 8080).

Usage:
    cd d:/AIOps_Incident_Management/backend
    python start_chatbot_server.py

Ports:
    8080 -- Existing Sentinel Dashboard API (backend/api/main.py)
    8001 -- QueryLangGraph02 AIOps Chat API  (this script)
    5173 -- Vite React frontend (npm run dev in frontend/)
"""
import os
import sys
from pathlib import Path

# ── Resolve absolute paths ─────────────────────────────────────────────
BACKEND_DIR    = Path(__file__).resolve().parent
QUERY_LG_ROOT  = BACKEND_DIR / "QueryLanggraph" / "QueryLangGraph02-main"

# Data source: live_feed_db has real live inference pipeline data
LIVE_FEED_DB   = BACKEND_DIR / "Simulator" / "live_feed_simulator" / "output" / "live_feed_db.sqlite"
SIMULATOR_DB   = BACKEND_DIR / "Simulator" / "app_data_generator_for_offline" / "output" / "simulator_db.sqlite"

# Use live DB if it has data (> 10KB), otherwise fall back to simulator DB
DATA_DB = LIVE_FEED_DB if LIVE_FEED_DB.exists() and LIVE_FEED_DB.stat().st_size > 10_000 else SIMULATOR_DB

if not QUERY_LG_ROOT.exists():
    print(f"[ERROR] QueryLanggraph directory not found: {QUERY_LG_ROOT}")
    sys.exit(1)

# ── Load .env from QueryLangGraph02-main ──────────────────────────────
try:
    from dotenv import load_dotenv
    env_file = QUERY_LG_ROOT / ".env"
    if env_file.exists():
        load_dotenv(str(env_file), override=True)
        print(f"[INFO] Loaded .env from {env_file}")
except ImportError:
    print("[WARN] python-dotenv not installed. Reading env vars from system environment.")

# ── Set DB path env var — ALWAYS force absolute path ──────────────────
# Never trust .env's AIOPS_DB_PATH value (relative paths break in uvicorn).
# DATA_DB is already an absolute Path resolved from BACKEND_DIR.
os.environ["AIOPS_DB_PATH"] = str(DATA_DB)

# ── Validate API key ───────────────────────────────────────────────────
gemini_key = os.environ.get("GEMINI_API_KEY", "")
openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

if gemini_key:
    print(f"[INFO] Gemini API key loaded: ****{gemini_key[-4:]}")
elif openrouter_key.startswith("sk-or-v1-"):
    print(f"[INFO] OpenRouter key loaded: sk-or-v1-****{openrouter_key[-4:]}")
else:
    print("[INFO] Operating in rule-based fallback mode (no Gemini/OpenRouter key found).")

# ── Validate DB ────────────────────────────────────────────────────────
active_db = Path(os.environ["AIOPS_DB_PATH"])
if active_db.exists() and active_db.stat().st_size > 0:
    print(f"[INFO] Using DB: {active_db} ({active_db.stat().st_size:,} bytes)")
else:
    print(f"[WARN] DB not found or empty: {active_db}")
    print("       Start inference pipeline (run_langgraph.py --live) to populate live data.")

print("\n" + "="*60)
print("  QueryLangGraph02 AIOps Chat API")
print(f"  Port:    8001")
print(f"  DB:      {os.environ.get('AIOPS_DB_PATH')}")
print(f"  Model:   {os.environ.get('LLM_MODEL', 'google/gemini-2.5-flash')}")
print("="*60 + "\n")

# ── Add QueryLangGraph02-main to sys.path so all its modules resolve ──
QUERY_LG_STR = str(QUERY_LG_ROOT)
if QUERY_LG_STR not in sys.path:
    sys.path.insert(0, QUERY_LG_STR)

# ── Change working directory so relative imports inside server.py work ─
os.chdir(QUERY_LG_STR)

# ── Import the FastAPI app object directly (avoids uvicorn string-import) ─
try:
    from api.server import app          # noqa: E402
except Exception as e:
    print(f"[ERROR] Failed to import api.server: {e}")
    print("        Check QueryLangGraph02 dependencies are installed.")
    raise

# ── Start Uvicorn with the app object (not a string) ─────────────────
import uvicorn                          # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        app,                            # pass object, not "api.server:app" string
        host="127.0.0.1",
        port=8001,
        reload=False,                   # reload=True requires string app path
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=True,
    )
