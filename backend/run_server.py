"""
backend/run_server.py
======================
Unified Backend Server Launcher.

Launches both API servers from a single terminal:
  1. Port 8001 -- QueryLangGraph02 Chat API (start_chatbot_server.py)
  2. Port 8080 -- Sentinel Dashboard API    (backend/api/main.py)

Usage:
    cd d:/AIOps_Incident_Management
    python backend/run_server.py
"""
import os
import sys
import time
import subprocess
import signal
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

chatbot_process: subprocess.Popen | None = None


def start_chatbot_api() -> subprocess.Popen:
    """Launch start_chatbot_server.py in a background subprocess."""
    script_path = BACKEND_DIR / "start_chatbot_server.py"
    print(f"[run_server] Starting QueryLangGraph Chatbot API (port 8001)...")
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(BACKEND_DIR),
    )
    return proc


def cleanup(signum=None, frame=None):
    """Gracefully terminate background processes on shutdown."""
    global chatbot_process
    print("\n[run_server] Shutting down backend servers...")
    if chatbot_process and chatbot_process.poll() is None:
        print("[run_server] Terminating Chatbot API (port 8001)...")
        chatbot_process.terminate()
        try:
            chatbot_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            chatbot_process.kill()
    print("[run_server] Shutdown complete.")
    sys.exit(0)


def main():
    global chatbot_process

    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 65)
    print("  Sentinel AIOps Unified Backend Server Launcher")
    print("  Port 8080 -- Sentinel Dashboard API (Main)")
    print("  Port 8001 -- QueryLangGraph Chatbot API (Chat)")
    print("=" * 65 + "\n")

    # 1. Start Port 8001 Chatbot API
    chatbot_process = start_chatbot_api()
    time.sleep(1.5)   # brief pause for chatbot startup logs

    # 2. Start Port 8080 Main Sentinel API
    print("[run_server] Starting Sentinel Dashboard API (port 8080)...")
    import uvicorn
    try:
        from api.main import app
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8080,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        cleanup()
    except Exception as exc:
        print(f"[run_server] Server error: {exc}")
        cleanup()


if __name__ == "__main__":
    main()
