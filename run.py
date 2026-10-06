"""
Convenience launcher: starts the FastAPI backend and the Streamlit
frontend together.

    python run.py

Equivalent to running these in two terminals:
    uvicorn app.api.main:app --host 127.0.0.1 --port 8000
    streamlit run app/frontend/streamlit_app.py

Use --api-only or --frontend-only to run just one of them (useful when
you want separate terminals/logs, which is also fine and often clearer
for development).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from app.config import settings

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanLens")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()

    processes = []

    try:
        if not args.frontend_only:
            print(f"Starting API on http://{settings.api_host}:{settings.api_port} ...")
            api_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "app.api.main:app",
                    "--host", settings.api_host, "--port", str(settings.api_port),
                ],
                cwd=str(ROOT),
            )
            processes.append(api_proc)
            time.sleep(2)  # give the API a moment to bind before Streamlit tries to call it

        if not args.api_only:
            print("Starting Streamlit frontend ...")
            frontend_proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app/frontend/streamlit_app.py"],
                cwd=str(ROOT),
            )
            processes.append(frontend_proc)

        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
