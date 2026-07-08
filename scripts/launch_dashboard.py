from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def normalized_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """
    Return a Windows-safe environment with one canonical Path key.

    Some shells can expose both Path and PATH. PowerShell's Env provider and
    Start-Process can crash when that happens, so this launcher passes a clean
    environment directly to subprocess.Popen.
    """
    source = dict(source or os.environ)
    env: dict[str, str] = {}
    path_value = ""
    for key, value in source.items():
        if key.lower() == "path":
            if value and len(value) > len(path_value):
                path_value = value
            continue
        if key.lower() not in {existing.lower() for existing in env}:
            env[key] = value
    if path_value:
        env["Path" if os.name == "nt" else "PATH"] = path_value
    return env


def _health_ok(port: int) -> bool:
    try:
        with urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as response:
            return response.status == 200 and response.read().decode("utf-8", errors="ignore").strip() == "ok"
    except (OSError, URLError):
        return False


def _stop_pid(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except OSError:
        pass


def launch(port: int, stop_existing: bool = True) -> int:
    pid_file = ROOT / f"streamlit-{port}.pid"
    out_log = ROOT / f"streamlit-{port}.out.log"
    err_log = ROOT / f"streamlit-{port}.err.log"

    if stop_existing:
        _stop_pid(pid_file)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    with out_log.open("w", encoding="utf-8") as stdout, err_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=normalized_env(),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    pid_file.write_text(str(process.pid), encoding="utf-8")

    for _ in range(30):
        if _health_ok(port):
            print(f"Streamlit is healthy on http://localhost:{port} pid={process.pid}")
            return 0
        time.sleep(1)

    print(f"Streamlit did not become healthy on port {port}. See {err_log.name} and {out_log.name}.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the PA-11R Streamlit dashboard with a normalized environment.")
    parser.add_argument("--port", type=int, default=8504)
    parser.add_argument("--no-stop", action="store_true", help="Do not stop the PID recorded for this port before launch.")
    args = parser.parse_args()
    return launch(args.port, stop_existing=not args.no_stop)


if __name__ == "__main__":
    raise SystemExit(main())
