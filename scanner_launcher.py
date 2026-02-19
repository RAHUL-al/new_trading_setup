"""
scanner_launcher.py — Master launcher for the stock scanning system.

Usage:
    python scanner_launcher.py

This script:
1. Runs stock_scanner_setup.py to prepare token lists
2. Launches one scanner_worker.py process per credential
3. Launches gap_analyzer.py process
4. Monitors all processes and restarts on crash
5. Stops everything at 3:35 PM
"""

import datetime
import json
import os
import subprocess
import sys
import signal
import time

import pytz
from logzero import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDIA_TZ = pytz.timezone("Asia/Kolkata")
PYTHON = sys.executable  # Use the same Python interpreter


def load_credentials() -> list:
    """Load credential names."""
    cred_path = os.path.join(BASE_DIR, "scanner_credentials.json")
    with open(cred_path, 'r') as f:
        return json.load(f)


def run_setup():
    """Run stock_scanner_setup.py to prepare token lists."""
    setup_script = os.path.join(BASE_DIR, "stock_scanner_setup.py")
    logger.info("Running stock_scanner_setup.py...")
    result = subprocess.run(
        [PYTHON, setup_script],
        capture_output=True, text=True, cwd=BASE_DIR
    )
    if result.returncode != 0:
        logger.error(f"Setup failed:\n{result.stderr}")
        sys.exit(1)
    logger.info("Setup completed successfully.")
    logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)


def start_worker(cred_name: str) -> subprocess.Popen:
    """Start a scanner_worker.py subprocess."""
    worker_script = os.path.join(BASE_DIR, "scanner_worker.py")
    proc = subprocess.Popen(
        [PYTHON, worker_script, "--credential", cred_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=BASE_DIR,
    )
    logger.info(f"Started worker [{cred_name}] PID={proc.pid}")
    return proc


def start_analyzer() -> subprocess.Popen:
    """Start gap_analyzer.py subprocess."""
    analyzer_script = os.path.join(BASE_DIR, "gap_analyzer.py")
    proc = subprocess.Popen(
        [PYTHON, analyzer_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=BASE_DIR,
    )
    logger.info(f"Started gap_analyzer PID={proc.pid}")
    return proc


def is_market_hours() -> bool:
    now = datetime.datetime.now(INDIA_TZ).time()
    return datetime.time(9, 10) <= now <= datetime.time(15, 36)


def kill_process(proc: subprocess.Popen):
    """Kill a subprocess."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    logger.info("=" * 60)
    logger.info("STOCK SCANNER LAUNCHER")
    logger.info(f"Time: {datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info("=" * 60)

    # Step 1: Run setup
    run_setup()

    # Step 2: Load credentials
    credentials = load_credentials()
    cred_names = [c['name'] for c in credentials]
    logger.info(f"Credentials: {cred_names}")

    # Step 3: Start worker processes
    workers = {}
    for name in cred_names:
        workers[name] = start_worker(name)
        time.sleep(2)  # Stagger starts to avoid rate limits

    # Step 4: Start analyzer
    analyzer = start_analyzer()

    # Step 5: Monitor loop
    logger.info("\nAll processes started. Monitoring...")
    all_processes = []

    try:
        while True:
            now = datetime.datetime.now(INDIA_TZ)

            # Stop after market close
            if now.time() > datetime.time(15, 36):
                logger.info("Market closed. Shutting down all processes...")
                break

            # Check and restart crashed workers
            for name in cred_names:
                proc = workers[name]
                if proc.poll() is not None:
                    # Process died
                    exit_code = proc.returncode
                    logger.warning(f"Worker [{name}] died with exit code {exit_code}")
                    if is_market_hours():
                        logger.info(f"Restarting worker [{name}]...")
                        time.sleep(3)
                        workers[name] = start_worker(name)

            # Check analyzer
            if analyzer.poll() is not None:
                logger.warning(f"Analyzer died with exit code {analyzer.returncode}")
                if is_market_hours():
                    logger.info("Restarting analyzer...")
                    time.sleep(3)
                    analyzer = start_analyzer()

            # Status log every 5 minutes
            if now.minute % 5 == 0 and now.second < 10:
                alive = sum(1 for w in workers.values() if w.poll() is None)
                analyzer_status = "running" if analyzer.poll() is None else "stopped"
                logger.info(
                    f"[STATUS] Workers: {alive}/{len(cred_names)} alive | "
                    f"Analyzer: {analyzer_status} | "
                    f"Time: {now.strftime('%H:%M:%S')}"
                )

            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt — shutting down...")

    finally:
        # Clean shutdown
        for name, proc in workers.items():
            logger.info(f"Stopping worker [{name}] PID={proc.pid}")
            kill_process(proc)

        logger.info(f"Stopping analyzer PID={analyzer.pid}")
        kill_process(analyzer)

        logger.info("All processes stopped. Scanner shutdown complete.")


if __name__ == "__main__":
    main()
