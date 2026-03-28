"""
simulator_routes.py — FastAPI endpoints + WebSocket for Market Replay Simulator.

Provides:
  - REST endpoints for simulation control and process monitoring
  - WebSocket endpoint for real-time candle/trade streaming to dashboard
"""

import sys
import os
import json
import asyncio
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

# Add parent directory to path for simulator import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from simulator import MarketSimulator

router = APIRouter(prefix="/api", tags=["simulator"])

# ─────────── Global state ───────────
_simulator: Optional[MarketSimulator] = None
_sim_task: Optional[asyncio.Task] = None
_ws_clients: List[WebSocket] = []

LOG_DIR = os.environ.get("LOG_DIR", "/root/Trading_setup_code/logs")
if os.name == "nt":
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

PROCESS_MAP = {
    "websocket": {"script": "angleone_websocket1.py", "pid_file": "/tmp/trading_websocket.pid"},
    "catboost": {"script": "catboost_live_engine", "pid_file": None},
    "pos_handler": {"script": "pos_handle_wts.py", "pid_file": "/tmp/trading_pos.pid"},
    "symbol_finder": {"script": "symbol_found.py", "pid_file": "/tmp/trading_symbol.pid"},
}


# ─────────── Helpers ───────────

async def broadcast(event_type: str, data: Dict[str, Any]):
    """Send event to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data})
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


def _check_process(script_name: str, pid_file: Optional[str] = None) -> Dict:
    """Check if a process is running."""
    result = {"name": script_name, "running": False, "pid": None}

    if os.name == "nt":
        # Windows: use tasklist
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq python*"], text=True, timeout=5
            )
            result["running"] = script_name.lower() in output.lower()
        except Exception:
            pass
    else:
        # Linux: check /proc or use pgrep
        if pid_file and os.path.exists(pid_file):
            try:
                with open(pid_file) as f:
                    pid = f.read().strip()
                if os.path.exists(f"/proc/{pid}"):
                    result["running"] = True
                    result["pid"] = int(pid)
                    return result
            except Exception:
                pass

        try:
            output = subprocess.check_output(
                ["pgrep", "-f", script_name], text=True, timeout=5
            )
            pids = output.strip().split("\n")
            if pids and pids[0]:
                result["running"] = True
                result["pid"] = int(pids[0])
        except (subprocess.CalledProcessError, Exception):
            pass

    return result


def _get_log_file(process_name: str) -> Optional[str]:
    """Find the latest log file for a process."""
    name_map = {
        "websocket": "websocket_",
        "catboost": "websocket_",  # catboost logs into websocket log
        "pos_handler": "pos_handler_",
        "symbol_finder": "symbol_found_",
    }
    prefix = name_map.get(process_name, process_name)
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(LOG_DIR, f"{prefix}{today}.log")

    if os.path.exists(log_file):
        return log_file

    # Fallback: find any matching file
    if os.path.isdir(LOG_DIR):
        files = sorted(
            [f for f in os.listdir(LOG_DIR) if f.startswith(prefix)],
            reverse=True
        )
        if files:
            return os.path.join(LOG_DIR, files[0])
    return None


# ─────────── Simulator Endpoints ───────────

@router.get("/simulator/dates")
async def get_available_dates():
    """Return available trading dates from CSV data."""
    global _simulator
    if _simulator is None:
        _simulator = MarketSimulator()

    if not _simulator.load_data():
        return JSONResponse(status_code=404, content={"error": "CSV data not found"})

    dates = _simulator.get_available_dates()
    return {
        "dates": dates,
        "total": len(dates),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
    }


@router.post("/simulator/start")
async def start_simulation(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD (default: same as start)"),
    speed: int = Query(10, description="Candles per second"),
):
    """Start a market replay simulation."""
    global _simulator, _sim_task

    if _simulator and _simulator.state.running:
        return JSONResponse(status_code=400, content={"error": "Simulation already running"})

    if end_date is None:
        end_date = start_date

    _simulator = MarketSimulator(on_event=broadcast)
    if not _simulator.load_data():
        return JSONResponse(status_code=404, content={"error": "CSV data or model not found"})

    # Run simulation in background task
    _sim_task = asyncio.create_task(
        _simulator.run_simulation(start_date, end_date, speed)
    )

    return {
        "status": "started",
        "start_date": start_date,
        "end_date": end_date,
        "speed": speed,
    }


@router.post("/simulator/stop")
async def stop_simulation():
    """Stop the current simulation."""
    global _simulator
    if _simulator and _simulator.state.running:
        _simulator.stop()
        return {"status": "stopping"}
    return {"status": "not_running"}


@router.post("/simulator/pause")
async def pause_simulation():
    """Toggle pause on the current simulation."""
    global _simulator
    if _simulator and _simulator.state.running:
        paused = _simulator.pause()
        return {"status": "paused" if paused else "resumed"}
    return {"status": "not_running"}


@router.post("/simulator/speed")
async def set_speed(speed: int = Query(..., description="New speed (candles/sec)")):
    """Change replay speed mid-simulation."""
    global _simulator
    if _simulator and _simulator.state.running:
        _simulator.set_speed(speed)
        return {"status": "speed_changed", "speed": speed}
    return {"status": "not_running"}


@router.get("/simulator/status")
async def get_simulation_status():
    """Get current simulation state."""
    global _simulator
    if _simulator is None:
        return {"running": False}

    s = _simulator.state
    return {
        "running": s.running,
        "paused": s.paused,
        "speed": s.speed,
        "current_index": s.current_index,
        "total_candles": s.total_candles,
        "progress": round(s.current_index / max(s.total_candles, 1) * 100, 1),
        "current_date": s.current_date,
        "trades": s.wins + s.losses,
        "wins": s.wins,
        "losses": s.losses,
        "daily_pnl": round(s.daily_pnl, 2),
    }


# ─────────── Process Monitor Endpoints ───────────

@router.get("/processes/status")
async def get_process_status():
    """Check which trading processes are currently running."""
    statuses = {}
    for name, config in PROCESS_MAP.items():
        statuses[name] = _check_process(config["script"], config.get("pid_file"))
    return {"processes": statuses, "timestamp": datetime.now().isoformat()}


@router.get("/processes/logs/{process_name}")
async def get_process_logs(
    process_name: str,
    lines: int = Query(100, description="Number of log lines to return"),
):
    """Get the last N lines of a process log."""
    log_file = _get_log_file(process_name)
    if not log_file:
        return JSONResponse(status_code=404, content={"error": f"No log file found for {process_name}"})

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {
            "process": process_name,
            "log_file": os.path.basename(log_file),
            "total_lines": len(all_lines),
            "lines": [l.rstrip() for l in tail],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─────────── WebSocket ───────────

@router.websocket("/ws/simulator")
async def simulator_websocket(ws: WebSocket):
    """WebSocket for real-time simulation streaming."""
    await ws.accept()
    _ws_clients.append(ws)

    try:
        while True:
            # Keep connection alive, handle client messages
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                data = json.loads(msg)

                if data.get("action") == "start":
                    # Start simulation via WebSocket
                    await start_simulation(
                        start_date=data["start_date"],
                        end_date=data.get("end_date", data["start_date"]),
                        speed=data.get("speed", 10),
                    )
                elif data.get("action") == "stop":
                    await stop_simulation()
                elif data.get("action") == "pause":
                    await pause_simulation()
                elif data.get("action") == "speed":
                    await set_speed(speed=data.get("speed", 10))
                elif data.get("action") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

            except asyncio.TimeoutError:
                # Send ping to keep alive
                try:
                    await ws.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


@router.websocket("/ws/logs")
async def log_stream_websocket(ws: WebSocket):
    """WebSocket for real-time log streaming."""
    await ws.accept()

    process_name = "websocket"  # Default
    try:
        # Wait for initial config
        msg = await asyncio.wait_for(ws.receive_text(), timeout=10)
        data = json.loads(msg)
        process_name = data.get("process", "websocket")
    except Exception:
        pass

    log_file = _get_log_file(process_name)
    if not log_file:
        await ws.send_text(json.dumps({"type": "error", "data": {"message": f"No log for {process_name}"}}))
        await ws.close()
        return

    try:
        # Initial tail
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            tail = lines[-50:] if len(lines) > 50 else lines
            await ws.send_text(json.dumps({
                "type": "log_init",
                "data": {"lines": [l.rstrip() for l in tail]}
            }))

        # Stream new lines
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)  # Seek to end
            while True:
                line = f.readline()
                if line:
                    await ws.send_text(json.dumps({
                        "type": "log_line",
                        "data": {"line": line.rstrip()}
                    }))
                else:
                    await asyncio.sleep(0.5)

                # Check for client disconnect
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("action") == "switch":
                        process_name = data.get("process", process_name)
                        log_file = _get_log_file(process_name)
                        if not log_file:
                            await ws.send_text(json.dumps({
                                "type": "error",
                                "data": {"message": f"No log for {process_name}"}
                            }))
                            continue
                        f.close()
                        f = open(log_file, "r", encoding="utf-8", errors="replace")
                        lines = f.readlines()
                        tail = lines[-50:] if len(lines) > 50 else lines
                        await ws.send_text(json.dumps({
                            "type": "log_init",
                            "data": {"lines": [l.rstrip() for l in tail]}
                        }))
                except asyncio.TimeoutError:
                    pass

    except WebSocketDisconnect:
        pass
