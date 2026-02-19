"""Bot process orchestrator — manages per-user trading bot lifecycles."""

import subprocess
import sys
import os
import signal
import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

import models
from encryption import decrypt

logger = logging.getLogger("BotManager")

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = sys.executable


def _kill_pid(pid: int):
    """Kill a process by PID (cross-platform)."""
    if pid is None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=5)
        else:
            os.kill(pid, signal.SIGTERM)
            import time
            time.sleep(0.5)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass  # already dead
    except (OSError, subprocess.TimeoutExpired):
        pass


def _pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    if pid is None:
        return False
    try:
        if os.name == 'nt':
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)  # signal 0 = check if process exists
            return True
    except (OSError, subprocess.TimeoutExpired):
        return False


class BotManager:
    def __init__(self):
        self._bots: dict[int, dict] = {}

    def start_bot(self, user_id: int, db: Session) -> dict:
        # Check in-memory first
        if user_id in self._bots and self._bots[user_id].get("status") == "running":
            # Verify processes are actually still alive
            if self.is_alive(user_id):
                return {"status": "running", "started_at": self._bots[user_id].get("started_at")}
            else:
                # Processes died, clean up
                del self._bots[user_id]

        # Check DB for existing running session (handles server restarts)
        session = db.query(models.BotSession).filter_by(
            user_id=user_id, status="running"
        ).order_by(models.BotSession.id.desc()).first()
        if session:
            pids = [session.pid_symbol, session.pid_websocket, session.pid_poshandle]
            if any(_pid_alive(p) for p in pids if p):
                return {"status": "running", "started_at": session.started_at}
            else:
                # Mark dead session as stopped
                session.status = "stopped"
                session.stopped_at = datetime.now(timezone.utc)
                db.commit()

        user = db.query(models.User).filter_by(id=user_id).first()
        if not user or not user.angelone_creds:
            return {"status": "error", "error_message": "AngelOne credentials not configured"}

        creds = user.angelone_creds
        api_key = decrypt(creds.api_key_enc)
        client_id = decrypt(creds.client_id_enc)
        password = decrypt(creds.password_enc)
        totp_secret = decrypt(creds.totp_secret_enc)

        settings = db.query(models.UserSettings).filter_by(user_id=user_id).first()
        quantity = settings.default_quantity if settings else 1
        price_min = settings.price_min if settings else 110
        price_max = settings.price_max if settings else 150

        env = os.environ.copy()
        env.update({
            "ANGELONE_API_KEY": api_key,
            "ANGELONE_CLIENT_ID": client_id,
            "ANGELONE_PASSWORD": password,
            "ANGELONE_TOTP_SECRET": totp_secret,
            "REDIS_PREFIX": f"user:{user_id}:",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "Rahul@7355",
            "TRADING_QUANTITY": str(quantity),
            "PRICE_MIN": str(price_min),
            "PRICE_MAX": str(price_max),
            "USER_ID": str(user_id),
        })

        started_at = datetime.now(timezone.utc)

        try:
            # Step 1: create_angleone_csv.py (synchronous)
            logger.info(f"[User {user_id}] Starting create_angleone_csv.py...")
            csv_proc = subprocess.run(
                [PYTHON_EXE, os.path.join(SCRIPTS_DIR, "create_angleone_csv.py")],
                env=env, capture_output=True, text=True, timeout=120, cwd=SCRIPTS_DIR
            )
            if csv_proc.returncode != 0:
                error_msg = f"create_angleone_csv failed: {csv_proc.stderr[:500]}"
                logger.error(f"[User {user_id}] {error_msg}")
                self._save_session(db, user_id, "error", started_at, error_msg)
                return {"status": "error", "error_message": error_msg}

            # Step 2: symbol_found.py (background)
            logger.info(f"[User {user_id}] Starting symbol_found.py...")
            symbol_proc = subprocess.Popen(
                [PYTHON_EXE, os.path.join(SCRIPTS_DIR, "symbol_found.py")],
                env=env, cwd=SCRIPTS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Step 3: angleone_websocket1.py (background)
            logger.info(f"[User {user_id}] Starting angleone_websocket1.py...")
            ws_proc = subprocess.Popen(
                [PYTHON_EXE, os.path.join(SCRIPTS_DIR, "angleone_websocket1.py")],
                env=env, cwd=SCRIPTS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Step 4: pos_handle_wts.py (background)
            logger.info(f"[User {user_id}] Starting pos_handle_wts.py...")
            pos_proc = subprocess.Popen(
                [PYTHON_EXE, os.path.join(SCRIPTS_DIR, "pos_handle_wts.py")],
                env=env, cwd=SCRIPTS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            self._bots[user_id] = {
                "processes": [symbol_proc, ws_proc, pos_proc],
                "status": "running",
                "started_at": started_at,
            }

            self._save_session(db, user_id, "running", started_at, None,
                               symbol_proc.pid, ws_proc.pid, pos_proc.pid)

            logger.info(f"[User {user_id}] All processes started (PIDs: {symbol_proc.pid}, {ws_proc.pid}, {pos_proc.pid})")
            return {"status": "running", "started_at": started_at}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[User {user_id}] Bot start error: {error_msg}")
            self._save_session(db, user_id, "error", started_at, error_msg)
            return {"status": "error", "error_message": error_msg}

    def stop_bot(self, user_id: int, db: Session) -> dict:
        stopped_any = False

        # 1. Kill from in-memory refs
        if user_id in self._bots:
            bot = self._bots[user_id]
            for proc in bot.get("processes", []):
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    stopped_any = True
                except Exception:
                    try:
                        proc.kill()
                        stopped_any = True
                    except Exception:
                        pass
            del self._bots[user_id]

        # 2. Kill from DB PIDs (handles server restart case)
        session = db.query(models.BotSession).filter_by(
            user_id=user_id, status="running"
        ).order_by(models.BotSession.id.desc()).first()

        if session:
            for pid in [session.pid_symbol, session.pid_websocket, session.pid_poshandle]:
                if pid and _pid_alive(pid):
                    _kill_pid(pid)
                    stopped_any = True
                    logger.info(f"[User {user_id}] Killed process PID={pid}")

            session.status = "stopped"
            session.stopped_at = datetime.now(timezone.utc)
            db.commit()
        elif not stopped_any:
            # No in-memory refs AND no running session in DB
            # Mark any remaining sessions as stopped just in case
            all_sessions = db.query(models.BotSession).filter_by(
                user_id=user_id, status="running"
            ).all()
            for s in all_sessions:
                s.status = "stopped"
                s.stopped_at = datetime.now(timezone.utc)
            db.commit()

        logger.info(f"[User {user_id}] Bot stopped")
        return {"status": "stopped"}

    def is_alive(self, user_id: int) -> bool:
        if user_id not in self._bots:
            return False
        return any(p.poll() is None for p in self._bots[user_id].get("processes", []))

    def _save_session(self, db: Session, user_id: int, status: str,
                      started_at: datetime, error: Optional[str],
                      pid_symbol: int = None, pid_ws: int = None, pid_pos: int = None):
        session = models.BotSession(
            user_id=user_id, status=status, started_at=started_at,
            error_message=error, pid_symbol=pid_symbol,
            pid_websocket=pid_ws, pid_poshandle=pid_pos,
        )
        db.add(session)
        db.commit()
