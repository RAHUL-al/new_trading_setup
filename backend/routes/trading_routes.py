"""Trading routes: bot control, portfolio, trade history, settings."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import redis
import json

from database import get_db
from auth import get_current_user
import models
import schemas
from bot_manager import BotManager

router = APIRouter(prefix="/api/trading", tags=["Trading"])

bot_manager = BotManager()


def _sync_trades_from_redis(user_id: int, db: Session):
    """Sync completed trades from Redis trade_history_* keys into SQLite."""
    try:
        r_conn = redis.StrictRedis(host="localhost", port=6379, password="Rahul@7355", db=0, decode_responses=True)
        prefix = f"user:{user_id}:"

        # Find all trade_history keys for this user
        pattern = f"{prefix}trade_history_*"
        keys = r_conn.keys(pattern)

        for key in keys:
            raw_data = r_conn.get(key)
            if not raw_data:
                continue

            try:
                trades = json.loads(raw_data)
                if not isinstance(trades, list):
                    trades = [trades]
            except (json.JSONDecodeError, TypeError):
                continue

            for t in trades:
                # Check if trade already exists (by entry_time + token to avoid duplicates)
                entry_time = t.get("entry_time", "")
                token = t.get("token", "")
                if not entry_time or not token:
                    continue

                existing = db.query(models.TradeRecord).filter_by(
                    user_id=user_id,
                    token=str(token),
                    entry_time=entry_time,
                ).first()

                if existing:
                    # Update exit info if trade was closed
                    if t.get("exit_price") and not existing.exit_price:
                        existing.exit_price = float(t.get("exit_price", 0))
                        existing.exit_time = t.get("exit_time", "")
                        existing.pnl = float(t.get("pnl", 0))
                        existing.close_reason = t.get("close_reason", "")
                    continue

                trade = models.TradeRecord(
                    user_id=user_id,
                    token=str(token),
                    trading_symbol=t.get("trading_symbol", ""),
                    option_type=t.get("option_type", ""),
                    position_type=t.get("position_type", ""),
                    entry_price=float(t.get("entry_price", 0)),
                    exit_price=float(t.get("exit_price")) if t.get("exit_price") else None,
                    quantity=int(t.get("quantity", 1)),
                    entry_time=entry_time,
                    exit_time=t.get("exit_time"),
                    stop_loss=float(t.get("stop_loss", 0)) if t.get("stop_loss") else None,
                    pnl=float(t.get("pnl", 0)),
                    close_reason=t.get("close_reason"),
                    trade_date=t.get("trade_date", datetime.now().strftime("%Y-%m-%d")),
                )
                db.add(trade)

            db.commit()
    except Exception as e:
        print(f"[TradeSync] Error syncing trades for user {user_id}: {e}")


@router.post("/bot/control", response_model=schemas.BotStatusResponse)
def control_bot(
    data: schemas.BotControlRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if data.action == "start":
        if not user.is_verified:
            raise HTTPException(status_code=400, detail="Complete email verification first")
        if not user.angelone_creds:
            raise HTTPException(status_code=400, detail="Configure AngelOne credentials first")
        result = bot_manager.start_bot(user.id, db)
        return schemas.BotStatusResponse(**result)
    elif data.action == "stop":
        result = bot_manager.stop_bot(user.id, db)
        return schemas.BotStatusResponse(**result)
    raise HTTPException(status_code=400, detail="Action must be 'start' or 'stop'")


@router.get("/bot/status", response_model=schemas.BotStatusResponse)
def get_bot_status(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(models.BotSession).filter_by(
        user_id=user.id
    ).order_by(models.BotSession.id.desc()).first()

    if not session:
        return schemas.BotStatusResponse(status="stopped")

    if session.status == "running":
        if not bot_manager.is_alive(user.id):
            session.status = "stopped"
            session.stopped_at = datetime.now(timezone.utc)
            db.commit()

    return schemas.BotStatusResponse(
        status=session.status, started_at=session.started_at, error_message=session.error_message,
    )


@router.get("/portfolio", response_model=schemas.PortfolioResponse)
def get_portfolio(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Sync Redis trades → SQLite on each portfolio fetch
    _sync_trades_from_redis(user.id, db)

    today = datetime.now().strftime("%Y-%m-%d")
    all_trades = db.query(models.TradeRecord).filter_by(user_id=user.id).all()
    today_trades = [t for t in all_trades if t.trade_date == today]

    total_pnl = sum(t.pnl for t in all_trades)
    today_pnl = sum(t.pnl for t in today_trades)

    wins = sum(1 for t in all_trades if t.pnl > 0) if all_trades else 0
    win_rate = (wins / len(all_trades) * 100) if all_trades else 0.0

    open_position = None
    try:
        r = redis.StrictRedis(host="localhost", port=6379, password="Rahul@7355", db=0, decode_responses=True)
        prefix = f"user:{user.id}:"
        pos_data = r.get(f"{prefix}active_positions")
        if pos_data:
            positions = json.loads(pos_data)
            if positions:
                token, pdata = next(iter(positions.items()))
                cur_price_str = r.get(f"{prefix}{token}")
                cur_price = float(cur_price_str) if cur_price_str else pdata.get("entry_price", 0)
                pnl = (cur_price - float(pdata["entry_price"])) * int(pdata.get("quantity", 1))
                open_position = {
                    "token": token,
                    "trading_symbol": pdata.get("trading_symbol", ""),
                    "option_type": pdata.get("option_type", ""),
                    "entry_price": float(pdata["entry_price"]),
                    "current_price": cur_price,
                    "quantity": int(pdata.get("quantity", 1)),
                    "stop_loss": float(pdata.get("stop_loss", 0)),
                    "pnl": round(pnl, 2),
                    "entry_time": pdata.get("entry_time", ""),
                }
    except Exception:
        pass

    session = db.query(models.BotSession).filter_by(
        user_id=user.id
    ).order_by(models.BotSession.id.desc()).first()
    bot_status = session.status if session else "stopped"

    return schemas.PortfolioResponse(
        total_trades=len(all_trades), today_trades=len(today_trades),
        today_pnl=round(today_pnl, 2), total_pnl=round(total_pnl, 2),
        win_rate=round(win_rate, 1), open_position=open_position, bot_status=bot_status,
    )


@router.get("/trades", response_model=list[schemas.TradeResponse])
def get_trades(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    return (
        db.query(models.TradeRecord)
        .filter_by(user_id=user.id)
        .order_by(models.TradeRecord.id.desc())
        .offset(offset).limit(limit).all()
    )


@router.get("/settings", response_model=schemas.UserSettingsResponse)
def get_settings(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    settings = db.query(models.UserSettings).filter_by(user_id=user.id).first()
    if not settings:
        settings = models.UserSettings(user_id=user.id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@router.put("/settings", response_model=schemas.UserSettingsResponse)
def update_settings(
    data: schemas.UserSettingsInput,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    settings = db.query(models.UserSettings).filter_by(user_id=user.id).first()
    if not settings:
        settings = models.UserSettings(user_id=user.id)
        db.add(settings)

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(settings, key, value)

    db.commit()
    db.refresh(settings)
    return settings


@router.get("/market-data")
def get_market_data(
    user: models.User = Depends(get_current_user),
):
    """
    Live market data from Redis — NIFTY price, ATR, CE/PE symbols + prices, signals.
    Polled every 1s by the frontend (no WebSocket dependency).
    """
    try:
        r = redis.StrictRedis(host="localhost", port=6379, password="Rahul@7355", db=0, decode_responses=True)
        prefix = f"user:{user.id}:"
        data = {}

        # NIFTY index price
        nifty = r.get(f"{prefix}99926000")
        data["nifty_price"] = float(nifty) if nifty else 0

        # ATR
        atr = r.get(f"{prefix}ATR_value")
        data["atr"] = float(atr) if atr else 0

        # Trading symbols with live prices
        ts_raw = r.get(f"{prefix}Trading_symbol")
        symbols = {}
        if ts_raw:
            ts = json.loads(ts_raw)
            for opt_type in ["CE", "PE"]:
                info = ts.get(opt_type, [None, None])
                if info and info[1]:
                    price_str = r.get(f"{prefix}{info[1]}")
                    symbols[opt_type] = {
                        "symbol": info[0],
                        "token": str(info[1]),
                        "price": float(price_str) if price_str else 0,
                    }
        data["symbols"] = symbols

        # Signals
        data["signals"] = {
            "buy": r.get(f"{prefix}buy_signal") == "true",
            "sell": r.get(f"{prefix}sell_signal") == "true",
        }

        # Last candle
        candle_raw = r.get(f"{prefix}last_candle")
        data["last_candle"] = json.loads(candle_raw) if candle_raw else None

        # Open position
        pos_raw = r.get(f"{prefix}active_positions")
        data["position"] = None
        if pos_raw:
            positions = json.loads(pos_raw)
            if positions:
                token_key, pdata = next(iter(positions.items()))
                cur_price_str = r.get(f"{prefix}{token_key}")
                cur_price = float(cur_price_str) if cur_price_str else float(pdata.get("entry_price", 0))
                pnl = (cur_price - float(pdata["entry_price"])) * int(pdata.get("quantity", 1))
                data["position"] = {
                    "token": token_key,
                    "trading_symbol": pdata.get("trading_symbol", ""),
                    "option_type": pdata.get("option_type", ""),
                    "entry_price": float(pdata["entry_price"]),
                    "current_price": cur_price,
                    "quantity": int(pdata.get("quantity", 1)),
                    "stop_loss": float(pdata.get("stop_loss", 0)),
                    "pnl": round(pnl, 2),
                    "entry_time": pdata.get("entry_time", ""),
                }

        return data
    except Exception as e:
        return {"error": str(e), "nifty_price": 0, "atr": 0, "symbols": {}, "signals": {}, "position": None}


@router.get("/candles/{symbol_key}")
def get_candles(
    symbol_key: str,
    user: models.User = Depends(get_current_user),
):
    """
    Get candle history from Redis for charting.
    symbol_key can be 'NIFTY' or an option trading symbol like 'NIFTY19FEB25C23500'.
    Returns array of {time (unix), open, high, low, close, volume} for lightweight-charts.
    """
    try:
        r = redis.StrictRedis(host="localhost", port=6379, password="Rahul@7355", db=0, decode_responses=True)
        prefix = f"user:{user.id}:"
        date_key = datetime.now().strftime("%Y-%m-%d")

        # IST timezone for timestamp conversion
        import pytz
        ist = pytz.timezone("Asia/Kolkata")

        history_key = f"{prefix}HISTORY:{symbol_key}:{date_key}"
        raw_candles = r.lrange(history_key, 0, -1)

        candles = []
        seen_times = set()
        for raw in raw_candles:
            try:
                c = json.loads(raw)
                ts_str = c.get("timestamp", "")
                if not ts_str:
                    continue

                # Parse "2025-02-19 10:30:00" as IST → Unix epoch
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    dt_ist = ist.localize(dt)
                    unix_ts = int(dt_ist.timestamp())
                except ValueError:
                    try:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                        dt_ist = ist.localize(dt)
                        unix_ts = int(dt_ist.timestamp())
                    except ValueError:
                        continue

                if unix_ts in seen_times:
                    continue
                seen_times.add(unix_ts)

                candles.append({
                    "time": unix_ts,
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": float(c.get("volume", 0)),
                })
            except (json.JSONDecodeError, ValueError):
                continue

        # Also include the current forming candle
        now = datetime.now(ist)
        candle_time = now.strftime("%Y-%m-%d %H:%M")
        current_key = f"{prefix}CANDLE:{symbol_key}:{candle_time}"
        current_raw = r.get(current_key)
        if current_raw:
            try:
                cc = json.loads(current_raw)
                ts_str = cc.get("timestamp", candle_time + ":00")
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                dt_ist = ist.localize(dt)
                unix_ts = int(dt_ist.timestamp())
                if unix_ts not in seen_times:
                    candles.append({
                        "time": unix_ts,
                        "open": float(cc.get("open", 0)),
                        "high": float(cc.get("high", 0)),
                        "low": float(cc.get("low", 0)),
                        "close": float(cc.get("close", 0)),
                        "volume": float(cc.get("volume", 0)),
                    })
            except (json.JSONDecodeError, ValueError):
                pass

        # Sort by time
        candles.sort(key=lambda x: x["time"])

        return {"symbol": symbol_key, "date": date_key, "candles": candles}
    except Exception as e:
        return {"symbol": symbol_key, "date": "", "candles": [], "error": str(e)}

