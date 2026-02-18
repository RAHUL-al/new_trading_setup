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


@router.post("/bot/control", response_model=schemas.BotStatusResponse)
def control_bot(
    data: schemas.BotControlRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if data.action == "start":
        if not user.is_verified:
            raise HTTPException(status_code=400, detail="Complete Aadhaar verification first")
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
                cur_price_str = r.get(f"{prefix}PRICE:{token}")
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
