"""WebSocket route for real-time dashboard updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis
import json
import asyncio
from datetime import datetime
from auth import decode_token

router = APIRouter(tags=["WebSocket"])

active_connections: dict[int, list[WebSocket]] = {}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = ""):
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        await websocket.close(code=4001)
        return

    user_id = int(payload["sub"])
    await websocket.accept()

    if user_id not in active_connections:
        active_connections[user_id] = []
    active_connections[user_id].append(websocket)

    try:
        r = redis.StrictRedis(
            host="localhost", port=6379, password="Rahul@7355",
            db=0, decode_responses=True
        )
        prefix = f"user:{user_id}:"

        while True:
            try:
                data = {}

                pos_raw = r.get(f"{prefix}active_positions")
                if pos_raw:
                    positions = json.loads(pos_raw)
                    if positions:
                        token_key, pdata = next(iter(positions.items()))
                        cur_price_str = r.get(f"{prefix}PRICE:{token_key}")
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

                nifty_price = r.get(f"{prefix}PRICE:99926000")
                if nifty_price:
                    data["nifty_price"] = float(nifty_price)

                ts_raw = r.get(f"{prefix}Trading_symbol")
                if ts_raw:
                    data["trading_symbols"] = json.loads(ts_raw)

                candle_raw = r.get(f"{prefix}last_candle")
                if candle_raw:
                    data["last_candle"] = json.loads(candle_raw)

                th_raw = r.get(f"{prefix}trade_history_{datetime.now().strftime('%Y-%m-%d')}")
                if th_raw:
                    trades = json.loads(th_raw)
                    data["today_pnl"] = round(sum(t.get("pnl", 0) for t in trades), 2)
                    data["today_trades"] = len(trades)

                if data:
                    await websocket.send_json({"type": "update", "data": data})

            except Exception as e:
                await websocket.send_json({"type": "error", "data": {"message": str(e)}})

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
    finally:
        if user_id in active_connections:
            active_connections[user_id].remove(websocket)
            if not active_connections[user_id]:
                del active_connections[user_id]
