from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import redis
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import pandas as pd
from redis.asyncio import Redis

# --------------------------- config ---------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "Rahul@7355"
REDIS_DB = 0

POSITIONS_KEY = "active_positions_by_10:30"
TRADE_HISTORY_KEY = f"trade_history_by_10:30_{datetime.now().date()}"
TRADING_SYMBOLS_KEY = "Trading_symbol"
CANDLE_CACHE_KEY = "candle:last"

CHAN_PRICE_PREFIX = "price:"
CHAN_SIGNAL_BUY = "signal:buy"
CHAN_SIGNAL_SELL = "signal:sell"

KEY_BUY_SIGNAL = "buy_signal"
KEY_SELL_SIGNAL = "sell_signal"

INDEX_TOKEN = "99926000"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradingBot")

r = redis.StrictRedis(
    host='localhost',
    port=6379,
    password='Rahul@7355',
    db=0,
    decode_responses=True
)

# --------------------------- data models ---------------------------
@dataclass
class Position:
    token: str
    trading_symbol: str
    position_type: str
    option_type: str
    entry_price: float
    quantity: int
    entry_time: str
    stop_loss: float
    status: str = "OPEN"

@dataclass
class Trade:
    token: str
    option_type: str
    position_type: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: Optional[str]
    exit_time: str
    stop_loss: float
    pnl: float
    close_reason: str

# --------------------------- helper utils ---------------------------
async def load_last_candle_from_csv(csv_path: str = "main_csv.csv") -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None
        row = df.tail(1).to_dict(orient='records')[0]
        high = row.get('high')
        low = row.get('low')
        return float(high) if high is not None else None, float(low) if low is not None else None
    except Exception as e:
        logger.warning(f"CSV load failed: {e}")
        return None, None

# --------------------------- core bot ---------------------------
class TradingBot:
    def __init__(self):
        self.r: Redis = Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=REDIS_DB, 
            password=REDIS_PASSWORD, 
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self.ce_symbol: Optional[str] = None
        self.ce_token: Optional[str] = None
        self.pe_symbol: Optional[str] = None
        self.pe_token: Optional[str] = None
        self.open_pos: Optional[Position] = None
        self._close_lock = asyncio.Lock()
        
        now = datetime.now()
        # Define trading sessions
        self.session1_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        self.session1_end = now.replace(hour=10, minute=30, second=0, microsecond=0)
        self.session2_start = now.replace(hour=14, minute=0, second=0, microsecond=0)
        self.session2_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        self.square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
        
        # Track if we're in active trading hours
        self._active_trading_hours = False

    # ---------- redis kv helpers ----------
    async def get_current_price(self, token: str) -> float:
        try:
            val = await self.r.get(token)
            return float(val) if val else 0.0
        except Exception as e:
            logger.error(f"Error getting price for token {token}: {e}")
            return 0.0

    async def get_index_price(self) -> float:
        return await self.get_current_price(INDEX_TOKEN)

    async def cache_candle(self, high: Optional[float], low: Optional[float]):
        if high is None or low is None:
            return
        data = {"high": high, "low": low, "ts": datetime.now().isoformat()}
        await self.r.set(CANDLE_CACHE_KEY, json.dumps(data), ex=120)

    async def get_candle(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            if await self.r.exists(CANDLE_CACHE_KEY):
                d = json.loads(await self.r.get(CANDLE_CACHE_KEY))
                return float(d.get("high")), float(d.get("low"))
        except Exception as e:
            logger.warning(f"Error getting cached candle: {e}")
        
        return await load_last_candle_from_csv()

    async def save_position(self, pos: Position):
        try:
            positions = {}
            if await self.r.exists(POSITIONS_KEY):
                positions = json.loads(await self.r.get(POSITIONS_KEY))
            positions[str(pos.token)] = asdict(pos)
            await self.r.set(POSITIONS_KEY, json.dumps(positions))
            logger.info(f"Saved position: {pos.token}")
        except Exception as e:
            logger.error(f"save_position error: {e}")

    async def remove_position(self, token: str):
        try:
            if await self.r.exists(POSITIONS_KEY):
                positions = json.loads(await self.r.get(POSITIONS_KEY))
                if str(token) in positions:
                    del positions[str(token)]
                    await self.r.set(POSITIONS_KEY, json.dumps(positions))
                    logger.info(f"Removed position: {token}")
        except Exception as e:
            logger.error(f"remove_position error: {e}")

    async def save_trade(self, trade: Trade):
        try:
            history = []
            if await self.r.exists(TRADE_HISTORY_KEY):
                history = json.loads(await self.r.get(TRADE_HISTORY_KEY))
            history.append(asdict(trade))
            if len(history) > 100:
                history = history[-100:]
            await self.r.set(TRADE_HISTORY_KEY, json.dumps(history))
            logger.info(f"Saved trade: {trade.token} pnl={trade.pnl}")
        except Exception as e:
            logger.error(f"save_trade error: {e}")

    # ---------- symbols / recovery ----------
    async def load_symbols(self):
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            try:
                if await self.r.exists(TRADING_SYMBOLS_KEY):
                    ts = json.loads(await self.r.get(TRADING_SYMBOLS_KEY))
                    self.ce_symbol, self.ce_token = ts.get("CE", [None, None])
                    self.pe_symbol, self.pe_token = ts.get("PE", [None, None])
                    if self.ce_token and self.pe_token:
                        logger.info(f"Symbols loaded: CE={self.ce_symbol}/{self.ce_token} PE={self.pe_symbol}/{self.pe_token}")
                        return True
                await asyncio.sleep(0.5)
                attempts += 1
            except Exception as e:
                logger.error(f"load_symbols error: {e}")
                await asyncio.sleep(0.5)
                attempts += 1
        logger.error("Failed to load trading symbols after multiple attempts")
        return False

    async def recover_open_position(self):
        try:
            if await self.r.exists(POSITIONS_KEY):
                positions = json.loads(await self.r.get(POSITIONS_KEY))
                if positions:
                    token, pdata = next(iter(positions.items()))
                    self.open_pos = Position(
                        token=token,
                        trading_symbol=pdata["trading_symbol"],
                        position_type=pdata["position_type"],
                        option_type=pdata["option_type"],
                        entry_price=float(pdata["entry_price"]),
                        quantity=int(pdata["quantity"]),
                        entry_time=pdata.get("entry_time", datetime.now().isoformat()),
                        stop_loss=float(pdata["stop_loss"]),
                        status=pdata.get("status", "OPEN"),
                    )
                    logger.info(f"Recovered open position: {self.open_pos}")
                    return True
        except Exception as e:
            logger.error(f"recover_open_position error: {e}")
        return False

    # ---------- trading hours management ----------
    def is_market_hours(self) -> bool:
        """Check if market is open (9:15 to 15:30)"""
        now = datetime.now()
        current_time = now.time()
        market_open = self.session1_start.time()
        market_close = self.session2_end.time()
        return market_open <= current_time <= market_close

    def is_active_trading_hours(self) -> bool:
        """Check if we're in active trading periods (9:15-10:30 or 14:00-15:30)"""
        now = datetime.now()
        current_time = now.time()
        
        session1_active = (self.session1_start.time() <= current_time <= self.session1_end.time())
        session2_active = (self.session2_start.time() <= current_time <= self.session2_end.time())
        
        return session1_active or session2_active

    def get_next_active_session_start(self) -> datetime:
        """Get the start time of the next active trading session"""
        now = datetime.now()
        
        # If before first session today
        if now.time() < self.session1_start.time():
            return now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # If between session1 end and session2 start
        if self.session1_end.time() <= now.time() < self.session2_start.time():
            return now.replace(hour=14, minute=0, second=0, microsecond=0)
        
        # If after session2 end, return first session tomorrow
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=9, minute=15, second=0, microsecond=0)

    # ---------- trade actions ----------
    async def take_buy(self, quantity: int = 1) -> bool:
        if not self.is_active_trading_hours():
            logger.warning("Cannot take position outside active trading hours")
            return False
            
        if not self.ce_token:
            logger.error("CE token not loaded")
            return False
        
        price = await self.get_current_price(self.ce_token)
        if price <= 0:
            logger.error(f"Invalid price for CE token {self.ce_token}: {price}")
            return False
        
        high, low = await self.get_candle()
        if low is None:
            logger.error("No candle low available for SL")
            return False
        
        pos = Position(
            token=str(self.ce_token),
            trading_symbol=str(self.ce_symbol),
            position_type="BUY",
            option_type="CE",
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=low,
        )
        self.open_pos = pos
        await self.save_position(pos)
        logger.info(f"BUY CE @ {price} SL {low}")
        return True

    async def take_sell(self, quantity: int = 1) -> bool:
        if not self.is_active_trading_hours():
            logger.warning("Cannot take position outside active trading hours")
            return False
            
        if not self.pe_token:
            logger.error("PE token not loaded")
            return False
        
        price = await self.get_current_price(self.pe_token)
        if price <= 0:
            logger.error(f"Invalid price for PE token {self.pe_token}: {price}")
            return False
        
        high, low = await self.get_candle()
        if high is None:
            logger.error("No candle high available for SL")
            return False
        
        pos = Position(
            token=str(self.pe_token),
            trading_symbol=str(self.pe_symbol),
            position_type="SELL",
            option_type="PE",
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=high,
        )
        self.open_pos = pos
        await self.save_position(pos)
        logger.info(f"SELL PE @ {price} SL {high}")
        return True

    async def close_position(self, reason: str) -> bool:
        async with self._close_lock:
            if not self.open_pos:
                return False
            
            pos = self.open_pos
            token = pos.token
            cur_price = await self.get_current_price(token)
            
            if cur_price <= 0:
                logger.error(f"No valid price for token {token}: {cur_price}")
                return False
            
            # Correct P&L calculation
            if pos.option_type == "CE":
                pnl = (cur_price - pos.entry_price) * pos.quantity
            else:
                pnl = (cur_price - pos.entry_price) * pos.quantity

            tr = Trade(
                token=token,
                option_type=pos.option_type,
                position_type=pos.position_type,
                entry_price=pos.entry_price,
                exit_price=cur_price,
                quantity=pos.quantity,
                entry_time=pos.entry_time,
                exit_time=datetime.now().isoformat(),
                stop_loss=pos.stop_loss,
                pnl=round(float(pnl), 2),
                close_reason=reason,
            )
            
            self.open_pos = None
            await self.save_trade(tr)
            await self.remove_position(token)
            logger.info(f"Closed {token} reason={reason} pnl={tr.pnl}")
            return True

    # ---------- event handlers ----------
    async def handle_index_tick(self, price: float):
        if not self.open_pos or not self.is_market_hours():
            return
        
        pos = self.open_pos
        print(f"Current price: {price}, Stop loss: {pos.stop_loss}")
        if pos.option_type == "CE" and price <= pos.stop_loss:
            await self.close_position("STOP_LOSS")
        elif pos.option_type == "PE" and price >= pos.stop_loss:
            await self.close_position("STOP_LOSS")

    async def on_buy_signal(self):
        if not self.is_active_trading_hours():
            logger.debug("Ignoring signal outside active trading hours")
            return
            
        if self.open_pos and self.open_pos.option_type == "PE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_buy(quantity=1)

    async def on_sell_signal(self):
        if not self.is_active_trading_hours():
            logger.debug("Ignoring signal outside active trading hours")
            return
            
        if self.open_pos and self.open_pos.option_type == "CE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_sell(quantity=1)

    # ---------- tasks ----------
    async def task_pubsub_listener(self):
        """Listens for Pub/Sub messages for prices and signals."""
        try:
            psub = self.r.pubsub()
            await psub.subscribe(
                f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}", 
                CHAN_SIGNAL_BUY, 
                CHAN_SIGNAL_SELL
            )
            logger.info("Subscribed to Pub/Sub channels")
            
            async for msg in psub.listen():
                if not self.is_market_hours():
                    continue
                    
                if msg is None or msg.get("type") != "message":
                    continue
                    
                channel = msg.get("channel")
                data = msg.get("data")
                
                if channel == f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}":
                    try:
                        price = float(data)
                        await self.handle_index_tick(price)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid price data: {data}, error: {e}")
                elif channel == CHAN_SIGNAL_BUY:
                    await self.on_buy_signal()
                elif channel == CHAN_SIGNAL_SELL:
                    await self.on_sell_signal()
                    
        except Exception as e:
            logger.error(f"PubSub listener error: {e}")

    async def task_key_fallback_poller(self):
        """Fallback for setups that only set Redis keys."""
        last_buy = None
        last_sell = None
        
        while True:
            if not self.is_market_hours():
                # Sleep until market opens again
                next_start = self.get_next_active_session_start()
                sleep_seconds = (next_start - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    logger.info(f"Market closed. Sleeping for {sleep_seconds/3600:.2f} hours")
                    await asyncio.sleep(sleep_seconds)
                continue

            try:
                # Always monitor stop loss regardless of active trading hours
                price = await self.get_index_price()
                if price > 0 and self.open_pos:
                    await self.handle_index_tick(price)

                # Only process signals during active trading hours
                if self.is_active_trading_hours():
                    buy_signal = await self.r.get(KEY_BUY_SIGNAL)
                    sell_signal = await self.r.get(KEY_SELL_SIGNAL)
                    
                    if buy_signal != last_buy and buy_signal == "true":
                        await self.on_buy_signal()
                    if sell_signal != last_sell and sell_signal == "true":
                        await self.on_sell_signal()
                        
                    last_buy, last_sell = buy_signal, sell_signal
                
            except Exception as e:
                logger.error(f"Key poller error: {e}")
                
            await asyncio.sleep(0.1)

    async def task_square_off_scheduler(self):
        while True:
            now = datetime.now()
            
            # Square off at 15:25 if market is open
            if now.time() >= self.square_off_time.time() and self.is_market_hours():
                if self.open_pos:
                    await self.close_position("SQUARE_OFF")
                logger.info("Square-off completed")
                
                # Calculate daily P&L
                data = await self.r.get(f"trade_history_by_10:30_{datetime.now().date()}")
                total_profit_or_loss = 0

                if data:
                    try:
                        trade_history = json.loads(data)
                        for trade in trade_history:
                            entry_time = datetime.fromisoformat(trade["entry_time"])
                            if entry_time.date() == datetime.today().date():
                                total_profit_or_loss += float(trade["pnl"])
                    except Exception as e:
                        logger.error(f"Error calculating P&L: {e}")

                logger.info(f"Total P&L for the day: {total_profit_or_loss}")
                
                # Sleep until market closes
                await asyncio.sleep((self.session2_end - now).total_seconds())
            
            # Check if market is closed and wait for next session
            elif not self.is_market_hours():
                next_start = self.get_next_active_session_start()
                sleep_seconds = (next_start - now).total_seconds()
                if sleep_seconds > 0:
                    logger.info(f"Market closed. Sleeping for {sleep_seconds/3600:.2f} hours")
                    await asyncio.sleep(sleep_seconds)
            
            else:
                await asyncio.sleep(5)

    async def task_candle_cache_refresh(self):
        while True:
            if self.is_market_hours():
                high, low = await load_last_candle_from_csv()
                if high is not None and low is not None:
                    await self.cache_candle(high, low)
            await asyncio.sleep(1)

    # ---------- main ----------
    async def run(self):
        logger.info("TradingBot starting…")
        
        if not await self.load_symbols():
            logger.error("Failed to load symbols, exiting")
            return
        
        # Recover any existing positions
        await self.recover_open_position()
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.task_pubsub_listener()),
            asyncio.create_task(self.task_key_fallback_poller()),
            asyncio.create_task(self.task_square_off_scheduler()),
            asyncio.create_task(self.task_candle_cache_refresh()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
        finally:
            for t in tasks:
                t.cancel()
            await self.r.close()

# --------------------------- entrypoint ---------------------------
if __name__ == "__main__":
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
