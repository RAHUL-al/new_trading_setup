# from __future__ import annotations
# import asyncio
# import json
# import logging
# import os
# from dataclasses import dataclass, asdict
# from datetime import datetime, timedelta
# from typing import Optional, Tuple, Dict, Any
# import redis
# try:
#     import uvloop
#     uvloop.install()
# except ImportError:
#     pass

# import pandas as pd
# from redis.asyncio import Redis

# # --------------------------- config ---------------------------
# REDIS_HOST = "localhost"
# REDIS_PORT = 6379
# REDIS_PASSWORD = "Rahul@7355"
# REDIS_DB = 0

# POSITIONS_KEY = "active_positions"
# TRADE_HISTORY_KEY = f"trade_history_{datetime.now().date()}"
# TRADING_SYMBOLS_KEY = "Trading_symbol"
# CANDLE_CACHE_KEY = "candle:last"

# CHAN_PRICE_PREFIX = "price:"
# CHAN_SIGNAL_BUY = "signal:buy"
# CHAN_SIGNAL_SELL = "signal:sell"

# KEY_BUY_SIGNAL = "buy_signal"
# KEY_SELL_SIGNAL = "sell_signal"

# INDEX_TOKEN = "99926000"

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("TradingBot")

# r = redis.StrictRedis(
#     host='localhost',
#     port=6379,
#     password='Rahul@7355',
#     db=0,
#     decode_responses=True
# )

# # --------------------------- data models ---------------------------
# @dataclass
# class Position:
#     token: str
#     trading_symbol: str
#     position_type: str
#     option_type: str
#     entry_price: float
#     quantity: int
#     entry_time: str
#     stop_loss: float
#     status: str = "OPEN"

# @dataclass
# class Trade:
#     token: str
#     option_type: str
#     position_type: str
#     entry_price: float
#     exit_price: float
#     quantity: int
#     entry_time: Optional[str]
#     exit_time: str
#     stop_loss: float
#     pnl: float
#     close_reason: str

# # --------------------------- helper utils ---------------------------
# async def load_last_candle_from_csv(csv_path: str = "main_csv.csv") -> Tuple[Optional[float], Optional[float]]:
#     if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
#         return None, None
#     try:
#         df = pd.read_csv(csv_path)
#         if df.empty:
#             return None, None
#         row = df.tail(1).to_dict(orient='records')[0]
#         high = row.get('high')
#         low = row.get('low')
#         return float(high) if high is not None else None, float(low) if low is not None else None
#     except Exception as e:
#         logger.warning(f"CSV load failed: {e}")
#         return None, None

# # --------------------------- core bot ---------------------------
# class TradingBot:
#     def __init__(self):
#         self.r: Redis = Redis(
#             host=REDIS_HOST, 
#             port=REDIS_PORT, 
#             db=REDIS_DB, 
#             password=REDIS_PASSWORD, 
#             decode_responses=True,
#             socket_timeout=5,  # Add timeout to prevent hanging
#             socket_connect_timeout=5
#         )
#         self.ce_symbol: Optional[str] = None
#         self.ce_token: Optional[str] = None
#         self.pe_symbol: Optional[str] = None
#         self.pe_token: Optional[str] = None
#         self.open_pos: Optional[Position] = None
#         self._close_lock = asyncio.Lock()
        
#         now = datetime.now()
#         self.market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
#         self.market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
#         self.square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
#         self._is_market_hours = False

#     # ---------- redis kv helpers ----------
#     async def get_current_price(self, token: str) -> float:
#         try:
#             val = await self.r.get(token)
#             return float(val) if val else 0.0
#         except Exception as e:
#             logger.error(f"Error getting price for token {token}: {e}")
#             return 0.0

#     async def get_index_price(self) -> float:
#         return await self.get_current_price(INDEX_TOKEN)

#     async def cache_candle(self, high: Optional[float], low: Optional[float]):
#         if high is None or low is None:
#             return
#         data = {"high": high, "low": low, "ts": datetime.now().isoformat()}
#         await self.r.set(CANDLE_CACHE_KEY, json.dumps(data), ex=120)

#     async def get_candle(self) -> Tuple[Optional[float], Optional[float]]:
#         try:
#             if await self.r.exists(CANDLE_CACHE_KEY):
#                 d = json.loads(await self.r.get(CANDLE_CACHE_KEY))
#                 return float(d.get("high")), float(d.get("low"))
#         except Exception as e:
#             logger.warning(f"Error getting cached candle: {e}")
        
#         return await load_last_candle_from_csv()

#     async def save_position(self, pos: Position):
#         try:
#             positions = {}
#             if await self.r.exists(POSITIONS_KEY):
#                 positions = json.loads(await self.r.get(POSITIONS_KEY))
#             positions[str(pos.token)] = asdict(pos)
#             await self.r.set(POSITIONS_KEY, json.dumps(positions))
#             logger.info(f"Saved position: {pos.token}")
#         except Exception as e:
#             logger.error(f"save_position error: {e}")

#     async def remove_position(self, token: str):
#         try:
#             if await self.r.exists(POSITIONS_KEY):
#                 positions = json.loads(await self.r.get(POSITIONS_KEY))
#                 if str(token) in positions:
#                     del positions[str(token)]
#                     await self.r.set(POSITIONS_KEY, json.dumps(positions))
#                     logger.info(f"Removed position: {token}")
#         except Exception as e:
#             logger.error(f"remove_position error: {e}")

#     async def save_trade(self, trade: Trade):
#         try:
#             history = []
#             if await self.r.exists(TRADE_HISTORY_KEY):
#                 history = json.loads(await self.r.get(TRADE_HISTORY_KEY))
#             history.append(asdict(trade))
#             if len(history) > 100:
#                 history = history[-100:]
#             await self.r.set(TRADE_HISTORY_KEY, json.dumps(history))
#             logger.info(f"Saved trade: {trade.token} pnl={trade.pnl}")
#         except Exception as e:
#             logger.error(f"save_trade error: {e}")

#     # ---------- symbols / recovery ----------
#     async def load_symbols(self):
#         max_attempts = 10
#         attempts = 0
#         while attempts < max_attempts:
#             try:
#                 if await self.r.exists(TRADING_SYMBOLS_KEY):
#                     ts = json.loads(await self.r.get(TRADING_SYMBOLS_KEY))
#                     self.ce_symbol, self.ce_token = ts.get("CE", [None, None])
#                     self.pe_symbol, self.pe_token = ts.get("PE", [None, None])
#                     if self.ce_token and self.pe_token:
#                         logger.info(f"Symbols loaded: CE={self.ce_symbol}/{self.ce_token} PE={self.pe_symbol}/{self.pe_token}")
#                         return True
#                 await asyncio.sleep(0.5)
#                 attempts += 1
#             except Exception as e:
#                 logger.error(f"load_symbols error: {e}")
#                 await asyncio.sleep(0.5)
#                 attempts += 1
#         logger.error("Failed to load trading symbols after multiple attempts")
#         return False

#     async def recover_open_position(self):
#         try:
#             if await self.r.exists(POSITIONS_KEY):
#                 positions = json.loads(await self.r.get(POSITIONS_KEY))
#                 if positions:
#                     token, pdata = next(iter(positions.items()))
#                     self.open_pos = Position(
#                         token=token,
#                         trading_symbol=pdata["trading_symbol"],
#                         position_type=pdata["position_type"],
#                         option_type=pdata["option_type"],
#                         entry_price=float(pdata["entry_price"]),
#                         quantity=int(pdata["quantity"]),
#                         entry_time=pdata.get("entry_time", datetime.now().isoformat()),
#                         stop_loss=float(pdata["stop_loss"]),
#                         status=pdata.get("status", "OPEN"),
#                     )
#                     logger.info(f"Recovered open position: {self.open_pos}")
#                     return True
#         except Exception as e:
#             logger.error(f"recover_open_position error: {e}")
#         return False

#     # ---------- trade actions ----------
#     async def take_buy(self, quantity: int = 1) -> bool:
#         if not self.ce_token:
#             logger.error("CE token not loaded")
#             return False
        
#         price = await self.get_current_price(self.ce_token)
#         if price <= 0:
#             logger.error(f"Invalid price for CE token {self.ce_token}: {price}")
#             return False
        
#         high, low = await self.get_candle()
#         if low is None:
#             logger.error("No candle low available for SL")
#             return False
        
#         pos = Position(
#             token=str(self.ce_token),
#             trading_symbol=str(self.ce_symbol),
#             position_type="BUY",
#             option_type="CE",
#             entry_price=price,
#             quantity=quantity,
#             entry_time=datetime.now().isoformat(),
#             stop_loss=low,
#         )
#         self.open_pos = pos
#         await self.save_position(pos)
#         logger.info(f"BUY CE @ {price} SL {low}")
#         return True

#     async def take_sell(self, quantity: int = 1) -> bool:
#         if not self.pe_token:
#             logger.error("PE token not loaded")
#             return False
        
#         price = await self.get_current_price(self.pe_token)
#         if price <= 0:
#             logger.error(f"Invalid price for PE token {self.pe_token}: {price}")
#             return False
        
#         high, low = await self.get_candle()
#         if high is None:
#             logger.error("No candle high available for SL")
#             return False
        
#         pos = Position(
#             token=str(self.pe_token),
#             trading_symbol=str(self.pe_symbol),
#             position_type="SELL",
#             option_type="PE",
#             entry_price=price,
#             quantity=quantity,
#             entry_time=datetime.now().isoformat(),
#             stop_loss=high,
#         )
#         self.open_pos = pos
#         await self.save_position(pos)
#         logger.info(f"SELL PE @ {price} SL {high}")
#         return True

#     async def close_position(self, reason: str) -> bool:
#         async with self._close_lock:
#             if not self.open_pos:
#                 return False
            
#             pos = self.open_pos
#             token = pos.token
#             cur_price = await self.get_current_price(token)
            
#             if cur_price <= 0:
#                 logger.error(f"No valid price for token {token}: {cur_price}")
#                 return False
            
#             # Correct P&L calculation
#             if pos.option_type == "CE":
#                 pnl = (cur_price - pos.entry_price) * pos.quantity
#             else:
#                 pnl = (cur_price - pos.entry_price) * pos.quantity

#             tr = Trade(
#                 token=token,
#                 option_type=pos.option_type,
#                 position_type=pos.position_type,
#                 entry_price=pos.entry_price,
#                 exit_price=cur_price,
#                 quantity=pos.quantity,
#                 entry_time=pos.entry_time,
#                 exit_time=datetime.now().isoformat(),
#                 stop_loss=pos.stop_loss,
#                 pnl=round(float(pnl), 2),
#                 close_reason=reason,
#             )
            
#             self.open_pos = None
#             await self.save_trade(tr)
#             await self.remove_position(token)
#             logger.info(f"Closed {token} reason={reason} pnl={tr.pnl}")
#             return True

#     # ---------- market hours check ----------
#     def is_market_hours(self) -> bool:
#         now = datetime.now()
#         current_time = now.time()
#         return (self.market_open_time.time() <= current_time <= self.market_close_time.time())

#     # ---------- event handlers ----------
#     async def handle_index_tick(self, price: float):
#         if not self.open_pos or not self.is_market_hours():
#             return
        
#         pos = self.open_pos
#         print(f"Current price: {price}, Stop loss: {pos.stop_loss}")
#         if pos.option_type == "CE" and price <= pos.stop_loss:
#             await self.close_position("STOP_LOSS")
#         elif pos.option_type == "PE" and price >= pos.stop_loss:
#             await self.close_position("STOP_LOSS")

#     async def on_buy_signal(self):
#         if not self.is_market_hours():
#             return
            
#         if self.open_pos and self.open_pos.option_type == "PE":
#             await self.close_position("OPPOSITE_SIGNAL")
        
#         if not self.open_pos:
#             await self.take_buy(quantity=1)

#     async def on_sell_signal(self):
#         if not self.is_market_hours():
#             return
            
#         if self.open_pos and self.open_pos.option_type == "CE":
#             await self.close_position("OPPOSITE_SIGNAL")
        
#         if not self.open_pos:
#             await self.take_sell(quantity=1)

#     # ---------- tasks ----------
#     async def task_pubsub_listener(self):
#         """Listens for Pub/Sub messages for prices and signals."""
#         try:
#             psub = self.r.pubsub()
#             await psub.subscribe(
#                 f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}", 
#                 CHAN_SIGNAL_BUY, 
#                 CHAN_SIGNAL_SELL
#             )
#             logger.info("Subscribed to Pub/Sub channels")
            
#             async for msg in psub.listen():
#                 if not self.is_market_hours():
#                     # await asyncio.sleep(1)############################################################################
#                     continue
                    
#                 if msg is None or msg.get("type") != "message":
#                     continue
                    
#                 channel = msg.get("channel")
#                 data = msg.get("data")
                
#                 if channel == f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}":
#                     try:
#                         price = float(data)
#                         await self.handle_index_tick(price)
#                     except (ValueError, TypeError) as e:
#                         logger.warning(f"Invalid price data: {data}, error: {e}")
#                 elif channel == CHAN_SIGNAL_BUY:
#                     await self.on_buy_signal()
#                 elif channel == CHAN_SIGNAL_SELL:
#                     await self.on_sell_signal()
                    
#         except Exception as e:
#             logger.error(f"PubSub listener error: {e}")

#     async def task_key_fallback_poller(self):
#         """Fallback for setups that only set Redis keys."""
#         last_buy = None
#         last_sell = None
        
#         while True:
#             if not self.is_market_hours():
#                 await asyncio.sleep(5)
#                 continue

#             try:
#                 # Check stop loss with current index price
#                 price = await self.get_index_price()
#                 if price > 0:
#                     await self.handle_index_tick(price)

#                 # Check for signal changes
#                 buy_signal = await self.r.get(KEY_BUY_SIGNAL)
#                 sell_signal = await self.r.get(KEY_SELL_SIGNAL)
                
#                 if buy_signal != last_buy and buy_signal == "true":
#                     await self.on_buy_signal()
#                 if sell_signal != last_sell and sell_signal == "true":
#                     await self.on_sell_signal()
                    
#                 last_buy, last_sell = buy_signal, sell_signal
                
#             except Exception as e:
#                 logger.error(f"Key poller error: {e}")
                
#             await asyncio.sleep(0.1)  # 100ms polling interval

#     async def task_square_off_scheduler(self):
#         while True:
#             now = datetime.now()
            
#             if now.time() >= self.square_off_time.time() and self.is_market_hours():
#                 if self.open_pos:
#                     await self.close_position("SQUARE_OFF")
#                 logger.info("Square-off completed")
#                 data = r.get(f"trade_history_{datetime.now().date()}")

#                 total_profit_or_loss = 0

#                 if data:
#                     trade_history = json.loads(data)
#                     for i in trade_history:
#                         entry_time = datetime.fromisoformat(i["entry_time"])
#                         if entry_time.date() == datetime.today().date():
#                             total_profit_or_loss += float(i["pnl"])

#                 logging.info(f"Total P&L for the day: {total_profit_or_loss}")
                
#                 await asyncio.sleep((self.market_close_time - now).total_seconds())
            
#             # Check if market is closed and wait for next open
#             elif not self.is_market_hours() and now.time() > self.market_close_time.time():
#                 tomorrow = (now + timedelta(days=1)).replace(
#                     hour=9, minute=15, second=0, microsecond=0
#                 )
#                 sleep_seconds = (tomorrow - now).total_seconds()
#                 logger.info(f"Market closed. Sleeping for {sleep_seconds/3600:.2f} hours")
#                 await asyncio.sleep(sleep_seconds)
            
#             else:
#                 await asyncio.sleep(5)

#     async def task_candle_cache_refresh(self):
#         while True:
#             if self.is_market_hours():
#                 high, low = await load_last_candle_from_csv()
#                 if high is not None and low is not None:
#                     await self.cache_candle(high, low)
#             await asyncio.sleep(1)

#     # ---------- main ----------
#     async def run(self):
#         logger.info("TradingBot starting…")
        
#         if not await self.load_symbols():
#             logger.error("Failed to load symbols, exiting")
#             return
        
#         # Recover any existing positions
#         await self.recover_open_position()
        
#         # Create tasks
#         tasks = [
#             asyncio.create_task(self.task_pubsub_listener()),
#             asyncio.create_task(self.task_key_fallback_poller()),
#             asyncio.create_task(self.task_square_off_scheduler()),
#             asyncio.create_task(self.task_candle_cache_refresh()),
#         ]
        
#         try:
#             await asyncio.gather(*tasks)
#         except asyncio.CancelledError:
#             logger.info("Tasks cancelled")
#         except Exception as e:
#             logger.exception(f"Unexpected error in main loop: {e}")
#         finally:
#             for t in tasks:
#                 t.cancel()
#             await self.r.close()

# # --------------------------- entrypoint ---------------------------
# if __name__ == "__main__":
#     bot = TradingBot()
#     try:
#         asyncio.run(bot.run())
#     except KeyboardInterrupt:
#         logger.info("Bot stopped by user")
#     except Exception as e:
#         logger.exception(f"Unexpected error: {e}")



from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
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

POSITIONS_KEY = "active_positions"
TRADE_HISTORY_KEY = f"trade_history_{datetime.now().date()}"
TRADING_SYMBOLS_KEY = "Trading_symbol"
CANDLE_CACHE_KEY = "candle:last"
TRAILING_DATA_KEY = "trailing_data"

CHAN_PRICE_PREFIX = "price:"
CHAN_SIGNAL_BUY = "signal:buy"
CHAN_SIGNAL_SELL = "signal:sell"
CHAN_CANDLE_CLOSE = "candle:close"  # New channel for candle close events

KEY_BUY_SIGNAL = "buy_signal"
KEY_SELL_SIGNAL = "sell_signal"

INDEX_TOKEN = "99926000"
TRAILING_OFFSET = 0  # Offset for trailing stop loss

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
class CandleData:
    open: float
    high: float
    low: float
    close: float
    timestamp: str

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
    highest_close: float = 0.0  # For CE positions
    lowest_close: float = 0.0   # For PE positions

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

@dataclass
class TrailingData:
    position_token: str
    extreme_price: float  # Highest close for CE, lowest close for PE
    current_stop_loss: float
    last_updated: str

# --------------------------- helper utils ---------------------------
async def load_last_candle_from_csv(csv_path: str = "main_csv.csv") -> Tuple[Optional[float], Optional[float]]:
    """Load only high and low from last candle (for backward compatibility)"""
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
        self.last_candle: Optional[CandleData] = None
        
        now = datetime.now()
        self.market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        self.market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        self.square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
        self._is_market_hours = False


    async def task_continuous_trailing_stop_loss(self):
        """Continuously monitor and update trailing stop loss based on latest candle data from CSV"""
        last_processed_timestamp = None
        
        while True:
            try:
                if not self.is_market_hours() or not self.open_pos:
                    await asyncio.sleep(1)
                    continue
                
                # Get the latest complete candle data from CSV
                candle_data = await self.load_last_complete_candle_from_csv()
                
                if candle_data and candle_data.timestamp != last_processed_timestamp:
                    last_processed_timestamp = candle_data.timestamp
                    
                    # Update trailing stop loss with actual candle data
                    await self.update_trailing_stop_loss(candle_data)
                    logger.info(f"Processed new candle for trailing SL: {candle_data.timestamp}")
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Continuous trailing stop loss error: {e}")
                await asyncio.sleep(1)

    async def load_last_complete_candle_from_csv(self, csv_path: str = "main_csv.csv") -> Optional[CandleData]:
        """Load the last complete candle with all OHLC data from CSV"""
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return None
            
            # Get the last row
            last_row = df.tail(1).to_dict(orient='records')[0]
            
            # Check if all required fields are present and valid
            required_fields = ['timestamp', 'open', 'high', 'low', 'close']
            for field in required_fields:
                if field not in last_row or pd.isna(last_row[field]):
                    return None
            
            # Create CandleData object with actual values
            return CandleData(
                open=float(last_row['open']),
                high=float(last_row['high']),
                low=float(last_row['low']),
                close=float(last_row['close']),
                timestamp=str(last_row['timestamp'])
            )
            
        except Exception as e:
            logger.warning(f"Failed to load complete candle from CSV: {e}")
            return None
        




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

    async def save_trailing_data(self, trailing_data: TrailingData):
        try:
            await self.r.set(
                f"{TRAILING_DATA_KEY}:{trailing_data.position_token}",
                json.dumps(asdict(trailing_data)),
                ex=3600  # Expire after 1 hour
            )
        except Exception as e:
            logger.error(f"save_trailing_data error: {e}")

    async def get_trailing_data(self, position_token: str) -> Optional[TrailingData]:
        try:
            data = await self.r.get(f"{TRAILING_DATA_KEY}:{position_token}")
            if data:
                d = json.loads(data)
                return TrailingData(
                    position_token=d["position_token"],
                    extreme_price=float(d["extreme_price"]),
                    current_stop_loss=float(d["current_stop_loss"]),
                    last_updated=d["last_updated"]
                )
        except Exception as e:
            logger.error(f"get_trailing_data error: {e}")
        return None

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
                        highest_close=float(pdata.get("highest_close", 0.0)),
                        lowest_close=float(pdata.get("lowest_close", 0.0)),
                    )
                    logger.info(f"Recovered open position: {self.open_pos}")
                    return True
        except Exception as e:
            logger.error(f"recover_open_position error: {e}")
        return False


    async def update_trailing_stop_loss(self, candle: CandleData):
        """Update trailing stop loss based on complete candle data"""
        if not self.open_pos:
            return

        pos = self.open_pos
        current_close = candle.close

        if pos.option_type == "CE":
            # For CALL options (long position)
            if current_close > pos.highest_close:
                pos.highest_close = current_close
                new_stop_loss = candle.low - TRAILING_OFFSET
                
                logger.info(f"CE: New high {current_close:.2f}, candle low: {candle.low:.2f}, proposed SL: {new_stop_loss:.2f}, current SL: {pos.stop_loss:.2f}")
                
                if new_stop_loss > pos.stop_loss:
                    pos.stop_loss = new_stop_loss
                    logger.info(f"✅ Trailing SL updated for CE: {pos.stop_loss:.2f}")
                    await self.save_position(pos)
                else:
                    logger.info(f"ℹ️ CE: New SL {new_stop_loss:.2f} not higher than current SL {pos.stop_loss:.2f}")

        elif pos.option_type == "PE":
            # For PUT options (short position)
            if current_close < pos.lowest_close or pos.lowest_close == 0:
                pos.lowest_close = current_close
                new_stop_loss = candle.high + TRAILING_OFFSET
                
                logger.info(f"PE: New low {current_close:.2f}, candle high: {candle.high:.2f}, proposed SL: {new_stop_loss:.2f}, current SL: {pos.stop_loss:.2f}")
                
                if new_stop_loss < pos.stop_loss or pos.stop_loss == 0:
                    pos.stop_loss = new_stop_loss
                    logger.info(f"✅ Trailing SL updated for PE: {pos.stop_loss:.2f}")
                    await self.save_position(pos)
                else:
                    logger.info(f"ℹ️ PE: New SL {new_stop_loss:.2f} not lower than current SL {pos.stop_loss:.2f}")
    # ---------- trailing stop loss logic ----------
    # async def update_trailing_stop_loss(self, candle: CandleData):
    #     print("coming inside the update trailing stop loss")
    #     if not self.open_pos:
    #         return

    #     pos = self.open_pos
    #     current_close = candle.close

    #     if pos.option_type == "CE":
    #         # Update highest close if current close is higher
    #         if current_close > pos.highest_close:
    #             pos.highest_close = current_close
                
    #             logging.info(f"New stoploss should be : {candle.low}")
    #             new_stop_loss = candle.low
    #             if new_stop_loss > pos.stop_loss:
    #                 pos.stop_loss = new_stop_loss
    #                 logger.info(f"Trailing SL updated for CE: {pos.stop_loss}")
    #                 await self.save_position(pos)

    #     elif pos.option_type == "PE":  # For PUT options (short position)
    #         # Update lowest close if current close is lower
    #         if current_close < pos.lowest_close or pos.lowest_close == 0:
    #             pos.lowest_close = current_close
    #             # Trail stop loss to the high of the signal candle with offset
    #             new_stop_loss = candle.high
    #             logging.info(f"New stoploss should be : {candle.high}")
    #             if new_stop_loss < pos.stop_loss or pos.stop_loss == 0:
    #                 pos.stop_loss = new_stop_loss
    #                 logger.info(f"Trailing SL updated for PE: {pos.stop_loss}")
    #                 await self.save_position(pos)

    async def handle_candle_close(self, candle_data: Dict[str, Any]):
        """Handle new candle close event for trailing stop loss"""
        try:
            candle = CandleData(
                open=float(candle_data['open']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                close=float(candle_data['close']),
                timestamp=candle_data['timestamp']
            )
            self.last_candle = candle
            
            if self.open_pos:
                await self.update_trailing_stop_loss(candle)
                
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid candle data: {candle_data}, error: {e}")

    # ---------- trade actions ----------
    async def take_buy(self, quantity: int = 1, signal_candle: Optional[CandleData] = None) -> bool:
        if not self.ce_token:
            logger.error("CE token not loaded")
            return False
        
        price = await self.get_current_price(self.ce_token)
        if price <= 0:
            logger.error(f"Invalid price for CE token {self.ce_token}: {price}")
            return False
        
        # Use provided signal candle or get current candle data
        if signal_candle is None:
            high, low = await self.get_candle()
            if low is None:
                logger.error("No candle data available for SL")
                return False
            initial_sl = low - TRAILING_OFFSET
        else:
            initial_sl = signal_candle.low - TRAILING_OFFSET
        
        pos = Position(
            token=str(self.ce_token),
            trading_symbol=str(self.ce_symbol),
            position_type="BUY",
            option_type="CE",
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=initial_sl,
            highest_close=price,  # Initialize with entry price
        )
        self.open_pos = pos
        await self.save_position(pos)
        logger.info(f"BUY CE @ {price} Initial SL {initial_sl}")
        return True

    async def take_sell(self, quantity: int = 1, signal_candle: Optional[CandleData] = None) -> bool:
        if not self.pe_token:
            logger.error("PE token not loaded")
            return False
        
        price = await self.get_current_price(self.pe_token)
        if price <= 0:
            logger.error(f"Invalid price for PE token {self.pe_token}: {price}")
            return False
        
        # Use provided signal candle or get current candle data
        if signal_candle is None:
            high, low = await self.get_candle()
            if high is None:
                logger.error("No candle data available for SL")
                return False
            initial_sl = high + TRAILING_OFFSET
        else:
            initial_sl = signal_candle.high + TRAILING_OFFSET
        
        pos = Position(
            token=str(self.pe_token),
            trading_symbol=str(self.pe_symbol),
            position_type="SELL",
            option_type="PE",
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=initial_sl,
            lowest_close=price,  # Initialize with entry price
        )
        self.open_pos = pos
        await self.save_position(pos)
        logger.info(f"SELL PE @ {price} Initial SL {initial_sl}")
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
            if pos.position_type == "BUY":
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

    # ---------- market hours check ----------
    def is_market_hours(self) -> bool:
        now = datetime.now()
        current_time = now.time()
        return (self.market_open_time.time() <= current_time <= self.market_close_time.time())

    # ---------- event handlers ----------
    async def handle_index_tick(self, price: float):

        if not self.open_pos or not self.is_market_hours():
            return
        
        pos = self.open_pos
        logger.info(f"Current price: {price}, Stop loss: {pos.stop_loss}")
        
        # Check if stop loss is hit
        if pos.option_type == "CE" and price <= pos.stop_loss:
            await self.close_position("STOP_LOSS")
        elif pos.option_type == "PE" and price >= pos.stop_loss:
            await self.close_position("STOP_LOSS")

    async def on_buy_signal(self, signal_candle: Optional[CandleData] = None):
        if not self.is_market_hours():
            return
            
        if self.open_pos and self.open_pos.option_type == "PE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_buy(quantity=1, signal_candle=signal_candle)

    async def on_sell_signal(self, signal_candle: Optional[CandleData] = None):
        if not self.is_market_hours():
            return
            
        if self.open_pos and self.open_pos.option_type == "CE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_sell(quantity=1, signal_candle=signal_candle)

    async def task_pubsub_listener(self):
        logging.info("coming inside the task pubsub listener function")
        """Listens for Pub/Sub messages for prices, signals, and candle closes."""
        try:
            psub = self.r.pubsub()
            await psub.subscribe(
                f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}", 
                CHAN_SIGNAL_BUY, 
                CHAN_SIGNAL_SELL,
                CHAN_CANDLE_CLOSE  # Subscribe to candle close events
            )
            logger.info("Subscribed to Pub/Sub channels")
            
            async for msg in psub.listen():

                logging.info(f"Received message : {msg}")
                
                if not self.is_market_hours():
                    logger.debug("Outside market hours, skipping message")
                    continue
                    
                if msg is None or msg.get("type") != "message":
                    logger.debug(f"Skipping non-message: {msg}")
                    continue
                    
                channel = msg.get("channel")
                logging.info(f"channel is : {channel}") 
                data = msg.get("data")
                print(data)
                
                logger.info(f"Processing channel: {channel}, data: {data}")
                
                try:
                    if channel == f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}":
                        logging.info("coming inside the channel price buy --------------->")
                        price = float(data)
                        await self.handle_index_tick(price)
                    elif channel == CHAN_SIGNAL_BUY:
                        logging.info("coming inside the channel signal buy --------------->")
                        # Parse candle data if provided with signal
                        try:
                            candle_data = json.loads(data) if data and isinstance(data, str) else {}
                            signal_candle = CandleData(**candle_data) if candle_data and 'open' in candle_data else None
                            await self.on_buy_signal(signal_candle)
                        except json.JSONDecodeError:
                            # If data is not JSON, assume it's a simple signal without candle data
                            await self.on_buy_signal(None)
                    elif channel == CHAN_SIGNAL_SELL:
                        logging.info("coming inside the channel signal sell --------------->")
                        # Parse candle data if provided with signal
                        try:
                            candle_data = json.loads(data) if data and isinstance(data, str) else {}
                            signal_candle = CandleData(**candle_data) if candle_data and 'open' in candle_data else None
                            await self.on_sell_signal(signal_candle)
                        except json.JSONDecodeError:
                            # If data is not JSON, assume it's a simple signal without candle data
                            await self.on_sell_signal(None)
                    elif channel == CHAN_CANDLE_CLOSE:
                        logging.info("coming inside the channel candle close --------------->")
                        # Handle candle close for trailing stop loss
                        try:
                            candle_data = json.loads(data)
                            await self.handle_candle_close(candle_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse candle data: {data}, error: {e}")
                    else:
                        logger.warning(f"Unknown channel: {channel}")

                    
                    
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(f"Invalid message data: {data}, error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"PubSub listener error: {e}")

    async def task_key_fallback_poller(self):
        """Fallback for setups that only set Redis keys."""
        last_buy = None
        last_sell = None
        
        while True:
            if not self.is_market_hours():
                await asyncio.sleep(5)
                continue

            try:
                # Check stop loss with current index price
                price = await self.get_index_price()
                if price > 0:
                    await self.handle_index_tick(price)

                # Check for signal changes
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
            
            if now.time() >= self.square_off_time.time() and self.is_market_hours():
                if self.open_pos:
                    await self.close_position("SQUARE_OFF")
                logger.info("Square-off completed")
                
                # Calculate daily P&L
                data = await self.r.get(f"trade_history_{datetime.now().date()}")
                total_profit_or_loss = 0

                if data:
                    try:
                        trade_history = json.loads(data)
                        for trade in trade_history:
                            entry_time = datetime.fromisoformat(trade["entry_time"])
                            if entry_time.date() == datetime.today().date():
                                total_profit_or_loss += float(trade["pnl"])
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Error calculating P&L: {e}")

                logger.info(f"Total P&L for the day: {total_profit_or_loss}")
                
                await asyncio.sleep((self.market_close_time - now).total_seconds())
            
            # Check if market is closed and wait for next open
            elif not self.is_market_hours() and now.time() > self.market_close_time.time():
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=9, minute=15, second=0, microsecond=0
                )
                sleep_seconds = (tomorrow - now).total_seconds()
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
        logger.info("TradingBot starting with trailing stop loss…")
        
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