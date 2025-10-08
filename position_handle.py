# from datetime import datetime, timedelta
# import time
# import redis
# import json
# import logging
# import os, pandas as pd
# import asyncio

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# class TradingBot:
#     def __init__(self):
#         self.positions_key = "active_positions"
#         self.trade_history_key = "trade_history"
#         self.trading_symbols_key = "Trading_symbol"
#         self.ce_trading_symbol = None
#         self.pe_trading_symbol = None
#         self.buy_signal_check = False
#         self.sell_signal_check = False
#         self.buy_trade_already_triggered = False
#         self.sell_trade_already_triggered = False
#         self.stoploss = 0
#         self.ce_token = 0
#         self.pe_token = 0
#         self.entry_price = 0
#         self.quantity = 1
#         self.option_signal = None
#         self.position_signal = None
#         self.entry_time = None
#         self.market_open_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
#         self.market_close_time = datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)
#         self.square_off_time = datetime.now().replace(hour=16, minute=25, second=0, microsecond=0)
    
#     def get_current_price(self, symbol):
#         try:
#             if r.exists(f"{symbol}"):
#                 price = r.get(f"{symbol}")
#                 return float(price) if price else 0
#             else:
#                 return 0
#         except:
#             return 0

#     def get_nifty_current_index(self):
#         try:
#             if r.exists("99926000"):
#                 current_index = r.get("99926000")
#                 return float(current_index) if current_index else 0
#             else:
#                 return 0
#         except:
#             return 0
    
#     def get_candle_data(self):
#         try:
#             data = self.get_last_row()
#             high = data.get('high')
#             low = data.get('low')
#             return high, low
#         except:
#             return 0, 0
    
#     def save_position_to_redis(self, token, position_data):
#         try:
#             positions = {}
#             if r.exists(self.positions_key):
#                 positions = json.loads(r.get(self.positions_key))
            
#             positions[str(token)] = position_data
            
#             r.set(self.positions_key, json.dumps(positions))
#             logger.info(f"Position saved to Redis: {token} - {position_data}")
            
#         except Exception as e:
#             logger.error(f"Error saving position to Redis: {e}")
    
#     def remove_position_from_redis(self, token):
#         try:
#             if r.exists(self.positions_key):
#                 positions = json.loads(r.get(self.positions_key))
#                 if str(token) in positions:
#                     del positions[str(token)]
#                     r.set(self.positions_key, json.dumps(positions))
#                     logger.info(f"Position removed from Redis: {token}")
#         except Exception as e:
#             logger.error(f"Error removing position from Redis: {e}")
    
#     def save_trade_to_history(self, trade_data):
#         try:
#             history = []
#             if r.exists(self.trade_history_key):
#                 history = json.loads(r.get(self.trade_history_key))
            
#             history.append(trade_data)
            
#             if len(history) > 100:
#                 history = history[-100:]
            
#             r.set(self.trade_history_key, json.dumps(history))
#             logger.info(f"Trade saved to history: {trade_data}")
            
#         except Exception as e:
#             logger.error(f"Error saving trade to history: {e}")

#     def get_last_row(self):
#         file_path = "main_csv.csv"
#         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#             return None
#         df = pd.read_csv(file_path)
#         if df.empty:
#             return None
#         return df.tail(1).to_dict(orient='records')[0]

#     def get_trading_symbols(self):
#         try:
#             if r.exists(self.trading_symbols_key):
#                 trading_symbols = json.loads(r.get(self.trading_symbols_key))
#                 if "CE" in trading_symbols and "PE" in trading_symbols:
#                     self.ce_trading_symbol = trading_symbols["CE"][0]
#                     self.ce_token = trading_symbols["CE"][1]
#                     self.pe_trading_symbol = trading_symbols["PE"][0]
#                     self.pe_token = trading_symbols["PE"][1]
#                     print(f"Trading symbols fetched: CE - {self.ce_trading_symbol}, PE - {self.pe_trading_symbol}")
#                     print(f"Tokens fetched: CE - {self.ce_token}, PE - {self.pe_token}")
#             else:
#                 while not r.exists(self.trading_symbols_key):
#                     time.sleep(1)
#         except Exception as e:
#             logger.error(f"Error fetching trading symbols: {e}")
#             return {}

#     def take_buy_position(self, quantity=1):
#         try:
#             current_price = self.get_current_price(self.ce_token)
#             print("This is a buy position and check the current_price",current_price)
#             candle_high, candle_low = self.get_candle_data()
            
#             if current_price == 0:
#                 logger.error(f"Invalid price for CE token: {self.ce_token}")
#                 return False
            
#             position_data = {
#                 "ce_token": self.ce_token,
#                 "ce_trading_symbol": self.ce_trading_symbol,
#                 "position_type": "BUY",
#                 "option_type": "CE",
#                 "entry_price": current_price,
#                 "quantity": quantity,
#                 "entry_time": datetime.now().isoformat(),
#                 "stop_loss": candle_low,
#                 "status": "OPEN"
#             }
#             self.quantity = quantity
#             self.position_signal = "BUY"
#             self.option_signal = "CE"
#             self.entry_price = current_price
#             self.stoploss = candle_low
#             self.buy_signal_check = True


#             self.save_position_to_redis(self.ce_token, position_data)

#             logger.info(f"BUY position taken: CE token {self.ce_token} at {current_price}, SL: {candle_low}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error taking buy position: {e}")
#             return False
    
#     def take_sell_position(self, quantity=1):
#         try:
#             current_price = self.get_current_price(self.pe_token)
#             candle_high, candle_low = self.get_candle_data()
            
#             if current_price == 0:
#                 logger.error(f"Invalid price for PE token: {self.pe_token}")
#                 return False
            
#             position_data = {
#                 "pe_token": self.pe_token,
#                 "pe_trading_symbol": self.pe_trading_symbol,
#                 "position_type": "SELL",
#                 "option_type": "PE",
#                 "entry_price": current_price,
#                 "quantity": quantity,
#                 "entry_time": datetime.now().isoformat(),
#                 "stop_loss": candle_high,
#                 "status": "OPEN"
#             }
#             self.quantity = quantity
#             self.position_signal = "SELL"
#             self.option_signal = "PE"
#             self.entry_price = current_price
#             self.stoploss = candle_high
#             self.sell_signal_check = True

#             self.save_position_to_redis(self.pe_token, position_data)

#             logger.info(f"SELL position taken: PE token {self.pe_token} at {current_price}, SL: {candle_high}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error taking sell position: {e}")
#             return False
    
#     async def check_stop_loss(self):
#         try:
#             if self.buy_signal_check == False and self.sell_signal_check == False:
#                     logger.info("No open positions to check for stop loss")
#                     return
            
#             current_index = self.get_nifty_current_index()
#             logging.info(f"Current Nifty index: {current_index}")

#             if current_index == 0:
#                 logger.warning(f"Could not get current index for token: 99926000")
#                 return
            
#             if self.option_signal == "CE":
#                 if current_index <= self.stoploss:
#                     self.close_position(self.ce_token, "STOP_LOSS")
#                     self.buy_trade_already_triggered = True
#                     logger.info(f"Stop loss hit for BUY position: token {self.ce_token} at {current_index}")

#             elif self.option_signal == "PE":
#                 if current_index >= self.stoploss:
#                     self.close_position(self.pe_token, "STOP_LOSS")
#                     self.sell_trade_already_triggered = True
#                     logger.info(f"Stop loss hit for SELL position: token {self.pe_token} at {current_index}")
                
#         except Exception as e:
#             logger.error(f"Error checking stop loss: {e}")

#     def close_position(self, token, reason="MANUAL"):
#         try:
#             current_price = self.get_current_price(token)
            
#             if current_price == 0:
#                 logger.error(f"Could not get current price for token: {token}")
#                 return False
            
#             if self.option_signal == "CE":
#                 pnl = (current_price - self.entry_price) * self.quantity
#             else:
#                 pnl = (self.entry_price - self.entry_price) * self.quantity
            
#             trade_data = {
#                 "token": token,
#                 "option_type": self.option_signal,
#                 "position_type": self.position_signal,
#                 "entry_price": self.entry_price,
#                 "exit_price": current_price,
#                 "quantity": self.quantity,
#                 "entry_time": self.entry_time,
#                 "exit_time": datetime.now().isoformat(),
#                 "stop_loss": self.stoploss,
#                 "pnl": round(pnl, 2),
#                 "close_reason": reason
#             }
#             self.buy_signal_check = False
#             self.sell_signal_check = False
#             self.option_signal = None
#             self.position_signal = None
#             self.entry_price = 0
#             self.stoploss = 0
#             self.quantity = 1
#             self.entry_time = None

#             self.save_trade_to_history(trade_data)     
#             self.remove_position_from_redis(token)

#             logger.info(f"Position closed: token {token}, P&L: {pnl}, Reason: {reason}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error closing position: {e}")
#             return False
    
#     def square_off_all_positions(self):
#         try:
#             if not r.exists(self.positions_key):
#                 logger.info("No positions to square off")
#                 return
            
#             positions = json.loads(r.get(self.positions_key))
            
#             for symbol in list(positions.keys()):
#                 self.close_position(symbol, "SQUARE_OFF")
            
#             logger.info("All positions squared off for market close")
            
#         except Exception as e:
#             logger.error(f"Error squaring off positions: {e}")
    
#     def has_open_position(self):
#         try:
#             if r.exists(self.positions_key):
#                 positions_data = json.loads(r.get(self.positions_key))
#                 positions_count = len(positions_data)
                
#                 if positions_count > 0:
#                     token, position_data = next(iter(positions_data.items()))
#                     self.get_trading_symbols()
                    
#                     if position_data["option_type"] == "CE":
#                         self.option_signal = "CE"
#                         self.buy_signal_check = True
#                         self.sell_signal_check = False
#                         self.entry_price = float(position_data["entry_price"])
#                         self.quantity = int(position_data["quantity"])
#                         self.stoploss = float(position_data["stop_loss"])
#                         self.position_signal = position_data["position_type"]
#                         self.entry_time = position_data["entry_time"]
#                         return "CE", positions_count

#                     elif position_data["option_type"] == "PE":
#                         self.option_signal = "PE"
#                         self.buy_signal_check = False
#                         self.sell_signal_check = True
#                         self.entry_price = float(position_data["entry_price"])
#                         self.quantity = int(position_data["quantity"])
#                         self.stoploss = float(position_data["stop_loss"])
#                         self.position_signal = position_data["position_type"]
#                         self.entry_time = position_data["entry_time"]
#                         return "PE", positions_count

#                 return None, 0
#             return None, 0
#         except Exception as e:
#             logger.error(f"Error checking open positions: {e}")
#             return None, 0

#     async def check_opposite_signal_from_main_csv(self):
#         try:
#             if self.buy_signal_check == True:
#                 signal = r.get("sell_signal")
#                 if signal == "true":
#                     self.close_position(self.ce_token, "STOP_LOSS")
#                     self.buy_signal_check = False
#                     self.sell_signal_check = True
#                     self.quantity = 1
#                     self.position_signal = None
#                     self.option_signal = None
#                     self.entry_price = 0
#                     self.stoploss = 0
#                     logger.info("sell signal detected and coming outside from the buy position!")

#             elif self.sell_signal_check == True:
#                 signal = r.get("buy_signal")
#                 if signal == "true":
#                     self.close_position(self.pe_token, "STOP_LOSS")
#                     self.sell_signal_check = False
#                     self.buy_signal_check = True
#                     self.quantity = 1
#                     self.position_signal = None
#                     self.option_signal = None
#                     self.entry_price = 0
#                     self.stoploss = 0
#                     logger.info("Buy signal detected and coming outside from the sell position!")

#         except Exception as e:
#             logger.error(f"Error checking opposite signals: {e}")


#     async def run_async(self):
#         logger.info("Trading bot started")
#         logger.info(f"Market hours: {self.market_open_time.strftime('%H:%M')} - {self.market_close_time.strftime('%H:%M')}")
#         logger.info(f"Square off time: {self.square_off_time.strftime('%H:%M')}")
#         self.has_open_position()
#         while True:
#             datetime_now = datetime.now()
#             if not (self.market_open_time.time() <= datetime_now.time() <= self.market_close_time.time()):
#                 logger.info("Outside market hours, waiting...")
#                 await asyncio.sleep(60)
#                 continue
            
#             if datetime_now.time() >= self.square_off_time.time():
#                 self.square_off_all_positions()
#                 logger.info("Market close approaching, all positions squared off")
#                 break
            
#             try:
#                 if self.position_signal != None:
#                     await self.check_stop_loss()

#                 if r.exists("buy_signal") and not self.buy_signal_check and not self.buy_trade_already_triggered:
#                     print("print the self.buy signal",self.buy_signal_check)
#                     signal = r.get("buy_signal")
#                     if signal == "true":
#                         logger.info("Buy signal detected!")
#                         if self.take_buy_position():
#                             self.sell_trade_already_triggered = False
#                             pass

#                 elif r.exists("sell_signal") and not self.sell_signal_check and not self.sell_trade_already_triggered:
#                     signal = r.get("sell_signal")
#                     if signal == "true":
#                         logger.info("Sell signal detected!")
#                         if self.take_sell_position():
#                             self.buy_trade_already_triggered = False
#                             pass

#                 await self.check_opposite_signal_from_main_csv()

#             except Exception as e:
#                 logger.error(f"Error in main loop: {e}")
            
#             await asyncio.sleep(5)
        
#         logger.info("Trading bot stopped")

#     def run(self):
#         asyncio.run(self.run_async())


# if __name__ == "__main__":
#     bot = TradingBot()
#     try:
#         bot.get_trading_symbols()
#         bot.run()
#     except KeyboardInterrupt:
#         logger.info("Bot stopped by user")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")



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

POSITIONS_KEY = "active_positions"
TRADE_HISTORY_KEY = f"trade_history_{datetime.now().date()}"
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
            socket_timeout=5,  # Add timeout to prevent hanging
            socket_connect_timeout=5
        )
        self.ce_symbol: Optional[str] = None
        self.ce_token: Optional[str] = None
        self.pe_symbol: Optional[str] = None
        self.pe_token: Optional[str] = None
        self.open_pos: Optional[Position] = None
        self._close_lock = asyncio.Lock()
        
        now = datetime.now()
        self.market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        self.market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        self.square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
        self._is_market_hours = False

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

    # ---------- trade actions ----------
    async def take_buy(self, quantity: int = 1) -> bool:
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
        print(f"Current price: {price}, Stop loss: {pos.stop_loss}")
        if pos.option_type == "CE" and price <= pos.stop_loss:
            await self.close_position("STOP_LOSS")
        elif pos.option_type == "PE" and price >= pos.stop_loss:
            await self.close_position("STOP_LOSS")

    async def on_buy_signal(self):
        if not self.is_market_hours():
            return
            
        if self.open_pos and self.open_pos.option_type == "PE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_buy(quantity=1)

    async def on_sell_signal(self):
        if not self.is_market_hours():
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
                    # await asyncio.sleep(1)############################################################################
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
                
            await asyncio.sleep(0.1)  # 100ms polling interval

    async def task_square_off_scheduler(self):
        while True:
            now = datetime.now()
            
            if now.time() >= self.square_off_time.time() and self.is_market_hours():
                if self.open_pos:
                    await self.close_position("SQUARE_OFF")
                logger.info("Square-off completed")
                data = r.get(f"trade_history_{datetime.now().date()}")

                total_profit_or_loss = 0

                if data:
                    trade_history = json.loads(data)
                    for i in trade_history:
                        entry_time = datetime.fromisoformat(i["entry_time"])
                        if entry_time.date() == datetime.today().date():
                            total_profit_or_loss += float(i["pnl"])

                logging.info(f"Total P&L for the day: {total_profit_or_loss}")
                
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