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
import time

# --------------------------- config ---------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "Rahul@7355"
REDIS_DB = 0

POSITIONS_KEY = "active_positions_points"
TRADE_HISTORY_KEY = f"trade_history_points_{datetime.now().date()}"
TRADING_SYMBOLS_KEY = "Trading_symbol"
CANDLE_CACHE_KEY = "candle:last"
TRAILING_DATA_KEY = "trailing_data_points"
PNL_TRACKER_KEY = "pnl_tracker_points"
NEXT_QUANTITY_KEY = "next_quantity_points"

CHAN_PRICE_PREFIX = "price_points:"
CHAN_SIGNAL_BUY = "signal_points:buy"
CHAN_SIGNAL_SELL = "signal_points:sell"
CHAN_CANDLE_CLOSE = "candle_points:close"

KEY_BUY_SIGNAL = "buy_signal"
KEY_SELL_SIGNAL = "sell_signal"

INDEX_TOKEN = "99926000"
TRAILING_OFFSET = 0
ATR_THRESHOLD = 10
ATR_REDIS_KEY = "ATR_value"

ATR_MULTIPLIER_LOW_VOL = 1.1
ATR_MULTIPLIER_HIGH_VOL = 1.5
ATR_VOLATILITY_THRESHOLD = 12

# Recovery system thresholds
LOSS_8_THRESHOLD = 8
LOSS_20_THRESHOLD = 20
LOSS_40_THRESHOLD = 40
LOSS_60_THRESHOLD = 60
PROFIT_TARGET = 35
PARTIAL_EXIT_PROFIT = 5

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
    highest_close: float = 0.0
    lowest_close: float = 0.0
    atr_value: float = 0.0
    atr_multiplier: float = 0.0
    recovery_position: bool = False
    base_quantity: int = 1

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
    extreme_price: float
    current_stop_loss: float
    last_updated: str

@dataclass
class PnLTracker:
    total_points: float = 0.0
    secured_profit: float = 0.0
    recovery_points: float = 0.0
    current_quantity: int = 1
    base_quantity: int = 1
    recovery_mode: bool = False
    last_updated: str = ""

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
        self.atr_value: float = 0.0
        self.pnl_tracker: PnLTracker = PnLTracker()
        self.next_trade_quantity: int = 1
        
        now = datetime.now()
        self.market_open_time = now.replace(hour=9, minute=17, second=1, microsecond=0)
        self.market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        self.market_denied_position = now.replace(hour=15, minute=5, second=0, microsecond=0)
        self.square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
        self._is_market_hours = False

    async def get_current_atr(self) -> float:
        """Get current ATR value from Redis""" 
        try:
            atr_str = await self.r.get(ATR_REDIS_KEY)
            if atr_str:
                return float(atr_str)
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing ATR value from Redis: {e}")
        except Exception as e:
            logger.error(f"Error getting ATR from Redis: {e}")
        return 0.0

    def get_dynamic_atr_multiplier(self, atr_value: float) -> float:
        if atr_value <= ATR_VOLATILITY_THRESHOLD:
            return ATR_MULTIPLIER_LOW_VOL
        elif atr_value < 15:
            return 1.2
        elif atr_value < 17:
            return 1.3
        elif atr_value < 20:
            return 1.4
        else:
            return ATR_MULTIPLIER_HIGH_VOL

    async def is_atr_above_threshold(self) -> bool:
        """Check if ATR is above the threshold for trading"""
        self.atr_value = await self.get_current_atr()
        return self.atr_value > ATR_THRESHOLD
    
    async def load_last_complete_candle_from_csv(self, csv_path: str = "main_csv.csv") -> Optional[CandleData]:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        
        try:
            df = pd.read_csv(csv_path).tail(5)
            if df.empty:
                return None
            last_row = df.tail(1).to_dict(orient='records')[0]
            
            required_fields = ['timestamp', 'open', 'high', 'low', 'close']
            for field in required_fields:
                if field not in last_row or pd.isna(last_row[field]):
                    return None
            
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

    # ==================== PNL TRACKING & RECOVERY SYSTEM ====================
    
    async def load_pnl_tracker(self) -> PnLTracker:
        """Load PnL tracker from Redis"""
        try:
            data = await self.r.get(PNL_TRACKER_KEY)
            if data:
                d = json.loads(data)
                return PnLTracker(
                    total_points=float(d.get("total_points", 0.0)),
                    secured_profit=float(d.get("secured_profit", 0.0)),
                    recovery_points=float(d.get("recovery_points", 0.0)),
                    current_quantity=int(d.get("current_quantity", 1)),
                    base_quantity=int(d.get("base_quantity", 1)),
                    recovery_mode=bool(d.get("recovery_mode", False)),
                    last_updated=d.get("last_updated", datetime.now().isoformat())
                )
        except Exception as e:
            logger.error(f"Error loading PnL tracker: {e}")
        return PnLTracker(last_updated=datetime.now().isoformat())

    async def save_pnl_tracker(self):
        """Save PnL tracker to Redis"""
        try:
            self.pnl_tracker.last_updated = datetime.now().isoformat()
            await self.r.set(PNL_TRACKER_KEY, json.dumps(asdict(self.pnl_tracker)))
            logger.info(f"💰 PnL Tracker: Total={self.pnl_tracker.total_points:.2f}, "
                       f"Secured={self.pnl_tracker.secured_profit:.2f}, "
                       f"Recovery={self.pnl_tracker.recovery_points:.2f}, "
                       f"NextQty={self.pnl_tracker.current_quantity}")
        except Exception as e:
            logger.error(f"Error saving PnL tracker: {e}")

    def calculate_next_quantity(self, total_points: float) -> int:
        """Calculate next quantity based on total points (loss-based)"""
        loss = abs(total_points) if total_points < 0 else 0
        
        if loss >= LOSS_60_THRESHOLD:
            return 5
        elif loss >= LOSS_40_THRESHOLD:
            return 4
        elif loss >= LOSS_20_THRESHOLD:
            return 3
        elif loss >= LOSS_8_THRESHOLD:
            return 2
        else:
            return 1

    async def update_pnl_after_trade(self, trade_pnl: float):
        """Update PnL tracking after a trade closes"""
        self.pnl_tracker.total_points += trade_pnl
        
        # Check if we're in recovery mode (loss greater than 8)
        loss = abs(self.pnl_tracker.total_points) if self.pnl_tracker.total_points < 0 else 0
        
        if loss >= LOSS_8_THRESHOLD:
            self.pnl_tracker.recovery_mode = True
            self.pnl_tracker.recovery_points = loss
            
            # Calculate next quantity for recovery
            self.pnl_tracker.current_quantity = self.calculate_next_quantity(self.pnl_tracker.total_points)
            self.next_trade_quantity = self.pnl_tracker.current_quantity
            
            logger.warning(f"⚠️ RECOVERY MODE: Loss={loss:.2f} points, "
                          f"Next Quantity={self.pnl_tracker.current_quantity}x")
        
        elif self.pnl_tracker.recovery_mode and self.pnl_tracker.total_points >= 0:
            # Exited recovery mode - back to profit or breakeven
            logger.info(f"✅ Recovery successful! Current points: {self.pnl_tracker.total_points:.2f}")
            self.pnl_tracker.recovery_points = 0.0
            self.pnl_tracker.recovery_mode = False
            self.pnl_tracker.current_quantity = 1
            self.next_trade_quantity = 1
        
        # Add positive points to secured profit when total points positive
        if self.pnl_tracker.total_points > 0 and not self.pnl_tracker.recovery_mode:
            self.pnl_tracker.secured_profit += self.pnl_tracker.total_points
            logger.info(f"💎 Profit added to secured: {self.pnl_tracker.total_points:.2f}, "
                       f"Total secured: {self.pnl_tracker.secured_profit:.2f}")
            self.pnl_tracker.total_points = 0.0
            self.pnl_tracker.current_quantity = 1
            self.next_trade_quantity = 1
        
        # Save updated tracker
        await self.save_pnl_tracker()
        await self.r.set(NEXT_QUANTITY_KEY, str(self.next_trade_quantity))

    async def check_profit_target_reached(self) -> bool:
        """Check if secured profit target is reached"""
        if self.pnl_tracker.secured_profit >= PROFIT_TARGET:
            logger.info(f"🎯 PROFIT TARGET REACHED! Secured: {self.pnl_tracker.secured_profit:.2f}")
            return True
        return False

    async def handle_partial_exit(self, position: Position, current_price: float):
        """Handle partial exit when in recovery mode with profit"""
        if not self.pnl_tracker.recovery_mode:
            return False
            
        if position.quantity <= 1:
            return False
        
        # Calculate current P&L per unit
        if position.position_type == "BUY":
            pnl_per_unit = current_price - position.entry_price
        else:
            # pnl_per_unit = position.entry_price - current_price
            pnl_per_unit = current_price - position.entry_price
        
        # Required recovery per unit = total loss / current quantity
        # Example: Loss 25 points, quantity 3 → need 8.33 points per unit
        required_recovery_per_unit = self.pnl_tracker.recovery_points / position.quantity
        
        logger.info(f"📊 Recovery Check: PnL per unit: {pnl_per_unit:.2f}, "
                   f"Required: {required_recovery_per_unit:.2f}, "
                   f"Quantity: {position.quantity}")
        
        if pnl_per_unit >= required_recovery_per_unit:
            # Exit all extra quantities, keep only 1
            exit_quantity = position.quantity - 1
            remaining_quantity = 1
            
            # Calculate partial PnL for exited units
            partial_pnl = pnl_per_unit * exit_quantity
            
            # Save partial exit trade
            tr = Trade(
                token=position.token,
                option_type=position.option_type,
                position_type=position.position_type,
                entry_price=position.entry_price,
                exit_price=current_price,
                quantity=exit_quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now().isoformat(),
                stop_loss=position.stop_loss,
                pnl=round(float(partial_pnl), 2),
                close_reason="PARTIAL_RECOVERY_EXIT",
            )
            await self.save_trade(tr)
            
            # Update position to keep only 1 quantity
            position.quantity = remaining_quantity
            position.base_quantity = remaining_quantity
            await self.save_position(position)
            
            # Update PnL tracker with the partial exit profit
            await self.update_pnl_after_trade(partial_pnl)
            
            logger.info(f"🔄 PARTIAL EXIT: Closed {exit_quantity} unit(s) @ {current_price:.2f}, "
                       f"Remaining: {remaining_quantity} unit, "
                       f"PnL from exit: {partial_pnl:.2f}, "
                       f"Required recovery per unit: {required_recovery_per_unit:.2f}")
            
            return True
        
        return False

    async def task_pnl_monitor(self):
        """Continuously monitor PnL and manage recovery system"""
        while True:
            try:
                if not self.is_market_hours():
                    await asyncio.sleep(5)
                    continue
                
                # Check profit target
                if await self.check_profit_target_reached():
                    logger.info("🛑 Profit target reached. No new positions allowed.")
                
                # Check for partial exits if in recovery mode - check every 0.5 seconds for fast response
                if self.open_pos and self.pnl_tracker.recovery_mode and self.open_pos.quantity > 1:
                    current_price = await self.get_current_price(self.open_pos.token)
                    if current_price > 0:
                        await self.handle_partial_exit(self.open_pos, current_price)
                
                await asyncio.sleep(0.5)  # Check more frequently for fast partial exits
                
            except Exception as e:
                logger.error(f"PnL monitor error: {e}")
                await asyncio.sleep(2)

    async def task_quantity_calculator(self):
        """Background task to pre-calculate next trade quantity"""
        while True:
            try:
                if self.is_market_hours():
                    # Pre-calculate next quantity based on current total points
                    next_qty = self.calculate_next_quantity(self.pnl_tracker.total_points)
                    
                    if next_qty != self.next_trade_quantity:
                        self.next_trade_quantity = next_qty
                        await self.r.set(NEXT_QUANTITY_KEY, str(next_qty))
                        logger.info(f"📊 Next trade quantity updated: {next_qty}x")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Quantity calculator error: {e}")
                await asyncio.sleep(5)

    # ==================== END PNL TRACKING & RECOVERY SYSTEM ====================

    async def task_continuous_trailing_stop_loss(self):
        last_processed_timestamp = None
        last_file_mod_time = 0
        
        while True:
            try:
                if not self.is_market_hours() or not self.open_pos:
                    await asyncio.sleep(1)
                    continue
                
                current_mod_time = os.path.getmtime("main_csv.csv") if os.path.exists("main_csv.csv") else 0
                if current_mod_time <= last_file_mod_time:
                    await asyncio.sleep(0.5)
                    continue
                    
                last_file_mod_time = current_mod_time
                
                candle_data = await self.load_last_complete_candle_from_csv()
                
                if candle_data and candle_data.timestamp != last_processed_timestamp:
                    last_processed_timestamp = candle_data.timestamp
                    
                    await self.update_trailing_stop_loss(candle_data)
                    logger.info(f"Processed new candle for trailing SL: {candle_data.timestamp}")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Continuous trailing stop loss error: {e}")
                await asyncio.sleep(1)

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
                ex=3600
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
                        atr_value=float(pdata.get("atr_value", 0.0)),
                        atr_multiplier=float(pdata.get("atr_multiplier", 0.0)),
                        recovery_position=bool(pdata.get("recovery_position", False)),
                        base_quantity=int(pdata.get("base_quantity", 1)),
                    )
                    logger.info(f"Recovered open position: {self.open_pos}")
                    return True
        except Exception as e:
            logger.error(f"recover_open_position error: {e}")
        return False

    async def update_trailing_stop_loss(self, candle: CandleData):
        if not self.open_pos:
            return

        pos = self.open_pos
        current_close = candle.close

        if pos.option_type == "CE":
            if current_close > pos.highest_close:
                pos.atr_value = await self.get_current_atr()
                pos.atr_multiplier = self.get_dynamic_atr_multiplier(pos.atr_value)
                pos.highest_close = current_close
                atr_offset = pos.atr_value * pos.atr_multiplier
                new_stop_loss = current_close - atr_offset
                
                logger.info(f"CE: New high {current_close:.2f}, ATR: {pos.atr_value:.2f}, Multiplier: {pos.atr_multiplier:.2f}, proposed SL: {new_stop_loss:.2f}, current SL: {pos.stop_loss:.2f}")
                
                if new_stop_loss > pos.stop_loss:
                    pos.stop_loss = new_stop_loss
                    logger.info(f"✅ Trailing SL updated for CE: {pos.stop_loss:.2f}")
                    await self.save_position(pos)
                else:
                    logger.info(f"ℹ️ CE: New SL {new_stop_loss:.2f} not higher than current SL {pos.stop_loss:.2f}")

        elif pos.option_type == "PE":
            if current_close < pos.lowest_close or pos.lowest_close == 0:
                pos.atr_value = await self.get_current_atr()
                pos.atr_multiplier = self.get_dynamic_atr_multiplier(pos.atr_value)
                pos.lowest_close = current_close
                atr_offset = pos.atr_value * pos.atr_multiplier
                new_stop_loss = current_close + atr_offset
                
                logger.info(f"PE: New low {current_close:.2f}, ATR: {pos.atr_value:.2f}, Multiplier: {pos.atr_multiplier:.2f}, proposed SL: {new_stop_loss:.2f}, current SL: {pos.stop_loss:.2f}")
                
                if new_stop_loss < pos.stop_loss or pos.stop_loss == 0:
                    pos.stop_loss = new_stop_loss
                    logger.info(f"✅ Trailing SL updated for PE: {pos.stop_loss:.2f}")
                    await self.save_position(pos)
                else:
                    logger.info(f"ℹ️ PE: New SL {new_stop_loss:.2f} not lower than current SL {pos.stop_loss:.2f}")

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

    async def take_buy(self, quantity: int = 1, signal_candle: Optional[CandleData] = None) -> bool:
        now = datetime.now()
        current_time = now.time()
        
        # Check profit target
        if await self.check_profit_target_reached():
            logger.info("🛑 Profit target reached. No new BUY positions.")
            return False
        
        if current_time < self.market_denied_position.time():
            if not await self.is_atr_above_threshold():
                logger.info(f"ATR {self.atr_value:.2f} below threshold {ATR_THRESHOLD}, skipping BUY")
                return False
                
            if not self.ce_token:
                logger.error("CE token not loaded")
                return False
            
            # Use pre-calculated quantity
            actual_quantity = self.next_trade_quantity
            
            price = await self.get_current_price(self.ce_token)
            current_index = await self.get_index_price()
            if price <= 0:
                logger.error(f"Invalid price for CE token {self.ce_token}: {price}")
                return False

            current_atr = await self.get_current_atr()
            atr_multiplier = self.get_dynamic_atr_multiplier(current_atr)
            
            if signal_candle is None:
                high, low = await self.get_candle()
                if low is None:
                    logger.error("No candle data available for SL")
                    return False
                initial_sl = current_index - (current_atr * atr_multiplier)
            else:
                initial_sl = current_index - (current_atr * atr_multiplier)
            
            pos = Position(
                token=str(self.ce_token),
                trading_symbol=str(self.ce_symbol),
                position_type="BUY",
                option_type="CE",
                entry_price=price,
                quantity=actual_quantity,
                entry_time=datetime.now().isoformat(),
                stop_loss=initial_sl,
                highest_close=signal_candle.close if signal_candle else price,
                atr_value=current_atr,
                atr_multiplier=atr_multiplier,
                recovery_position=self.pnl_tracker.recovery_mode,
                base_quantity=actual_quantity,
            )
            self.open_pos = pos
            await self.save_position(pos)
            logger.info(f"🟢 BUY CE @ {price} Qty={actual_quantity}x Initial SL {initial_sl:.2f} "
                       f"(ATR: {current_atr:.2f}, Multiplier: {atr_multiplier:.2f}) "
                       f"Recovery: {self.pnl_tracker.recovery_mode}")
            return True
        return False

    async def take_sell(self, quantity: int = 1, signal_candle: Optional[CandleData] = None) -> bool:
        now = datetime.now()
        current_time = now.time()
        
        # Check profit target
        if await self.check_profit_target_reached():
            logger.info("🛑 Profit target reached. No new SELL positions.")
            return False
        
        if current_time < self.market_denied_position.time():
            if not await self.is_atr_above_threshold():
                logger.info(f"ATR {self.atr_value:.2f} below threshold {ATR_THRESHOLD}, skipping SELL")
                return False
                
            if not self.pe_token:
                logger.error("PE token not loaded")
                return False
            
            # Use pre-calculated quantity
            actual_quantity = self.next_trade_quantity
            
            price = await self.get_current_price(self.pe_token)
            current_index = await self.get_index_price()
            if price <= 0:
                logger.error(f"Invalid price for PE token {self.pe_token}: {price}")
                return False
            
            current_atr = await self.get_current_atr()
            atr_multiplier = self.get_dynamic_atr_multiplier(current_atr)
            
            if signal_candle is None:
                high, low = await self.get_candle()
                if high is None:
                    logger.error("No candle data available for SL")
                    return False
                initial_sl = current_index + (current_atr * atr_multiplier)
            else:
                initial_sl = current_index + (current_atr * atr_multiplier)
            
            pos = Position(
                token=str(self.pe_token),
                trading_symbol=str(self.pe_symbol),
                position_type="SELL",
                option_type="PE",
                entry_price=price,
                quantity=actual_quantity,
                entry_time=datetime.now().isoformat(),
                stop_loss=initial_sl,
                lowest_close=signal_candle.close if signal_candle else price,
                atr_value=current_atr,
                atr_multiplier=atr_multiplier,
                recovery_position=self.pnl_tracker.recovery_mode,
                base_quantity=actual_quantity,
            )
            self.open_pos = pos
            await self.save_position(pos)
            logger.info(f"🔴 SELL PE @ {price} Qty={actual_quantity}x Initial SL {initial_sl:.2f} "
                       f"(ATR: {current_atr:.2f}, Multiplier: {atr_multiplier:.2f}) "
                       f"Recovery: {self.pnl_tracker.recovery_mode}")
            return True
        return False

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
            
            # Update PnL tracking after trade closes
            await self.update_pnl_after_trade(pnl)
            
            logger.info(f"Closed {token} reason={reason} pnl={tr.pnl} "
                       f"Total Points: {self.pnl_tracker.total_points:.2f}")
            return True

    async def task_atr_monitor(self):
        """Continuously monitor ATR values (for logging only)"""
        while True:
            try:
                if self.is_market_hours():
                    current_atr = await self.get_current_atr()
                    atr_multiplier = self.get_dynamic_atr_multiplier(current_atr)
                
            except Exception as e:
                logger.error(f"ATR monitor error: {e}")
                await asyncio.sleep(10)

    def is_market_hours(self) -> bool:
        now = datetime.now()
        current_time = now.time()
        return (self.market_open_time.time() <= current_time <= self.market_close_time.time())

    async def handle_index_tick(self, price: float):
        if not self.open_pos or not self.is_market_hours():
            return
        
        pos = self.open_pos
        
        # Check if stop loss is hit
        if pos.option_type == "CE" and price <= pos.stop_loss:
            await self.close_position("STOP_LOSS")
        elif pos.option_type == "PE" and price >= pos.stop_loss:
            await self.close_position("STOP_LOSS")

    async def on_buy_signal(self, signal_candle: Optional[CandleData] = None):
        if not await self.is_atr_above_threshold():
            logger.info(f"ATR below threshold, ignoring BUY signal")
            return
            
        signal_candle = await self.load_last_complete_candle_from_csv()
        if signal_candle is None:
            logger.warning("No complete candle data available for buy signal")
            return
            
        if not self.is_market_hours():
            return
            
        if self.open_pos and self.open_pos.option_type == "PE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_buy(quantity=1, signal_candle=signal_candle)

    async def on_sell_signal(self, signal_candle: Optional[CandleData] = None):
        if not await self.is_atr_above_threshold():
            logger.info(f"ATR below threshold, ignoring SELL signal")
            return
            
        signal_candle = await self.load_last_complete_candle_from_csv()
        if signal_candle is None:
            logger.warning("No complete candle data available for sell signal")
            return

        if not self.is_market_hours():
            return
            
        if self.open_pos and self.open_pos.option_type == "CE":
            await self.close_position("OPPOSITE_SIGNAL")
        
        if not self.open_pos:
            await self.take_sell(quantity=1, signal_candle=signal_candle)

    async def task_pubsub_listener(self):
        """Listens for Pub/Sub messages for prices, signals, and candle closes."""
        try:
            psub = self.r.pubsub()
            await psub.subscribe(
                f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}", 
                CHAN_SIGNAL_BUY, 
                CHAN_SIGNAL_SELL,
                CHAN_CANDLE_CLOSE
            )
            logger.info("Subscribed to Pub/Sub channels")
            
            async for msg in psub.listen():
                if not self.is_market_hours():
                    continue
                    
                if msg is None or msg.get("type") != "message":
                    continue
                    
                channel = msg.get("channel")
                data = msg.get("data")
                
                try:
                    if channel == f"{CHAN_PRICE_PREFIX}{INDEX_TOKEN}":
                        price = float(data)
                        await self.handle_index_tick(price)
                    
                    elif channel == CHAN_SIGNAL_BUY:
                        signal_candle = await self.load_last_complete_candle_from_csv()
                        if signal_candle is None:
                            logger.warning("No complete candle data available for buy signal")
                            continue
                        await self.on_buy_signal(signal_candle)
                    
                    elif channel == CHAN_SIGNAL_SELL:
                        signal_candle = await self.load_last_complete_candle_from_csv()
                        if signal_candle is None:
                            logger.warning("No complete candle data available for sell signal")
                            continue
                        await self.on_sell_signal(signal_candle)
                    
                    elif channel == CHAN_CANDLE_CLOSE:
                        candle_data = await self.load_last_complete_candle_from_csv()
                        if candle_data is None:
                            logger.warning("No complete candle data available for candle close")
                            continue
                        await self.handle_candle_close(asdict(candle_data))
                    
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
                price = await self.get_index_price()
                if price > 0:
                    await self.handle_index_tick(price)

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

                for key in r.scan_iter("active_positio*"):
                    r.delete(key)
                
                for key in r.scan_iter("CANDLE*"):
                    r.delete(key)
                
                # Calculate daily P&L
                data = await self.r.get(f"trade_history_atr_{datetime.now().date()}")
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

                logger.info(f"📊 DAILY SUMMARY:")
                logger.info(f"   Total P&L: {total_profit_or_loss:.2f}")
                logger.info(f"   Secured Profit: {self.pnl_tracker.secured_profit:.2f}")
                logger.info(f"   Final Total Points: {self.pnl_tracker.total_points:.2f}")
                
                # Reset daily PnL tracker for next day
                self.pnl_tracker = PnLTracker(last_updated=datetime.now().isoformat())
                await self.save_pnl_tracker()
                
                await asyncio.sleep((self.market_close_time - now).total_seconds())
            
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

    async def run(self):
        logger.info("=" * 80)
        logger.info("🚀 TradingBot starting with Advanced Recovery System")
        logger.info("=" * 80)
        logger.info("📋 Recovery Strategy:")
        logger.info(f"   Loss > 8 points → Take next trade with 2x quantity")
        logger.info(f"   Loss > 20 points → Take next trade with 3x quantity")
        logger.info(f"   Loss > 40 points → Take next trade with 4x quantity")
        logger.info(f"   Loss > 60 points → Take next trade with 5x quantity")
        logger.info("")
        logger.info("📋 Partial Exit Logic:")
        logger.info(f"   When profit per unit ≥ (Total Loss / Quantity):")
        logger.info(f"   → Exit ALL extra quantities (keep only 1)")
        logger.info(f"   → Continue trading with 1 quantity only")
        logger.info(f"   Example: Loss 25 pts, 3x qty → need 8.33 pts/unit")
        logger.info(f"            When achieved → exit 2 units, keep 1")
        logger.info("")
        logger.info(f"   Secured profit ≥ {PROFIT_TARGET} → Stop taking new positions")
        logger.info("=" * 80)
        
        if not await self.load_symbols():
            logger.error("Failed to load symbols, exiting")
            return
        
        # Load PnL tracker
        self.pnl_tracker = await self.load_pnl_tracker()
        logger.info(f"💰 Loaded PnL Tracker: Total={self.pnl_tracker.total_points:.2f}, "
                   f"Secured={self.pnl_tracker.secured_profit:.2f}, "
                   f"Recovery Mode={self.pnl_tracker.recovery_mode}")
        
        # Calculate initial next quantity
        self.next_trade_quantity = self.calculate_next_quantity(self.pnl_tracker.total_points)
        await self.r.set(NEXT_QUANTITY_KEY, str(self.next_trade_quantity))
        logger.info(f"📊 Initial next trade quantity: {self.next_trade_quantity}x")
        
        # Recover any existing positions
        await self.recover_open_position()
        
        # Initial ATR check
        initial_atr = await self.get_current_atr()
        initial_multiplier = self.get_dynamic_atr_multiplier(initial_atr)
        logger.info(f"Initial ATR: {initial_atr:.2f}, Multiplier: {initial_multiplier:.2f}")
        
        tasks = [
            asyncio.create_task(self.task_pubsub_listener()),
            asyncio.create_task(self.task_key_fallback_poller()),
            asyncio.create_task(self.task_square_off_scheduler()),
            asyncio.create_task(self.task_candle_cache_refresh()),
            asyncio.create_task(self.task_continuous_trailing_stop_loss()),
            asyncio.create_task(self.task_atr_monitor()),
            asyncio.create_task(self.task_pnl_monitor()),
            asyncio.create_task(self.task_quantity_calculator()),
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
    
    while True:
        if bot.is_market_hours():
            try:
                asyncio.run(bot.run())
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                time.sleep(5)
        else:
            print("⏳ Market closed. Waiting for market hours...")
            time.sleep(1)