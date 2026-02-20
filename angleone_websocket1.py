"""
angleone_websocket1.py — WebSocket feeder with:
- 1-minute candle aggregation for NIFTY index + stocks
- Dynamic NFO option token subscription (CE/PE from symbol_found.py)
- UT Bot Alerts signal engine on NIFTY candles
- Market close cleanup at 3:30 PM (archive candles, clear Redis)
"""

from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from SmartApi import SmartConnect
from logzero import logger
import pyotp
import pandas as pd
import numpy as np
import redis
import threading
import time
import ujson as json
import datetime
import pytz
import os
from multiprocessing import Process


# ─────────── Config (read from environment, set by BotManager) ───────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "")

r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0, decode_responses=True)

TOTP_TOKEN = os.environ.get("ANGELONE_TOTP_SECRET", "33OUTDUE57WS3TUPHPLFUCGHFM")
API_KEY = os.environ.get("ANGELONE_API_KEY", "Ytt1NkKD")
CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "R865920")
PWD = os.environ.get("ANGELONE_PASSWORD", "7355")
CORRELATION_ID = f"user_{CLIENT_ID}"

TRADING_SYMBOLS_KEY = f"{REDIS_PREFIX}Trading_symbol"
OPTION_TOKENS_CHANNEL = f"{REDIS_PREFIX}option_tokens_updated"

INDIA_TZ = pytz.timezone("Asia/Kolkata")

ltp_data = {}
tokens = []
symbol_map = {}

# Track currently subscribed option tokens
current_option_tokens = {"CE": None, "PE": None}


def load_tokens_from_csv():
    global tokens, symbol_map
    df = pd.read_csv("stocks_csv_1.csv")
    df["pSymbolName"] = df["pSymbolName"].astype(str)
    df["pSymbol"] = df["pSymbol"].astype(str)
    tokens = df["pSymbol"].tolist()
    symbol_map = dict(zip(df["pSymbol"], df["pSymbolName"]))
    logger.info(f"Loaded {len(tokens)} tokens with symbol names.")


def connect_api():
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smartApi = SmartConnect(API_KEY)
    smartApi.generateSession(CLIENT_ID, PWD, totp)
    FEED_TOKEN = smartApi.getfeedToken()
    return FEED_TOKEN


last_candle_time_map = {}
last_candle_map = {}


# ─────────── WebSocket ───────────

def run_websocket():
    """Main WebSocket process — handles index/stock ticks + option ticks."""

    def on_data(wsapp, message):
        if message != b'\x00':
            try:
                tick = message
                token = str(tick.get('token'))
                ltp = tick.get('last_traded_price')
                pSymbolName = symbol_map.get(token, token)

                if token and ltp:
                    price = ltp / 100
                    now = datetime.datetime.now(INDIA_TZ)
                    minute = now.minute  # 1-minute candles
                    candle_time_obj = now.replace(minute=minute, second=0, microsecond=0)
                    candle_time = candle_time_obj.strftime('%Y-%m-%d %H:%M')
                    date_key = now.strftime('%Y-%m-%d')

                    redis_key = f"{REDIS_PREFIX}CANDLE:{pSymbolName}:{candle_time}"
                    list_key = f"{REDIS_PREFIX}HISTORY:{pSymbolName}:{date_key}"

                    last_candle_time = last_candle_time_map.get(pSymbolName)
                    if last_candle_time and last_candle_time != candle_time:
                        prev_candle = last_candle_map.get(pSymbolName)
                        if prev_candle:
                            r.rpush(list_key, json.dumps(prev_candle))

                    existing = r.get(redis_key)
                    if existing:
                        candle = json.loads(existing)
                        candle['high'] = max(candle['high'], price)
                        candle['low'] = min(candle['low'], price)
                        candle['close'] = price
                        candle['volume'] += tick.get('last_traded_quantity', 0)
                        candle['total_buy_quantity'] += tick.get('total_buy_quantity', 0)
                        candle['total_sell_quantity'] += tick.get('total_sell_quantity', 0)
                    else:
                        candle = {
                            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'open': price,
                            'high': price,
                            'low': price,
                            'close': price,
                            'volume': tick.get('last_traded_quantity', 0),
                            'total_buy_quantity': tick.get('total_buy_quantity', 0),
                            'total_sell_quantity': tick.get('total_sell_quantity', 0),
                        }

                    # Store LTP by token (for pos_handle_wts.py price lookups)
                    r.set(f"{REDIS_PREFIX}{token}", price)
                    # Also store by symbol name
                    r.set(f"{REDIS_PREFIX}{pSymbolName}", price)
                    r.set(redis_key, json.dumps(candle))

                    last_candle_time_map[pSymbolName] = candle_time
                    last_candle_map[pSymbolName] = candle

                    logger.info(f"[CANDLE] {pSymbolName} @ {candle_time} -> {candle}")

            except Exception as e:
                logger.error(f"Error processing tick: {e}")

    def on_open(wsapp):
        logger.info("WebSocket Opened")
        # Subscribe NSE Cash Market tokens (index + stocks)
        batch_size = 50
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            token_list = [{"exchangeType": 1, "tokens": batch_tokens}]
            sws.subscribe(CORRELATION_ID, 3, token_list)

        # Subscribe to any existing option tokens from Redis
        _subscribe_option_tokens()

    def _subscribe_option_tokens():
        """Read Trading_symbol from Redis and subscribe to CE/PE option tokens."""
        try:
            ts_data = r.get(TRADING_SYMBOLS_KEY)
            if ts_data:
                ts = json.loads(ts_data)
                nfo_tokens = []

                ce_info = ts.get("CE", [None, None])
                pe_info = ts.get("PE", [None, None])

                if ce_info and ce_info[1]:
                    ce_token = str(ce_info[1])
                    nfo_tokens.append(ce_token)
                    symbol_map[ce_token] = ce_info[0]
                    current_option_tokens["CE"] = ce_token

                if pe_info and pe_info[1]:
                    pe_token = str(pe_info[1])
                    nfo_tokens.append(pe_token)
                    symbol_map[pe_token] = pe_info[0]
                    current_option_tokens["PE"] = pe_token

                if nfo_tokens:
                    nfo_token_list = [{"exchangeType": 2, "tokens": nfo_tokens}]
                    sws.subscribe(CORRELATION_ID, 3, nfo_token_list)
                    logger.info(f"Subscribed to NFO option tokens: {nfo_tokens}")
        except Exception as e:
            logger.error(f"Error subscribing option tokens: {e}")

    def _handle_token_update():
        """Listen for option_tokens_updated channel and re-subscribe."""
        pubsub = r.pubsub()
        pubsub.subscribe(OPTION_TOKENS_CHANNEL)
        logger.info("Listening for option token updates...")

        for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                new_data = json.loads(msg["data"])
                logger.info(f"Option tokens updated: {new_data}")

                old_tokens = []
                if current_option_tokens["CE"]:
                    old_tokens.append(current_option_tokens["CE"])
                if current_option_tokens["PE"]:
                    old_tokens.append(current_option_tokens["PE"])

                if old_tokens:
                    try:
                        old_token_list = [{"exchangeType": 2, "tokens": old_tokens}]
                        sws.unsubscribe(CORRELATION_ID, 3, old_token_list)
                        logger.info(f"Unsubscribed old NFO tokens: {old_tokens}")
                    except Exception as e:
                        logger.warning(f"Error unsubscribing old tokens: {e}")

                _subscribe_option_tokens()

            except Exception as e:
                logger.error(f"Error handling token update: {e}")

    def on_error(wsapp, error):
        logger.error(f"WebSocket Error: {error}")
        logger.info("Reconnecting in 3 seconds...")
        time.sleep(3)
        threading.Thread(target=run_websocket).start()

    def on_close(wsapp):
        logger.info("WebSocket Closed")
        logger.info("Reconnecting in 3 seconds...")
        time.sleep(3)
        threading.Thread(target=run_websocket).start()

    FEED_TOKEN = connect_api()
    sws = SmartWebSocketV2(TOTP_TOKEN, API_KEY, CLIENT_ID, FEED_TOKEN)
    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error
    sws.on_close = on_close

    token_update_thread = threading.Thread(target=_handle_token_update, daemon=True)
    token_update_thread.start()

    logger.info("Connecting WebSocket...")
    sws.connect()


# ─────────── Market Close Cleanup ───────────

def market_close_cleanup():
    """At 3:30 PM: smart cleanup — keep last 10 days of history data."""
    while True:
        now = datetime.datetime.now(INDIA_TZ)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now >= market_close:
            try:
                all_keys = r.keys("*")

                # Patterns to ALWAYS preserve
                keep_prefixes = ("trade_history_", "SCAN:")

                # Date-keyed historical data patterns
                history_prefixes = ("CANDLE:", "HISTORY:")

                # Build set of valid date suffixes (last 10 days)
                valid_dates = set()
                for i in range(10):
                    d = (now - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                    valid_dates.add(d)

                keys_to_delete = []
                for k in all_keys:
                    if any(k.startswith(pfx) for pfx in keep_prefixes):
                        continue
                    if any(pfx in k for pfx in history_prefixes):
                        if any(d in k for d in valid_dates):
                            continue
                    keys_to_delete.append(k)

                if keys_to_delete:
                    r.delete(*keys_to_delete)
                    logger.info(f"🗑️ Deleted {len(keys_to_delete)} session/old keys (kept {len(all_keys) - len(keys_to_delete)} history keys)")
                logger.info("✅ Market close cleanup completed.")

            except Exception as e:
                logger.error(f"Error during market close cleanup: {e}")

            # Sleep until next day
            tomorrow_open = (now + datetime.timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            sleep_seconds = (tomorrow_open - now).total_seconds()
            logger.info(f"Sleeping until next market open ({sleep_seconds/3600:.1f} hrs)")
            time.sleep(sleep_seconds)
        else:
            time.sleep(1)


# ─────────── UT Bot Alerts Signal Engine (Async, Low Latency) ───────────
# Matches user's exact UT Bot indicator code (a=2, c=100)

import asyncio
from redis.asyncio import Redis as AsyncRedis

NIFTY_SYMBOL = "NIFTY"
SIGNAL_COOLDOWN = 60  # Min seconds between same-direction signals
ATR_MIN_THRESHOLD = 6.9  # Minimum ATR to take new positions


def calculate_true_range(data):
    """
    True Range = max(High-Low, abs(High-prev_Close), abs(Low-prev_Close))
    Uses full candle data (High/Low), not just Close.
    """
    high = data['High']
    low = data['Low']
    prev_close = data['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # First row: just High-Low since no prev_close
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    return true_range


def rma(series, period):
    """
    RMA (Wilder's Smoothing / Running Moving Average).
    RMA = (prev_RMA * (period-1) + current_value) / period
    Equivalent to EWM with alpha=1/period.
    """
    return series.ewm(alpha=1/period, adjust=False).mean()


def calculate_ut_bot(data, a=2, c=100, h=False):
    """
    UT Bot Alert indicator with proper ATR:
    - True Range using High/Low/prev_Close (full candle)
    - RMA smoothing (Wilder's method, alpha=1/c)
    - a = ATR multiplier (key value), c = ATR period (sensitivity)
    """
    # ATR using True Range + RMA (not close-to-close EWM)
    true_range = calculate_true_range(data)
    xATR = rma(true_range, c)
    nLoss = a * xATR
    src = data['Close']
    xATRTrailingStop = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        if i == 0:
            xATRTrailingStop.iloc[i] = src.iloc[i]
        elif src.iloc[i] > xATRTrailingStop.iloc[i - 1] and src.iloc[i - 1] > xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = max(xATRTrailingStop.iloc[i - 1], src.iloc[i] - nLoss.iloc[i])
        elif src.iloc[i] < xATRTrailingStop.iloc[i - 1] and src.iloc[i - 1] < xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = min(xATRTrailingStop.iloc[i - 1], src.iloc[i] + nLoss.iloc[i])
        elif src.iloc[i] > xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = src.iloc[i] - nLoss.iloc[i]
        else:
            xATRTrailingStop.iloc[i] = src.iloc[i] + nLoss.iloc[i]

    pos = np.zeros(len(data))

    for i in range(len(data)):
        if i == 0:
            pos[i] = 0
        elif src.iloc[i - 1] < xATRTrailingStop.iloc[i - 1] and src.iloc[i] > xATRTrailingStop.iloc[i - 1]:
            pos[i] = 1
        elif src.iloc[i - 1] > xATRTrailingStop.iloc[i - 1] and src.iloc[i] < xATRTrailingStop.iloc[i - 1]:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    buy = (pos == 1)
    sell = (pos == -1)

    signals = pd.DataFrame(index=data.index)
    signals['buy'] = buy
    signals['sell'] = sell

    return signals, xATR


async def nifty_signal_engine():
    """
    Async signal engine — polls at 50ms, only recalculates when new candles arrive.
    Uses async Redis for non-blocking I/O.
    ATR published with every signal for pos_handle_wts to check threshold.
    """
    ar = AsyncRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0, decode_responses=True)

    last_signal_time = 0
    last_signal = "NONE"
    prev_buy = False
    prev_sell = False
    last_candle_count = 0
    last_published_candle_time = None

    logger.info("⚡ Starting async NIFTY signal engine (UT Bot a=2, c=100, RMA ATR, threshold=6.9)...")

    while True:
        try:
            now_time = datetime.datetime.now(INDIA_TZ).time()
            if now_time >= datetime.time(15, 30):
                logger.info("Market closed. Stopping signal engine.")
                break

            if now_time < datetime.time(9, 16):
                await asyncio.sleep(0.5)
                continue

            # Read candle count first — skip if unchanged
            date_key = datetime.date.today().strftime('%Y-%m-%d')
            history_key = f"{REDIS_PREFIX}HISTORY:{NIFTY_SYMBOL}:{date_key}"
            candle_count = await ar.llen(history_key)

            if candle_count < 5:
                await asyncio.sleep(0.05)
                continue

            # Skip recalculation if no new candles
            if candle_count == last_candle_count:
                await asyncio.sleep(0.05)  # 50ms poll
                continue

            last_candle_count = candle_count

            # New candle detected — recalculate
            history_data = await ar.lrange(history_key, 0, -1)
            candles = [json.loads(x) for x in history_data]
            df = pd.DataFrame(candles)
            df = df.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
                "volume": "Volume",
            })
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col].astype(float)
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].astype(float)

            # Run UT Bot (now with True Range + RMA)
            signals, xATR = calculate_ut_bot(df, a=2, c=100)

            # Store ATR in Redis (non-blocking) — pos_handle_wts reads this
            current_atr = xATR.iloc[-1]
            if pd.notna(current_atr):
                await ar.set(f"{REDIS_PREFIX}ATR_value", str(round(current_atr, 2)))

            # Store last candle in Redis (NO CSV — pure Redis for low latency)
            last_row = df.iloc[-1]
            last_candle_data = json.dumps({
                "timestamp": str(last_row.get("timestamp", "")),
                "open": float(last_row["Open"]),
                "high": float(last_row["High"]),
                "low": float(last_row["Low"]),
                "close": float(last_row["Close"]),
            })
            await ar.set(f"{REDIS_PREFIX}last_candle", last_candle_data)

            # Edge detection for signals
            curr_buy = bool(signals['buy'].iloc[-1])
            curr_sell = bool(signals['sell'].iloc[-1])
            now_ts = time.time()
            atr_ok = pd.notna(current_atr) and current_atr > ATR_MIN_THRESHOLD

            if curr_buy and not prev_buy:
                if now_ts - last_signal_time > SIGNAL_COOLDOWN:
                    # Always publish signal — pos_handle_wts decides whether to open new position
                    signal_data = json.dumps({"signal": "buy", "atr": round(current_atr, 2) if pd.notna(current_atr) else 0})
                    await ar.publish(f"{REDIS_PREFIX}signal:buy", signal_data)
                    await ar.set(f"{REDIS_PREFIX}buy_signal", "true")
                    await ar.set(f"{REDIS_PREFIX}sell_signal", "false")
                    last_signal = "BUY"
                    last_signal_time = now_ts
                    atr_status = f"ATR={current_atr:.2f} ✅" if atr_ok else f"ATR={current_atr:.2f} ⚠️ <{ATR_MIN_THRESHOLD}"
                    logger.info(f"🟢 BUY SIGNAL — UT Bot | NIFTY Close={last_row['Close']} | {atr_status}")

            elif curr_sell and not prev_sell:
                if now_ts - last_signal_time > SIGNAL_COOLDOWN:
                    signal_data = json.dumps({"signal": "sell", "atr": round(current_atr, 2) if pd.notna(current_atr) else 0})
                    await ar.publish(f"{REDIS_PREFIX}signal:sell", signal_data)
                    await ar.set(f"{REDIS_PREFIX}sell_signal", "true")
                    await ar.set(f"{REDIS_PREFIX}buy_signal", "false")
                    last_signal = "SELL"
                    last_signal_time = now_ts
                    atr_status = f"ATR={current_atr:.2f} ✅" if atr_ok else f"ATR={current_atr:.2f} ⚠️ <{ATR_MIN_THRESHOLD}"
                    logger.info(f"🔴 SELL SIGNAL — UT Bot | NIFTY Close={last_row['Close']} | {atr_status}")

            prev_buy = curr_buy
            prev_sell = curr_sell

            # Publish candle close only on new candles
            candle_ts = str(last_row.get("timestamp", ""))
            if candle_ts != last_published_candle_time:
                last_published_candle_time = candle_ts
                await ar.publish(f"{REDIS_PREFIX}candle:close", json.dumps({
                    "timestamp": candle_ts,
                    "open": float(last_row["Open"]),
                    "high": float(last_row["High"]),
                    "low": float(last_row["Low"]),
                    "close": float(last_row["Close"]),
                }))

        except Exception as e:
            logger.error(f"Signal engine error: {e}")
            await asyncio.sleep(1)


def run_signal_engine():
    """Wrapper to run async signal engine in a new event loop (for multiprocessing)."""
    asyncio.run(nifty_signal_engine())


# ─────────── Entry Point ───────────

if __name__ == "__main__":
    load_tokens_from_csv()

    P0 = Process(target=run_websocket)
    P1 = Process(target=run_signal_engine)
    P2 = Process(target=market_close_cleanup)

    P0.start()
    P1.start()
    P2.start()

    P0.join()
    P1.join()
    P2.join()