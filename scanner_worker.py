"""
scanner_worker.py — Generic WebSocket worker for stock scanning.

Usage:
    python scanner_worker.py --credential ws3

Connects to AngelOne WebSocket with the specified credential,
subscribes to assigned stock tokens, and builds 5-minute candles
stored in Redis with SCAN: prefix.

Redis keys written:
    SCAN:LTP:{symbol}                — Latest traded price
    SCAN:CANDLE:{symbol}:{time}      — Current forming 5-min candle
    SCAN:HISTORY:{symbol}:{date}     — Completed candles list
    SCAN:OPEN:{symbol}:{date}        — Day's opening price (first tick)
    SCAN:DAY_HIGH:{symbol}:{date}    — Intraday high
    SCAN:DAY_LOW:{symbol}:{date}     — Intraday low
    SCAN:VOLUME:{symbol}:{date}      — Cumulative volume
"""

import argparse
import datetime
import json
import os
import sys
import threading
import time

import pandas as pd
import pyotp
import pytz
import redis
from logzero import logger
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDIA_TZ = pytz.timezone("Asia/Kolkata")

# Redis config
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")

r = redis.StrictRedis(
    host=REDIS_HOST, port=REDIS_PORT,
    password=REDIS_PASSWORD, db=0, decode_responses=True
)

# In-memory candle tracking
last_candle_time_map = {}
last_candle_map = {}
symbol_map = {}
tokens = []


def load_credential(cred_name: str) -> dict:
    """Load a specific credential from scanner_credentials.json."""
    cred_path = os.path.join(BASE_DIR, "scanner_credentials.json")
    with open(cred_path, 'r') as f:
        creds = json.load(f)
    for c in creds:
        if c['name'] == cred_name:
            return c
    raise ValueError(f"Credential '{cred_name}' not found in scanner_credentials.json")


def load_tokens(cred_name: str):
    """Load assigned token list for this credential."""
    global tokens, symbol_map
    csv_path = os.path.join(BASE_DIR, f"scanner_tokens_{cred_name}.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Token file not found: {csv_path}. Run stock_scanner_setup.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["pSymbol"] = df["pSymbol"].astype(str)
    df["pSymbolName"] = df["pSymbolName"].astype(str)
    tokens = df["pSymbol"].tolist()
    symbol_map = dict(zip(df["pSymbol"], df["pSymbolName"]))
    logger.info(f"[{cred_name}] Loaded {len(tokens)} stock tokens.")


def connect_api(cred: dict) -> str:
    """Login to AngelOne and get feed token."""
    totp = pyotp.TOTP(cred['totp_secret']).now()
    smart_api = SmartConnect(cred['api_key'])
    smart_api.generateSession(cred['client_id'], cred['password'], totp)
    feed_token = smart_api.getfeedToken()
    logger.info(f"[{cred['name']}] API login successful, got feed token.")
    return feed_token


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:15 AM - 3:35 PM)."""
    now = datetime.datetime.now(INDIA_TZ).time()
    return datetime.time(9, 15) <= now <= datetime.time(15, 35)


def run_websocket(cred: dict):
    """Main WebSocket loop — subscribes to stocks and builds candles."""

    def on_data(wsapp, message):
        """Process incoming tick data into 5-minute candles."""
        if message == b'\x00':
            return
        try:
            tick = message
            token_id = str(tick.get('token'))
            ltp_raw = tick.get('last_traded_price')
            p_symbol = symbol_map.get(token_id, token_id)

            if not token_id or not ltp_raw:
                return

            price = ltp_raw / 100
            now = datetime.datetime.now(INDIA_TZ)
            date_key = now.strftime('%Y-%m-%d')

            # ── Store LTP ──
            r.set(f"SCAN:LTP:{p_symbol}", price)

            # ── Track day open / high / low ──
            open_key = f"SCAN:OPEN:{p_symbol}:{date_key}"
            if not r.exists(open_key):
                r.set(open_key, price)

            high_key = f"SCAN:DAY_HIGH:{p_symbol}:{date_key}"
            cur_high = r.get(high_key)
            if not cur_high or price > float(cur_high):
                r.set(high_key, price)

            low_key = f"SCAN:DAY_LOW:{p_symbol}:{date_key}"
            cur_low = r.get(low_key)
            if not cur_low or price < float(cur_low):
                r.set(low_key, price)

            # ── Track volume ──
            vol = tick.get('last_traded_quantity', 0)
            if vol:
                vol_key = f"SCAN:VOLUME:{p_symbol}:{date_key}"
                r.incr(vol_key, vol)

            # ── 5-minute candle aggregation ──
            minute = (now.minute // 5) * 5
            candle_time_obj = now.replace(minute=minute, second=0, microsecond=0)
            candle_time = candle_time_obj.strftime('%Y-%m-%d %H:%M')

            redis_key = f"SCAN:CANDLE:{p_symbol}:{candle_time}"
            list_key = f"SCAN:HISTORY:{p_symbol}:{date_key}"

            # Check if a new candle started → push previous candle to history
            last_ct = last_candle_time_map.get(p_symbol)
            if last_ct and last_ct != candle_time:
                prev = last_candle_map.get(p_symbol)
                if prev:
                    r.rpush(list_key, json.dumps(prev))

            # Get or create current candle
            existing = r.get(redis_key)
            if existing:
                candle = json.loads(existing)
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] = candle.get('volume', 0) + (vol or 0)
            else:
                candle = {
                    'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': vol or 0,
                }

            r.set(redis_key, json.dumps(candle))
            last_candle_time_map[p_symbol] = candle_time
            last_candle_map[p_symbol] = candle

        except Exception as e:
            logger.error(f"[{cred['name']}] Tick error: {e}")

    def on_open(wsapp):
        """Subscribe to all assigned tokens on connect."""
        logger.info(f"[{cred['name']}] WebSocket opened — subscribing {len(tokens)} stocks")

        # AngelOne allows max 50 tokens per subscribe call, batch them
        BATCH = 50
        for i in range(0, len(tokens), BATCH):
            batch = tokens[i:i + BATCH]
            token_list = [{"exchangeType": 1, "tokens": batch}]  # 1 = NSE
            try:
                sws.subscribe(cred['name'], 1, token_list)
                logger.info(f"  Subscribed batch {i // BATCH + 1}: {len(batch)} tokens")
            except Exception as e:
                logger.error(f"  Subscribe batch error: {e}")
            time.sleep(0.2)  # Small delay between batches

    def on_error(wsapp, error):
        logger.error(f"[{cred['name']}] WebSocket error: {error}")
        if is_market_hours():
            logger.info("Reconnecting in 5 seconds...")
            time.sleep(5)
            threading.Thread(target=run_websocket, args=(cred,), daemon=True).start()

    def on_close(wsapp):
        logger.info(f"[{cred['name']}] WebSocket closed.")
        if is_market_hours():
            logger.info("Reconnecting in 5 seconds...")
            time.sleep(5)
            threading.Thread(target=run_websocket, args=(cred,), daemon=True).start()

    try:
        feed_token = connect_api(cred)
        sws = SmartWebSocketV2(
            cred['totp_secret'], cred['api_key'],
            cred['client_id'], feed_token
        )
        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close

        logger.info(f"[{cred['name']}] Connecting WebSocket...")
        sws.connect()
    except Exception as e:
        logger.error(f"[{cred['name']}] WebSocket startup failed: {e}")
        if is_market_hours():
            time.sleep(10)
            run_websocket(cred)


def flush_remaining_candles():
    """Push any remaining in-memory candles to Redis history at shutdown."""
    date_key = datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d')
    for symbol, candle in last_candle_map.items():
        list_key = f"SCAN:HISTORY:{symbol}:{date_key}"
        r.rpush(list_key, json.dumps(candle))
    logger.info(f"Flushed {len(last_candle_map)} remaining candles to history.")


def main():
    parser = argparse.ArgumentParser(description="Stock Scanner WebSocket Worker")
    parser.add_argument("--credential", required=True, help="Credential name from scanner_credentials.json (e.g. ws3)")
    args = parser.parse_args()

    cred_name = args.credential
    logger.info(f"{'=' * 50}")
    logger.info(f"SCANNER WORKER [{cred_name}] starting...")
    logger.info(f"{'=' * 50}")

    cred = load_credential(cred_name)
    load_tokens(cred_name)

    if not tokens:
        logger.error("No tokens to subscribe. Run stock_scanner_setup.py first.")
        sys.exit(1)

    # Run websocket
    run_websocket(cred)

    # After market close / disconnect, flush remaining candles
    flush_remaining_candles()


if __name__ == "__main__":
    main()
