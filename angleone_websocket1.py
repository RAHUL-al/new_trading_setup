"""
angleone_websocket1.py — WebSocket feeder with:
- 1-minute candle aggregation for NIFTY index + stocks
- Dynamic NFO option token subscription (CE/PE from symbol_found.py)
- Market close cleanup at 3:30 PM (archive candles, clear Redis)
"""

from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from SmartApi import SmartConnect
from logzero import logger
import pyotp
import pandas as pd
import redis
import threading
import time
import ujson as json
import datetime
import pytz
from multiprocessing import Process
import ta


# ─────────── Config ───────────
r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

TOTP_TOKEN = "33OUTDUE57WS3TUPHPLFUCGHFM"
API_KEY = "Ytt1NkKD"
CLIENT_ID = "R865920"
PWD = '7355'
CORRELATION_ID = "Rahul_7355"

TRADING_SYMBOLS_KEY = "Trading_symbol"
OPTION_TOKENS_CHANNEL = "option_tokens_updated"

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

                    redis_key = f"CANDLE:{pSymbolName}:{candle_time}"
                    list_key = f"HISTORY:{pSymbolName}:{date_key}"

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
                    r.set(token, price)
                    # Also store by symbol name
                    r.set(pSymbolName, price)
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
                    symbol_map[ce_token] = ce_info[0]  # trading symbol as name
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

                # Unsubscribe old tokens
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

                # Subscribe new tokens
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

    # Start token update listener in background thread
    token_update_thread = threading.Thread(target=_handle_token_update, daemon=True)
    token_update_thread.start()

    logger.info("Connecting WebSocket...")
    sws.connect()


# ─────────── Market Close Cleanup ───────────

def market_close_cleanup():
    """
    At 3:30 PM:
    1. Archive all NIFTY candle history to NIFTY_CANDLES:{date}
    2. Clear all CANDLE:* and HISTORY:* keys
    3. Clear Trading_symbol
    """
    while True:
        now = datetime.datetime.now(INDIA_TZ)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now >= market_close:
            date_key = now.strftime('%Y-%m-%d')
            archive_key = f"NIFTY_CANDLES:{date_key}"

            try:
                # Archive NIFTY index candle history
                nifty_history_key = f"HISTORY:NIFTY:{date_key}"
                nifty_candles = r.lrange(nifty_history_key, 0, -1)
                if nifty_candles:
                    for candle in nifty_candles:
                        r.rpush(archive_key, candle)
                    logger.info(f"Archived {len(nifty_candles)} NIFTY candles to {archive_key}")

                # Also archive all other symbols' candles
                try:
                    df = pd.read_csv("stocks_csv_1.csv")
                    symbols = list(df["pSymbolName"].str.strip())
                    for sym in symbols:
                        hist_key = f"HISTORY:{sym}:{date_key}"
                        sym_candles = r.lrange(hist_key, 0, -1)
                        if sym_candles:
                            sym_archive = f"{sym}_CANDLES:{date_key}"
                            for c in sym_candles:
                                r.rpush(sym_archive, c)
                            logger.info(f"Archived {len(sym_candles)} candles for {sym}")
                except Exception as e:
                    logger.error(f"Error archiving symbol candles: {e}")

                # Delete all CANDLE:* keys
                candle_keys = r.keys("CANDLE:*")
                if candle_keys:
                    r.delete(*candle_keys)
                    logger.info(f"Deleted {len(candle_keys)} CANDLE keys")

                # Delete all HISTORY:* keys
                history_keys = r.keys("HISTORY:*")
                if history_keys:
                    r.delete(*history_keys)
                    logger.info(f"Deleted {len(history_keys)} HISTORY keys")

                # Delete Trading_symbol
                r.delete(TRADING_SYMBOLS_KEY)
                logger.info("Deleted Trading_symbol key")

                # Delete signal keys for today
                signal_keys = r.keys(f"SIGNAL:*")
                if signal_keys:
                    r.delete(*signal_keys)
                    logger.info(f"Deleted {len(signal_keys)} SIGNAL keys")

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
            # Check every 10 seconds near market close
            time.sleep(10)


# ─────────── Indicators & Signals ───────────

def calculate_indicators(df):
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    return df


def generate_signal(df):
    latest = df.iloc[-1]
    if df.shape[0] < 2:
        return "NO_SIGNAL", latest.to_dict()
    if (
        latest["ema9"] > latest["ema21"] and
        latest["macd"] > latest["macd_signal"] and
        latest["rsi"] > 60 and
        latest["adx"] > 25
    ):
        return "STRONG_BUY", latest.to_dict()
    elif (
        latest["ema9"] < latest["ema21"] and
        latest["macd"] < latest["macd_signal"] and
        latest["rsi"] < 40 and
        latest["adx"] > 25
    ):
        return "STRONG_SELL", latest.to_dict()
    return "NO_SIGNAL", latest.to_dict()


def update_indicators_and_signal():
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    while True:
        try:
            now_time = datetime.datetime.now(INDIA_TZ).time()
            market_close = datetime.time(15, 30)

            if now_time >= market_close:
                print("Market closed. Stopping indicator process.")
                break
            else:
                now = datetime.date.today()
                df = pd.read_csv("stocks_csv_1.csv")
                list_data = list(df["pSymbolName"].str.strip())
                for data in list_data:
                    history_key = f"HISTORY:{data}:{now}"
                    history_data = r.lrange(history_key, 0, -1)
                    if len(history_data) < 30:
                        return
                    df = pd.DataFrame([json.loads(x) for x in history_data])
                    df = df.astype({
                        "open": float, "high": float, "low": float,
                        "close": float, "volume": float,
                        "total_buy_quantity": float, "total_sell_quantity": float
                    })
                    df = calculate_indicators(df)
                    signal, data_dict = generate_signal(df)
                    data_to_store = {
                        "Signal": signal,
                        "dat": data_dict,
                    }
                    if signal != "NO_SIGNAL":
                        if not r.exists(f"SIGNAL:{data}:{today_date}"):
                            r.set(f"SIGNAL:{data}:{today_date}", json.dumps(data_to_store))
                            logger.info(f"[SIGNAL] {data}: {signal}")

        except Exception as e:
            logger.error(f"Error in update_indicators_and_signal: {e}")


# ─────────── Entry Point ───────────

if __name__ == "__main__":
    load_tokens_from_csv()

    P0 = Process(target=run_websocket)
    P1 = Process(target=update_indicators_and_signal)
    P2 = Process(target=market_close_cleanup)

    P0.start()
    P1.start()
    P2.start()

    P0.join()
    P1.join()
    P2.join()