from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from SmartApi import SmartConnect
from logzero import logger
import pyotp
import pandas as pd
import redis
import threading
import time
import ujson as json
import pytz
from multiprocessing import Process
import ta
import numpy as np  
from neo_api_client import NeoAPI
import datetime
import os
import asyncio

# Your credentials and configuration
api_key = 'SsUDlNA9 '
clientId = 'A1079871'
pwd = '0465'
token = "OIN6QBZAYV4I26Q55OYASIEQVY"
correlation_id = "Anil"


# token = "33OUTDUE57WS3TUPHPLFUCGHFM"
# api_key = "Ytt1NkKD"
# clientId = "R865920"
# pwd = '7355'
# correlation_id = "Rahul_7355"

ltp_data = {}
tokens = []
symbol_map = {}

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# Market hours configuration
now = datetime.datetime.now()
market_open_time = now.replace(hour=9, minute=14, second=59, microsecond=59)
market_close_time = now.replace(hour=15, minute=30, second=5, microsecond=0)
_is_market_hours = False

def connect_api():
    totp = pyotp.TOTP(token).now()
    smartApi = SmartConnect(api_key)
    smartApi.generateSession(clientId, pwd, totp)
    FEED_TOKEN = smartApi.getfeedToken()
    return FEED_TOKEN

last_candle_time_map = {}
last_candle_map = {}

def is_market_hours() -> bool:
    now = datetime.datetime.now()
    current_time = now.time()
    return (market_open_time.time() <= current_time <= market_close_time.time())

def get_odd_minute_candle_time(timestamp):
    minute = timestamp.minute
    odd_minute = minute if minute % 2 == 1 else minute - 1
    if odd_minute < 0:
        odd_minute = 59
        hour = timestamp.hour - 1
        if hour < 0:
            hour = 23
        candle_time = timestamp.replace(hour=hour, minute=odd_minute, second=0, microsecond=0)
    else:
        candle_time = timestamp.replace(minute=odd_minute, second=0, microsecond=0)
    return candle_time

def run_websocket():
    def on_data(wsapp, message):
        if message != b'\x00':
            try:
                if is_market_hours():
                    tick = message
                    print(tick)
                    
                    token = str(tick.get('token'))
                    ltp = tick.get('last_traded_price')
                    pSymbolName = symbol_map.get(token, token)

                    if token and ltp:
                        price = ltp / 100
                        india_timezone = pytz.timezone("Asia/Kolkata")

                        now = datetime.datetime.now(india_timezone)
                        candle_time_obj = get_odd_minute_candle_time(now)
                        candle_time = candle_time_obj.strftime('%Y-%m-%d %H:%M')
                        date_key = now.strftime('%Y-%m-%d')

                        redis_key = f"CANDLE:{pSymbolName}:{candle_time}"
                        list_key = f"HISTORY:{pSymbolName}"

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
                        else:
                            candle = {
                                'timestamp': candle_time_obj.strftime('%Y-%m-%d %H:%M:%S'),
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                            }

                        r.set(pSymbolName, price)
                        r.set(redis_key, json.dumps(candle))

                        last_candle_time_map[pSymbolName] = candle_time
                        last_candle_map[pSymbolName] = candle

                        logger.info(f"[CANDLE] {pSymbolName} @ {candle_time} -> {candle}")

                    else:
                        print("this is not market hours, please wait for market hour")

            except Exception as e:
                logger.error(f"Error processing tick: {e}")

    def on_open(wsapp):
        logger.info("WebSocket Opened")
        token_list = [{"exchangeType": 1, "tokens": ["99926000"]}]
        sws.subscribe(correlation_id, 1, token_list)

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
    sws = SmartWebSocketV2(token, api_key, clientId, FEED_TOKEN)
    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error
    sws.on_close = on_close

    logger.info("Connecting WebSocket...")
    sws.connect()




def atr_calculate(data):
    try:
        atr_series = ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=11
        ).average_true_range()

        atr_value_latest = atr_series.iloc[-1]

        r.set("ATR_value", float(atr_value_latest))
        print("ATR set in Redis:", atr_value_latest)

    except Exception as e:
        print("Error in atr_calculate:", e)


def calculate_indicators(data, a=2, c=100, h=False):
    xATR = data['close'].diff().abs().ewm(span=c, adjust=False).mean()
    nLoss = a * xATR
    src = data['close']
    xATRTrailingStop = pd.Series(index=data.index)

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

    return signals

file_path = "main_csv.csv"
file_exists = os.path.isfile(file_path)

def update_indicators_and_signal():
    india_timezone = pytz.timezone("Asia/Kolkata")
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    while True:
        try:
            now_time = datetime.datetime.now(india_timezone).time()
            market_close = datetime.time(15, 30)
            signal_time = datetime.time(15, 20)

            if now_time >= market_close:
                print("Market closed. Stopping indicator process.")
                break

            df_tokens = pd.read_csv("future_and_options_token.csv")
            psymbol = df_tokens["pSymbol"].values[0]
            list_data = [psymbol]

            for symbol in list_data:
                history_key = f"HISTORY:{symbol}"
                history_data = r.lrange(history_key, 0, -1)

                if len(history_data) < 30:
                    continue

                df = pd.DataFrame([json.loads(x) for x in history_data])
                df = df.astype({
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float
                })

                signal = calculate_indicators(df)
                atr_calculate(df)
                data = pd.concat([df, signal], axis=1)
                last_row_df = data.tail(1)

                if last_row_df.empty or "timestamp" not in last_row_df.columns:
                    continue

                redis_last_row_timestamp = last_row_df["timestamp"].values[0]

                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    last_row_df.to_csv(file_path, mode="w", index=False, header=True, line_terminator="n")
                else:
                    main_df_file = pd.read_csv(file_path)

                    if main_df_file.empty or "timestamp" not in main_df_file.columns:
                        timestamp_main = None
                    else:
                        main_tail = main_df_file.tail(1)
                        if not main_tail.empty:
                            timestamp_main = main_tail["timestamp"].values[0]
                        else:
                            timestamp_main = None

                    if timestamp_main is None or redis_last_row_timestamp != timestamp_main:
                        last_row_df.to_csv(file_path, mode="a", index=False, header=False)
                    
        except Exception as e:
            logger.error(f"Error in update_indicators_and_signal for : {e}")       

def run_update_indicators():
    update_indicators_and_signal()

if __name__ == "__main__":
    P0 = Process(target=run_websocket)
    P1 = Process(target=run_update_indicators)
    
    P0.start()
    P1.start()
    
    P0.join()
    P1.join()