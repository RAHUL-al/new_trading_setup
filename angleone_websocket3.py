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
import requests
import numpy as np

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

api_key = 'SsUDlNA9 '
clientId = 'A1079871'
pwd = '0465'
token = "OIN6QBZAYV4I26Q55OYASIEQVY"
totp = pyotp.TOTP(token).now()
correlation_id = "Anil"

ltp_data = {}

tokens = []
symbol_map = {}


def load_tokens_from_csv():
    today_date_str = datetime.datetime.now().strftime("%Y%m%d")
    global tokens, symbol_map
    df = pd.read_csv("future_and_options_token.csv")
    df["pSymbolName"] = df["pSymbolName"].astype(str)
    # df["pSymbol"] = df["pSymbol"].astype(str)
    tokens = df["pSymbol"].astype(int).tolist()
    symbol_map = dict(zip(df["pSymbol"], df["pSymbolName"]))
    logger.info(f"Loaded {len(tokens)} tokens with symbol names.")

def connect_api():
    totp = pyotp.TOTP(token).now()
    smartApi = SmartConnect(api_key)
    smartApi.generateSession(clientId, pwd, totp)
    FEED_TOKEN = smartApi.getfeedToken()
    return FEED_TOKEN

last_candle_time_map = {}
last_candle_map = {}


def run_websocket():
    def on_data(wsapp, message):
        if message != b'\x00':
            try:
                if isinstance(message, bytes):
                    message = message.decode()
                tick = json.loads(message)
                print(tick)
                # tick = message
                token = str(tick.get('token'))
                ltp = tick.get('last_traded_price')
                pSymbolName = symbol_map.get(token, token)

                if token and ltp:
                    price = ltp / 100
                    india_timezone = pytz.timezone("Asia/Kolkata")
                    now = datetime.datetime.now(india_timezone)
                    minute = (now.minute // 5) * 5
                    candle_time_obj = now.replace(minute=minute, second=0, microsecond=0)
                    candle_time = candle_time_obj.strftime('%Y-%m-%d %H:%M')
                    date_key = now.strftime('%Y-%m-%d')

                    redis_key = f"CANDLE:{pSymbolName}:{candle_time}"
                    list_key = f"HISTORY:{pSymbolName}:{date_key}"

                    # Check if a new 5-minute candle has started
                    last_candle_time = last_candle_time_map.get(pSymbolName)
                    if last_candle_time and last_candle_time != candle_time:
                        prev_candle = last_candle_map.get(pSymbolName)
                        if prev_candle:
                            r.rpush(list_key, json.dumps(prev_candle))

                    # Get or create the current candle
                    existing = r.get(redis_key)
                    if existing:
                        candle = json.loads(existing)
                        candle['high'] = max(candle['high'], price)
                        candle['low'] = min(candle['low'], price)
                        candle['close'] = price
                        # candle['volume'] += tick.get('last_traded_quantity', 0)
                        # candle['total_buy_quantity'] += tick.get('total_buy_quantity',0)
                        # candle['total_sell_quantity'] += tick.get('total_sell_quantity',0)
                    else:
                        candle = {
                            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'open': price,
                            'high': price,
                            'low': price,
                            'close': price,
                            # 'volume': tick.get('last_traded_quantity', 0),
                            # 'total_buy_quantity':tick.get('total_buy_quantity',0),
                            # 'total_sell_quantity':tick.get('total_sell_quantity',0),
                        }
                    # print(candle)
                    # Save updated candle to Redis
                    r.set(pSymbolName,price)
                    r.set(redis_key, json.dumps(candle))

                    # Update tracking for next tick
                    last_candle_time_map[pSymbolName] = candle_time
                    last_candle_map[pSymbolName] = candle

                    logger.info(f"[CANDLE] {pSymbolName} @ {candle_time} -> {candle}")

            except Exception as e:
                logger.error(f"Error processing tick: {e}")



    def on_open(wsapp):
        logger.info("WebSocket Opened")
        batch_size = 1
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            print("this is betch token : ",batch_tokens)
            token_list = [{"exchangeType": 3, "tokens": batch_tokens}]
            
            sws.subscribe(correlation_id, 3, token_list)

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
    


def calculate_indicators(data, a=2, c=100, h=False):
    xATR = data['Close'].diff().abs().ewm(span=c, adjust=False).mean()
    nLoss = a * xATR
    src = data['Close']
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



def update_indicators_and_signal():
    india_timezone = pytz.timezone("Asia/Kolkata")
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    while True:
        try:
            now_time = datetime.datetime.now(india_timezone).time()
            market_close = datetime.time(15, 30)
            signal_time = datetime.time(15,20)

            if now_time >= market_close:
                print("Market closed. Stopping indicator process.")
                break
            else:
                now = datetime.date.today()
                df = pd.read_csv("future_and_options_token.csv")
                list_data = list(df["pSymbolName"].str.strip())
                for data in list_data:
                    history_key = f"HISTORY:{data}:{now}"
                    history_data = r.lrange(history_key, 0, -1)
                    if len(history_data) < 30:
                        return
                    df = pd.DataFrame([json.loads(x) for x in history_data])
                    df = df.astype({
                        "open": float, "high": float, "low": float,
                        "close": float
                    })
                    signal = calculate_indicators(df)
                    data = pd.concat([df,signal],axis=1)
                    
                            
        except Exception as e:
            logger.error(f"Error in update_indicators_and_signal for {data}: {e}")
            

if __name__ == "__main__":
    load_tokens_from_csv()
    P0 = Process(target=run_websocket)
    P1 = Process(target=update_indicators_and_signal)
    
    P0.start()
    P1.start()
    
    P0.join()
    P1.join()