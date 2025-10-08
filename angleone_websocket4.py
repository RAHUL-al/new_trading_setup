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
import numpy as np  
from neo_api_client import NeoAPI
import pandas as pd
import json
import threading
import time
from datetime import datetime
from kotak_login import get_kotak_client
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text
from neo_api_client import NeoAPI
import datetime as dt
from dateutil.relativedelta import relativedelta
from threading import Thread
import pandas as pd
import threading
import time
import dbload
from datetime import datetime


response_data = None
response_event = threading.Event()

client = get_kotak_client()

username = 'trading_user'
password = 'Rahul@7355'
host = 'localhost'
port = 3306



access_token = None
sid = None
secretkey = None
neo_fin_key = None

password = quote_plus(password)

engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
connection = engine.connect()

login_handle = f"session_{datetime.now().strftime('%d%m%Y')}"

login_handle_exists = sa.inspect(engine).has_table("%s"%(f"{login_handle}"))

if login_handle_exists:
    query = f"SELECT * FROM `{login_handle}` LIMIT 1"
    result = connection.execute(text(query)).fetchone()
    if result:
        access_token = result[0]
        sid = result[1]
        secretkey = result[2]
        neo_fin_key = result[3]




mpin = "735596"
mobileNumber = "9815767797"
login_password = "Rahul@7355"
consumer_key = "cR7ZW_66Z5zmvEj_35GDBKpuxYga"
consumer_secret = secretkey
neo_fin_key = neo_fin_key
access_token = access_token
sid = sid
neo_fin_key = neo_fin_key

client = NeoAPI(consumer_key=consumer_key, consumer_secret=consumer_secret, environment="prod", access_token=access_token, neo_fin_key=neo_fin_key)
client.login(mobilenumber=mobileNumber, password=login_password)
client.session_2fa(OTP=mpin)

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# token = "33OUTDUE57WS3TUPHPLFUCGHFM"
# api_key = "Ytt1NkKD"
# clientId = "R865920"
# pwd = '7355'
# correlation_id = "Rahul_7355"

# ltp_data = {}

# tokens = []
symbol_map = {}


token = "EAUTTJX3764X7IU32OLXX6D2SM"
api_key = "haALEYIV "
clientId = "M833928"
pwd = '4041'
correlation_id = "mITHESH_7355"
ltp_data = {}

tokens = []
symbol_map = {}


# api_key = 'SsUDlNA9 '
# clientId = 'A1079871'
# pwd = '0465'
# token = "OIN6QBZAYV4I26Q55OYASIEQVY"
# totp = pyotp.TOTP(token).now()
# correlation_id = "Anil"
# symbol_map = {}

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
                
                tick = message
                print(tick)
                
                token = str(tick.get('token'))
                ltp = tick.get('last_traded_price')
                pSymbolName = symbol_map.get(token, token)

                if token and ltp:
                    price = ltp / 100
                    india_timezone = pytz.timezone("Asia/Kolkata")

                    now = datetime.datetime.now(india_timezone)
                    minute = (now.minute // 2) * 2
                    candle_time_obj = now.replace(minute=minute, second=0, microsecond=0)
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
                            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'open': price,
                            'high': price,
                            'low': price,
                            'close': price,
                        }

                    r.set(pSymbolName,price)
                    r.set(redis_key, json.dumps(candle))

                    last_candle_time_map[pSymbolName] = candle_time
                    last_candle_map[pSymbolName] = candle

                    logger.info(f"[CANDLE] {pSymbolName} @ {candle_time} -> {candle}")

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

import os
file_path = "main_csv.csv"
file_exists = os.path.isfile(file_path)

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
                data = pd.concat([df, signal], axis=1)
                last_row_df = data.tail(1)

                if last_row_df.empty or "timestamp" not in last_row_df.columns:
                    continue

                redis_last_row_timestamp = last_row_df["timestamp"].values[0]

                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    last_row_df.to_csv(file_path, mode="w", index=False, header=True, line_terminator="\n")
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


def create_kotak_df():
    url = client.scrip_master(exchange_segment="nse_fo")
    df = pd.read_csv(url,low_memory=False)
    return df


def get_weekly_df(index):
    df = create_kotak_df()
    df = df[df['pInstType'] == 'OPTIDX']
    df = df[df['pSymbolName'] == index]
    df["lExpiryDate "] = df["lExpiryDate "].apply(lambda x: dt.datetime.fromtimestamp(x).date() + relativedelta(years=10))
    exp = df["lExpiryDate "].to_list()
    td = dt.date.today()
    cur_exp = min(exp, key=lambda x: (x - td))
    df = df[df['lExpiryDate '] == cur_exp]
    df['dStrikePrice;'] = df["dStrikePrice;"].apply(lambda x: int(str(x)[:5]))
    df = df[['pTrdSymbol', 'pSymbol', 'pOptionType', 'dStrikePrice;', 'lExpiryDate ']]
    df = df.reset_index(drop=True)
    return df


def select_trading_symbol(index, ltp, CE_or_PE):
    df = get_weekly_df(index)
    minus_part = ltp % 100
    ltp = ltp - minus_part
    
    data = df[(df["dStrikePrice;"] == ltp) & (df["pOptionType"] == f"{CE_or_PE}")]
    if data.empty:
        return None, None
    
    pTrdSymbol = data["pTrdSymbol"].values[0]
    pSymbol = data["pSymbol"].values[0]
    return pTrdSymbol, pSymbol


def trading_symbol():
    market_open_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    current_time = datetime.now()
    market_close_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)

    while market_open_time < current_time < market_close_time:
        try:
            data = r.get("99926000")
            output = int(float(data))

            CE = select_trading_symbol("NIFTY",output - 100,"CE")
            PE = select_trading_symbol("NIFTY",output + 100,"PE")
            data_dict = {
                "CE":CE,
                "PE":PE
            }
            r.set("Trading_symbol", json.dumps(data_dict, default=str))
            time.sleep(60)
        except Exception as e:
            print(f"Error in trading symbol function is : {e}")

if __name__ == "__main__":
    P0 = Process(target=run_websocket)
    P1 = Process(target=update_indicators_and_signal)
    P2 = Process(target=trading_symbol)
    
    P0.start()
    P1.start()
    P2.start()
    
    P0.join()
    P1.join()
    P2.join()