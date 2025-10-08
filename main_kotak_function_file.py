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
import threading
import time
from datetime import datetime
from kotak_login import get_kotak_client
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text


response_data = None
response_event = threading.Event()

client = get_kotak_client()

username = 'trading_user'
password = 'Rahul@7355'
host = 'localhost'
port = 3306

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

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
    # ltp = int(float(ltp))
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
        data = r.get("99926000")
        output = int(float(data))

        """For found CE symbol"""
        CE = select_trading_symbol("NIFTY",output - 100,"CE")
        PE = select_trading_symbol("NIFTY",output + 100,"PE")
        data_dict = {
            "CE":CE,
            "PE":PE
        }
        r.set("Trading_symbol", json.dumps(data_dict, default=str))
        # data = r.get("Trading_symbol")
        # data_dict = json.loads(data)
        # print(data_dict["CE"][1])
        # print(data_dict["CE"][0])
        time.sleep(60)


if __name__ == "__main__":
    trading_symbol()
# def select_ltp_price(index, ltp, CE_or_PE):
#     df = get_weekly_df(index)
#     # ltp = int(float(ltp))
#     minus_part = ltp % 100
#     ltp = ltp - minus_part
    
#     data = df[(df["dStrikePrice;"] == ltp) & (df["pOptionType"] == f"{CE_or_PE}")]
#     if data.empty:
#         return None, None
    
#     pTrdSymbol = data["pTrdSymbol"].values[0]
#     pSymbol = data["pSymbol"].values[0]
#     price = fetch_ltp(pSymbol)
#     price = int(float(price))
    
#     if 140 < price < 180:
#         print(price)
    
#     else:
#         ltp = ltp - 100
#         select_ltp_price("BANKNIFTY",ltp,"PE")
#     return pTrdSymbol, pSymbol


# def modify_Order(order_id,price,trigger_price,order_type,quantity):
#     try:
#         modifyorder = kl.client.modify_order(
#             order_id=f'{order_id}',
#             price=f"{price}",
#             trigger_price=f"{trigger_price}",
#             order_type=f"{order_type}", 
#             quantity=f'{quantity}', 
#             validity="DAY", 
#         )
#     except ValueError as e:
#         print(f"ValueError: {e}")
#         return None
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return None
    
#     return modifyorder




# def place_order_for_buy_index(trading_symbol,quantity):
#     order_response = kl.client.place_order(
#         exchange_segment="nse_fo",
#         product="MIS",
#         price="0",
#         order_type="MKT",
#         quantity=f"{quantity}",
#         validity="DAY",
#         trading_symbol=trading_symbol,
#         transaction_type="B"
#     )
#     return order_response
    
    
    
# def place_order_for_sell_index(trading_symbol,quantity):
#     order_response = kl.client.place_order(
#         exchange_segment="nse_fo",
#         product="MIS",
#         price="0",
#         order_type="MKT",
#         quantity=f"{quantity}",
#         validity="DAY",
#         trading_symbol=trading_symbol,
#         transaction_type="S"
#     )
#     return order_response



# def place_order_for_buy(trading_symbol):
#     order_response = kl.client.place_order(
#         exchange_segment="nse_cm",
#         product="MIS",
#         price="0",
#         order_type="MKT",
#         quantity="1",
#         validity="DAY",
#         trading_symbol=f"{trading_symbol}",
#         transaction_type="B"
#     )
#     return order_response




# def place_order_for_sell(trading_symbol):
#     order_response = kl.client.place_order(
#         exchange_segment="nse_cm",
#         product="MIS",
#         price="0",
#         order_type="MKT",
#         quantity="1",
#         validity="DAY",
#         trading_symbol=f"{trading_symbol}",
#         transaction_type="S"
#     )
#     return order_response



# def Stoploss_Order(price, trigger_price, trading_symbol, transaction_type, order_type):
#     order_response = kl.client.place_order(
#         exchange_segment="nse_cm",
#         product="MIS",
#         price=price,
#         trigger_price=trigger_price,
#         order_type=order_type,
#         quantity="1",
#         validity="DAY",
#         trading_symbol=trading_symbol,
#         transaction_type=f"{transaction_type}"
#     )
#     return order_response



# # Example call
# # data = Stoploss_Order("507.50", "507.50", "COALINDIA-EQ", "S", "SL")
# # print(data)



# def square_off_position(symbol,transaction_type,quantity):
#     try:
#         square_off_order = kl.client.place_order(exchange_segment="NFO", product="MIS", price="0", order_type="MKT", quantity=f"{quantity}", validity="DAY", trading_symbol=str(symbol),
#                    transaction_type=f"{transaction_type}")
#         return square_off_order
#     except Exception as e:
#         return {'Error': str(e)}
    


# def position():
#     data = kl.client.positions()
#     return data

    

