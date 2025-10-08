    
from neo_api_client import NeoAPI
import datetime as dt
from dateutil.relativedelta import relativedelta
from threading import Thread
import pandas as pd
import json
import threading
import time
import dbload
from datetime import datetime
from kotak_login import get_kotak_client
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI


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
print(client.configuration.bearer_token)


client



# def on_message(message):
#     global response_data
#     response_data = message
#     response_event.set()
#     print('[Res]: ', message)


# def on_error(message):
#     print('[OnError]: ', message)
    
    
# def on_open():
#     print('[OnOpen]: WebSocket connection opened.')


# def on_close():
#     print('[OnClose]: WebSocket connection closed.')


# client.on_message = on_message
# client.on_error = on_error  
# client.on_close = None  
# client.on_open = None 




# def fetch_ltp(pSymbol):
#     global response_data
#     response_data = None 
#     response_event.clear() 

#     inst_tokens = [{"instrument_token": pSymbol, "exchange_segment": "nse_cm"}]
#     client.quotes(instrument_tokens=inst_tokens, quote_type="ltp", isIndex=False)

    
#     if response_event.wait(timeout=1):
#         if response_data and response_data.get('type') == 'quotes' and 'data' in response_data:
#             ltp_value = response_data['data'][0].get('ltp')
#             return ltp_value

#     return None


# value = fetch_ltp("27839")
# print(value)

# def modify_Order(order_id,price,trigger_price,order_type,quantity):
#     try:
#         modifyorder = client.modify_order(
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




# def place_order_for_buy_index(trading_symbol,quantity=1):
#     order_response = client.place_order(
#         exchange_segment="nse_cm",
#         product="MIS",
#         price="0",
#         order_type="MKT",
#         quantity=f"{quantity}",
#         validity="DAY",
#         trading_symbol=trading_symbol,
#         transaction_type="B"
#     )
#     return order_response
    
# # data = place_order_for_buy_index("OBEROIRLTY-EQ")
# # print(data)
    
# def place_order_for_sell_index(trading_symbol,quantity):
#     order_response = client.place_order(
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
#     try:
#         order_response = client.place_order(
#             exchange_segment="nse_cm",
#             product="NRML",
#             order_type="L",
#             quantity="1",
#             price="765",
#             validity="DAY",
#             trading_symbol=trading_symbol,
#             transaction_type="B"
#         )
#         return order_response
#     except Exception as e:
#         print("Error placing order:", e)
#         return None


# # data = place_order_for_buy("NYKAA-EQ")
# # print(data)


# def place_order_for_sell(trading_symbol):
#     order_response = client.place_order(
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
#     order_response = client.place_order(
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
#         square_off_order = client.place_order(exchange_segment="NFO", product="MIS", price="0", order_type="MKT", quantity=f"{quantity}", validity="DAY", trading_symbol=str(symbol),
#                    transaction_type=f"{transaction_type}")
#         return square_off_order
#     except Exception as e:
#         return {'Error': str(e)}
    


# def position():
#     data = client.positions()
#     return data

    

