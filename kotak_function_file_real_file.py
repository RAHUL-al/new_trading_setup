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



response_data = None
response_event = threading.Event()


def create_kotak_df():
    url = kl.client.scrip_master(exchange_segment="nse_fo")
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



def select_ltp_price(index, ltp, CE_or_PE):
    df = get_weekly_df(index)
    # ltp = int(float(ltp))
    minus_part = ltp % 100
    ltp = ltp - minus_part
    
    data = df[(df["dStrikePrice;"] == ltp) & (df["pOptionType"] == f"{CE_or_PE}")]
    if data.empty:
        return None, None
    
    pTrdSymbol = data["pTrdSymbol"].values[0]
    pSymbol = data["pSymbol"].values[0]
    price = fetch_ltp(pSymbol)
    price = int(float(price))
    
    if 140 < price < 180:
        print(price)
    
    else:
        ltp = ltp - 100
        select_ltp_price("BANKNIFTY",ltp,"PE")
    return pTrdSymbol, pSymbol



def on_message(message):
    global response_data
    response_data = message
    response_event.set()
    print('[Res]: ', message)


def on_error(message):
    print('[OnError]: ', message)
    
    
def on_open():
    print('[OnOpen]: WebSocket connection opened.')


def on_close():
    print('[OnClose]: WebSocket connection closed.')


kl.client.on_message = on_message
kl.client.on_error = on_error  
kl.client.on_close = None  
kl.client.on_open = None 




def fetch_ltp(pSymbol):
    global response_data
    response_data = None 
    response_event.clear() 

    inst_tokens = [{"instrument_token": pSymbol, "exchange_segment": "nse_fo"}]
    kl.client.quotes(instrument_tokens=inst_tokens, quote_type="ltp", isIndex=False)

    
    if response_event.wait(timeout=1):
        if response_data and response_data.get('type') == 'quotes' and 'data' in response_data:
            ltp_value = response_data['data'][0].get('ltp')
            return ltp_value

    return None


def modify_Order(order_id,price,trigger_price,order_type,quantity):
    try:
        modifyorder = kl.client.modify_order(
            order_id=f'{order_id}',
            price=f"{price}",
            trigger_price=f"{trigger_price}",
            order_type=f"{order_type}", 
            quantity=f'{quantity}', 
            validity="DAY", 
        )
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
    return modifyorder




def place_order_for_buy_index(trading_symbol,quantity):
    order_response = kl.client.place_order(
        exchange_segment="nse_fo",
        product="MIS",
        price="0",
        order_type="MKT",
        quantity=f"{quantity}",
        validity="DAY",
        trading_symbol=trading_symbol,
        transaction_type="B"
    )
    return order_response
    
    
    
def place_order_for_sell_index(trading_symbol,quantity):
    order_response = kl.client.place_order(
        exchange_segment="nse_fo",
        product="MIS",
        price="0",
        order_type="MKT",
        quantity=f"{quantity}",
        validity="DAY",
        trading_symbol=trading_symbol,
        transaction_type="S"
    )
    return order_response



def place_order_for_buy(trading_symbol):
    order_response = kl.client.place_order(
        exchange_segment="nse_cm",
        product="MIS",
        price="0",
        order_type="MKT",
        quantity="1",
        validity="DAY",
        trading_symbol=f"{trading_symbol}",
        transaction_type="B"
    )
    return order_response




def place_order_for_sell(trading_symbol):
    order_response = kl.client.place_order(
        exchange_segment="nse_cm",
        product="MIS",
        price="0",
        order_type="MKT",
        quantity="1",
        validity="DAY",
        trading_symbol=f"{trading_symbol}",
        transaction_type="S"
    )
    return order_response



def Stoploss_Order(price, trigger_price, trading_symbol, transaction_type, order_type):
    order_response = kl.client.place_order(
        exchange_segment="nse_cm",
        product="MIS",
        price=price,
        trigger_price=trigger_price,
        order_type=order_type,
        quantity="1",
        validity="DAY",
        trading_symbol=trading_symbol,
        transaction_type=f"{transaction_type}"
    )
    return order_response



# Example call
# data = Stoploss_Order("507.50", "507.50", "COALINDIA-EQ", "S", "SL")
# print(data)



def square_off_position(symbol,transaction_type,quantity):
    try:
        square_off_order = kl.client.place_order(exchange_segment="NFO", product="MIS", price="0", order_type="MKT", quantity=f"{quantity}", validity="DAY", trading_symbol=str(symbol),
                   transaction_type=f"{transaction_type}")
        return square_off_order
    except Exception as e:
        return {'Error': str(e)}
    


def position():
    data = kl.client.positions()
    return data

    

