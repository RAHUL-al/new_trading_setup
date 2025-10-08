import kotak_function_file
import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import requests
from datetime import datetime,timedelta,date
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from SmartApi import SmartConnect
import pyotp
import kotak_login as kl
from neo_api_client import NeoAPI
from dateutil.relativedelta import relativedelta
from threading import Thread
import dbload
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker



username = 'root'
password = 'Rahul@7355'
host = 'localhost'
port = 3306

    



from urllib.parse import quote_plus
password = quote_plus(password)

engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
connection = engine.connect()
dbname = "stocks_data"


totalpoint_taken = f"totalpoint_taken_{datetime.now().strftime('%d%m%Y')}"
totalpoint_taken_exists = sa.inspect(engine).has_table("%s"%(f"{totalpoint_taken}"))

final_point = 0

if totalpoint_taken_exists:
    query = f"SELECT * FROM `{totalpoint_taken}` LIMIT 1"
    result = connection.execute(text(query)).fetchone()
    if result:
        enter_exit_point = result[0]
        final_point = enter_exit_point
        print(enter_exit_point)
else:
    enterdata = {
                    "enter_exit_point":0,
                }
    df = pd.DataFrame([enterdata])
    dbload.createtable(totalpoint_taken,df)
    




api_key = 'Ytt1NkKD'
clientId = 'R865920'
pwd = '7355'
token = "33OUTDUE57WS3TUPHPLFUCGHFM"
totp = pyotp.TOTP(token).now()
correlation_id = "Rahul_7355"

# Login to the Smart API
smartApi = SmartConnect(api_key)
data1 = smartApi.generateSession(clientId, pwd, totp)
authToken = data1['data']['jwtToken']
refreshToken = data1['data']['refreshToken']
feedToken = smartApi.getfeedToken()
res = smartApi.getProfile(refreshToken)
smartApi.generateToken(refreshToken)

current_date = datetime.now()
formatted_date = current_date.strftime("%Y-%m-%d")


def history(exchange, symboltoken):
    try:
        historicParam = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": "THREE_MINUTE",
            "fromdate": "2024-03-01 09:15",
            "todate": f"{formatted_date} 15:30"
        }
        CandleData = smartApi.getCandleData(historicParam)
        columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
        CandleData = pd.DataFrame(CandleData["data"], columns=columns)
    except Exception as e:
        print("Historic Api failed: {}".format(e))
    return CandleData




#################################################################################################################################################



# Generate UT Bot alerts
def bot_alerts(data, a=2, c=100, h=False):
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



######################################################################################################################################################




# Initialize trading signals
def strategy1(data):
    data['buy_signal'] = 0
    data['sell_signal'] = 0

    ut_buy_condition = data["buy"]
    ut_sell_condition = data["sell"]
    flag = 0

    for i in range(len(data)):
        if ut_buy_condition[i] == True and flag == 0:
            data.loc[i, 'buy_signal'] = 1
            flag = 1
            
        elif ut_sell_condition[i] == True and flag == 1:
            data.loc[i, 'sell_signal'] = -1
            flag = 0

    return data


####################################################################################################################





market_open_time = datetime.now().replace(hour=4, minute=54, second=0, microsecond=0)
# current_time = datetime.now().replace(hour=10, minute=15, second=0, microsecond=0)
market_close_time = datetime.now().replace(hour=11, minute=50, second=0, microsecond=0)

def start_process():
    longPoint = 0
    sellpoint = 0
    ShortPoint = 0
    GainPoint = 0
    TotalCount = 0
    increaseShort = 0
    increaseGain = 0
    short_flag = 0
    flag = 0
    total_number_of_trade_taken = 0
    short_trade_taken = 0
    long_trade_taken = 0
    loss_gain_in_trade = 0
    Stoploss_hit_in_long_trade = 0
    StopLoss_hit_in_short_trade = 0
    Loss_booked_in_Long_trade = 0
    Loss_booked_in_Short_trade = 0
    stoploss_candle_for_short = 0
    stoploss_candle_for_long = 0
    short_trade_continue = 0
    long_trade_continue = 0
    short_trade_continue_handle = 0
    long_trade_continue_handle = 0
    StopLossForShort = 0
    StopLossForLong = 0
    Stoploss_trail_for_short = 0
    Stoploss_trail_for_long = 0
    order_id_for_short = 0
    order_id_for_long = 0
    price = 0
    CandleSize = 0
    StopLoss_point = 0
    Quantity = 0
    last_traded_value_for_short = 0
   
    last_traded_value_for_long = 0
    
    TotalPointGain = 0
    current_traded_value_for_short = 0
    current_traded_value_for_long = 0
    pSymbol = str(0)
    
    
    
    
    
    
    shorttable = f"banknifty_short_{datetime.now().strftime('%d%m%Y')}"
    longtable = f"banknifty_long_{datetime.now().strftime('%d%m%Y')}"
    position_handle = f"position_{datetime.now().strftime('%d%m%Y')}"
    
    
    positiondata = {
                    "Quantity":Quantity,
                    "last_traded_value_for_short":last_traded_value_for_short,
                    "last_traded_value_for_long":last_traded_value_for_long,
                    "TotalPointGain":TotalPointGain,
                    "StopLoss_point":StopLoss_point,
                    "pSymbol":pSymbol,
                }
    df = pd.DataFrame([positiondata])
    dbload.createtable(position_handle,df)
    
    shorttable_exists = sa.inspect(engine).has_table("%s"%(f"{shorttable}"))
    longtable_exists = sa.inspect(engine).has_table("%s"%(f"{longtable}"))
    position_handle_exists = sa.inspect(engine).has_table("%s"%(f"{position_handle}"))
    
    if position_handle_exists:
        query = f"SELECT * FROM `{position_handle}` LIMIT 1"
        result = connection.execute(text(query)).fetchone()
        if result:
            Quantity = result[0]
            last_traded_value_for_short = result[1]
            last_traded_value_for_long = result[2]
            TotalPointGain = result[3]
            StopLoss_point = result[4]
            pSymbol = result[5]

    
    if shorttable_exists:
        query = f"SELECT * FROM `{shorttable}` LIMIT 1"
        result = connection.execute(text(query)).fetchone()
        if result:
            order_id_for_short = result[0]
            stoploss_candle_for_short = result[1]
            StopLossForShort = result[2]
            sellpoint = result[3]
            low = result[4]
            short_flag = result[5]
            long_trade_continue_handle = result[6]
            Stoploss_trail_for_short = result[7]
            short_trade_continue = result[8]
            price = result[9]
            CandleSize = result[10]
            print("Short variables reinitialized from database.")
            
        
    if longtable_exists:
        query = f"SELECT * FROM `{longtable}` LIMIT 1"
        result = connection.execute(text(query)).fetchone()

        if result:
            order_id_for_long = result[0] 
            stoploss_candle_for_long = result[1] 
            StopLossForLong = result[2]  
            longPoint = result[3] 
            high = result[4] 
            flag = result[5] 
            short_trade_continue_handle = result[6] 
            long_trade_continue_handle=result[7]
            Stoploss_trail_for_long = result[8] 
            long_trade_continue = result[9]
            price = result[10] 
            CandleSize = result[11] 
            print("Long variables reinitialized from database.")
            
    


    next_action_time = datetime.now() + timedelta(minutes=30)
    trading_symbol = str(0)
    i = -1
    j = -2


    while True:
        current_time = datetime.now()

        if market_open_time <= current_time <= market_close_time:
            exchange = "NSE"
            symboltoken = "99926009"
            
            maindata = history(exchange,symboltoken)
            
            if maindata.empty:
                print("No data available. Waiting for the next trading session.")
                time.sleep(2)
                continue
            
            signals = bot_alerts(maindata)
            data = pd.concat([maindata,signals],axis=1)
            strategy1(data)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(data.tail(15))
            
            if len(data) > abs(j) and len(data) > abs(i):
                
                
                
                
                
                if data["sell_signal"].iloc[j] == -1 and short_flag == 0:  # Enter short position
                    if StopLoss_point == 1:
                        Quantity = 15
                        
                    else:
                        Quantity = 15
                    
                    ltp = data["Close"].iloc[j] - 300
                    trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"PE")
                    order_response = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
                    
                    last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    while last_traded_value_for_short is None:
                        last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                        
                    last_traded_value_for_short = int(float(last_traded_value_for_short))
                    
                    if sa.inspect(engine).has_table(position_handle):
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            Quantity = :Quantity_value, 
                            last_traded_value_for_short = :last_traded_value_for_short_value,
                            pSymbol = :pSymbol_value
                            """
                        )
                        connection.execute(update_query, {
                            'Quantity_value':Quantity,
                            'last_traded_value_for_short_value': last_traded_value_for_short,
                            'pSymbol_value':pSymbol,
                        })
                        connection.commit()
                    
        
                    print(order_response)
                    if order_id_for_short is not None:
                        stoploss_candle_for_short = data["Open"].iloc[j] - data["Close"].iloc[j]
                        StopLossForShort = data["High"].iloc[j]
                        
                        sellpoint = data["Close"].iloc[j]
                        low = data["Low"].iloc[j]
                        short_flag = 1
                        long_trade_continue_handle = 0
                        total_number_of_trade_taken += 1
                        short_trade_taken += 1
                        
                        stockdata = {
                                "order_id_for_short":order_id_for_short,
                                "stoploss_candle_for_short":stoploss_candle_for_short,
                                "StopLossForShort":StopLossForShort,
                                "sellpoint":sellpoint,
                                "low":low,
                                "short_flag":short_flag,
                                "long_trade_continue_handle":long_trade_continue_handle,
                                "Stoploss_trail_for_short":Stoploss_trail_for_short,
                                "short_trade_continue":short_trade_continue,
                                "price":price,
                                "CandleSize":CandleSize,
                            }
                        df = pd.DataFrame([stockdata])
                        dbload.createtable(shorttable,df)
                        
                        print(f"taking position at the point {sellpoint} at index {j}")
                    
                    else:
                        print("Order id is None for Short.")
                        break
                    
                    
                elif (data["Close"].iloc[i] > StopLossForShort) and short_flag == 1:  # Stoploss condition in short position
                    square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                    current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_short is None:
                        current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_short = int(float(current_traded_value_for_short))
                    TotalPointGain += current_traded_value_for_short - last_traded_value_for_short
                    StopLoss_point = 1
                    
                    if sa.inspect(engine).has_table(position_handle):
                        update_query = text(
                            f"""
                            UPDATE `{position_handle}`
                            SET
                                StopLoss_point = :StopLoss_point_value,
                                TotalPointGain = :TotalPointGain_value
                            """
                        )
                        connection.execute(update_query, {
                            'TotalPointGain_value': TotalPointGain,
                            'StopLoss_point_value': StopLoss_point,
                        })
                        
                        connection.commit()
                    
                    increaseShort = sellpoint - StopLossForShort
                    ShortPoint += increaseShort
                    TotalCount += increaseShort
                    Stoploss_trail_for_short = 0
                    short_flag = 0
                    point2 = data["Close"].iloc[i]
                    short_trade_continue_handle = 1
                    short_trade_continue = 1
                    StopLoss_hit_in_short_trade += 1
                    connection.execute(text(f"DROP TABLE `{shorttable}`"))
                    print("Enter in stoploss condition.")
                    time.sleep(2)
                    print(f"StopLoss hit at point : {point2} and index is {i} and difference is {increaseShort}")
                    
                    
                    
                elif (data["buy_signal"].iloc[j] == 1) and (short_flag == 1):  # Exit short position
                    square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                    current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_short is None:
                        current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_short = int(float(current_traded_value_for_short))
                    TotalPointGain += current_traded_value_for_short - last_traded_value_for_short
                    StopLoss_point = 1
                    
                    if sa.inspect(engine).has_table(position_handle):
                        
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            TotalPointGain = :TotalPointGain_value, 
                            StopLoss_point = :StopLoss_point_value
                            """
                        )
                        connection.execute(update_query, {
                            'TotalPointGain_value': TotalPointGain,
                            'StopLoss_point_value': StopLoss_point,
                        })
                        connection.commit()
                    
                    increaseShort = sellpoint - data["Close"].iloc[j]
                    ShortPoint += increaseShort
                    TotalCount += increaseShort
                    short_flag = 0
                    point = data["Close"].iloc[j]
                    connection.execute(text(f"DROP TABLE `{shorttable}`"))
                    print("Enter in exit short position condition")
                    time.sleep(2)
                    
                    print(f"Coming outside from the trade at the point {point} at index {j} and difference is {increaseShort}")
                    
                    
                    
                    
                elif (short_flag == 1) and (data["sell_signal"].iloc[j] != -1):  # Trailing StopLoss for short
                    CandleSize = data["Open"].iloc[j] - data["Close"].iloc[j]
                    if CandleSize > 0:
                        if data["Close"].iloc[j] < low:
                            StopLossForShort = data["High"].iloc[j] - (CandleSize * 0.40)
                            stoploss_candle_for_short = CandleSize
                            low = data["Low"].iloc[j]
                            action_time = data["Time"].iloc[j]
                            Stoploss_trail_for_short = 1  # Ensure this is defined before usage
                            
                            if sa.inspect(engine).has_table(shorttable):
                                try:
                                    
                                    update_query = text(
                                        f"""UPDATE `{shorttable}` SET 
                                        CandleSize = :CandleSize_value, 
                                        StopLossForShort = :StopLossForShort_value, 
                                        low = :low_value, 
                                        StopLoss_trail_for_short = :StopLoss_trail_for_short_value, 
                                        stoploss_candle_for_short = :stoploss_candle_for_short_value 
                                        """
                                    )
                                    connection.execute(update_query, {
                                        'CandleSize_value': CandleSize,
                                        'StopLossForShort_value': StopLossForShort,
                                        'low_value': low,
                                        'StopLoss_trail_for_short_value': Stoploss_trail_for_short,
                                        'stoploss_candle_for_short_value': stoploss_candle_for_short
                                    })
                                    
                                    connection.commit()
                                    
                                except Exception as e:
                                    print(f"An error occurred: {e}")
                                    connection.rollback()
                                finally:
                                    connection.close()
                                
                            print(f"Stoploss trail to {StopLossForShort} at the time of candle {action_time}.")

                            
                            
                        
                elif (short_trade_continue == 1) and data["Close"].iloc[j] < low and (short_trade_continue_handle == 1):
                    if StopLoss_point == 1:
                        Quantity = 15
                        
                    else:
                        Quantity = 15
                        
                    ltp = data["Close"].iloc[j] - 300
                    trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"PE")
                    order_response1 = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
                    last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    while last_traded_value_for_short is None:
                        last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    last_traded_value_for_short = int(float(last_traded_value_for_short))
                    
                    
                    
                    if sa.inspect(engine).has_table(position_handle):
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            Quantity = :Quantity_value, 
                            last_traded_value_for_short = :last_traded_value_for_short_value,
                            pSymbol = :pSymbol_value
                            """)
                        connection.execute(update_query, {
                            'Quantity_value':Quantity,
                            'last_traded_value_for_short_value': last_traded_value_for_short,
                            'pSymbol_value':pSymbol,
                        })
                        connection.commit()
                    
                    
                    print(order_response1)
                    time.sleep(2)
                    order_id_for_short = order_response1.get('nOrdNo')
                    if order_id_for_short is not None:
                        stoploss_candle_for_short = data["Open"].iloc[j]-data["Close"].iloc[j]
                        StopLossForShort = data["High"].iloc[j]
                        short_flag = 1
                        sellpoint = data["Close"].iloc[j]
                        low = data["Low"].iloc[j]
                        action_time = data["Time"].iloc[j]
                        short_trade_continue_handle = 0
                        
                        
                        stockdata = {
                            "order_id_for_short":order_id_for_short,
                            "stoploss_candle_for_short":stoploss_candle_for_short,
                            "StopLossForShort":StopLossForShort,
                            "sellpoint":sellpoint,
                            "low":low,
                            "short_flag":short_flag,
                            "long_trade_continue_handle":long_trade_continue_handle,
                            "Stoploss_trail_for_short":Stoploss_trail_for_short,
                            "short_trade_continue":short_trade_continue,
                            "price":price,
                            "CandleSize":CandleSize,
                        }
                        df = pd.DataFrame([stockdata])
                        dbload.createtable(shorttable,df)
                        print(f"Again Start short position at index {i} at the time {action_time}.")
                    else:
                        print("Order is None in again take short position")
                        break
                    
                    
                    
                if short_flag == 1:
                    current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_short is None:
                        current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_short = int(float(current_traded_value_for_short))
                    TotalPointGain = current_traded_value_for_short - last_traded_value_for_short
                    if TotalPointGain >= 5:
                        square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                        connection.execute(text(f"DROP TABLE `{shorttable}`"))
                        connection.execute(text(f"DROP TABLE `{position_handle}`"))
                        print("Deleted both shorttable and position table")
                        print("total 22 point Gain")
                        update_query = text(
                            f"""UPDATE `{totalpoint_taken}` SET 
                            enter_exit_point = :enter_exit_point_value
                            """ 
                        )
                        connection.execute(update_query, {
                            'enter_exit_point_value':1,
                        })
                        connection.commit()
                        break
                        
                    

    
                    
                if data["buy_signal"].iloc[j] == 1 and flag == 0: # Enter Long position
                    if StopLoss_point == 1:
                        Quantity = 15
                    else:
                        Quantity = 15
       
                    ltp = data["Close"].iloc[j] + 300
                    trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"CE")
                    order_response1 = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
                    last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    while last_traded_value_for_long is None:
                        last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    last_traded_value_for_long = int(float(last_traded_value_for_long))
                    
                    if sa.inspect(engine).has_table(position_handle):
                        
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            Quantity = :Quantity_value, 
                            last_traded_value_for_long = :last_traded_value_for_long_value,
                            pSymbol = :pSymbol_value
                            """
                        )
                        connection.execute(update_query, {
                            'Quantity_value':Quantity,
                            'last_traded_value_for_long_value': last_traded_value_for_long,
                            'pSymbol_value':pSymbol,
                        })
                        connection.commit()
                    
                    
                    print(order_response1)
                    time.sleep(3)
                    order_id_for_long = order_response1.get('nOrdNo')
                    if order_id_for_long is not None:
                        stoploss_candle_for_long = data["Close"].iloc[j] - data["Open"].iloc[j]
                        
                        StopLossForLong = data["Low"].iloc[j]
                        longPoint = data["Close"].iloc[j]
                        high = data["High"].iloc[j]
                        flag = 1
                        total_number_of_trade_taken += 1
                        long_trade_taken += 1
                        short_trade_continue_handle = 0
                        
                        
                        stockdata = {
                            "order_id_for_long":order_id_for_long,
                            "stoploss_candle_for_long":stoploss_candle_for_long,
                            "StopLossForLong":StopLossForLong,
                            "longPoint":longPoint,
                            "high":high,
                            "flag":flag,
                            "short_trade_continue_handle":short_trade_continue_handle,
                            "long_trade_continue_handle":long_trade_continue_handle,
                            "Stoploss_trail_for_long":Stoploss_trail_for_long,
                            "long_trade_continue":long_trade_continue,
                            "price":price,
                            "CandleSize":CandleSize,
                        }
                        df = pd.DataFrame([stockdata])
                        dbload.createtable(longtable,df)
                        
                        print(f"Taking position at point {longPoint} at index {j}")
                        
                    else:
                        print("Order id is None in Long position.")
                        break
                    
                    
                    
                elif data["Close"].iloc[i] < StopLossForLong and flag == 1: # stoploss condition for long position
                    square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                    current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_long is None:
                        current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_long = int(float(current_traded_value_for_long))
                    TotalPointGain += current_traded_value_for_long - last_traded_value_for_long
                    StopLoss_point = 1
                    if sa.inspect(engine).has_table(position_handle):
                        
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            TotalPointGain = :TotalPointGain_value, 
                            StopLoss_point = :StopLoss_point_value
                            """
                        )
                        connection.execute(update_query, {
                            'TotalPointGain_value': TotalPointGain,
                            'StopLoss_point_value': StopLoss_point,
                        })
                        
                        connection.commit()
                    increaseGain = StopLossForLong - longPoint
                    GainPoint += increaseGain
                    TotalCount += increaseGain
                    Stoploss_trail_for_long = 0
                    long_trade_continue = 1
                    long_trade_continue_handle = 1
                    flag = 0
                    point4 = data["Close"].iloc[i]
                    connection.execute(text(f"DROP TABLE `{longtable}`"))
                    print("drop the longtable because enter in stoploss condition")
                    time.sleep(3)
                    Stoploss_hit_in_long_trade += 1
                    print(f"Coming outside from the trade at the point {point4} at the index {i} and difference is {increaseGain}")  
                    
                    
                    
                    
                elif data["sell_signal"].iloc[j] == -1 and flag == 1: # Exit long position
                    square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                    current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_long is None:
                        current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_long = int(float(current_traded_value_for_long))
                    TotalPointGain += current_traded_value_for_long - last_traded_value_for_long
                    StopLoss_point = 1
                    if sa.inspect(engine).has_table(position_handle):
                        
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            TotalPointGain = :TotalPointGain_value, 
                            StopLoss_point = :StopLoss_point_value
                            """
                        )
                        connection.execute(update_query, {
                            'TotalPointGain_value': TotalPointGain,
                            'StopLoss_point_value': StopLoss_point,
                        })
                        connection.commit()
                    increaseGain = data["Close"].iloc[j] - longPoint
                    GainPoint += increaseGain
                    TotalCount += increaseGain
                    flag = 0
                    point3 = data["Close"].iloc[j]
                    connection.execute(text(f"DROP TABLE `{longtable}`"))
                    print("Drop the longtable becasue i am in exit long positon.")
                    time.sleep(2)
                        
                    print(f"Coming outside from the trade at the point {point3} at the index {j} and difference is {increaseGain}")
                    
                        
                
                    
                    
                elif (flag == 1) and data["buy_signal"].iloc[j] != 1: # Trailing StopLoss for long
                    CandleSize = data["Close"].iloc[j] - data["Open"].iloc[j]
                    if CandleSize > 0:
                        if (data["Close"].iloc[j] > high):
                            StopLossForLong = data["Low"].iloc[j] + CandleSize * 0.20
                            stoploss_candle_for_long = CandleSize
                            high = data["High"].iloc[j]
                            Stoploss_trail_for_long = 1

                            if sa.inspect(engine).has_table(longtable):
                                update_query = text(
                                    f"""UPDATE `{longtable}` SET 
                                    CandleSize = :CandleSize_value, 
                                    StopLossForLong = :StopLossForLong_value, 
                                    high = :high_value, 
                                    StopLoss_trail_for_long = :StopLoss_trail_for_long_value, 
                                    stoploss_candle_for_long = :stoploss_candle_for_long_value
                                    """
                                )
                                connection.execute(update_query, {
                                    'CandleSize_value': CandleSize,
                                    'StopLossForLong_value': StopLossForLong,
                                    'high_value': high,
                                    'StopLoss_trail_for_long_value': Stoploss_trail_for_long,
                                    'stoploss_candle_for_long_value': stoploss_candle_for_long
                                })
                                connection.commit()
                                
                            else:
                                print("longtable does not exist in trailing stoploss for long.")
                            action_time = data["Time"].iloc[j]
                            print(f"Stoploss trail to {StopLossForLong} at the time of candle {action_time}.")   

                
                
                elif long_trade_continue == 1 and data["Close"].iloc[j] > high and long_trade_continue_handle == 1:
                    if StopLoss_point == 1:
                        Quantity = Quantity + 15
                        
                    else:
                        Quantity = 15
                        
                    ltp = data["Close"].iloc[j] + 300
                    trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"CE")
                    order_response1 = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
                    last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    while last_traded_value_for_long is None:
                        last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    last_traded_value_for_long = int(float(last_traded_value_for_long))
                    if sa.inspect(engine).has_table(position_handle):
                       
                        update_query = text(
                            f"""UPDATE `{position_handle}` SET 
                            Quantity = :Quantity_value, 
                            last_traded_value_for_long = :last_traded_value_for_long_value,
                            pSymbol = :pSymbol_value
                            """ 
                        )
                        connection.execute(update_query, {
                            'Quantity_value':Quantity,
                            'last_traded_value_for_long_value': last_traded_value_for_long,
                            'pSymbol_value':pSymbol,
                        })
                        connection.commit()
                        
                    print(order_response1)
                    order_id_for_long = order_response1.get('nOrdNo')
                    
                    if order_id_for_long is not None:
                        stoploss_candle_for_long = data["Close"].iloc[j] - data["Open"].iloc[j]
                        StopLossForLong = data["Low"].iloc[j]
                        longPoint = data["Close"].iloc[j]
                        high = data["High"].iloc[j]
                        action_time = data["Time"].iloc[j]
                        long_trade_continue_handle = 0
                        flag = 1
                        
                        stockdata = {
                            "order_id_for_long":order_id_for_long,
                            "stoploss_candle_for_long":stoploss_candle_for_long,
                            "StopLossForLong":StopLossForLong,
                            "longPoint":longPoint,
                            "high":high,
                            "flag":flag,
                            "short_trade_continue_handle":short_trade_continue_handle,
                            "long_trade_continue_handle":long_trade_continue_handle,
                            "Stoploss_trail_for_long":Stoploss_trail_for_long,
                            "long_trade_continue":long_trade_continue,
                            "price":price,
                            "CandleSize":CandleSize,
                        }
                        
                        df = pd.DataFrame([stockdata])
                        dbload.createtable(longtable,df)
                        
                        
                        print(f"Again start long position at index {j} at the time {action_time}.")  
                    
                    else:
                        print("Order id is None in Long position")
                        break
                    
                    
                if flag == 1:
                    current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    while current_traded_value_for_long is None:
                        current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
                    current_traded_value_for_long = int(float(current_traded_value_for_long))
                    TotalPointGain = current_traded_value_for_long - last_traded_value_for_long
                    if TotalPointGain >= 5:
                        square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
                        connection.execute(text(f"DROP TABLE `{longtable}`"))
                        connection.execute(text(f"DROP TABLE `{position_handle}`"))
                        print("Deleted both longtable and position table")
                        print("Total 22 point Gain.")
                        update_query = text(
                            f"""UPDATE `{totalpoint_taken}` SET 
                            enter_exit_point = :enter_exit_point_value
                            """ 
                        )
                        connection.execute(update_query, {
                            'enter_exit_point_value':1,
                        })
                        connection.commit()
                        break

        else:
            print("Market is closed. Waiting for the next trading session.")
            break
    
        time.sleep(0.5)

# start_process()
if final_point != 1:
    while True:
        try:
            start_process()
            time.sleep(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting the process...")
            time.sleep(2)
            continue
        

