import pymysql
import json
from datetime import datetime
from multiprocessing import Process
import redis
import ujson as json
from threading import Thread
import pandas as pd
import datetime
# from kotak_function_file import *

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

mysql = pymysql.connect(
    host='localhost',
    user='trading_user',
    password='Rahul@7355',
)
cursor = mysql.cursor()
cursor.execute("create database if not exists stock_analysis")
mysql.database = "stock_analysis"

def store_data_in_mysql(symbol, data):
    try:
        cursor.execute("CREATE TABLE IF NOT EXISTS stock_data (symbol VARCHAR(255), data JSON, timestamp DATETIME)")
        cursor.execute("INSERT INTO stock_data (symbol, data, timestamp) VALUES (%s, %s, %s)",
                       (symbol, json.dumps(data), datetime.now()))
        mysql.commit()
    except Exception as e:
        print(f"Error storing data in MySQL: {e}")
        mysql.rollback()
        
    
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

def strategy(data):
    data['ut_bot_buy_signal'] = 0
    data['ut_bot_sell_signal'] = 0

    ut_buy_condition = data["buy"]
    ut_sell_condition = data["sell"]
    # flag = 0

    for i in range(len(data)):
        if ut_buy_condition[i] == True:
            data.loc[i, 'ut_bot_buy_signal'] = 1
            # flag = 1
            
        elif ut_sell_condition[i] == True:
            data.loc[i, 'ut_bot_sell_signal'] = -1
            # flag = 0

    return data

stocks_dict = {}
def get_stocks_data():
    while True:
        stocks = r.scan_iter("SIGNAL*")
        for stock in stocks:
            data = stock.split(":")
            stock_name = data[1]
            if stock_name not in stocks_dict:
                stocks_dict[stock_name] = json.loads(r.get(stock))

def run_stocks():
    for stock_name, stock_data in stocks_dict.items():
        if "dat" in stock_data:
            now = datetime.date.today()
            history_key = f"HISTORY:{stock_name}:{now}"
            history_data = r.lrange(history_key, 0, -1)
            if len(history_data) < 30:
                return
            df = pd.DataFrame([json.loads(x) for x in history_data])
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float,"total_buy_quantity":float,"total_sell_quantity":float
            })

            signals = bot_alerts(df)
            maindata = pd.concat([df, signals], axis=1)
            maindata = strategy(maindata)
            signal = maindata.iloc[-1]
            signal_from_redis = stock_data["Signal"]
            
            store_data_in_mysql(stock_name, stock_data)
                
t1 = Thread(target=get_stocks_data)
t1.start()
t1.join()               


def human_readable_quantity(qty):
    if qty >= 1_000_000:
        return f"{qty / 1_000_000:.1f}M"
    elif qty >= 1_000:
        return f"{qty / 1_000:.1f}K"
    else:
        return str(qty)

for key in r.scan_iter("SIGNAL*"):
    keydata = key.split(":")
    keydata = keydata[1]
    
    if r.exists(f"place_order_{keydata}"):
        pass
            
    else:
        signal_data = r.get(key)
        json_data = json.loads(signal_data)
        signal_data = json_data["Signal"]
        full_data = json_data["dat"]
        total_buy_quantity = full_data["total_buy_quantity"]
        total_sell_quantity = full_data["total_sell_quantity"]
        signal = json_data["Signal"]

        data = r.get(f"fo_{keydata}")
        if data is None:
            continue
        
        fo_json_data = json.loads(data)

        ce_position = fo_json_data["CE"]
        pe_position = fo_json_data["PE"]
        # if signal == "STRONG_BUY":
        if signal == "STRONG_BUY" and total_buy_quantity > 2 * total_sell_quantity:
            data = ce_position["pTrdSymbol"]
            # print(keydata,signal,human_readable_quantity(total_buy_quantity),human_readable_quantity(total_sell_quantity))
            print(keydata,signal,data)
            
        # elif signal_data == "STRONG_SELL":
        elif signal_data == "STRONG_SELL" and total_sell_quantity > 2 * total_buy_quantity:
            data = pe_position["pTrdSymbol"]
            # print(keydata,signal,human_readable_quantity(total_buy_quantity),human_readable_quantity(total_sell_quantity))
            print(keydata,signal,data)
        #     # position = ce_position["Position"]
        #     # if position != 1:
        #     #     ce_position["Position"] = 1
        #     #     r.set(f"place_order_{keydata}",json.dumps(json_data))
        #     print(data)
            # position = pe_position["Position"]
            # if position != 1:
            #     pe_position["Position"] = 1
            #     r.set(f"place_order_{keydata}",json.dumps(json_data))
             # data = r.get(f"fo_{keydata}")
        # r.set(f"place_order_{keydata}",data)

if __name__ == "__main__":
    P1 = Process(target=update_indicators_and_signal)
    
    P1.start()
    
    P1.join()