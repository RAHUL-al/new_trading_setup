import pandas as pd
import redis
import threading
import ujson as json
from multiprocessing import Process
from neo_api_client import NeoAPI
import datetime
from kotak_login import get_kotak_client
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from dateutil.relativedelta import relativedelta
from symbol_ltp import TestCases
from concurrent.futures import ThreadPoolExecutor
import time

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

login_handle = f"session_{datetime.datetime.now().strftime('%d%m%Y')}"

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


def create_kotak_df():
    url = client.scrip_master(exchange_segment="nse_fo")
    df = pd.read_csv(url,low_memory=False)
    return df


def get_weekly_df(index):
    df = create_kotak_df()
    df = df[df['pInstType'] == 'OPTIDX']
    df = df[df['pSymbolName'] == index]
    df["lExpiryDate "] = df["lExpiryDate "].apply(lambda x: datetime.datetime.fromtimestamp(x).date() + relativedelta(years=10))
    exp = df["lExpiryDate "].to_list()
    td = datetime.date.today()
    cur_exp = min(exp, key=lambda x: (x - td))
    df = df[df['lExpiryDate '] == cur_exp]
    df['dStrikePrice;'] = df["dStrikePrice;"].apply(lambda x: int(str(x)[:5]))
    df = df[['pTrdSymbol', 'pSymbol', 'pOptionType', 'dStrikePrice;', 'lExpiryDate ']]
    df = df.reset_index(drop=True)
    return df


def select_trading_symbol(index, ltp, CE_or_PE):
    df = get_weekly_df(index)
    minus_part = ltp % 50
    ltp = ltp - minus_part
    
    data = df[(df["dStrikePrice;"] == ltp) & (df["pOptionType"] == f"{CE_or_PE}")]
    if data.empty:
        return None, None
    
    pTrdSymbol = data["pTrdSymbol"].values[0]
    pSymbol = data["pSymbol"].values[0]

    suite = TestCases()
    ltp_data = suite.test_ltpdata(pSymbol)
    print(ltp_data)
    if ltp_data["data"]["ltp"] is None:
        return None, None
    elif ltp_data["data"]["ltp"] < 150:
        return pTrdSymbol, pSymbol
    elif CE_or_PE == "CE":
        return select_trading_symbol("NIFTY", ltp + 50, CE_or_PE)

    elif CE_or_PE == "PE":
        return select_trading_symbol("NIFTY", ltp - 50, CE_or_PE)


def trading_symbol():
    try:
        data = r.get("99926000")
        output = int(float(data))

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_CE = executor.submit(select_trading_symbol, "NIFTY", output - 100, "CE")
            future_PE = executor.submit(select_trading_symbol, "NIFTY", output + 100, "PE")
            CE = future_CE.result()
            PE = future_PE.result()

        data_dict = {
            "CE":CE,
            "PE":PE
        }
        r.set("Trading_symbol", json.dumps(data_dict, default=str))
        print("coming inside the trading symbol")
    except Exception as e:
        print(f"Error in trading symbol function is : {e}")         

if __name__ == "__main__":
    market_open_time = datetime.datetime.now().replace(hour=9, minute=15, second=5, microsecond=0)
    current_time = datetime.datetime.now()
    market_close_time = datetime.datetime.now().replace(hour=15, minute=25, second=0, microsecond=0)
    while current_time < market_open_time or current_time > market_close_time:
        if market_open_time < current_time < market_close_time:
            P2 = Process(target=trading_symbol)
            P2.start()
            P2.join()
        else:
            print("Market is closed now")

        time.sleep(1)
        current_time = datetime.datetime.now()