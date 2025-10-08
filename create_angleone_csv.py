import pandas as pd
import numpy as np
import datetime as dt
import os
from datetime import datetime,timedelta,date
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text
from neo_api_client import NeoAPI
import threading
from kotak_login import get_kotak_client
from urllib.parse import quote_plus

response_event = threading.Event()
response_data = None

client = get_kotak_client()

password = 'Rahul@7355'
host = 'localhost'
port = 3306



access_token = None
sid = None
secretkey = None
neo_fin_key = None

password = quote_plus(password)

# engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
engine = create_engine(f"mysql+pymysql://trading_user:{password}@localhost/stocks_data")

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




consumer_key = 'cR7ZW_66Z5zmvEj_35GDBKpuxYga'
secretkey = 'sbtJCz6vbHrSURnjXBGSIcSIMmka'


username = 'client49515'
password = '0FT01pJT'

user_id = '4f24f544-7268-4e6d-86df-f89fb7436628'
mobileNumber = '+919815767797'
login_password = 'Rahul@7355'
mpin = '735596'


consumer_secret = secretkey
access_token = access_token
neo_fin_key = "X6Nk8cQhUgGmJ2vBdWw4sfzrz4L5En"
client = NeoAPI(consumer_key=consumer_key, consumer_secret=consumer_secret, environment="prod", access_token=access_token, neo_fin_key=neo_fin_key)
client.login(mobilenumber=mobileNumber, password=login_password)
client.session_2fa(OTP=mpin)



# def create_kotak_dataframe(file_path):
#     url = client.scrip_master(exchange_segment="nse_fo")
#     df = pd.read_csv(url, low_memory=False) 
#     df.to_csv(file_path, index=False)  

def create_kotak_dataframe(file_path_fo):
    
    file_path_all_stocks = f"stocks_{file_path_fo}"
    print("inside the create kotak dataframe")
    print(file_path_all_stocks)
    
    url_fo = client.scrip_master(exchange_segment="nse_fo")
    print(url_fo)
    df_fo = pd.read_csv(url_fo, low_memory=False)
    df_fo.to_csv(file_path_fo, index=False)
    print(df_fo.head())

    url_nse_cm = client.scrip_master(exchange_segment="nse_cm")
    url_bse_cm = client.scrip_master(exchange_segment="bse_cm")
    print(url_bse_cm)
    print(url_nse_cm)

    # df_nse = pd.read_csv(url_nse_cm, low_memory=False)
    # df_bse = pd.read_csv(url_bse_cm, low_memory=False)


    # df_all_stocks = pd.concat([df_nse, df_bse], ignore_index=True)
    # df_all_stocks.to_csv(file_path_all_stocks, index=False)
    




def delete_old_files(today_file_name):
    print("coming inside te delete old files")
    today_date_str = datetime.now().strftime("%Y%m%d")
    folder_path = "." 
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and file_name != today_file_name and file_name != "stocks_name.csv" and file_name != f"stocks_data_{today_date_str}.csv" and file_name != f"final_stock_csv_{today_date_str}.csv" and file_name != f"stocks_csv_{today_date_str}.csv":
            try:
                os.remove(os.path.join(folder_path, file_name))
            except Exception as e:
                print(f"Error deleting file {file_name}: {e}")
                
                

def create_kotak_df():
    today_date_str = datetime.now().strftime("%Y%m%d")
    file_path = f'data_{today_date_str}.csv'

    delete_old_files(f"data_{today_date_str}.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        create_kotak_dataframe(file_path)   

        df = pd.read_csv(file_path, low_memory=False) 

    return df
create_kotak_df()
