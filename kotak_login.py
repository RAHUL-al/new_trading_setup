# kotak_login.py
from neo_api_client import NeoAPI
import json
import os
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import dbload



class KotakLogin:
    def __init__(self):
        self.consumer_key = 'cR7ZW_66Z5zmvEj_35GDBKpuxYga'
        self.secretkey = 'sbtJCz6vbHrSURnjXBGSIcSIMmka'
        self.mobileNumber = '+919815767797'
        self.login_password = 'Rahul@7355'
        self.mpin = '735596'
        self.access_token = None
        self.sid = None
        self.neo_fin_key = None
    
    def login(self):
        """Logs in and stores JWT token in a file."""
        client = NeoAPI(consumer_key=self.consumer_key, consumer_secret=self.secretkey, environment='prod')
        
        # Login process
        client.login(mobilenumber=self.mobileNumber, password=self.login_password)
        data = client.session_2fa(OTP=self.mpin)
        print("this is session data : ",data)
        
        
        username = 'trading_user'
        password = 'Rahul@7355'
        host = 'localhost'
        port = 3306
        
        password = quote_plus(password)
        
        engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
        connection = engine.connect()
        
        login_handle = f"session_{datetime.now().strftime('%d%m%Y')}"
        
        login_data = {
                    "access_token":client.configuration.bearer_token,
                    "sid":client.configuration.edit_sid,
                    "secretkey":self.secretkey,
                    "neo_fin_key":client.configuration.get_neo_fin_key(),
                }
        
        df = pd.DataFrame([login_data])
        dbload.createtable(login_handle,df)
        
        
        return client
    
    def store_jwt(self, jwt_token):
        """Stores the JWT token in a file."""
        with open(self.jwt_file, 'w') as f:
            json.dump({'jwt': jwt_token}, f)

    def load_jwt(self):
        username = 'trading_user'
        password = 'Rahul@7355'
        host = 'localhost'
        port = 3306
        
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
                
                return access_token,sid,secretkey,neo_fin_key
        else:
            return None
                

    def get_client(self):
        """Returns the NeoAPI client, reusing the JWT token if available."""
        all_token_data = self.load_jwt()
        
        if all_token_data:
            print("already is all token are found.")
            access_token = all_token_data[0]
            sid = all_token_data[1]
            secretkey = all_token_data[2]
            neo_fin_key = all_token_data[3]
            client = NeoAPI(consumer_key=self.consumer_key, consumer_secret=self.secretkey, environment='prod',
                            access_token=access_token)
            return client
        else:
            return self.login()


def get_kotak_client():
    login_manager = KotakLogin()
    return login_manager.get_client()
