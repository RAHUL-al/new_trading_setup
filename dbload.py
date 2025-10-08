from datetime import datetime, timezone
import json
import pandas as pd
import pytz
import requests
import sqlalchemy as sa
from sqlalchemy import create_engine ,MetaData, Table, update,text,inspect,insert
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker 
from random import randint, randrange
import asyncio
import os


username = 'trading_user'
password = 'Rahul@7355'
host = 'localhost'
port = 3306

from urllib.parse import quote_plus
password = quote_plus(password)



def createdb(dbname):
    
    # conn_string = 'mysql+pymysql://avinash:soft22len@localhost:3306/%s?charset=utf8mb4' %(dbname)
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    if not database_exists(engine.url):
        create_database(engine.url)
        print("Created Database for ", str(dbname))
        engine.dispose()
        return True
    else:
        print("Database exists")
        engine.dispose()
        return False
    
    
    
    
def insertnewrow(df, tablename):
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    meta = MetaData()
    conn = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    # Reflect the table from the database
    meta.reflect(bind=engine, only=[tablename])
    table = Table(tablename, meta, autoload_with=engine)

    # Convert DataFrame to list of dictionaries
    datadict = df.to_dict(orient='records')

    try:
        # Insert new rows
        insert_stmt = insert(table).values(datadict)
        session.execute(insert_stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error inserting rows: {str(e)}")
        return False
    finally:
        session.close()
        conn.close()
        engine.dispose()
    
    return True


    

def read_table(tablename):
    db = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    Session = sessionmaker(db)
    mysession = Session()
    meta = MetaData() 
    MetaData.reflect(meta,bind=db,extend_existing=True,only=[tablename]) 
    table = sa.Table(tablename,meta)
    # results = accounts.
    results = mysession.query(table).all()
    df = pd.DataFrame(results)
    mysession.close()

    return df



def append_table(tablename, df):
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    conn = engine.connect()
    Session = sessionmaker(engine)
    
    df = pd.DataFrame(df)
    df.to_sql(tablename, con=conn, if_exists='append', index=False)
    
    conn.close()
    engine.dispose()
    return "Success Append"



def addbulkrow(df,tablename):
    
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    tablex = sa.inspect(engine).has_table("%s"%(tablename))
    conn = engine.connect()
    Session = sessionmaker(engine)
    session = Session()
    meta = MetaData()
    MetaData.reflect(meta,bind=engine,extend_existing=True,only=[tablename])

    df = pd.DataFrame(df)
    datadict = df.to_dict(orient='records')
    datadict = datadict
    
    # print(datadict)
    table = sa.Table(tablename,meta)
    insertrow = sa.insert(table)
    insertrow = insertrow.values(datadict)

    results = session.execute(insertrow)
    session.commit()
    session.close()
    conn.close()
    engine.dispose()
    return True


def updatedbtable(df, tablename):
    db = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    Session = sessionmaker(db)
    mysession = Session()
    meta = MetaData()
    MetaData.reflect(meta, bind=db, extend_existing=True, only=[tablename])
    conn = db.connect()
    table = sa.Table(tablename, meta)

    if df.empty:
        print("The DataFrame is empty. No updates will be made.")
        mysession.close()
        conn.close()
        db.dispose()
        return False

    if len(df) != 1:
        print("The DataFrame contains multiple rows. Only single row updates are supported.")
        mysession.close()
        conn.close()
        db.dispose()
        return False

    datadict = df.to_dict(orient='records')[0]
    id = df['PerformanceId'].values[0]

    updaterow = sa.update(table).where(table.c.PerformanceId == id)
    updaterow = updaterow.values(datadict)
    results = mysession.execute(updaterow)
    mysession.commit()
    mysession.close()
    conn.close()
    db.dispose()
    return True



def replace_table(tablename,df):

    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    conn = engine.connect()
    Session = sessionmaker(engine)
    tablename = str(tablename) 

    print("Creating new table")
    df = pd.DataFrame(df)
    df.to_sql(tablename,con=conn,if_exists='replace',index=False)
    
    conn.close()
    engine.dispose()
    return "Success New"


def createtable(tablename,df):

    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    conn = engine.connect()
    Session = sessionmaker(engine)
    tablename = str(tablename) 
    exist=sa.inspect(engine).has_table("%s"%(tablename))

    # print("Creating new table")
    df = pd.DataFrame(df)
    df.to_sql(tablename,engine,if_exists='replace',index=False)
    conn.close()
    engine.dispose()
    return "Success New"



def existtable(tablename, schema=None):
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    tablex = sa.inspect(engine).has_table(tablename, schema=schema)
    engine.dispose()
    return tablex


def readdb(tablename):
    db = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/stocks_data')
    Session = sessionmaker(db)
    mysession = Session()
    meta = MetaData()
    MetaData.reflect(meta,bind=db,extend_existing=True,only=[tablename]) 
    table = sa.Table(tablename,meta)
    results = mysession.query(table).all()
    df = pd.DataFrame(results)
    mysession.close()
    db.dispose()
    return df