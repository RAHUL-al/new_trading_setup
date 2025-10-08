import redis
import pandas as pd
from multiprocessing import Process

r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

def update_future_and_options():
    for key in r.scan_iter("trade_history*"):
        r.delete(key)
        print(f"Deleted: {key}")
    
# df = pd.read_csv("stocks_csv_20250610.csv")
# symbol_list = df["pSymbolName"].astype(str).tolist()
# print(symbol_list)
# "HISTORY:APOLLOHOSP:2025-06-10"

if __name__ == "__main__":
    P2 = Process(target=update_future_and_options)
    P2.start()
