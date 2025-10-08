from fastapi import FastAPI
import pandas as pd
import time
import threading
import os
import redis

app = FastAPI()

CSV_FILE = "main_csv.csv"
last_row_cache = None
r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)
def get_last_row(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        return None
    return df.tail(1).to_dict(orient='records')[0]

def monitor_csv_file():
    global last_row_cache

    while True:
        try:
            current_last_row = get_last_row(CSV_FILE)

            if current_last_row and current_last_row != last_row_cache:
                if current_last_row["buy"] == True:
                    r.set("buy_signal", "true")
                    r.delete("sell_signal")
                elif current_last_row["sell"] == True:
                    r.set("sell_signal", "true")
                    r.delete("buy_signal")

                print("🔔 New row detected:", current_last_row)
                last_row_cache = current_last_row


        except Exception as e:
            print(f"Error in CSV monitor: {e}")
            time.sleep(5)

@app.on_event("startup")
def start_background_task():
    thread = threading.Thread(target=monitor_csv_file)
    thread.daemon = True
    thread.start()

@app.get("/")
def root():
    return {"message": "Webhook listener running!"}