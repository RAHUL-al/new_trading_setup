"""
fetch_rf_multi_tf.py — Fetch NIFTY 2-min data from AngelOne SmartAPI.
                        Fetches 1-min data and resamples to 2-min (AngelOne doesn't have native 2-min).

Usage:
    python fetch_rf_multi_tf.py

Output:
    nifty_2min_data.csv
"""

from SmartApi import SmartConnect
import pyotp
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from logzero import logger

# ─────────── Config ───────────
TOTP_TOKEN = os.environ.get("ANGELONE_TOTP_SECRET", "33OUTDUE57WS3TUPHPLFUCGHFM")
API_KEY = os.environ.get("ANGELONE_API_KEY", "eGoFh2vK")
CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "R865920")
PWD = os.environ.get("ANGELONE_PASSWORD", "7355")

NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "nifty_2min_data.csv")
FROM_DATE = "2024-04-01"
CHUNK_DAYS = 5


def connect_api():
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smart_api = SmartConnect(API_KEY)
    data = smart_api.generateSession(CLIENT_ID, PWD, totp)
    if not data or not data.get("data"):
        raise Exception(f"Login failed: {data}")
    logger.info(f"AngelOne API connected. Client: {CLIENT_ID}")
    return smart_api


def fetch_candles(smart_api, from_date, to_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            params = {
                "exchange": NIFTY_EXCHANGE,
                "symboltoken": NIFTY_TOKEN,
                "interval": "ONE_MINUTE",
                "fromdate": f"{from_date} 09:15",
                "todate": f"{to_date} 15:30",
            }
            response = smart_api.getCandleData(params)
            if response and response.get("data"):
                columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
                return pd.DataFrame(response["data"], columns=columns)
            else:
                msg = response.get("message", "No data") if response else "No response"
                if "TooManyRequests" in str(msg):
                    time.sleep(5 * (attempt + 1))
                    continue
                return pd.DataFrame()
        except Exception as e:
            if "TooManyRequests" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            logger.error(f"API error {from_date}-{to_date}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    smart_api = connect_api()

    start = datetime.strptime(FROM_DATE, "%Y-%m-%d")
    end = datetime.now()

    logger.info(f"Fetching 1-min data: {FROM_DATE} → {end.strftime('%Y-%m-%d')}")

    chunks = []
    curr = start
    while curr < end:
        chunk_end = min(curr + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((curr.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        curr = chunk_end + timedelta(days=1)

    all_data = []
    for i, (f, t) in enumerate(chunks):
        logger.info(f"[{i+1}/{len(chunks)}] {f} → {t}")
        df = fetch_candles(smart_api, f, t)
        if not df.empty:
            all_data.append(df)
        if i < len(chunks) - 1:
            time.sleep(2)

    if not all_data:
        logger.error("No data fetched!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined["Time"] = pd.to_datetime(combined["Time"])
    combined = combined.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Market hours
    mkt_open = datetime.strptime("09:15", "%H:%M").time()
    mkt_close = datetime.strptime("15:30", "%H:%M").time()
    combined["_t"] = combined["Time"].dt.time
    combined = combined[(combined["_t"] >= mkt_open) & (combined["_t"] <= mkt_close)].drop(columns=["_t"])

    # Resample to 2-min
    logger.info(f"Resampling {len(combined)} 1-min candles → 2-min...")
    combined = combined.set_index('Time')
    resampled = combined.resample('2min', label='left', closed='left').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
    }).dropna().reset_index()

    resampled["_t"] = resampled["Time"].dt.time
    resampled = resampled[(resampled["_t"] >= mkt_open) & (resampled["_t"] <= mkt_close)].drop(columns=["_t"])

    resampled.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"✅ Done: {len(resampled)} candles | {resampled['Time'].dt.date.nunique()} days → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
