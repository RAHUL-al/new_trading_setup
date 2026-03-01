"""
fetch_stock_data.py — Fetch historical candle data for NIFTY 50 stocks from AngelOne

Steps:
  1. Downloads AngelOne scrip master (symbol tokens)
  2. Matches NIFTY 50 stock names to their NSE Cash Market tokens
  3. Fetches 3-min historical data for each stock (6 months)
  4. Saves to stock_data/{SYMBOL}_3min.csv

Usage:
    python fetch_stock_data.py                    # Fetch all NIFTY 50
    python fetch_stock_data.py --stock RELIANCE   # Fetch one stock
    python fetch_stock_data.py --interval 2       # 2-min data
"""

from SmartApi import SmartConnect
import pyotp
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime, timedelta
from logzero import logger


# ─────────── Config ───────────
TOTP_TOKEN = os.environ.get("ANGELONE_TOTP_SECRET", "OIN6QBZAYV4I26Q55OYASIEQVY")
API_KEY = os.environ.get("ANGELONE_API_KEY", "SsUDlNA9")
CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "A1079871")
PWD = os.environ.get("ANGELONE_PASSWORD", "0465")

DATA_DIR = "stock_data"
MONTHS_BACK = 6
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

# NIFTY 50 stock symbols (NSE trading symbols)
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "TITAN", "SUNPHARMA", "NTPC", "TATAMOTORS",
    "WIPRO", "ULTRACEMCO", "POWERGRID", "NESTLEIND", "TECHM",
    "ONGC", "INDUSINDBK", "JSWSTEEL", "TATASTEEL", "ADANIPORTS",
    "M&M", "BAJAJFINSV", "COALINDIA", "GRASIM", "BPCL",
    "BRITANNIA", "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO",
    "APOLLOHOSP", "DIVISLAB", "BAJAJ-AUTO", "TRENT", "SBILIFE",
    "HDFCLIFE", "BEL", "SHRIRAMFIN", "TATACONSUM", "ADANIENT",
]


def connect_api():
    """Login to AngelOne SmartAPI."""
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smart_api = SmartConnect(API_KEY)
    data = smart_api.generateSession(CLIENT_ID, PWD, totp)
    if not data or not data.get("data"):
        raise Exception(f"Login failed: {data}")
    logger.info(f"AngelOne API connected. Client: {CLIENT_ID}")
    return smart_api


def get_symbol_tokens():
    """Download scrip master and extract NSE Cash Market symbol tokens."""
    logger.info("Downloading AngelOne scrip master for symbol tokens...")
    response = requests.get(SCRIP_MASTER_URL, timeout=120)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)

    # Filter NSE Cash Market (equity segment)
    nse_eq = df[df['exch_seg'] == 'NSE'].copy()
    # Filter for EQ series (regular stocks)
    nse_eq = nse_eq[nse_eq['symbol'].str.endswith('-EQ')].copy()

    # Create mapping: clean symbol → token
    token_map = {}
    for _, row in nse_eq.iterrows():
        symbol = row['symbol'].replace('-EQ', '')
        token_map[symbol] = {
            'token': str(row['token']),
            'name': row.get('name', symbol),
            'symbol_full': row['symbol'],
        }

    logger.info(f"Found {len(token_map)} NSE equity symbols")
    return token_map


def fetch_candles(smart_api, symbol_token, from_date, to_date, interval="THREE_MINUTE"):
    """Fetch candle data for a stock."""
    try:
        params = {
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": f"{from_date} 09:15",
            "todate": f"{to_date} 15:30",
        }
        response = smart_api.getCandleData(params)
        if response and response.get("data"):
            columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
            return pd.DataFrame(response["data"], columns=columns)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"API error: {e}")
        return pd.DataFrame()


def fetch_stock(smart_api, symbol, token, interval="THREE_MINUTE", chunk_days=15, resample_to=None):
    """Fetch all historical data for one stock."""
    start_date = datetime.now() - timedelta(days=MONTHS_BACK * 30)
    end_date = datetime.now()

    all_data = []
    current = start_date
    total_chunks = ((end_date - start_date).days // chunk_days) + 1

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        from_d = current.strftime("%Y-%m-%d")
        to_d = chunk_end.strftime("%Y-%m-%d")

        df = fetch_candles(smart_api, token, from_d, to_d, interval)
        if not df.empty:
            all_data.append(df)

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # Rate limit

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined["Time"] = pd.to_datetime(combined["Time"])
    combined = combined.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Filter market hours
    combined["time_only"] = combined["Time"].dt.time
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    combined = combined[
        (combined["time_only"] >= market_open) & (combined["time_only"] <= market_close)
    ].drop(columns=["time_only"])

    # Resample if needed (e.g., 1-min to 2-min)
    if resample_to:
        combined = combined.set_index('Time')
        combined = combined.resample(f'{resample_to}min', label='left', closed='left').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum',
        }).dropna().reset_index()

    return combined


def main():
    import argparse
    global MONTHS_BACK

    parser = argparse.ArgumentParser(description="Fetch NIFTY 50 stock data from AngelOne")
    parser.add_argument("--stock", default=None, help="Fetch only this stock (e.g., RELIANCE)")
    parser.add_argument("--interval", type=int, default=3, help="Candle interval in minutes (default: 3)")
    parser.add_argument("--months", type=int, default=MONTHS_BACK, help="Months of history (default: 6)")
    args = parser.parse_args()

    MONTHS_BACK = args.months

    # Determine API interval and resampling
    interval_min = args.interval
    if interval_min == 1:
        api_interval = "ONE_MINUTE"
        resample_to = None
        chunk_days = 5
    elif interval_min == 2:
        api_interval = "ONE_MINUTE"
        resample_to = 2
        chunk_days = 5
    elif interval_min == 3:
        api_interval = "THREE_MINUTE"
        resample_to = None
        chunk_days = 15
    elif interval_min == 5:
        api_interval = "FIVE_MINUTE"
        resample_to = None
        chunk_days = 25
    else:
        api_interval = "ONE_MINUTE"
        resample_to = interval_min
        chunk_days = 5

    os.makedirs(DATA_DIR, exist_ok=True)

    # Get symbol tokens
    token_map = get_symbol_tokens()

    # Connect to API
    smart_api = connect_api()

    # Determine stocks to fetch
    stocks = [args.stock] if args.stock else NIFTY_50

    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING {len(stocks)} STOCKS | {interval_min}-min | {MONTHS_BACK} months")
    logger.info(f"{'='*60}")

    success = 0
    failed = []

    for idx, symbol in enumerate(stocks):
        if symbol not in token_map:
            logger.warning(f"[{idx+1}/{len(stocks)}] {symbol}: NOT FOUND in scrip master, skipping")
            failed.append(symbol)
            continue

        token = token_map[symbol]['token']
        output_file = os.path.join(DATA_DIR, f"{symbol}_{interval_min}min.csv")

        # Skip if already fetched recently
        if os.path.exists(output_file):
            mod_time = datetime.fromtimestamp(os.path.getmtime(output_file))
            if (datetime.now() - mod_time).days < 1:
                logger.info(f"[{idx+1}/{len(stocks)}] {symbol}: Already fetched today, skipping")
                success += 1
                continue

        logger.info(f"[{idx+1}/{len(stocks)}] {symbol} (token: {token})...")
        df = fetch_stock(smart_api, symbol, token, api_interval, chunk_days, resample_to)

        if not df.empty:
            df.to_csv(output_file, index=False)
            days = df['Time'].dt.date.nunique()
            logger.info(f"  ✅ {len(df)} candles | {days} days | Saved to {output_file}")
            success += 1
        else:
            logger.warning(f"  ❌ No data for {symbol}")
            failed.append(symbol)

        time.sleep(0.5)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE! {success}/{len(stocks)} stocks fetched")
    if failed:
        logger.info(f"Failed: {', '.join(failed)}")
    logger.info(f"Data saved to: {DATA_DIR}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
