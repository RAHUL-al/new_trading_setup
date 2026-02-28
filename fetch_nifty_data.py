"""
fetch_nifty_data.py — Fetch NIFTY 50 historical 1-minute candle data from AngelOne SmartAPI.

Usage:
    python fetch_nifty_data.py

Output: nifty_1min_data.csv with columns: Time, Open, High, Low, Close, Volume

Note: AngelOne limits ~30 days per request for 1-minute data.
      This script automatically paginates to fetch full date range.
"""

from SmartApi import SmartConnect
import pyotp
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from logzero import logger


# ─────────── Config ───────────
TOTP_TOKEN = os.environ.get("ANGELONE_TOTP_SECRET", "OIN6QBZAYV4I26Q55OYASIEQVY")
API_KEY = os.environ.get("ANGELONE_API_KEY", "SsUDlNA9")
CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "A1079871")
PWD = os.environ.get("ANGELONE_PASSWORD", "0465")

NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"
INTERVAL = "ONE_MINUTE"

# How far back to fetch (AngelOne allows ~2000 candles per request for 1-min)
# 1 trading day = ~375 candles (9:15 to 15:30 = 375 minutes)
# Safe chunk = 5 trading days (~1875 candles per request)
CHUNK_DAYS = 5

# Fetch data from this date
FROM_DATE = os.environ.get("FETCH_FROM_DATE", "2025-01-01")

OUTPUT_FILE = "nifty_1min_data.csv"


def connect_api():
    """Login to AngelOne SmartAPI."""
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smart_api = SmartConnect(API_KEY)
    data = smart_api.generateSession(CLIENT_ID, PWD, totp)

    if not data or not data.get("data"):
        raise Exception(f"Login failed: {data}")

    auth_token = data['data']['jwtToken']
    refresh_token = data['data']['refreshToken']
    feed_token = smart_api.getfeedToken()

    logger.info(f"AngelOne API connected. Client: {CLIENT_ID}")
    return smart_api


def fetch_candles(smart_api, from_date, to_date):
    """Fetch 1-minute candles for NIFTY between two dates."""
    try:
        params = {
            "exchange": NIFTY_EXCHANGE,
            "symboltoken": NIFTY_TOKEN,
            "interval": INTERVAL,
            "fromdate": f"{from_date} 09:15",
            "todate": f"{to_date} 15:30",
        }
        response = smart_api.getCandleData(params)

        if response and response.get("data"):
            columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
            df = pd.DataFrame(response["data"], columns=columns)
            return df
        else:
            msg = response.get("message", "No data") if response else "No response"
            logger.warning(f"No data for {from_date} to {to_date}: {msg}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"API failed for {from_date} to {to_date}: {e}")
        return pd.DataFrame()


def get_trading_days(start_date, end_date):
    """Generate date chunks for pagination (skip weekends)."""
    chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return chunks


def main():
    logger.info("=" * 60)
    logger.info("NIFTY 1-MINUTE DATA FETCHER")
    logger.info("=" * 60)

    # Parse date range
    start_date = datetime.strptime(FROM_DATE, "%Y-%m-%d")
    end_date = datetime.now()

    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Interval: {INTERVAL}")
    logger.info(f"Token: {NIFTY_TOKEN} (NIFTY 50)")

    # Connect to API
    smart_api = connect_api()

    # Generate date chunks
    chunks = get_trading_days(start_date, end_date)
    logger.info(f"Total chunks to fetch: {len(chunks)}")

    # Fetch all chunks
    all_data = []
    for i, (from_d, to_d) in enumerate(chunks):
        logger.info(f"[{i+1}/{len(chunks)}] Fetching {from_d} → {to_d}...")
        df = fetch_candles(smart_api, from_d, to_d)

        if not df.empty:
            all_data.append(df)
            logger.info(f"  Got {len(df)} candles")
        else:
            logger.info(f"  No data (weekend/holiday?)")

        # Rate limit: wait between requests
        if i < len(chunks) - 1:
            time.sleep(0.5)

    if not all_data:
        logger.error("No data fetched at all! Check credentials and date range.")
        return

    # Combine and deduplicate
    combined = pd.concat(all_data, ignore_index=True)
    combined["Time"] = pd.to_datetime(combined["Time"])
    combined = combined.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Filter to market hours only (9:15 to 15:30)
    combined["time_only"] = combined["Time"].dt.time
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    combined = combined[
        (combined["time_only"] >= market_open) &
        (combined["time_only"] <= market_close)
    ].drop(columns=["time_only"])

    # Save to CSV
    combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    total_days = combined["Time"].dt.date.nunique()
    logger.info("=" * 60)
    logger.info(f"FETCH COMPLETE")
    logger.info(f"  Total candles: {len(combined)}")
    logger.info(f"  Trading days:  {total_days}")
    logger.info(f"  Date range:    {combined['Time'].iloc[0]} → {combined['Time'].iloc[-1]}")
    logger.info(f"  Saved to:      {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
