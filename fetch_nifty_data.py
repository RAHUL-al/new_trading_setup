"""
fetch_nifty_data.py — Fetch NIFTY 50 historical candle data from AngelOne SmartAPI.

Usage:
    python fetch_nifty_data.py                          # Fetches all timeframes
    python fetch_nifty_data.py --interval ONE_MINUTE    # Only 1-min
    python fetch_nifty_data.py --interval THREE_MINUTE  # Only 3-min
    python fetch_nifty_data.py --interval FIVE_MINUTE   # Only 5-min

Output files:
    nifty_1min_data.csv   (from 2025-01-01)
    nifty_3min_data.csv   (from 2024-01-01 — more data per request)
    nifty_5min_data.csv   (from 2024-01-01 — even more data per request)

Note: AngelOne limits ~2000 candles per request.
      Larger intervals = more days per chunk = can go further back.
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

# Each interval config: chunk_days, default_from_date, output_file
# Note: AngelOne API supports: ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, TEN_MINUTE,
#       FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY
# TWO_MINUTE is NOT natively supported — we fetch 1-min and resample.
INTERVAL_CONFIG = {
    "ONE_MINUTE": {
        "chunk_days": 5,
        "from_date": "2024-01-01",
        "output_file": "nifty_1min_data.csv",
        "api_interval": "ONE_MINUTE",
        "resample_to": None,
    },
    "TWO_MINUTE": {
        "chunk_days": 5,          # Fetch 1-min, then resample to 2-min
        "from_date": "2024-01-01",
        "output_file": "nifty_2min_data.csv",
        "api_interval": "ONE_MINUTE",  # Fetch 1-min from API
        "resample_to": 2,              # Then resample to 2-min
    },
    "THREE_MINUTE": {
        "chunk_days": 15,
        "from_date": "2024-01-01",
        "output_file": "nifty_3min_data.csv",
        "api_interval": "THREE_MINUTE",
        "resample_to": None,
    },
    "FIVE_MINUTE": {
        "chunk_days": 25,
        "from_date": "2024-01-01",
        "output_file": "nifty_5min_data.csv",
        "api_interval": "FIVE_MINUTE",
        "resample_to": None,
    },
}


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


def fetch_candles(smart_api, from_date, to_date, interval):
    """Fetch candles for NIFTY between two dates."""
    try:
        params = {
            "exchange": NIFTY_EXCHANGE,
            "symboltoken": NIFTY_TOKEN,
            "interval": interval,
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


def get_date_chunks(start_date, end_date, chunk_days):
    """Generate date chunks for pagination."""
    chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return chunks


def fetch_interval(smart_api, interval):
    """Fetch all data for a specific interval."""
    config = INTERVAL_CONFIG[interval]
    from_override = os.environ.get("FETCH_FROM_DATE")
    from_date = from_override if from_override else config["from_date"]
    output_file = config["output_file"]
    chunk_days = config["chunk_days"]

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.now()

    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING {interval} DATA")
    logger.info(f"  Date range: {from_date} → {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Chunk size: {chunk_days} days")
    logger.info(f"  Output: {output_file}")
    logger.info(f"{'='*60}")

    # Generate chunks
    chunks = get_date_chunks(start_date, end_date, chunk_days)
    logger.info(f"Total chunks: {len(chunks)}")

    # Fetch all chunks
    all_data = []
    for i, (from_d, to_d) in enumerate(chunks):
        logger.info(f"[{i+1}/{len(chunks)}] Fetching {from_d} → {to_d}...")
        df = fetch_candles(smart_api, from_d, to_d, config["api_interval"])

        if not df.empty:
            all_data.append(df)
            logger.info(f"  Got {len(df)} candles")
        else:
            logger.info(f"  No data (weekend/holiday?)")

        # Rate limit
        if i < len(chunks) - 1:
            time.sleep(0.5)

    if not all_data:
        logger.error(f"No data fetched for {interval}!")
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

    # Resample if needed (e.g., TWO_MINUTE = fetch 1-min then resample)
    resample_to = config.get("resample_to")
    if resample_to:
        logger.info(f"Resampling to {resample_to}-minute candles...")
        combined = combined.set_index('Time')
        combined = combined.resample(f'{resample_to}min', label='left', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna().reset_index()
        # Re-filter market hours after resampling
        combined["time_only"] = combined["Time"].dt.time
        combined = combined[
            (combined["time_only"] >= market_open) &
            (combined["time_only"] <= market_close)
        ].drop(columns=["time_only"])
        logger.info(f"  After resampling: {len(combined)} candles")

    # Save to CSV
    combined.to_csv(output_file, index=False)

    # Summary
    total_days = combined["Time"].dt.date.nunique()
    logger.info(f"\n✅ {interval} FETCH COMPLETE")
    logger.info(f"  Total candles: {len(combined)}")
    logger.info(f"  Trading days:  {total_days}")
    logger.info(f"  Date range:    {combined['Time'].iloc[0]} → {combined['Time'].iloc[-1]}")
    logger.info(f"  Saved to:      {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch NIFTY historical candle data")
    parser.add_argument("--interval", default="TWO_MINUTE",
                        choices=["ONE_MINUTE", "TWO_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "ALL"],
                        help="Which interval to fetch (default: TWO_MINUTE)")
    args = parser.parse_args()

    # Connect once
    smart_api = connect_api()

    if args.interval == "ALL":
        for interval in ["ONE_MINUTE", "TWO_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"]:
            fetch_interval(smart_api, interval)
    else:
        fetch_interval(smart_api, args.interval)

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
