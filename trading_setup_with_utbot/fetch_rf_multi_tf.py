"""
fetch_rf_multi_tf.py — Fetch NIFTY 1-min, 3-min, and 5-min historical data from AngelOne SmartAPI.

Usage:
    python fetch_rf_multi_tf.py                          # Fetches all 3 timeframes
    python fetch_rf_multi_tf.py --interval ONE_MINUTE    # Only 1-min
    python fetch_rf_multi_tf.py --interval THREE_MINUTE  # Only 3-min
    python fetch_rf_multi_tf.py --interval FIVE_MINUTE   # Only 5-min

Output files (inside new_trading_setup/):
    nifty_1min_data.csv
    nifty_3min_data.csv
    nifty_5min_data.csv
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

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Interval configs: chunk_days, from_date, output_file, api_interval
INTERVAL_CONFIG = {
    "ONE_MINUTE": {
        "chunk_days": 5,
        "from_date": "2024-04-01",
        "output_file": os.path.join(SCRIPT_DIR, "nifty_1min_data.csv"),
        "api_interval": "ONE_MINUTE",
    },
    "THREE_MINUTE": {
        "chunk_days": 15,
        "from_date": "2024-04-01",
        "output_file": os.path.join(SCRIPT_DIR, "nifty_3min_data.csv"),
        "api_interval": "THREE_MINUTE",
    },
    "FIVE_MINUTE": {
        "chunk_days": 25,
        "from_date": "2024-04-01",
        "output_file": os.path.join(SCRIPT_DIR, "nifty_5min_data.csv"),
        "api_interval": "FIVE_MINUTE",
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
    feed_token = smart_api.getfeedToken()

    logger.info(f"AngelOne API connected. Client: {CLIENT_ID}")
    return smart_api


def fetch_candles(smart_api, from_date, to_date, interval, max_retries=3):
    """Fetch candles for NIFTY between two dates. Retries on rate limit."""
    for attempt in range(max_retries):
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
                if "TooManyRequests" in str(msg):
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                logger.warning(f"No data for {from_date} to {to_date}: {msg}")
                return pd.DataFrame()

        except Exception as e:
            if "TooManyRequests" in str(e):
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            logger.error(f"API failed for {from_date} to {to_date}: {e}")
            return pd.DataFrame()

    logger.error(f"Failed after {max_retries} retries: {from_date} to {to_date}")
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

        # Rate limit — 2 seconds between API calls
        if i < len(chunks) - 1:
            time.sleep(2)

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
    parser = argparse.ArgumentParser(description="Fetch NIFTY multi-timeframe data for RF strategy")
    parser.add_argument("--interval", default="ALL",
                        choices=["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "ALL"],
                        help="Which interval to fetch (default: ALL)")
    args = parser.parse_args()

    # Connect once
    smart_api = connect_api()

    if args.interval == "ALL":
        for interval in INTERVAL_CONFIG:
            fetch_interval(smart_api, interval)
    else:
        fetch_interval(smart_api, args.interval)

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
