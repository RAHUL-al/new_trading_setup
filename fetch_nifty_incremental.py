"""
fetch_nifty_incremental.py — Smart incremental NIFTY data fetcher.

Checks existing CSV files for the last available date and only fetches
missing days instead of re-downloading everything from scratch.

Usage:
    python fetch_nifty_incremental.py                          # Update all timeframes
    python fetch_nifty_incremental.py --interval ONE_MINUTE    # Only 1-min
    python fetch_nifty_incremental.py --interval TWO_MINUTE    # Only 2-min
    python fetch_nifty_incremental.py --full                   # Force full re-fetch

How it works:
    1. Reads existing CSV (e.g., nifty_1min_data.csv)
    2. Finds the last date in the data
    3. Only fetches from (last_date + 1) to today
    4. Appends new data and saves
    5. If CSV doesn't exist, does a full fetch from the configured start date
"""

from SmartApi import SmartConnect
import pyotp
import pandas as pd
import os
import sys
import time
from datetime import datetime, timedelta
from logzero import logger


# ─────────── Config ───────────
TOTP_TOKEN = os.environ.get("ANGELONE_TOTP_SECRET", "33OUTDUE57WS3TUPHPLFUCGHFM")
API_KEY = os.environ.get("ANGELONE_API_KEY", "7355")
CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "R865920")
PWD = os.environ.get("ANGELONE_PASSWORD", "7355")

NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"

# Interval configurations
INTERVAL_CONFIG = {
    "ONE_MINUTE": {
        "chunk_days": 5,
        "from_date": "2019-01-01",
        "output_file": "nifty_1min_data.csv",
        "api_interval": "ONE_MINUTE",
        "resample_to": None,
    },
    "TWO_MINUTE": {
        "chunk_days": 5,
        "from_date": "2019-01-01",
        "output_file": "nifty_2min_data.csv",
        "api_interval": "ONE_MINUTE",   # Fetch 1-min, then resample to 2-min
        "resample_to": 2,
    },
}

MARKET_OPEN = datetime.strptime("09:15", "%H:%M").time()
MARKET_CLOSE = datetime.strptime("15:30", "%H:%M").time()


# ─────────── API ───────────

def connect_api():
    """Login to AngelOne SmartAPI."""
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smart_api = SmartConnect(API_KEY)
    data = smart_api.generateSession(CLIENT_ID, PWD, totp)

    if not data or not data.get("data"):
        raise Exception(f"Login failed: {data}")

    logger.info(f"✅ AngelOne API connected. Client: {CLIENT_ID}")
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


# ─────────── Core: Incremental Fetch ───────────

def get_last_date_in_csv(output_file):
    """Read existing CSV and return the last date available."""
    if not os.path.exists(output_file):
        return None

    try:
        df = pd.read_csv(output_file)
        if df.empty or "Time" not in df.columns:
            return None

        df["Time"] = pd.to_datetime(df["Time"])
        last_date = df["Time"].max().date()
        total_rows = len(df)
        total_days = df["Time"].dt.date.nunique()
        first_date = df["Time"].min().date()

        logger.info(f"  📂 Existing data found: {output_file}")
        logger.info(f"     Rows: {total_rows:,} | Days: {total_days}")
        logger.info(f"     Range: {first_date} → {last_date}")
        return last_date
    except Exception as e:
        logger.warning(f"  ⚠️ Could not read {output_file}: {e}")
        return None


def filter_market_hours(df):
    """Filter to market hours only (9:15 to 15:30)."""
    df["time_only"] = df["Time"].dt.time
    df = df[
        (df["time_only"] >= MARKET_OPEN) &
        (df["time_only"] <= MARKET_CLOSE)
    ].drop(columns=["time_only"])
    return df


def resample_to_nmin(df, n):
    """Resample 1-min candles to N-min candles."""
    df = df.set_index('Time')
    df = df.resample(f'{n}min', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna().reset_index()
    return filter_market_hours(df)


def fetch_interval_incremental(smart_api, interval, force_full=False):
    """Smart incremental fetch — only downloads missing days."""
    config = INTERVAL_CONFIG[interval]
    output_file = config["output_file"]
    chunk_days = config["chunk_days"]

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 {interval} — INCREMENTAL UPDATE")
    logger.info(f"{'='*60}")

    # Step 1: Check existing data
    existing_df = None
    if not force_full:
        last_date = get_last_date_in_csv(output_file)
    else:
        last_date = None
        logger.info("  🔄 Force full re-fetch requested")

    # Step 2: Determine start date
    if last_date:
        # Start from the day after the last date in the CSV
        # (re-fetch last day too in case it was incomplete)
        start_date = datetime.combine(last_date, datetime.min.time())
        logger.info(f"  📅 Fetching from {last_date} (re-fetch last day for completeness)")

        # Load existing data (will merge later)
        try:
            existing_df = pd.read_csv(output_file)
            existing_df["Time"] = pd.to_datetime(existing_df["Time"])
            # Remove last day (will be re-fetched fresh)
            existing_df = existing_df[existing_df["Time"].dt.date < last_date]
            logger.info(f"  📦 Kept {len(existing_df):,} existing rows (before {last_date})")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load existing data: {e}")
            existing_df = None
    else:
        from_date = config["from_date"]
        start_date = datetime.strptime(from_date, "%Y-%m-%d")
        logger.info(f"  📅 No existing data — full fetch from {from_date}")

    end_date = datetime.now()

    # Check if already up to date
    if last_date and last_date >= end_date.date():
        logger.info(f"  ✅ Already up to date! Last date: {last_date}")
        return

    # Step 3: Fetch missing chunks
    chunks = get_date_chunks(start_date, end_date, chunk_days)
    days_to_fetch = (end_date.date() - start_date.date()).days
    logger.info(f"  📥 Days to fetch: ~{days_to_fetch} | API chunks: {len(chunks)}")

    new_data = []
    for i, (from_d, to_d) in enumerate(chunks):
        logger.info(f"  [{i+1}/{len(chunks)}] {from_d} → {to_d}...")
        df = fetch_candles(smart_api, from_d, to_d, config["api_interval"])

        if not df.empty:
            new_data.append(df)
            logger.info(f"    ✓ {len(df)} candles")
        else:
            logger.info(f"    · No data (weekend/holiday?)")

        # Rate limit
        if i < len(chunks) - 1:
            time.sleep(2)

    if not new_data:
        if existing_df is not None and not existing_df.empty:
            logger.info(f"  ℹ️ No new data fetched (probably up to date)")
            return
        else:
            logger.error(f"  ❌ No data fetched at all!")
            return

    # Step 4: Combine new data
    new_combined = pd.concat(new_data, ignore_index=True)
    new_combined["Time"] = pd.to_datetime(new_combined["Time"])
    new_combined = filter_market_hours(new_combined)

    # Step 5: Merge with existing data
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_combined], ignore_index=True)
        logger.info(f"  🔗 Merged: {len(existing_df):,} existing + {len(new_combined):,} new")
    else:
        combined = new_combined

    # Deduplicate and sort
    combined = combined.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Step 6: Resample if needed (e.g., TWO_MINUTE)
    resample_to = config.get("resample_to")
    if resample_to:
        logger.info(f"  🔄 Resampling to {resample_to}-min candles...")
        combined = resample_to_nmin(combined, resample_to)
        logger.info(f"    After resampling: {len(combined):,} candles")

    # Step 7: Save
    combined.to_csv(output_file, index=False)

    # Summary
    total_days = combined["Time"].dt.date.nunique()
    first = combined['Time'].iloc[0]
    last = combined['Time'].iloc[-1]
    logger.info(f"\n  ✅ {interval} UPDATE COMPLETE")
    logger.info(f"     Total candles: {len(combined):,}")
    logger.info(f"     Trading days:  {total_days}")
    logger.info(f"     Range: {first} → {last}")
    logger.info(f"     Saved: {output_file}")


# ─────────── Main ───────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Smart incremental NIFTY data fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_nifty_incremental.py                        # Update all (1-min + 2-min)
  python fetch_nifty_incremental.py --interval ONE_MINUTE  # Only update 1-min
  python fetch_nifty_incremental.py --full                 # Force re-download everything
  python fetch_nifty_incremental.py --status               # Just show what's in the CSVs
        """
    )
    parser.add_argument("--interval", default="ALL",
                        choices=["ONE_MINUTE", "TWO_MINUTE", "ALL"],
                        help="Which interval to fetch (default: ALL)")
    parser.add_argument("--full", action="store_true",
                        help="Force full re-fetch from start date (ignores existing data)")
    parser.add_argument("--status", action="store_true",
                        help="Just show status of existing CSV files, don't fetch")
    args = parser.parse_args()

    # Status check only
    if args.status:
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 DATA STATUS CHECK")
        logger.info(f"{'='*60}")
        for interval, config in INTERVAL_CONFIG.items():
            output_file = config["output_file"]
            logger.info(f"\n  📊 {interval} → {output_file}")
            last = get_last_date_in_csv(output_file)
            if last:
                gap = (datetime.now().date() - last).days
                if gap <= 1:
                    logger.info(f"     ✅ Up to date!")
                else:
                    logger.info(f"     ⚠️ Missing ~{gap} days (last: {last})")
            else:
                logger.info(f"     ❌ No data file found")
        return

    # Connect and fetch
    smart_api = connect_api()

    intervals = INTERVAL_CONFIG.keys() if args.interval == "ALL" else [args.interval]
    for interval in intervals:
        fetch_interval_incremental(smart_api, interval, force_full=args.full)

    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 ALL DONE!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
