#!/bin/bash
# ─────────────────────────────────────────────────────────
# Pre-Market Setup (Cron: 08:55 IST, Mon-Fri)
# Runs BEFORE market opens. Sequential execution order:
#   1. stocks_name.py          — Download F&O stocks list from NSE
#   2. create_angleone_csv.py  — Download AngelOne scrip master
#   3. fetch_nifty_incremental — Sync missing NIFTY 1-min candle data
# ─────────────────────────────────────────────────────────

PROJECT_DIR="/root/Trading_setup_code"
LOGFILE="$PROJECT_DIR/logs/pre_market_$(date +%Y%m%d).log"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
mkdir -p logs

echo "========================================" >> "$LOGFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pre-market setup starting..." >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"

# Step 1: Download F&O stocks list from NSE (stocks_name.csv)
echo "[$(date '+%H:%M:%S')] [1/3] Running stocks_name.py..." >> "$LOGFILE"
python3 stocks_name.py >> "$LOGFILE" 2>&1
if [ $? -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✅ stocks_name.csv downloaded." >> "$LOGFILE"
else
    echo "[$(date '+%H:%M:%S')] ❌ stocks_name.py failed!" >> "$LOGFILE"
fi

# Step 2: Download AngelOne scrip master (data_YYYYMMDD.csv)
echo "[$(date '+%H:%M:%S')] [2/3] Running create_angleone_csv.py..." >> "$LOGFILE"
python3 create_angleone_csv.py >> "$LOGFILE" 2>&1
if [ $? -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✅ Scrip master downloaded." >> "$LOGFILE"
else
    echo "[$(date '+%H:%M:%S')] ❌ create_angleone_csv.py failed!" >> "$LOGFILE"
fi

# Step 3: Sync NIFTY 1-min historical data (nifty_1min_data.csv)
echo "[$(date '+%H:%M:%S')] [3/3] Running fetch_nifty_incremental.py..." >> "$LOGFILE"
python3 fetch_nifty_incremental.py --interval ONE_MINUTE >> "$LOGFILE" 2>&1
if [ $? -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✅ NIFTY data synced." >> "$LOGFILE"
else
    echo "[$(date '+%H:%M:%S')] ❌ fetch_nifty_incremental.py failed!" >> "$LOGFILE"
fi

echo "[$(date '+%H:%M:%S')] Pre-market setup complete." >> "$LOGFILE"
echo "" >> "$LOGFILE"
