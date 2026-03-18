#!/bin/bash
# ─────────────────────────────────────────────────────────
# Stop Trading Processes (Cron: 15:35 IST, Mon-Fri)
# Gracefully kills all trading processes after market close.
# ─────────────────────────────────────────────────────────

PROJECT_DIR="/root/Trading_setup_code"
LOGDIR="$PROJECT_DIR/logs"

echo "========================================" >> "$LOGDIR/cron.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping trading processes..." >> "$LOGDIR/cron.log"
echo "========================================" >> "$LOGDIR/cron.log"

# Send SIGTERM first (graceful shutdown)
pkill -f "angleone_websocket1.py" 2>/dev/null
pkill -f "symbol_found.py" 2>/dev/null
pkill -f "pos_handle_wts.py" 2>/dev/null
pkill -f "catboost_live_engine" 2>/dev/null

sleep 3

# Force kill any survivors
pkill -9 -f "angleone_websocket1.py" 2>/dev/null
pkill -9 -f "symbol_found.py" 2>/dev/null
pkill -9 -f "pos_handle_wts.py" 2>/dev/null
pkill -9 -f "catboost_live_engine" 2>/dev/null

# Clean up PID files
rm -f /tmp/trading_websocket.pid
rm -f /tmp/trading_symbol.pid
rm -f /tmp/trading_pos.pid

echo "[$(date '+%H:%M:%S')] ✅ All trading processes stopped." >> "$LOGDIR/cron.log"
echo "" >> "$LOGDIR/cron.log"
