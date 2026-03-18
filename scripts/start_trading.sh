#!/bin/bash
# ─────────────────────────────────────────────────────────
# Start Trading Processes (Cron: 09:10 IST, Mon-Fri)
#
# Execution order:
#   1. symbol_found.py         — Select CE/PE option contracts (background)
#   2. angleone_websocket1.py  — WebSocket feeder + CatBoost ML engine (background)
#      ↳ internally spawns: catboost_live_engine.py (as subprocess)
#      ↳ internally spawns: market_close_cleanup (as subprocess)
#   3. pos_handle_wts.py       — Paper trading bot / position handler (background)
#
# NOTE: catboost_live_engine.py does NOT need to be run separately.
#       angleone_websocket1.py already launches it via multiprocessing.
# ─────────────────────────────────────────────────────────

PROJECT_DIR="/root/Trading_setup_code"
LOGDIR="$PROJECT_DIR/logs"
DATE=$(date +%Y%m%d)

cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
mkdir -p "$LOGDIR"

echo "========================================" >> "$LOGDIR/cron.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting trading processes..." >> "$LOGDIR/cron.log"
echo "========================================" >> "$LOGDIR/cron.log"

# Kill any leftover processes from previous day
pkill -f "angleone_websocket1.py" 2>/dev/null
pkill -f "symbol_found.py" 2>/dev/null
pkill -f "pos_handle_wts.py" 2>/dev/null
pkill -f "catboost_live_engine" 2>/dev/null
sleep 2

# Process 1: Option contract selector (finds CE/PE in ₹200-250 range, stores in Redis)
# Starts first so Trading_symbol is available in Redis before trading bot needs it
nohup python3 symbol_found.py >> "$LOGDIR/symbol_found_${DATE}.log" 2>&1 &
SYM_PID=$!
echo "$SYM_PID" > /tmp/trading_symbol.pid
echo "[$(date '+%H:%M:%S')] [1/3] Symbol finder started (PID: $SYM_PID)" >> "$LOGDIR/cron.log"

sleep 3  # Give symbol_found time to populate Redis with CE/PE tokens

# Process 2: WebSocket feeder + CatBoost engine + market close cleanup
# This is the MAIN process — it internally spawns:
#   P0: run_websocket()        — tick data → 1-min candles → Redis
#   P1: run_catboost_engine()  — ML predictions → signals → Redis
#   P2: market_close_cleanup() — cleanup at 3:30 PM
nohup python3 angleone_websocket1.py >> "$LOGDIR/websocket_${DATE}.log" 2>&1 &
WS_PID=$!
echo "$WS_PID" > /tmp/trading_websocket.pid
echo "[$(date '+%H:%M:%S')] [2/3] WebSocket + CatBoost started (PID: $WS_PID)" >> "$LOGDIR/cron.log"

sleep 5  # Let websocket connect and start building candles before pos handler

# Process 3: Paper trading bot (listens to Redis signals, manages positions)
nohup python3 pos_handle_wts.py >> "$LOGDIR/pos_handler_${DATE}.log" 2>&1 &
POS_PID=$!
echo "$POS_PID" > /tmp/trading_pos.pid
echo "[$(date '+%H:%M:%S')] [3/3] Position handler started (PID: $POS_PID)" >> "$LOGDIR/cron.log"

echo "" >> "$LOGDIR/cron.log"
echo "  All trading processes running:" >> "$LOGDIR/cron.log"
echo "    Symbol Finder              : PID $SYM_PID  (symbol_found.py)" >> "$LOGDIR/cron.log"
echo "    WebSocket + CatBoost       : PID $WS_PID  (angleone_websocket1.py)" >> "$LOGDIR/cron.log"
echo "    Position Handler (Paper)   : PID $POS_PID  (pos_handle_wts.py)" >> "$LOGDIR/cron.log"
echo "" >> "$LOGDIR/cron.log"
