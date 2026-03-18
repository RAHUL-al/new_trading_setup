#!/bin/bash
# ─────────────────────────────────────────────────────────
# Check status of all trading processes
# Usage: bash scripts/status.sh
# ─────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════╗"
echo "║        Paper Trading System Status                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Redis
if pgrep -x "redis-server" > /dev/null; then
    echo "  ✅ Redis              : RUNNING (PID: $(pgrep -x redis-server))"
else
    echo "  ❌ Redis              : NOT RUNNING"
fi

# Symbol Finder
SYM_PID=$(pgrep -f "symbol_found.py")
if [ -n "$SYM_PID" ]; then
    echo "  ✅ Symbol Finder      : RUNNING (PID: $SYM_PID)"
else
    echo "  ❌ Symbol Finder      : NOT RUNNING"
fi

# WebSocket + CatBoost
WS_PID=$(pgrep -f "angleone_websocket1.py")
if [ -n "$WS_PID" ]; then
    echo "  ✅ WebSocket+CatBoost : RUNNING (PID: $WS_PID)"
else
    echo "  ❌ WebSocket+CatBoost : NOT RUNNING"
fi

# Position Handler
POS_PID=$(pgrep -f "pos_handle_wts.py")
if [ -n "$POS_PID" ]; then
    echo "  ✅ Position Handler   : RUNNING (PID: $POS_PID)"
else
    echo "  ❌ Position Handler   : NOT RUNNING"
fi

echo ""
echo "  📊 Cron jobs installed:"
crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" | while read -r line; do
    echo "     $line"
done

echo ""
echo "  📁 Today's logs:"
TODAY=$(date +%Y%m%d)
ls -lt /root/Trading_setup_code/logs/*${TODAY}* 2>/dev/null | head -5 | awk '{print "     " $6, $7, $8, $9}'
if [ $? -ne 0 ] || [ -z "$(ls /root/Trading_setup_code/logs/*${TODAY}* 2>/dev/null)" ]; then
    echo "     (no logs for today yet)"
fi
echo ""
