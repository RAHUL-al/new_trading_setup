#!/bin/bash
# ─────────────────────────────────────────────────────────
# Cleanup old logs (Cron: Sunday midnight)
# Deletes log files older than 7 days to prevent disk bloat.
# ─────────────────────────────────────────────────────────

LOGDIR="/root/Trading_setup_code/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up logs older than 7 days..." >> "$LOGDIR/cron.log"

DELETED=$(find "$LOGDIR" -name "*.log" -mtime +7 -type f -print -delete | wc -l)

echo "[$(date '+%H:%M:%S')] ✅ Deleted $DELETED old log files." >> "$LOGDIR/cron.log"
