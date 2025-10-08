import redis
import json
from datetime import datetime

r = redis.StrictRedis(
    host='localhost',
    port=6379,
    password='Rahul@7355',
    db=0,
    decode_responses=True
)

data = r.get("trade_history_2025-08-29")

total_profit_or_loss = 0

if data:
    trade_history = json.loads(data)
    for i in trade_history:
        entry_time = datetime.fromisoformat(i["entry_time"])
        if entry_time.date() == datetime.today().date():
            total_profit_or_loss += float(i["pnl"])

print("Total PnL for today:", total_profit_or_loss)
