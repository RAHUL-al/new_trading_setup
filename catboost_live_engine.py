"""
catboost_live_engine.py — Live CatBoost ML Signal Engine for NIFTY (Pure 2-Minute)

Architecture:
  1. Pre-market: Sync CSV data (fill missing days via AngelOne API)
  2. Trading: Aggregates fully closed 2-minute candles and predicts signals
  3. Background: Cross-check Redis candles vs API every minute until close

Same Redis interface as UT Bot engine — pos_handle_wts.py works unchanged.
"""

import asyncio
import datetime
import time
import os
import threading
import numpy as np
import pandas as pd
import ujson as json
import pytz
from collections import deque
from logzero import logger

try:
    from catboost import CatBoostClassifier
except ImportError:
    logger.error("❌ CatBoost not installed. Run: pip install catboost")
    exit(1)

from redis.asyncio import Redis as AsyncRedis
import redis

# Import incremental fetch functions
try:
    from fetch_nifty_incremental import (
        connect_api as api_connect,
        fetch_candles as api_fetch_candles,
        get_last_date_in_csv,
        filter_market_hours,
        get_date_chunks,
        INTERVAL_CONFIG,
    )
    HAS_FETCH_MODULE = True
except ImportError:
    HAS_FETCH_MODULE = False
    logger.warning("⚠️ fetch_nifty_incremental.py not found — pre-market sync disabled")

# Import purely 2-minute backtest feature builder
try:
    from catboost_strategy import build_features_2m
    HAS_BACKTEST_FEATURES = True
    logger.info("✅ Using build_features_2m from catboost_strategy.py")
except ImportError:
    HAS_BACKTEST_FEATURES = False
    logger.error("❌ catboost_strategy.py not found or doesn't have build_features_2m")

# ─────────── Config ───────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "")

NIFTY_SYMBOL = "NIFTY"
MODEL_PATH = os.environ.get("CATBOOST_MODEL", "catboost_nifty_model.cbm")

# Only 2-minute CSV is fully needed for warm-up
CSV_2M = os.environ.get("CSV_2M", "nifty_2min_data.csv")
WARMUP_CANDLES = 500  # Load last N 2-minute candles for EWM convergence

ATR_KEY_VALUE = 1.0
MIN_ATR = float(os.environ.get("MIN_ATR", "6.5"))
SIGNAL_COOLDOWN = 2

# Trading window (configurable via env)
WINDOW_START = datetime.time(
    int(os.environ.get("WINDOW_START_H", "9")),
    int(os.environ.get("WINDOW_START_M", "20"))
)
WINDOW_END = datetime.time(
    int(os.environ.get("WINDOW_END_H", "15")),
    int(os.environ.get("WINDOW_END_M", "15"))
)
SQUARE_OFF_TIME = datetime.time(15, 24)

INDIA_TZ = pytz.timezone("Asia/Kolkata")

# Feature column names (strictly 2M)
FEATURE_COLS_2M = [
    'atr_2m', 'rsi_2m', 'ut_dir_2m', 'close_vs_trail_2m',
    'mom_3_2m', 'mom_5_2m', 'mom_10_2m',
    'body_2m', 'body_pct_2m', 'upper_wick_2m', 'lower_wick_2m', 'range_2m',
    'std_5_2m', 'std_10_2m',
    'sma_5_2m', 'sma_10_2m', 'sma_20_2m',
    'close_vs_sma5_2m', 'close_vs_sma10_2m', 'sma5_vs_sma10_2m',
    'high_5_2m', 'low_5_2m', 'close_vs_high5_2m', 'close_vs_low5_2m',
]

ALL_FEATURE_COLS = FEATURE_COLS_2M

# ─────────── Pre-Market CSV Sync ───────────
# We still sync 1-min data with API, because live engine resamples it to 2-min
# This saves having to sync two distinct files if 1-min covers everything.

def pre_market_csv_sync():
    """
    Sync 1-minute CSV data. Live engine relies on resampling 1-min to 2-min.
    """
    if not HAS_FETCH_MODULE:
        logger.warning("⚠️ Pre-market sync skipped")
        return None

    logger.info("=" * 60)
    logger.info("📡 PRE-MARKET CSV SYNC")
    logger.info("=" * 60)

    try:
        smart_api = api_connect()
    except Exception as e:
        logger.error(f"❌ API login failed for pre-market sync: {e}")
        return None

    # Sync 1-min data
    config_1m = INTERVAL_CONFIG["ONE_MINUTE"]
    output_1m = config_1m["output_file"]
    last_date = get_last_date_in_csv(output_1m)
    today = datetime.date.today()

    if last_date and last_date >= today:
        logger.info(f"  ✅ CSV already up to date (last: {last_date})")
    else:
        if last_date:
            start_date = datetime.datetime.combine(last_date, datetime.datetime.min.time())
            days_missing = (today - last_date).days
            logger.info(f"  📅 Missing ~{days_missing} days (last: {last_date})")
        else:
            start_date = datetime.datetime.strptime(config_1m["from_date"], "%Y-%m-%d")
            logger.info(f"  📅 No CSV — fetching from {config_1m['from_date']}")

        # Fetch missing chunks
        end_date = datetime.datetime.now()
        chunks = get_date_chunks(start_date, end_date, config_1m["chunk_days"])
        new_data = []

        for i, (from_d, to_d) in enumerate(chunks):
            df = api_fetch_candles(smart_api, from_d, to_d, config_1m["api_interval"])
            if not df.empty:
                new_data.append(df)
            if i < len(chunks) - 1:
                time.sleep(2)

        if new_data:
            new_combined = pd.concat(new_data, ignore_index=True)
            new_combined["Time"] = pd.to_datetime(new_combined["Time"])
            new_combined = filter_market_hours(new_combined)

            if last_date and os.path.exists(output_1m):
                existing_df = pd.read_csv(output_1m)
                existing_df["Time"] = pd.to_datetime(existing_df["Time"])
                existing_df = existing_df[existing_df["Time"].dt.date < last_date]
                combined = pd.concat([existing_df, new_combined], ignore_index=True)
            else:
                combined = new_combined

            combined = combined.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
            combined.to_csv(output_1m, index=False)
            logger.info(f"  ✅ 1-min CSV synced: {len(combined):,} candles")
            
            # Auto-generate 2-min CSV to keep backtester aligned
            combined_2m = combined.set_index('Time').resample('2min', label='left', closed='left').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna().reset_index()
            combined_2m.to_csv(CSV_2M, index=False)
            logger.info(f"  ✅ 2-min CSV auto-generated: {len(combined_2m):,} candles → {CSV_2M}")

    # Load warm-up cache directly from 2-min CSV
    cached_df_2m = None
    if os.path.exists(CSV_2M):
        try:
            full_2m = pd.read_csv(CSV_2M)
            full_2m["Time"] = pd.to_datetime(full_2m["Time"])
            tail = full_2m.tail(WARMUP_CANDLES).copy()
            cached_df_2m = tail.reset_index(drop=True)
            for col in ["Open", "High", "Low", "Close"]:
                 cached_df_2m[col] = cached_df_2m[col].astype(float)
            logger.info(f"  📦 Loaded {len(cached_df_2m)} 2-min candles for warm-up")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load CSV cache: {e}")

    logger.info("=" * 60)
    return cached_df_2m

# ─────────── Background Cross-Check ───────────
_cross_check_api = None

def _ensure_cross_check_api():
    global _cross_check_api
    if _cross_check_api is None:
        try:
            _cross_check_api = api_connect()
        except Exception as e:
            return None
    return _cross_check_api

def _do_cross_check(date_str, redis_prefix):
    if not HAS_FETCH_MODULE: return
    smart_api = _ensure_cross_check_api()
    if not smart_api: return
    r = redis.Redis(host=REDIS_HOST, port=int(REDIS_PORT), password=REDIS_PASSWORD, db=0, decode_responses=True)
    try:
        api_df = api_fetch_candles(smart_api, date_str, date_str, "ONE_MINUTE")
        if api_df.empty: return
        api_df["Time"] = pd.to_datetime(api_df["Time"])
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        api_df = api_df[(api_df["Time"].dt.time >= market_open) & (api_df["Time"].dt.time <= market_close)]
        history_key = f"{redis_prefix}HISTORY:NIFTY:{date_str}"
        redis_raw = r.lrange(history_key, 0, -1)
        if not redis_raw: return
        redis_candles = [json.loads(x) for x in redis_raw]
        api_count = len(api_df)
        redis_count = len(redis_candles)
        mismatches = 0
        for i, (_, api_row) in enumerate(api_df.iterrows()):
            if i >= redis_count: break
            rc = redis_candles[i]
            api_close = float(api_row["Close"])
            redis_close = float(rc.get("close", 0))
            if abs(api_close - redis_close) > 0.5:
                mismatches += 1
                fixed = {
                    "timestamp": rc.get("timestamp", ""),
                    "open": float(api_row["Open"]), "high": float(api_row["High"]),
                    "low": float(api_row["Low"]), "close": api_close,
                    "volume": int(api_row.get("Volume", 0)),
                }
                r.lset(history_key, i, json.dumps(fixed))
    except Exception as e:
        pass
    finally:
        try: r.close()
        except: pass

async def trigger_cross_check(date_str, redis_prefix):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_cross_check, date_str, redis_prefix)

# ─────────── Helper: Print Daily Summary ───────────
def _print_daily_summary(trades, daily_pnl, wins, losses, predictions):
    logger.info("=" * 60)
    logger.info("  DAILY SUMMARY (PURE 2-MIN MODEL)")
    logger.info("=" * 60)
    logger.info(f"Total P&L:   {daily_pnl:+.2f} pts")
    logger.info(f"Trades:      {len(trades)} (W:{wins} L:{losses})")
    logger.info(f"Predictions: {predictions}")
    logger.info("=" * 60)

# ─────────── Main Engine ───────────

async def catboost_signal_engine():
    ar = AsyncRedis(
        host=REDIS_HOST, port=REDIS_PORT,
        password=REDIS_PASSWORD, db=0, decode_responses=True
    )

    logger.info(f"🧠 Loading Pure-2M CatBoost model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        return

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded ({model.tree_count_} trees)")

    try:
        model_features = model.feature_names_
    except:
        model_features = ALL_FEATURE_COLS

    # ── Live position tracking ──
    position = None
    trades = []
    daily_pnl = 0.0
    wins = 0
    losses = 0
    prediction_count = 0
    last_prediction = 0

    STATE_KEY = f"{REDIS_PREFIX}catboost_state"
    TRADES_KEY = f"{REDIS_PREFIX}catboost_trades"

    async def save_state():
        state = {
            "position": position, "daily_pnl": daily_pnl,
            "wins": wins, "losses": losses,
            "prediction_count": prediction_count, "last_prediction": last_prediction,
            "date": datetime.date.today().strftime('%Y-%m-%d'),
        }
        await ar.set(STATE_KEY, json.dumps(state))
        await ar.set(TRADES_KEY, json.dumps(trades))

    async def load_state():
        nonlocal position, daily_pnl, wins, losses, prediction_count, last_prediction, trades
        try:
            raw = await ar.get(STATE_KEY)
            if raw:
                state = json.loads(raw)
                if state.get("date") == datetime.date.today().strftime('%Y-%m-%d'):
                    position = state.get("position")
                    daily_pnl = state.get("daily_pnl", 0)
                    wins = state.get("wins", 0)
                    losses = state.get("losses", 0)
                    prediction_count = state.get("prediction_count", 0)
                    last_prediction = state.get("last_prediction", 0)
                    trades_raw = await ar.get(TRADES_KEY)
                    if trades_raw: trades = json.loads(trades_raw)
                    if position:
                        logger.info(f"  🔄 RECOVERED {position['dir']} @ {position['entry']:.2f} | SL={position['sl']:.2f}")
                else:
                    await ar.delete(STATE_KEY, TRADES_KEY)
        except Exception as e:
            pass

    await load_state()

    last_candle_count = 0
    last_completed_2m_count = 0
    last_status_log = 0
    cached_df_2m_warmup = None
    pre_market_synced = False
    cross_check_task = None

    logger.info(f"⚡ Pure 2-Min CatBoost Engine matching Live Reality")

    async def do_close_position(exit_price, reason, exit_time_str):
        nonlocal position, daily_pnl, wins, losses
        if not position: return
        pnl = round((exit_price - position['entry']) if position['dir'] == 'LONG' else (position['entry'] - exit_price), 2)
        daily_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
        trade = {
            'dir': position['dir'], 'entry': position['entry'], 'exit': exit_price, 'sl': position['sl'],
            'entry_time': position['entry_time'], 'exit_time': exit_time_str, 'pnl': pnl, 'reason': reason,
        }
        trades.append(trade)
        icon = "✅" if pnl > 0 else "❌"
        logger.info(f"  {'🟢' if position['dir']=='LONG' else '🔴'} CLOSED {position['dir']} | Entry={position['entry']:.2f} → Exit={exit_price:.2f} | P&L={pnl:+.2f} {icon} | Reason={reason}")
        position = None
        await save_state()

    while True:
        try:
            now = datetime.datetime.now(INDIA_TZ)
            now_time = now.time()

            if now_time >= datetime.time(15, 30):
                if position:
                    live_ltp = await ar.get(f"{REDIS_PREFIX}NIFTY")
                    cp = float(live_ltp) if live_ltp else position['entry']
                    logger.info("🕐 Market close — squaring off")
                    await do_close_position(cp, "MARKET_CLOSE", now.strftime('%H:%M'))
                logger.info("Market closed.")
                _print_daily_summary(trades, daily_pnl, wins, losses, prediction_count)
                break

            if now_time < datetime.time(9, 20):
                if not pre_market_synced and now_time >= datetime.time(9, 17):
                    logger.info("🚀 Loading 2-Min CSV sync...")
                    cached_df_2m_warmup = await asyncio.get_event_loop().run_in_executor(None, pre_market_csv_sync)
                    pre_market_synced = True
                
                if time.time() - last_status_log > 10:
                    logger.info(f"⏳ Waiting for 09:20...")
                    last_status_log = time.time()
                await asyncio.sleep(0.5)
                continue

            date_key = now.strftime('%Y-%m-%d')
            history_key = f"{REDIS_PREFIX}HISTORY:{NIFTY_SYMBOL}:{date_key}"
            candle_count = await ar.llen(history_key)

            if candle_count < 1:
                await asyncio.sleep(0.5)
                continue

            if candle_count == last_candle_count:
                live_ltp = await ar.get(f"{REDIS_PREFIX}NIFTY")
                if not live_ltp: 
                    await asyncio.sleep(0.5)
                    continue
                live_price = float(live_ltp)

                # Dashboard print
                if position and time.time() - last_status_log > 5:
                    unr = round((live_price - position['entry']) if position['dir'] == 'LONG' else (position['entry'] - live_price), 2)
                    logger.info(f"📊 {now.strftime('%H:%M:%S')} | NIFTY={live_price:.2f} | Pos={position['dir']} | Unreal={unr:+.2f} | P&L={daily_pnl+unr:+.2f}")
                    last_status_log = time.time()

                await asyncio.sleep(0.5)
                continue

            last_candle_count = candle_count

            if cross_check_task is None or cross_check_task.done():
                cross_check_task = asyncio.create_task(trigger_cross_check(date_key, REDIS_PREFIX))

            history_data = await ar.lrange(history_key, 0, -1)
            candles = [json.loads(x) for x in history_data]
            df_today = pd.DataFrame(candles)
            df_today = df_today.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
            for col in ["Open", "High", "Low", "Close"]: df_today[col] = df_today[col].astype(float)

            if 'timestamp' in df_today.columns:
                df_today['Time'] = pd.to_datetime(df_today['timestamp'])
            
            # --- Enforce STRICT 2-Minute Logic (No Future Data Leak) ---
            df_2m_today = df_today.set_index('Time').resample('2min', label='left', closed='left').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            }).dropna().reset_index()

            current_minute = df_today['Time'].iloc[-1].replace(second=0, microsecond=0)
            
            # Extract ONLY completed 2-min blocks safely.
            # E.g. If current_minute is 09:16, (09:16 - 2m) is 09:14. Block 09:14 covers 09:14 and 09:15. It is CLOSED.
            completed_mask = df_2m_today['Time'] <= (current_minute - pd.Timedelta(minutes=2))
            closed_2m = df_2m_today[completed_mask]

            if len(closed_2m) <= last_completed_2m_count:
                # We haven't fully finished painting the new 2-minute candle yet. 
                await asyncio.sleep(0.2)
                continue

            last_completed_2m_count = len(closed_2m)

            # We have a brand new completely closed 2-min candle!!
            if cached_df_2m_warmup is not None and not cached_df_2m_warmup.empty:
                today_start = closed_2m['Time'].iloc[0]
                warmup_filt = cached_df_2m_warmup[cached_df_2m_warmup['Time'] < today_start]
                df_2m_combined = pd.concat([warmup_filt, closed_2m], ignore_index=True)
            else:
                df_2m_combined = closed_2m.copy()

            if not HAS_BACKTEST_FEATURES:
                logger.error("Skipping prediction: catboost_strategy functions missing")
                await asyncio.sleep(2)
                continue

            # Build 2-minute features exactly like backtest
            features_df = build_features_2m(df_2m_combined)
            last_feat = features_df.iloc[-1]

            feature_vector = [float(last_feat.get(f, 0)) for f in model_features]
            feature_vector = [0 if (v != v or abs(v) == float('inf')) else v for v in feature_vector]

            # PREDICT 
            pred = model.predict([feature_vector]).flatten()[0].item()
            prediction_count += 1
            last_prediction = time.time()

            close_price = float(closed_2m.iloc[-1]['Close'])
            high_price = float(closed_2m.iloc[-1]['High'])
            low_price = float(closed_2m.iloc[-1]['Low'])
            time_str = closed_2m.iloc[-1]['Time'].strftime('%H:%M')
            
            current_atr = float(last_feat.get('atr_2m', 0))

            t = now.time()
            in_window = WINDOW_START <= t <= WINDOW_END
            atr_ok = current_atr >= MIN_ATR

            signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            signal_name = signal_map.get(pred, "HOLD")
            logger.info(f"🔎 2-Min Eval [@ {time_str}] | NIFTY={close_price:.2f} | Predicted={signal_name} | ATR={current_atr:.2f}")

            # ── Trading Logic (On 2-min Close) ──
            if position and t >= SQUARE_OFF_TIME:
                await do_close_position(close_price, "SQUARE_OFF", time_str)

            if position:
                pos = position
                if pos['dir'] == 'LONG' and low_price <= pos['sl']:
                    await do_close_position(pos['sl'], "TRAIL_SL", time_str)
                elif pos['dir'] == 'SHORT' and high_price >= pos['sl']:
                    await do_close_position(pos['sl'], "TRAIL_SL", time_str)

            if position:
                pos = position
                if pos['dir'] == 'LONG':
                    new_sl = close_price - current_atr * ATR_KEY_VALUE
                    if new_sl > pos['sl']: position['sl'] = new_sl
                elif pos['dir'] == 'SHORT':
                    new_sl = close_price + current_atr * ATR_KEY_VALUE
                    if new_sl < pos['sl']: position['sl'] = new_sl
                await save_state()

            if pred == 1 and position and position['dir'] == 'SHORT':
                await do_close_position(close_price, "OPPOSITE", time_str)
            elif pred == -1 and position and position['dir'] == 'LONG':
                await do_close_position(close_price, "OPPOSITE", time_str)

            if not position and pred != 0 and in_window and atr_ok and t < SQUARE_OFF_TIME:
                if pred == 1:
                    sl = close_price - current_atr * ATR_KEY_VALUE
                    position = {'dir': 'LONG', 'entry': close_price, 'sl': sl, 'entry_time': time_str}
                elif pred == -1:
                    sl = close_price + current_atr * ATR_KEY_VALUE
                    position = {'dir': 'SHORT', 'entry': close_price, 'sl': sl, 'entry_time': time_str}
                
                logger.info(f"  🟢 2-MIN NEW POSITION: {position['dir']} @ {position['entry']:.2f}")
                
                # Transmit signal exactly like UT_Bot 
                payload = {
                    "signal": "BUY" if pred == 1 else "SELL",
                    "time": datetime.datetime.now().strftime("%I:%M %p"),
                    "close": close_price,
                    "target": 0,
                    "stop_loss": position['sl']
                }
                ar.publish(f"{REDIS_PREFIX}WTS_NIFTY_SIGNALS", json.dumps(payload))
                logger.info(f"  📢 Published WTS Signal: {payload}")
                await save_state()

        except Exception as e:
            logger.error(f"⚠️ Engine Error: {e}", exc_info=True)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(catboost_signal_engine())
