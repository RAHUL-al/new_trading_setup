"""
catboost_live_engine.py — Live CatBoost ML Signal Engine for NIFTY

Architecture:
  1. Pre-market: Sync CSV data (fill missing days via AngelOne API)
  2. Trading: Hybrid prediction (cached history + live websocket candle)
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

# Import backtest-identical feature builders
try:
    from catboost_strategy import build_features_1min, build_features_2min
    HAS_BACKTEST_FEATURES = True
    logger.info("✅ Using backtest-identical feature builders from catboost_strategy.py")
except ImportError:
    HAS_BACKTEST_FEATURES = False
    logger.warning("⚠️ catboost_strategy.py not found — using built-in feature builder")

# ─────────── Config ───────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "")

NIFTY_SYMBOL = "NIFTY"
MODEL_PATH = os.environ.get("CATBOOST_MODEL", "catboost_nifty_model.cbm")

FIXED_SL_PTS = float(os.environ.get("FIXED_SL_PTS", "15.0"))
SIGNAL_COOLDOWN = 2  # seconds between signals (fast for ML)

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

# Feature column names (must match training order exactly)
FEATURE_COLS_1M = [
    'mom_3', 'mom_5', 'mom_10',
    'body_1m', 'body_pct_1m', 'upper_wick_1m', 'lower_wick_1m', 'range_1m',
    'std_5', 'std_10',
    'sma_5', 'sma_10', 'sma_20',
    'close_vs_sma5', 'close_vs_sma10', 'sma5_vs_sma10',
    'high_5', 'low_5', 'close_vs_high5', 'close_vs_low5',
]

FEATURE_COLS_2M = [
    'mom_3_2m', 'mom_5_2m', 'range_2m', 'body_2m',
]

ALL_FEATURE_COLS = FEATURE_COLS_1M + FEATURE_COLS_2M


# ─────────── Feature Builder ───────────

def build_all_features(df_1m, df_2m=None):
    """
    Build all features from 1-min (and optional 2-min) DataFrame.
    Returns the feature row for the LAST candle only.
    """
    close = df_1m['Close'].astype(float)
    high = df_1m['High'].astype(float)
    low = df_1m['Low'].astype(float)
    opn = df_1m['Open'].astype(float)

    features = {}
    i = len(df_1m) - 1  # last row

    # Momentum
    features['mom_3'] = float((close.iloc[i] / close.iloc[max(0, i-3)] - 1) * 100) if i >= 3 else 0
    features['mom_5'] = float((close.iloc[i] / close.iloc[max(0, i-5)] - 1) * 100) if i >= 5 else 0
    features['mom_10'] = float((close.iloc[i] / close.iloc[max(0, i-10)] - 1) * 100) if i >= 10 else 0

    # Candle
    features['body_1m'] = float(close.iloc[i] - opn.iloc[i])
    features['body_pct_1m'] = float((close.iloc[i] - opn.iloc[i]) / opn.iloc[i] * 100) if opn.iloc[i] != 0 else 0
    upper_ref = max(close.iloc[i], opn.iloc[i])
    lower_ref = min(close.iloc[i], opn.iloc[i])
    features['upper_wick_1m'] = float(high.iloc[i] - upper_ref)
    features['lower_wick_1m'] = float(lower_ref - low.iloc[i])
    features['range_1m'] = float(high.iloc[i] - low.iloc[i])

    # Volatility
    features['std_5'] = float(close.iloc[max(0, i-4):i+1].std()) if i >= 4 else 0
    features['std_10'] = float(close.iloc[max(0, i-9):i+1].std()) if i >= 9 else 0

    # Moving averages
    sma_5 = float(close.iloc[max(0, i-4):i+1].mean()) if i >= 4 else float(close.iloc[i])
    sma_10 = float(close.iloc[max(0, i-9):i+1].mean()) if i >= 9 else float(close.iloc[i])
    sma_20 = float(close.iloc[max(0, i-19):i+1].mean()) if i >= 19 else float(close.iloc[i])

    features['sma_5'] = sma_5
    features['sma_10'] = sma_10
    features['sma_20'] = sma_20
    features['close_vs_sma5'] = float(close.iloc[i]) - sma_5
    features['close_vs_sma10'] = float(close.iloc[i]) - sma_10
    features['sma5_vs_sma10'] = sma_5 - sma_10

    # High/Low channels
    high_5 = float(high.iloc[max(0, i-4):i+1].max()) if i >= 4 else float(high.iloc[i])
    low_5 = float(low.iloc[max(0, i-4):i+1].min()) if i >= 4 else float(low.iloc[i])
    features['high_5'] = high_5
    features['low_5'] = low_5
    features['close_vs_high5'] = float(close.iloc[i]) - high_5
    features['close_vs_low5'] = float(close.iloc[i]) - low_5

    # 2-min features
    if df_2m is not None and len(df_2m) >= 5:
        close_2m = df_2m['Close'].astype(float)
        high_2m = df_2m['High'].astype(float)
        low_2m = df_2m['Low'].astype(float)
        opn_2m = df_2m['Open'].astype(float)

        j = len(df_2m) - 1
        features['mom_3_2m'] = float((close_2m.iloc[j] / close_2m.iloc[max(0, j-3)] - 1) * 100) if j >= 3 else 0
        features['mom_5_2m'] = float((close_2m.iloc[j] / close_2m.iloc[max(0, j-5)] - 1) * 100) if j >= 5 else 0
        features['range_2m'] = float(high_2m.iloc[j] - low_2m.iloc[j])
        features['body_2m'] = float(close_2m.iloc[j] - opn_2m.iloc[j])
    else:
        for col in FEATURE_COLS_2M:
            features[col] = 0.0

    return features


def resample_to_2min(df_1m):
    """Resample 1-min candles to 2-min candles."""
    if len(df_1m) < 2:
        return pd.DataFrame()

    df = df_1m.copy()
    df['Time'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.to_datetime(df.index)
    df = df.set_index('Time')

    resampled = df.resample('2min', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna().reset_index()

    return resampled

# ─────────── Background Cross-Check ───────────

_cross_check_api = None  # Cached API connection for cross-checks

def _ensure_cross_check_api():
    """Get or create API connection for cross-checks (sync, runs in thread)."""
    global _cross_check_api
    if _cross_check_api is None:
        try:
            _cross_check_api = api_connect()
        except Exception as e:
            logger.warning(f"  ⚠️ Cross-check API login failed: {e}")
            return None
    return _cross_check_api


def _do_cross_check(date_str, redis_prefix):
    """
    Cross-check today's Redis candles vs API (runs in thread).
    Fetches today's 1-min data from API, compares with Redis HISTORY,
    fixes any mismatches.
    """
    if not HAS_FETCH_MODULE:
        return

    smart_api = _ensure_cross_check_api()
    if not smart_api:
        return

    r = redis.Redis(
        host=REDIS_HOST, port=int(REDIS_PORT),
        password=REDIS_PASSWORD, db=0, decode_responses=True
    )

    try:
        # Fetch today's data from API
        api_df = api_fetch_candles(smart_api, date_str, date_str, "ONE_MINUTE")
        if api_df.empty:
            logger.info(f"  🔄 Cross-check: No API data for {date_str} (market may not have started)")
            return

        api_df["Time"] = pd.to_datetime(api_df["Time"])
        # Filter market hours
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        api_df = api_df[
            (api_df["Time"].dt.time >= market_open) &
            (api_df["Time"].dt.time <= market_close)
        ]

        # Get Redis HISTORY candles
        history_key = f"{redis_prefix}HISTORY:NIFTY:{date_str}"
        redis_raw = r.lrange(history_key, 0, -1)
        if not redis_raw:
            logger.info(f"  🔄 Cross-check: No Redis candles yet for {date_str}")
            return

        redis_candles = [json.loads(x) for x in redis_raw]

        # Compare counts
        api_count = len(api_df)
        redis_count = len(redis_candles)

        mismatches = 0
        # Compare each candle that exists in both
        for i, (_, api_row) in enumerate(api_df.iterrows()):
            if i >= redis_count:
                break
            rc = redis_candles[i]
            # Check close price mismatch (> 0.5 points = significant)
            api_close = float(api_row["Close"])
            redis_close = float(rc.get("close", 0))
            if abs(api_close - redis_close) > 0.5:
                mismatches += 1
                # Fix Redis candle with API data
                fixed = {
                    "timestamp": rc.get("timestamp", ""),
                    "open": float(api_row["Open"]),
                    "high": float(api_row["High"]),
                    "low": float(api_row["Low"]),
                    "close": api_close,
                    "volume": int(api_row.get("Volume", 0)),
                }
                r.lset(history_key, i, json.dumps(fixed))

        if mismatches > 0:
            logger.info(f"  🔧 Cross-check: Fixed {mismatches}/{min(api_count, redis_count)} mismatches | API={api_count} Redis={redis_count}")
        else:
            logger.info(f"  ✅ Cross-check: OK ({min(api_count, redis_count)} candles match) | API={api_count} Redis={redis_count}")

    except Exception as e:
        logger.warning(f"  ⚠️ Cross-check error: {e}")
    finally:
        try:
            r.close()
        except:
            pass


async def trigger_cross_check(date_str, redis_prefix):
    """Run cross-check in background thread (non-blocking)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_cross_check, date_str, redis_prefix)


# ─────────── Main Engine ───────────

async def catboost_signal_engine():
    """
    Async CatBoost signal engine — polls Redis for new candles,
    builds features, predicts BUY/SELL/HOLD, publishes signals.
    """
    ar = AsyncRedis(
        host=REDIS_HOST, port=REDIS_PORT,
        password=REDIS_PASSWORD, db=0, decode_responses=True
    )

    # Load model
    logger.info(f"🧠 Loading CatBoost model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        logger.error("Train first: python catboost_strategy.py --min-atr 6.5")
        return

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded ({model.tree_count_} trees)")

    # Get feature names from model
    try:
        model_features = model.feature_names_
        if model_features:
            logger.info(f"  Model features: {len(model_features)}")
    except:
        model_features = ALL_FEATURE_COLS
        logger.info(f"  Using default feature list: {len(model_features)}")

    # ── Live position tracking ──
    position = None
    trades = []
    daily_pnl = 0.0
    wins = 0
    losses = 0
    prediction_count = 0
    last_prediction = 0

    # ── Redis keys for state persistence ──
    STATE_KEY = f"{REDIS_PREFIX}catboost_state"
    TRADES_KEY = f"{REDIS_PREFIX}catboost_trades"

    async def save_state():
        """Save current state to Redis (survives restarts)."""
        state = {
            "position": position,
            "daily_pnl": daily_pnl,
            "wins": wins,
            "losses": losses,
            "prediction_count": prediction_count,
            "last_prediction": last_prediction,
            "date": datetime.date.today().strftime('%Y-%m-%d'),
        }
        await ar.set(STATE_KEY, json.dumps(state))
        await ar.set(TRADES_KEY, json.dumps(trades))

    async def load_state():
        """Load previous state from Redis on startup."""
        nonlocal position, daily_pnl, wins, losses, prediction_count, last_prediction, trades
        try:
            raw = await ar.get(STATE_KEY)
            if raw:
                state = json.loads(raw)
                # Only load if same trading day
                if state.get("date") == datetime.date.today().strftime('%Y-%m-%d'):
                    position = state.get("position")
                    daily_pnl = state.get("daily_pnl", 0)
                    wins = state.get("wins", 0)
                    losses = state.get("losses", 0)
                    prediction_count = state.get("prediction_count", 0)
                    last_prediction = state.get("last_prediction", 0)
                    # Load trades
                    trades_raw = await ar.get(TRADES_KEY)
                    if trades_raw:
                        trades = json.loads(trades_raw)
                    if position:
                        cname = position.get('contract', '')
                        logger.info(
                            f"  🔄 RECOVERED {position['dir']} @ {position['entry']:.2f} | "
                            f"SL={position['sl']:.2f} | Contract={cname}"
                        )
                    logger.info(
                        f"  🔄 Restored: {len(trades)} trades | W:{wins} L:{losses} | "
                        f"Day P&L={daily_pnl:+.2f} | Predictions={prediction_count}"
                    )
                else:
                    logger.info("  📅 New trading day — starting fresh")
                    await ar.delete(STATE_KEY, TRADES_KEY)
        except Exception as e:
            logger.warning(f"  ⚠️ State load failed: {e}")

    # Load previous state
    await load_state()

    # ── Live position tracking defaults (overridden by load_state if data exists) ──
    SQUARE_OFF_TIME = datetime.time(15, 24)
    last_signal_time = 0
    last_candle_count = 0
    last_status_log = 0
    cached_df_1m = None          # Cached today's DataFrame
    last_live_predict = 0
    cross_check_task = None

    logger.info(f"⚡ CatBoost signal engine started")
    logger.info(f"  Window: {WINDOW_START.strftime('%H:%M')} - {WINDOW_END.strftime('%H:%M')}")
    logger.info(f"  Fixed SL: {FIXED_SL_PTS} pts")
    logger.info(f"  Signal cooldown: {SIGNAL_COOLDOWN}s")

    async def do_close_position(exit_price, reason, exit_time_str):
        nonlocal position, daily_pnl, wins, losses
        if not position:
            return
        if position['dir'] == 'LONG':
            pnl = exit_price - position['entry']
        else:
            pnl = position['entry'] - exit_price
        pnl = round(pnl, 2)
        daily_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        trade = {
            'dir': position['dir'], 'entry': position['entry'],
            'exit': exit_price, 'sl': position['sl'],
            'entry_time': position['entry_time'], 'exit_time': exit_time_str,
            'pnl': pnl, 'reason': reason,
        }
        trades.append(trade)
        icon = "✅" if pnl > 0 else "❌"
        # Get contract exit price
        opt_exit_str = ""
        cname = position.get('contract', '')
        opt_entry = position.get('contract_entry', 0)
        if cname:
            try:
                opt_exit_raw = await ar.get(f"{REDIS_PREFIX}{cname}")
                if opt_exit_raw:
                    opt_exit_str = f" | OPT: {cname} Entry=₹{opt_entry:.2f} Exit=₹{float(opt_exit_raw):.2f}"
            except:
                pass
        logger.info(
            f"  {'🟢' if position['dir']=='LONG' else '🔴'} CLOSED {position['dir']} | "
            f"Entry={position['entry']:.2f} → Exit={exit_price:.2f} | "
            f"P&L={pnl:+.2f} {icon} | Reason={reason}{opt_exit_str}"
        )
        position = None
        await save_state()

    while True:
        try:
            now = datetime.datetime.now(INDIA_TZ)
            now_time = now.time()

            if now_time >= datetime.time(15, 30):
                # Square off any open position
                if position:
                    logger.info("🕐 Market close — squaring off")
                    await do_close_position(close_price, "MARKET_CLOSE", now.strftime('%H:%M'))
                logger.info("Market closed. Stopping CatBoost engine.")
                _print_daily_summary(trades, daily_pnl, wins, losses, prediction_count)
                break

            if now_time < datetime.time(9, 20):
                if time.time() - last_status_log > 10:
                    logger.info(f"⏳ Waiting for 09:20 (now: {now_time.strftime('%H:%M:%S')})...")
                    last_status_log = time.time()
                await asyncio.sleep(0.5)
                continue

            # Read candle history
            date_key = now.strftime('%Y-%m-%d')
            history_key = f"{REDIS_PREFIX}HISTORY:{NIFTY_SYMBOL}:{date_key}"
            candle_count = await ar.llen(history_key)

            if candle_count < 20:
                if time.time() - last_status_log > 10:
                    logger.info(f"⏳ Waiting for candles: {candle_count}/20 in Redis ({history_key})")
                    last_status_log = time.time()
                await asyncio.sleep(0.1)
                continue

            if candle_count == last_candle_count:
                # ── Between candles: ONLY SL check + dashboard (NO predictions) ──
                live_ltp = await ar.get(f"{REDIS_PREFIX}NIFTY")
                if not live_ltp:
                    await asyncio.sleep(0.5)
                    continue
                live_price = float(live_ltp)

                # Live dashboard (every 1s)
                if position and time.time() - last_status_log > 1:
                    if position['dir'] == 'LONG':
                        unrealized = round(live_price - position['entry'], 2)
                    else:
                        unrealized = round(position['entry'] - live_price, 2)
                    sl_dist = round(abs(live_price - position['sl']), 2)
                    pos_icon = "🟢" if position['dir'] == 'LONG' else "🔴"
                    cname = position.get('contract', '')
                    live_pnl = daily_pnl + unrealized
                    total_trades = wins + losses
                    wr = (wins / total_trades * 100) if total_trades > 0 else 0
                    opt_str = ""
                    if cname:
                        opt_ltp = await ar.get(f"{REDIS_PREFIX}{cname}")
                        opt_entry = position.get('contract_entry', 0)
                        if opt_ltp:
                            opt_str = f" | OPT: Entry={opt_entry:.2f} Now={float(opt_ltp):.2f}"
                    logger.info(
                        f"📊 {now.strftime('%H:%M:%S')} | NIFTY={live_price:.2f} | "
                        f"Pos={pos_icon}{position['dir']} @ {position['entry']:.2f} [{cname}]{opt_str} | "
                        f"Unreal={unrealized:+.2f} | SL={position['sl']:.2f} (dist={sl_dist:.2f}) | "
                        f"Trades={total_trades} (W:{wins} L:{losses} {wr:.0f}%) | "
                        f"Day P&L={live_pnl:+.2f} | LIVE"
                    )
                    last_status_log = time.time()

                await asyncio.sleep(0.5)
                continue

            last_candle_count = candle_count
            t_start = time.perf_counter()

            # ── Trigger background cross-check (runs in thread, non-blocking) ──
            if cross_check_task is None or cross_check_task.done():
                cross_check_task = asyncio.create_task(trigger_cross_check(date_key, REDIS_PREFIX))

            # Load candles from Redis HISTORY (today's data)
            history_data = await ar.lrange(history_key, 0, -1)
            candles = [json.loads(x) for x in history_data]
            df_today = pd.DataFrame(candles)
            df_today = df_today.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
            })
            for col in ["Open", "High", "Low", "Close"]:
                df_today[col] = df_today[col].astype(float)

            # Add Time column
            if 'timestamp' in df_today.columns:
                df_today['Time'] = pd.to_datetime(df_today['timestamp'])
            else:
                df_today['Time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_today), freq='1min')

            cached_df_1m = df_today.copy()

            # ── Build features using BACKTEST-IDENTICAL approach ──
            if HAS_BACKTEST_FEATURES:
                feat_1m = build_features_1min(df_today)

                # Build 2-min features using today's resampled
                df_2m_today = df_today.set_index('Time').resample('2min', label='left', closed='left').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                }).dropna().reset_index()

                feat_2m = build_features_2min(df_2m_today, df_today)

                # Combine features and take ONLY the last row (= current candle)
                all_features = pd.concat([feat_1m, feat_2m], axis=1)
                last_feat = all_features.iloc[-1]

                if model_features and len(model_features) > 0:
                    feature_vector = [float(last_feat.get(f, 0)) for f in model_features]
                else:
                    feature_vector = [float(last_feat.get(f, 0)) for f in ALL_FEATURE_COLS]
                feature_vector = [0 if (v != v or abs(v) == float('inf')) else v for v in feature_vector]

                logger.info(f"  📐 Features: backtest-identical built on {len(df_today)} candles")
            else:
                # Fallback: use built-in feature builder (no warm-up)
                df_2m = resample_to_2min(df_today)
                features = build_all_features(df_today, df_2m)

                if model_features and len(model_features) > 0:
                    feature_vector = [features.get(f, 0) for f in model_features]
                else:
                    feature_vector = [features.get(f, 0) for f in ALL_FEATURE_COLS]
                feature_vector = [0 if (v != v or abs(v) == float('inf')) else v for v in feature_vector]

            last_row = df_today.iloc[-1]
            close_price = float(last_row["Close"])
            high_price = float(last_row["High"])
            low_price = float(last_row["Low"])
            candle_time = str(last_row.get("timestamp", ""))[-8:-3]  # HH:MM

            last_candle_data = json.dumps({
                "timestamp": str(last_row.get("timestamp", "")),
                "open": float(last_row["Open"]),
                "high": float(last_row["High"]),
                "low": float(last_row["Low"]),
                "close": close_price,
            })
            await ar.set(f"{REDIS_PREFIX}last_candle", last_candle_data)
            await ar.publish(f"{REDIS_PREFIX}candle:close", last_candle_data)

            pred = model.predict([feature_vector]).flatten()[0].item()
            prediction_count += 1
            t_elapsed = (time.perf_counter() - t_start) * 1000

            in_window = WINDOW_START <= now_time <= WINDOW_END
            now_ts = time.time()

            signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            signal_name = signal_map.get(pred, "HOLD")

            # ── Square off at 15:24 ──
            if position and now_time >= SQUARE_OFF_TIME:
                await do_close_position(close_price, "SQUARE_OFF", candle_time)

            # ── Trail SL check ──
            if position:
                sl_hit = False
                if position['dir'] == 'LONG' and low_price <= position['sl']:
                    await do_close_position(position['sl'], "TRAIL_SL", candle_time)
                elif position['dir'] == 'SHORT' and high_price >= position['sl']:
                    await do_close_position(position['sl'], "TRAIL_SL", candle_time)

            # ── Update trailing SL ──
            if position:
                if position['dir'] == 'LONG':
                    new_sl = close_price - FIXED_SL_PTS
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                elif position['dir'] == 'SHORT':
                    new_sl = close_price + FIXED_SL_PTS
                    if new_sl < position['sl']:
                        position['sl'] = new_sl

            # ── Opposite signal — close existing ──
            if pred == 1 and position and position['dir'] == 'SHORT':
                await do_close_position(close_price, "OPPOSITE", candle_time)
            elif pred == -1 and position and position['dir'] == 'LONG':
                await do_close_position(close_price, "OPPOSITE", candle_time)

            # ── New entry ──
            if not position and pred != 0 and in_window and now_time < SQUARE_OFF_TIME:
                if now_ts - last_signal_time > SIGNAL_COOLDOWN:
                    # Read active contract from Redis
                    contract_name = ""
                    try:
                        ts_raw = await ar.get(f"{REDIS_PREFIX}Trading_symbol")
                        if ts_raw:
                            ts_data = json.loads(ts_raw)
                            if pred == 1:  # BUY → PE (sell put) or CE
                                ce_info = ts_data.get("CE", [None])
                                contract_name = ce_info[0] if ce_info and ce_info[0] else ""
                            elif pred == -1:  # SELL → PE
                                pe_info = ts_data.get("PE", [None])
                                contract_name = pe_info[0] if pe_info and pe_info[0] else ""
                    except:
                        pass

                    # Get contract entry price from Redis
                    contract_entry_price = 0
                    if contract_name:
                        try:
                            cep = await ar.get(f"{REDIS_PREFIX}{contract_name}")
                            if cep:
                                contract_entry_price = float(cep)
                        except:
                            pass

                    if pred == 1:
                        sl = close_price - FIXED_SL_PTS
                        position = {'dir': 'LONG', 'entry': close_price, 'sl': sl, 'entry_time': candle_time, 'contract': contract_name, 'contract_entry': contract_entry_price}
                        logger.info(f"  🟢 ENTERED LONG @ {close_price:.2f} | SL={sl:.2f} | Contract={contract_name} @ ₹{contract_entry_price:.2f}")
                        await save_state()
                    elif pred == -1:
                        sl = close_price + FIXED_SL_PTS
                        position = {'dir': 'SHORT', 'entry': close_price, 'sl': sl, 'entry_time': candle_time, 'contract': contract_name, 'contract_entry': contract_entry_price}
                        logger.info(f"  🔴 ENTERED SHORT @ {close_price:.2f} | SL={sl:.2f} | Contract={contract_name} @ ₹{contract_entry_price:.2f}")
                        await save_state()

            # ── Publish signals via Redis (for pos_handle_wts) ──
            if pred != 0 and pred != last_prediction and in_window:
                if now_ts - last_signal_time > SIGNAL_COOLDOWN:
                    signal_data = json.dumps({
                        "signal": "buy" if pred == 1 else "sell",
                        "source": "catboost",
                        "prediction": pred,
                        "latency_ms": round(t_elapsed, 1),
                    })
                    if pred == 1:
                        await ar.publish(f"{REDIS_PREFIX}signal:buy", signal_data)
                        await ar.set(f"{REDIS_PREFIX}buy_signal", "true")
                        await ar.set(f"{REDIS_PREFIX}sell_signal", "false")
                    elif pred == -1:
                        await ar.publish(f"{REDIS_PREFIX}signal:sell", signal_data)
                        await ar.set(f"{REDIS_PREFIX}sell_signal", "true")
                        await ar.set(f"{REDIS_PREFIX}buy_signal", "false")
                    last_signal_time = now_ts
                    last_prediction = pred

            # ── Live Dashboard ──
            unrealized = 0
            pos_str = "FLAT"
            sl_dist = 0
            if position:
                if position['dir'] == 'LONG':
                    unrealized = close_price - position['entry']
                else:
                    unrealized = position['entry'] - close_price
                unrealized = round(unrealized, 2)
                sl_dist = round(abs(close_price - position['sl']), 2)
                pos_icon = "🟢" if position['dir'] == 'LONG' else "🔴"
                cname = position.get('contract', '')
                pos_str = f"{pos_icon}{position['dir']} @ {position['entry']:.2f} [{cname}]" if cname else f"{pos_icon}{position['dir']} @ {position['entry']:.2f}"

            total_trades = wins + losses
            wr = (wins / total_trades * 100) if total_trades > 0 else 0
            live_pnl = daily_pnl + unrealized  # realized + unrealized

            if position:
                pos_detail = (
                    f"Unreal={unrealized:+.2f} | SL_dist={sl_dist:.2f} | "
                )
            else:
                pos_detail = ""

            logger.info(
                f"📊 {candle_time} | NIFTY={close_price:.2f} | "
                f"Pred={signal_name} | "
                f"Pos={pos_str} | {pos_detail}"
                f"Trades={total_trades} (W:{wins} L:{losses} {wr:.0f}%) | "
                f"Day P&L={live_pnl:+.2f} | {t_elapsed:.0f}ms"
            )

        except Exception as e:
            logger.error(f"CatBoost engine error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


def _print_daily_summary(trades, daily_pnl, wins, losses, predictions):
    """Print end-of-day summary."""
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    logger.info(f"\n{'='*70}")
    logger.info(f"  📋 DAILY SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  Total trades:    {total}")
    logger.info(f"  Wins: {wins} | Losses: {losses} | Win rate: {wr:.0f}%")
    logger.info(f"  Day P&L:         {daily_pnl:+.2f} pts")
    logger.info(f"  Predictions:     {predictions}")
    if trades:
        logger.info(f"\n  {'#':>3} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'SL':>10} {'P&L':>8} {'Reason':<12}")
        logger.info(f"  {'-'*62}")
        for i, t in enumerate(trades, 1):
            icon = "✅" if t['pnl'] > 0 else "❌"
            logger.info(
                f"  {i:>3} {t['dir']:<6} {t['entry']:>10.2f} {t['exit']:>10.2f} "
                f"{t['sl']:>10.2f} {t['pnl']:>+7.2f}{icon} {t['reason']:<12}"
            )
        logger.info(f"  {'-'*62}")
        logger.info(f"  {'':>3} {'':>6} {'':>10} {'':>10} {'TOTAL':>10} {daily_pnl:>+7.2f}")
    logger.info(f"{'='*70}")


def run_catboost_engine():
    """Wrapper to run async CatBoost engine in a new event loop."""
    asyncio.run(catboost_signal_engine())


if __name__ == "__main__":
    run_catboost_engine()
