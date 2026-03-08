"""
catboost_live_engine.py — Live CatBoost ML Signal Engine for NIFTY

Replaces UT Bot signal engine. Loads trained CatBoost model and generates
BUY/SELL signals from live 1-min candle data via Redis.

Architecture:
  Redis HISTORY:NIFTY → build features (incremental) → CatBoost predict → Redis pub/sub

Same Redis interface as UT Bot engine — pos_handle_wts.py works unchanged.
"""

import asyncio
import datetime
import time
import os
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

# ─────────── Config ───────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "")

NIFTY_SYMBOL = "NIFTY"
MODEL_PATH = os.environ.get("CATBOOST_MODEL", "catboost_nifty_model.cbm")

ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = float(os.environ.get("MIN_ATR", "6.5"))
SIGNAL_COOLDOWN = 2  # seconds between signals (fast for ML)

# Trading window (configurable via env)
WINDOW_START = datetime.time(
    int(os.environ.get("WINDOW_START_H", "14")),
    int(os.environ.get("WINDOW_START_M", "0"))
)
WINDOW_END = datetime.time(
    int(os.environ.get("WINDOW_END_H", "15")),
    int(os.environ.get("WINDOW_END_M", "3"))
)

INDIA_TZ = pytz.timezone("Asia/Kolkata")

# Feature column names (must match training order exactly)
FEATURE_COLS_1M = [
    'atr_1m', 'rsi_1m', 'ut_dir_1m', 'close_vs_trail_1m',
    'mom_3', 'mom_5', 'mom_10',
    'body_1m', 'body_pct_1m', 'upper_wick_1m', 'lower_wick_1m', 'range_1m',
    'std_5', 'std_10',
    'sma_5', 'sma_10', 'sma_20',
    'close_vs_sma5', 'close_vs_sma10', 'sma5_vs_sma10',
    'high_5', 'low_5', 'close_vs_high5', 'close_vs_low5',
]

FEATURE_COLS_2M = [
    'atr_2m', 'rsi_2m', 'ut_dir_2m', 'close_vs_trail_2m',
    'mom_3_2m', 'mom_5_2m', 'range_2m', 'body_2m',
]

ALL_FEATURE_COLS = FEATURE_COLS_1M + FEATURE_COLS_2M


# ─────────── Indicator Functions (same as catboost_strategy.py) ───────────

def calc_rma(series, period):
    """RMA (Wilder's smoothed moving average)."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_atr(df, period=14):
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    return calc_rma(tr, period)


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = calc_rma(gain, period)
    avg_loss = calc_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_ut_bot_direction(close, atr, key_value=1.0):
    """Returns trail_stop and direction arrays."""
    n = len(close)
    trail_stop = np.zeros(n)
    direction = np.zeros(n)
    trail_stop[0] = close[0]
    direction[0] = 1

    for i in range(1, n):
        nloss = atr[i] * key_value
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]

        if prev_dir == 1:
            new_ts = close[i] - nloss
            trail_stop[i] = max(new_ts, prev_ts)
            if close[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close[i] + nloss
            else:
                direction[i] = 1
        else:
            new_ts = close[i] + nloss
            trail_stop[i] = min(new_ts, prev_ts)
            if close[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
            else:
                direction[i] = -1

    return trail_stop, direction


# ─────────── Feature Builder ───────────

def build_all_features(df_1m, df_2m=None):
    """
    Build all 32 features from 1-min (and optional 2-min) DataFrame.
    Returns the feature row for the LAST candle only.
    """
    close = df_1m['Close'].astype(float)
    high = df_1m['High'].astype(float)
    low = df_1m['Low'].astype(float)
    opn = df_1m['Open'].astype(float)

    atr = calc_atr(df_1m, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    trail, dirn = calc_ut_bot_direction(close.values, atr.values, ATR_KEY_VALUE)

    features = {}
    i = len(df_1m) - 1  # last row

    features['atr_1m'] = float(atr.iloc[i])
    features['rsi_1m'] = float(rsi.iloc[i])
    features['ut_dir_1m'] = float(dirn[i])
    features['close_vs_trail_1m'] = float(close.iloc[i] - trail[i])

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

        atr_2m = calc_atr(df_2m, ATR_PERIOD)
        rsi_2m = calc_rsi(close_2m, 14)
        trail_2m, dir_2m = calc_ut_bot_direction(close_2m.values, atr_2m.values, ATR_KEY_VALUE)

        j = len(df_2m) - 1
        features['atr_2m'] = float(atr_2m.iloc[j])
        features['rsi_2m'] = float(rsi_2m.iloc[j])
        features['ut_dir_2m'] = float(dir_2m[j])
        features['close_vs_trail_2m'] = float(close_2m.iloc[j] - trail_2m[j])
        features['mom_3_2m'] = float((close_2m.iloc[j] / close_2m.iloc[max(0, j-3)] - 1) * 100) if j >= 3 else 0
        features['mom_5_2m'] = float((close_2m.iloc[j] / close_2m.iloc[max(0, j-5)] - 1) * 100) if j >= 5 else 0
        features['range_2m'] = float(high_2m.iloc[j] - low_2m.iloc[j])
        features['body_2m'] = float(close_2m.iloc[j] - opn_2m.iloc[j])
    else:
        for col in FEATURE_COLS_2M:
            features[col] = 0.0

    return features, float(atr.iloc[-1]), float(trail[i])


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

    last_signal_time = 0
    last_candle_count = 0
    last_prediction = 0
    prediction_count = 0

    logger.info(f"⚡ CatBoost signal engine started")
    logger.info(f"  Window: {WINDOW_START.strftime('%H:%M')} - {WINDOW_END.strftime('%H:%M')}")
    logger.info(f"  Min ATR: {MIN_ATR}")
    logger.info(f"  Signal cooldown: {SIGNAL_COOLDOWN}s")

    while True:
        try:
            now = datetime.datetime.now(INDIA_TZ)
            now_time = now.time()

            if now_time >= datetime.time(15, 30):
                logger.info("Market closed. Stopping CatBoost engine.")
                break

            if now_time < datetime.time(9, 16):
                await asyncio.sleep(0.5)
                continue

            # Read candle history
            date_key = now.strftime('%Y-%m-%d')
            history_key = f"{REDIS_PREFIX}HISTORY:{NIFTY_SYMBOL}:{date_key}"
            candle_count = await ar.llen(history_key)

            if candle_count < 20:  # Need enough history for features
                await asyncio.sleep(0.1)
                continue

            # Skip if no new candle
            if candle_count == last_candle_count:
                await asyncio.sleep(0.05)  # 50ms poll
                continue

            last_candle_count = candle_count
            t_start = time.perf_counter()

            # Load candles
            history_data = await ar.lrange(history_key, 0, -1)
            candles = [json.loads(x) for x in history_data]
            df_1m = pd.DataFrame(candles)
            df_1m = df_1m.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
            })
            for col in ["Open", "High", "Low", "Close"]:
                df_1m[col] = df_1m[col].astype(float)

            # Resample to 2-min
            df_2m = resample_to_2min(df_1m)

            # Build features
            features, current_atr, trail_stop = build_all_features(df_1m, df_2m)

            # Store ATR in Redis (for pos_handle_wts)
            await ar.set(f"{REDIS_PREFIX}ATR_value", str(round(current_atr, 2)))

            # Store last candle in Redis
            last_row = df_1m.iloc[-1]
            last_candle_data = json.dumps({
                "timestamp": str(last_row.get("timestamp", "")),
                "open": float(last_row["Open"]),
                "high": float(last_row["High"]),
                "low": float(last_row["Low"]),
                "close": float(last_row["Close"]),
            })
            await ar.set(f"{REDIS_PREFIX}last_candle", last_candle_data)

            # Publish candle close
            await ar.publish(f"{REDIS_PREFIX}candle:close", last_candle_data)

            # Build feature vector in correct order
            if model_features and len(model_features) > 0:
                feature_vector = [features.get(f, 0) for f in model_features]
            else:
                feature_vector = [features.get(f, 0) for f in ALL_FEATURE_COLS]

            # Replace NaN/inf
            feature_vector = [0 if (v != v or abs(v) == float('inf')) else v for v in feature_vector]

            # Predict
            pred = int(model.predict([feature_vector])[0])
            prediction_count += 1

            t_elapsed = (time.perf_counter() - t_start) * 1000  # ms

            # Check trading window
            in_window = WINDOW_START <= now_time <= WINDOW_END
            atr_ok = current_atr > MIN_ATR
            now_ts = time.time()

            close_price = float(last_row["Close"])

            # Signal labels
            signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            signal_name = signal_map.get(pred, "HOLD")

            # Log every prediction
            if pred != 0:
                logger.info(
                    f"🤖 CatBoost: {signal_name} | NIFTY={close_price:.2f} | "
                    f"ATR={current_atr:.2f} | Trail={trail_stop:.2f} | "
                    f"Window={'✅' if in_window else '❌'} | "
                    f"Latency={t_elapsed:.1f}ms | #{prediction_count}"
                )

            # Only publish signals within window + ATR ok + cooldown
            if pred != 0 and pred != last_prediction and in_window and atr_ok:
                if now_ts - last_signal_time > SIGNAL_COOLDOWN:
                    signal_data = json.dumps({
                        "signal": "buy" if pred == 1 else "sell",
                        "atr": round(current_atr, 2),
                        "source": "catboost",
                        "prediction": pred,
                        "latency_ms": round(t_elapsed, 1),
                    })

                    if pred == 1:
                        await ar.publish(f"{REDIS_PREFIX}signal:buy", signal_data)
                        await ar.set(f"{REDIS_PREFIX}buy_signal", "true")
                        await ar.set(f"{REDIS_PREFIX}sell_signal", "false")
                        logger.info(f"🟢 BUY SIGNAL PUBLISHED | NIFTY={close_price:.2f} | ATR={current_atr:.2f}")
                    elif pred == -1:
                        await ar.publish(f"{REDIS_PREFIX}signal:sell", signal_data)
                        await ar.set(f"{REDIS_PREFIX}sell_signal", "true")
                        await ar.set(f"{REDIS_PREFIX}buy_signal", "false")
                        logger.info(f"🔴 SELL SIGNAL PUBLISHED | NIFTY={close_price:.2f} | ATR={current_atr:.2f}")

                    last_signal_time = now_ts
                    last_prediction = pred

        except Exception as e:
            logger.error(f"CatBoost engine error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


def run_catboost_engine():
    """Wrapper to run async CatBoost engine in a new event loop."""
    asyncio.run(catboost_signal_engine())


if __name__ == "__main__":
    run_catboost_engine()
