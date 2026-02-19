"""
gap_analyzer.py — Gap-Up / Gap-Down scoring engine.

Runs continuously during market hours. At 3:20 PM, performs full analysis
on all 212 F&O stocks to identify high-probability gap-up and gap-down
candidates for overnight futures trading.

Scoring factors:
  1. Day momentum      — % change from open to current (strength of intraday move)
  2. Last-hour momentum — price change in final hour (3:00-3:20 PM, late conviction)
  3. Close position     — where price sits relative to day range (near high=strong)
  4. RSI level          — RSI > 70 (overbought → gap-down risk), RSI < 30 (oversold → gap-up)
  5. EMA trend          — EMA(9) vs EMA(21) crossover direction
  6. Volume surge       — last 30 min avg vs day avg (accumulation/distribution signal)

Results stored in Redis:
  SCAN:RANKINGS:{date} — JSON with ranked gap-up and gap-down candidates
"""

import datetime
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pytz
import redis
from logzero import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDIA_TZ = pytz.timezone("Asia/Kolkata")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "Rahul@7355")

r = redis.StrictRedis(
    host=REDIS_HOST, port=REDIS_PORT,
    password=REDIS_PASSWORD, db=0, decode_responses=True
)


# ─── Indicator helpers ───────────────────────────────────────────────

def calc_ema(values: list, period: int) -> list:
    """Calculate EMA."""
    if len(values) < period:
        return [None] * len(values)
    ema = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema.append(sma)
    k = 2 / (period + 1)
    for i in range(period, len(values)):
        ema.append(values[i] * k + ema[-1] * (1 - k))
    return ema


def calc_rsi(values: list, period: int = 14) -> float:
    """Calculate latest RSI value."""
    if len(values) < period + 1:
        return 50.0  # neutral default

    gains, losses = [], []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    if len(gains) < period:
        return 50.0

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_vwap(candles_df: pd.DataFrame) -> float:
    """Simple VWAP approximation using candle data."""
    if candles_df.empty or 'volume' not in candles_df.columns:
        return 0
    typical = (candles_df['high'] + candles_df['low'] + candles_df['close']) / 3
    vol = candles_df['volume'].replace(0, 1)  # avoid division by zero
    return (typical * vol).sum() / vol.sum()


# ─── Core analysis ───────────────────────────────────────────────────

def load_all_symbols() -> tuple:
    """Load all stock symbols and token map from scanner_all_tokens.csv."""
    csv_path = os.path.join(BASE_DIR, "scanner_all_tokens.csv")
    if not os.path.exists(csv_path):
        logger.error("scanner_all_tokens.csv not found. Run stock_scanner_setup.py first.")
        return [], {}
    df = pd.read_csv(csv_path)
    symbols = df['pSymbolName'].tolist()
    token_map = dict(zip(df['pSymbolName'], df['pSymbol'].astype(str)))
    return symbols, token_map


def get_candle_df(symbol: str, date_key: str) -> pd.DataFrame:
    """Retrieve candle history from Redis and return as DataFrame."""
    history_key = f"SCAN:HISTORY:{symbol}:{date_key}"
    raw_candles = r.lrange(history_key, 0, -1)

    if not raw_candles:
        return pd.DataFrame()

    candles = []
    for raw in raw_candles:
        try:
            c = json.loads(raw)
            candles.append({
                'timestamp': c.get('timestamp', ''),
                'open': float(c.get('open', 0)),
                'high': float(c.get('high', 0)),
                'low': float(c.get('low', 0)),
                'close': float(c.get('close', 0)),
                'volume': float(c.get('volume', 0)),
            })
        except (json.JSONDecodeError, ValueError):
            continue

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def score_stock(symbol: str, date_key: str) -> dict:
    """
    Analyze a single stock and return gap-up / gap-down scores.

    Returns:
        dict with symbol, ltp, change_pct, gap_up_score, gap_down_score,
        rsi, ema_trend, close_position, reasons, etc.
    """
    result = {
        'symbol': symbol,
        'token': '',
        'ltp': 0,
        'day_open': 0,
        'day_high': 0,
        'day_low': 0,
        'change_pct': 0,
        'gap_up_score': 0,
        'gap_down_score': 0,
        'rsi': 50,
        'ema_trend': 'neutral',
        'close_position': 50,
        'last_hour_pct': 0,
        'volume_surge': 1.0,
        'num_candles': 0,
        'reasons': [],
    }

    # Get LTP
    ltp = r.get(f"SCAN:LTP:{symbol}")
    if not ltp:
        return result
    result['ltp'] = float(ltp)

    # Get day open
    day_open = r.get(f"SCAN:OPEN:{symbol}:{date_key}")
    if day_open:
        result['day_open'] = float(day_open)

    # Get day high / low
    day_high = r.get(f"SCAN:DAY_HIGH:{symbol}:{date_key}")
    day_low = r.get(f"SCAN:DAY_LOW:{symbol}:{date_key}")
    if day_high:
        day_high = float(day_high)
    else:
        day_high = result['ltp']
    if day_low:
        day_low = float(day_low)
    else:
        day_low = result['ltp']

    result['day_high'] = day_high
    result['day_low'] = day_low

    # Get candle data
    df = get_candle_df(symbol, date_key)
    if df.empty or len(df) < 5:
        return result

    result['num_candles'] = len(df)

    closes = df['close'].tolist()
    current_price = result['ltp']

    # ── Factor 1: Day momentum ──
    if result['day_open'] > 0:
        result['change_pct'] = ((current_price - result['day_open']) / result['day_open']) * 100
    day_momentum = result['change_pct']

    # ── Factor 2: Last-hour momentum ──
    # Get candles from the last 60 minutes
    if len(df) >= 12:  # 12 x 5-min = 1 hour
        last_hour = df.tail(12)
        first_close = last_hour.iloc[0]['close']
        last_close = last_hour.iloc[-1]['close']
        if first_close > 0:
            result['last_hour_pct'] = ((last_close - first_close) / first_close) * 100

    # ── Factor 3: Close position (0=at day low, 100=at day high) ──
    day_range = day_high - day_low
    if day_range > 0:
        result['close_position'] = ((current_price - day_low) / day_range) * 100
    else:
        result['close_position'] = 50

    # ── Factor 4: RSI ──
    result['rsi'] = calc_rsi(closes, 14)

    # ── Factor 5: EMA trend ──
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    if ema9[-1] is not None and ema21[-1] is not None:
        if ema9[-1] > ema21[-1]:
            result['ema_trend'] = 'bullish'
        elif ema9[-1] < ema21[-1]:
            result['ema_trend'] = 'bearish'

    # ── Factor 6: Volume surge ──
    if 'volume' in df.columns and len(df) >= 6:
        day_avg_vol = df['volume'].mean()
        last30_avg = df.tail(6)['volume'].mean()  # 6 x 5-min = 30 min
        if day_avg_vol > 0:
            result['volume_surge'] = last30_avg / day_avg_vol

    # ─────────────────── SCORING ───────────────────

    gap_up = 0
    gap_down = 0
    reasons_up = []
    reasons_down = []

    # 1. Day momentum (max ±25 points)
    momentum_score = min(abs(day_momentum) * 5, 25)
    if day_momentum > 0.5:
        gap_up += momentum_score
        reasons_up.append(f"Strong day +{day_momentum:.1f}%")
    elif day_momentum < -0.5:
        gap_down += momentum_score
        reasons_down.append(f"Weak day {day_momentum:.1f}%")

    # 2. Last-hour momentum (max ±20 points) — critical for gap prediction
    lh = result['last_hour_pct']
    lh_score = min(abs(lh) * 8, 20)
    if lh > 0.3:
        gap_up += lh_score
        reasons_up.append(f"Last hr push +{lh:.2f}%")
    elif lh < -0.3:
        gap_down += lh_score
        reasons_down.append(f"Last hr sell {lh:.2f}%")

    # 3. Close position (max ±15 points)
    cp = result['close_position']
    if cp > 75:
        gap_up += min((cp - 50) * 0.3, 15)
        reasons_up.append(f"Close near high ({cp:.0f}%)")
    elif cp < 25:
        gap_down += min((50 - cp) * 0.3, 15)
        reasons_down.append(f"Close near low ({cp:.0f}%)")

    # 4. RSI (max ±15 points)
    rsi = result['rsi']
    if rsi < 35:
        gap_up += min((40 - rsi) * 0.75, 15)
        reasons_up.append(f"Oversold RSI={rsi:.0f}")
    elif rsi > 65:
        gap_down += min((rsi - 60) * 0.75, 15)
        reasons_down.append(f"Overbought RSI={rsi:.0f}")

    # 5. EMA trend (max ±10 points)
    if result['ema_trend'] == 'bullish':
        gap_up += 10
        reasons_up.append("EMA bullish cross")
    elif result['ema_trend'] == 'bearish':
        gap_down += 10
        reasons_down.append("EMA bearish cross")

    # 6. Volume surge (max ±15 points)
    vs = result['volume_surge']
    if vs > 1.5:
        vol_pts = min((vs - 1) * 10, 15)
        # Volume with direction
        if day_momentum > 0:
            gap_up += vol_pts
            reasons_up.append(f"Volume surge {vs:.1f}x (buying)")
        else:
            gap_down += vol_pts
            reasons_down.append(f"Volume surge {vs:.1f}x (selling)")

    result['gap_up_score'] = round(min(gap_up, 100), 1)
    result['gap_down_score'] = round(min(gap_down, 100), 1)
    result['reasons'] = reasons_up if gap_up >= gap_down else reasons_down

    return result


def run_analysis():
    """Run full gap analysis on all stocks."""
    date_key = datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d')
    symbols, token_map = load_all_symbols()

    if not symbols:
        logger.error("No symbols to analyze.")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"GAP ANALYSIS — {date_key} — Analyzing {len(symbols)} stocks")
    logger.info(f"{'=' * 60}")

    results = []
    for i, symbol in enumerate(symbols):
        try:
            score = score_stock(symbol, date_key)
            score['token'] = token_map.get(symbol, '')
            if score['ltp'] > 0:  # Only include stocks with live data
                results.append(score)
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")

        if (i + 1) % 50 == 0:
            logger.info(f"  Scored {i + 1}/{len(symbols)} stocks...")

    logger.info(f"Scored {len(results)} stocks with live data.")

    # Rank gap-up candidates
    gap_up_sorted = sorted(results, key=lambda x: x['gap_up_score'], reverse=True)
    top_gap_up = [
        {
            'symbol': s['symbol'],
            'score': s['gap_up_score'],
            'ltp': s['ltp'],
            'change_pct': round(s['change_pct'], 2),
            'rsi': round(s['rsi'], 1),
            'close_position': round(s['close_position'], 1),
            'last_hour_pct': round(s['last_hour_pct'], 2),
            'volume_surge': round(s['volume_surge'], 2),
            'ema_trend': s['ema_trend'],
            'reason': ' | '.join(s['reasons']),
        }
        for s in gap_up_sorted[:20]
        if s['gap_up_score'] >= 15
    ]

    # Rank gap-down candidates
    gap_down_sorted = sorted(results, key=lambda x: x['gap_down_score'], reverse=True)
    top_gap_down = [
        {
            'symbol': s['symbol'],
            'score': s['gap_down_score'],
            'ltp': s['ltp'],
            'change_pct': round(s['change_pct'], 2),
            'rsi': round(s['rsi'], 1),
            'close_position': round(s['close_position'], 1),
            'last_hour_pct': round(s['last_hour_pct'], 2),
            'volume_surge': round(s['volume_surge'], 2),
            'ema_trend': s['ema_trend'],
            'reason': ' | '.join(s['reasons']),
        }
        for s in gap_down_sorted[:20]
        if s['gap_down_score'] >= 15
    ]

    # Build rankings output
    now_str = datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M:%S')
    rankings = {
        'timestamp': now_str,
        'total_stocks_analyzed': len(results),
        'gap_up': top_gap_up,
        'gap_down': top_gap_down,
    }

    # Store in Redis
    rankings_key = f"SCAN:RANKINGS:{date_key}"
    r.set(rankings_key, json.dumps(rankings))
    r.expire(rankings_key, 86400)  # 24h TTL

    # Log results
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📈 TOP GAP-UP CANDIDATES:")
    logger.info(f"{'=' * 60}")
    for i, s in enumerate(top_gap_up[:10], 1):
        logger.info(
            f"  {i:2d}. {s['symbol']:15s} Score={s['score']:5.1f}  "
            f"LTP=₹{s['ltp']:.2f}  Chg={s['change_pct']:+.2f}%  "
            f"RSI={s['rsi']:.0f}  {s['reason']}"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📉 TOP GAP-DOWN CANDIDATES:")
    logger.info(f"{'=' * 60}")
    for i, s in enumerate(top_gap_down[:10], 1):
        logger.info(
            f"  {i:2d}. {s['symbol']:15s} Score={s['score']:5.1f}  "
            f"LTP=₹{s['ltp']:.2f}  Chg={s['change_pct']:+.2f}%  "
            f"RSI={s['rsi']:.0f}  {s['reason']}"
        )

    # ─── Save final CSV with all stocks ───
    save_results_csv(results, date_key)

    return rankings


def save_results_csv(results: list, date_key: str):
    """Save all scored stocks to a CSV file sorted by best gap signal."""
    if not results:
        return

    rows = []
    for s in results:
        # Determine primary signal direction
        if s['gap_up_score'] > s['gap_down_score']:
            signal = 'GAP_UP'
            best_score = s['gap_up_score']
        elif s['gap_down_score'] > s['gap_up_score']:
            signal = 'GAP_DOWN'
            best_score = s['gap_down_score']
        else:
            signal = 'NEUTRAL'
            best_score = 0

        rows.append({
            'Token': s.get('token', ''),
            'Symbol': s['symbol'],
            'Signal': signal,
            'Score': round(best_score, 1),
            'Gap_Up_Score': round(s['gap_up_score'], 1),
            'Gap_Down_Score': round(s['gap_down_score'], 1),
            'LTP': round(s['ltp'], 2),
            'Day_Open': round(s['day_open'], 2),
            'Day_High': round(s.get('day_high', 0), 2),
            'Day_Low': round(s.get('day_low', 0), 2),
            'Change_Pct': round(s['change_pct'], 2),
            'Last_Hour_Pct': round(s['last_hour_pct'], 2),
            'RSI': round(s['rsi'], 1),
            'EMA_Trend': s['ema_trend'],
            'Close_Position': round(s['close_position'], 1),
            'Volume_Surge': round(s['volume_surge'], 2),
            'Num_Candles': s.get('num_candles', 0),
            'Reason': ' | '.join(s.get('reasons', [])),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)

    csv_filename = f"scanner_results_{date_key}.csv"
    csv_path = os.path.join(BASE_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    logger.info(f"\n📄 Saved results CSV: {csv_filename} ({len(df)} stocks)")


def main():
    """Main loop — run analysis at 3:20 PM (and every minute until 3:25 for updates)."""
    logger.info("Gap Analyzer started — waiting for 3:20 PM analysis window...")

    analysis_done = False

    while True:
        now = datetime.datetime.now(INDIA_TZ)
        now_time = now.time()

        # Stop after market close
        if now_time > datetime.time(15, 35):
            logger.info("Market closed. Final analysis complete.")
            break

        # Analysis window: 3:15 PM to 3:25 PM
        if datetime.time(15, 15) <= now_time <= datetime.time(15, 25):
            logger.info(f"Analysis window active — running at {now.strftime('%H:%M:%S')}")
            run_analysis()

            if now_time >= datetime.time(15, 20):
                analysis_done = True

            # Update every 2 minutes during the window
            time.sleep(120)
        elif analysis_done:
            # Post-analysis: update once more at 3:28
            if datetime.time(15, 27) <= now_time <= datetime.time(15, 29):
                logger.info("Final update at 3:28 PM...")
                run_analysis()
                time.sleep(300)  # sleep until market close
            else:
                time.sleep(30)
        else:
            # Before 3:15 — periodic health check
            if now_time >= datetime.time(9, 30) and now.minute % 30 == 0 and now.second < 5:
                symbols, _ = load_all_symbols()
                date_key = now.strftime('%Y-%m-%d')
                live_count = 0
                for s in symbols:
                    if r.exists(f"SCAN:LTP:{s}"):
                        live_count += 1
                logger.info(f"Health check: {live_count}/{len(symbols)} stocks have live data")

            time.sleep(5)


if __name__ == "__main__":
    main()
