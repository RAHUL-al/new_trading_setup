"""
stock_scanner_strategy.py — Scan NIFTY 50 → Pick Top Stocks → Backtest

STRATEGY v2: "Pullback in Trend + ORB" (Higher Win Rate Design)

The fundamental problem with crossover strategies: they're LAGGING.
By the time EMAs cross, the move already happened → low win rate.

THIS strategy is different — it trades pullbacks within established trends:

MORNING (9:15 - 10:30): OPENING RANGE BREAKOUT (ORB)
  1. Wait 15 min → define Opening Range (OR high/low)
  2. If price breaks above OR high:
     + Volume surge (1.5x avg)
     + EMA(21) trending up
     → BUY (SL = OR low or ATR-based)
  3. Mirror for sells
  Win rate boost: ORB trades WITH first momentum, not against it.

AFTERNOON (1:00 - 3:15): VWAP PULLBACK
  1. Identify strong trend (EMA 5 > EMA 21 by enough gap)
  2. Wait for price to pull back to VWAP or EMA(21)
  3. Enter when price bounces off VWAP with:
     + Volume surge on bounce candle
     + RSI not extreme
     + Strong bullish/bearish candle body
  Win rate boost: Buying at support in uptrend = high probability.

KEY DIFFERENCE: This doesn't chase crossovers —
               it WAITS for pullbacks in trends.

Usage:
    python stock_scanner_strategy.py
    python stock_scanner_strategy.py --stock RELIANCE --detailed
    python stock_scanner_strategy.py --top 5
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
DATA_DIR = "stock_data"
RESULTS_DIR = "scan_results"

ORB_MINUTES = 15                   # First 15 min for Opening Range
WINDOW_1_START = dt_time(9, 15)
WINDOW_1_END = dt_time(10, 30)
WINDOW_2_START = dt_time(13, 0)
WINDOW_2_END = dt_time(15, 15)
SQUARE_OFF_TIME = dt_time(15, 24)

EMA_FAST = 5
EMA_SLOW = 21

SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.0              # Tighter trail for stocks
PARTIAL_EXIT_RR = 1.0             # Take 50% profit at 1:1 R:R
MIN_HOLD = 2
BODY_RATIO_MIN = 0.40
VOL_SURGE_MULT = 1.3              # Volume > 1.3x avg (lowered from 1.5)
VWAP_TOLERANCE = 0.002            # 0.2% tolerance for VWAP touch


# ─────────── Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    initial_risk: float
    entry_idx: int
    entry_atr: float
    partial_taken: bool = False

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_pts: float
    pnl_pct: float
    close_reason: str
    signal_type: str              # "ORB" or "VWAP_PB"

@dataclass
class StockScore:
    symbol: str
    total_candles: int
    trading_days: int
    avg_volume: float
    avg_atr: float
    atr_pct: float
    trend_score: float
    volume_score: float
    signal_score: float
    liquidity_score: float
    total_score: float
    total_trades: int
    win_rate: float
    total_pnl_pct: float
    profit_factor: float
    orb_trades: int
    orb_win_rate: float
    vwap_trades: int
    vwap_win_rate: float


# ─────────── Indicators ───────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr.ewm(alpha=1/period, adjust=False).mean()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))


def calc_vwap(df):
    """Intraday VWAP, resets daily."""
    df = df.copy()
    df['date'] = df['Time'].dt.date
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (tp * df['Volume']).groupby(df['date']).cumsum()
    cum_vol = df['Volume'].groupby(df['date']).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ─────────── Opening Range Detection ───────────

def get_opening_range(df):
    """
    For each day, compute the high and low of the first ORB_MINUTES candles.
    Returns: or_high (Series), or_low (Series), or_defined (bool Series)
    """
    df = df.copy()
    df['date'] = df['Time'].dt.date

    or_high = pd.Series(np.nan, index=df.index)
    or_low = pd.Series(np.nan, index=df.index)
    or_defined = pd.Series(False, index=df.index)

    for date, group in df.groupby('date'):
        # First candles that form the opening range
        market_start = datetime.combine(date, WINDOW_1_START)

        # Determine interval from data
        if len(group) >= 2:
            interval_mins = (group['Time'].iloc[1] - group['Time'].iloc[0]).total_seconds() / 60
        else:
            interval_mins = 3

        candles_needed = max(1, int(ORB_MINUTES / interval_mins))
        or_candles = group.head(candles_needed)

        if len(or_candles) > 0:
            high = or_candles['High'].max()
            low = or_candles['Low'].min()

            # Set OR values for all candles AFTER the opening range
            remaining = group.iloc[candles_needed:]
            for idx in remaining.index:
                or_high.loc[idx] = high
                or_low.loc[idx] = low
                or_defined.loc[idx] = True

    return or_high, or_low, or_defined


# ─────────── Signal Generation ───────────

def generate_signals(df):
    """
    Two signal types:
    1. ORB — Opening Range Breakout (morning window)
    2. VWAP_PB — VWAP Pullback in trend (afternoon window)
    """
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    n = len(df)

    # Indicators
    ema_fast = calc_ema(close, EMA_FAST)
    ema_slow = calc_ema(close, EMA_SLOW)
    atr_val = calc_atr(df)
    rsi_val = calc_rsi(close)
    vwap_val = calc_vwap(df)
    obv_val = calc_obv(close, volume)
    obv_ema = calc_ema(obv_val, 10)
    vol_avg = volume.rolling(20).mean()

    # Opening Range
    or_high, or_low, or_defined = get_opening_range(df)

    # Candle quality
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    body_ratio = body / full_range

    # Trend strength
    ema_gap_pct = ((ema_fast - ema_slow) / close * 100).abs()

    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    signal_type = [''] * n

    for i in range(5, n):
        t = df.iloc[i]['Time'].time() if hasattr(df.iloc[i]['Time'], 'time') else None
        if t is None:
            continue

        curr_atr = atr_val.iloc[i]
        curr_vol = volume.iloc[i]
        avg_vol = vol_avg.iloc[i]
        r = rsi_val.iloc[i]

        if np.isnan(curr_atr) or np.isnan(avg_vol) or avg_vol == 0:
            continue

        # Dynamic ATR gate
        if curr_atr < atr_val.rolling(100).quantile(0.2).iloc[i]:
            continue

        vol_surge = curr_vol >= (avg_vol * VOL_SURGE_MULT)
        br = body_ratio.iloc[i]
        if np.isnan(br) or br < BODY_RATIO_MIN:
            continue

        # ─── MORNING: ORB SIGNALS (9:30 - 10:30) ───
        if dt_time(9, 30) <= t <= WINDOW_1_END and or_defined.iloc[i]:
            orh = or_high.iloc[i]
            orl = or_low.iloc[i]

            if not np.isnan(orh) and not np.isnan(orl):
                # BUY: Close breaks above OR high
                if close.iloc[i] > orh and close.iloc[i] > open_.iloc[i]:
                    if vol_surge:
                        if ema_fast.iloc[i] > ema_slow.iloc[i]:  # Trend aligned
                            if np.isnan(r) or r < 75:
                                buy_signal[i] = True
                                signal_type[i] = 'ORB'

                # SELL: Close breaks below OR low
                if close.iloc[i] < orl and close.iloc[i] < open_.iloc[i]:
                    if vol_surge:
                        if ema_fast.iloc[i] < ema_slow.iloc[i]:
                            if np.isnan(r) or r > 25:
                                sell_signal[i] = True
                                signal_type[i] = 'ORB'

        # ─── AFTERNOON: VWAP PULLBACK SIGNALS (1:00 - 3:15) ───
        if WINDOW_2_START <= t <= WINDOW_2_END:
            curr_vwap = vwap_val.iloc[i] if i < len(vwap_val) else np.nan
            if np.isnan(curr_vwap):
                continue

            # Need established trend (EMA gap > 0.05%)
            gap = ema_gap_pct.iloc[i]
            if np.isnan(gap) or gap < 0.05:
                continue

            # Check OBV
            obv_up = obv_val.iloc[i] > obv_ema.iloc[i] if not np.isnan(obv_val.iloc[i]) else False
            obv_down = obv_val.iloc[i] < obv_ema.iloc[i] if not np.isnan(obv_val.iloc[i]) else False

            vwap_dist = (close.iloc[i] - curr_vwap) / curr_vwap

            # BUY: Uptrend + price pulls back near VWAP + bounces
            if ema_fast.iloc[i] > ema_slow.iloc[i]:
                # Price was near or touched VWAP recently (within last 3 candles)
                touched_vwap = False
                for lb in range(1, 4):
                    j = i - lb
                    if j >= 0 and not np.isnan(vwap_val.iloc[j]):
                        if low.iloc[j] <= vwap_val.iloc[j] * (1 + VWAP_TOLERANCE):
                            touched_vwap = True
                            break

                if touched_vwap and close.iloc[i] > curr_vwap:  # Now above VWAP = bounce
                    if close.iloc[i] > open_.iloc[i] and vol_surge:  # Bullish + volume
                        if (np.isnan(r) or r < 70) and obv_up:
                            buy_signal[i] = True
                            signal_type[i] = 'VWAP_PB'

            # SELL: Downtrend + price pulls back to VWAP + drops
            if ema_fast.iloc[i] < ema_slow.iloc[i]:
                touched_vwap = False
                for lb in range(1, 4):
                    j = i - lb
                    if j >= 0 and not np.isnan(vwap_val.iloc[j]):
                        if high.iloc[j] >= vwap_val.iloc[j] * (1 - VWAP_TOLERANCE):
                            touched_vwap = True
                            break

                if touched_vwap and close.iloc[i] < curr_vwap:
                    if close.iloc[i] < open_.iloc[i] and vol_surge:
                        if (np.isnan(r) or r > 30) and obv_down:
                            sell_signal[i] = True
                            signal_type[i] = 'VWAP_PB'

    return buy_signal, sell_signal, signal_type, atr_val, or_high, or_low


# ─────────── Backtest ───────────

def run_backtest(df, buy_sig, sell_sig, sig_type, atr_val, or_high, or_low):
    trades = []
    pos = None
    prev_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time()
        curr_date = row['Time'].date()
        close = float(row['Close'])
        high_v = float(row['High'])
        low_v = float(row['Low'])
        curr_atr = float(atr_val.iloc[i])

        # Day reset
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(df.iloc[i-1]['Close'])
                pnl = _pnl(pos, prev_close)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, prev_close,
                                    pos.entry_time, df.iloc[i-1]['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "DAY_END",
                                    "ORB" if pos.entry_time.time() < WINDOW_2_START else "VWAP_PB"))
                pos = None
        prev_date = curr_date

        # Window 1 close at 10:30
        if pos and t > WINDOW_1_END and t < WINDOW_2_START:
            if pos.entry_time.time() <= WINDOW_1_END:
                pnl = _pnl(pos, close)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, close,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "W1_CLOSE", "ORB"))
                pos = None
            continue

        # Square off
        if pos and t >= SQUARE_OFF_TIME:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "SQUARE_OFF",
                                "ORB" if pos.entry_time.time() < WINDOW_2_START else "VWAP_PB"))
            pos = None
            continue

        # Skip outside windows
        in_w1 = WINDOW_1_START <= t <= WINDOW_1_END
        in_w2 = WINDOW_2_START <= t <= SQUARE_OFF_TIME
        if not in_w1 and not in_w2:
            continue

        # SL check
        if pos:
            sl_hit = False
            sl_price = pos.stop_loss
            if pos.direction == "LONG" and low_v <= pos.stop_loss:
                sl_hit = True
            elif pos.direction == "SHORT" and high_v >= pos.stop_loss:
                sl_hit = True

            if sl_hit:
                pnl = _pnl(pos, sl_price)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, sl_price,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "STOP_LOSS",
                                    "ORB" if pos.entry_time.time() < WINDOW_2_START else "VWAP_PB"))
                pos = None

        # Trail SL
        if pos and curr_atr > 0:
            if pos.direction == "LONG":
                new_sl = high_v - curr_atr * TRAIL_ATR_MULT
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
            elif pos.direction == "SHORT":
                new_sl = low_v + curr_atr * TRAIL_ATR_MULT
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl

        # Opposite signal
        is_buy = bool(buy_sig[i])
        is_sell = bool(sell_sig[i])

        if is_buy and pos and pos.direction == "SHORT" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "OPPOSITE",
                                "ORB" if pos.entry_time.time() < WINDOW_2_START else "VWAP_PB"))
            pos = None
        elif is_sell and pos and pos.direction == "LONG" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "OPPOSITE",
                                "ORB" if pos.entry_time.time() < WINDOW_2_START else "VWAP_PB"))
            pos = None

        # New entry
        can_enter = (in_w1 and t >= dt_time(9, 30)) or (in_w2 and t <= WINDOW_2_END)
        if not pos and can_enter:
            if is_buy and curr_atr > 0:
                # For ORB: SL = OR low; For VWAP_PB: SL = ATR-based
                if sig_type[i] == 'ORB' and not np.isnan(or_low.iloc[i]):
                    sl = or_low.iloc[i] - curr_atr * 0.3  # OR low - buffer
                else:
                    sl = close - curr_atr * SL_ATR_MULT
                risk = close - sl
                pos = Position("LONG", close, row['Time'], sl, risk, i, curr_atr)
            elif is_sell and curr_atr > 0:
                if sig_type[i] == 'ORB' and not np.isnan(or_high.iloc[i]):
                    sl = or_high.iloc[i] + curr_atr * 0.3
                else:
                    sl = close + curr_atr * SL_ATR_MULT
                risk = sl - close
                pos = Position("SHORT", close, row['Time'], sl, risk, i, curr_atr)

    return trades


def _pnl(pos, exit_price):
    return (exit_price - pos.entry_price) if pos.direction == "LONG" else (pos.entry_price - exit_price)


# ─────────── Scoring ───────────

def score_stock(symbol, df, trades):
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    # Trend score
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    ema_dir = (ema20 > ema50).astype(int)
    runs = (ema_dir != ema_dir.shift(1)).cumsum()
    avg_run = ema_dir.groupby(runs).count().mean()
    trend_score = min(avg_run / 50 * 100, 100)

    # Volume score
    vol_cv = volume.std() / volume.mean() if volume.mean() > 0 else 10
    vol_nonzero = (volume > 0).mean()
    volume_score = min(vol_nonzero * 60 + (1 - min(vol_cv, 3) / 3) * 40, 100)

    # Signal/Win rate score
    win_rate = 0
    if trades:
        win_rate = sum(1 for t in trades if t.pnl_pts > 0) / len(trades) * 100
    signal_score = min(win_rate * 1.5, 100)

    # Liquidity
    avg_turnover = volume.mean() * close.mean()
    liquidity_score = min(avg_turnover / 1e7 * 100, 100)

    # ATR %
    atr_val = calc_atr(df)
    atr_pct = (atr_val / close * 100).mean()

    total_score = trend_score * 0.20 + volume_score * 0.20 + signal_score * 0.40 + liquidity_score * 0.20

    # Backtest metrics
    pf = 0
    if trades:
        wins = [t for t in trades if t.pnl_pts > 0]
        losses = [t for t in trades if t.pnl_pts <= 0]
        gp = sum(t.pnl_pts for t in wins) if wins else 0
        gl = abs(sum(t.pnl_pts for t in losses)) if losses else 1
        pf = gp / gl if gl > 0 else 0

    # ORB vs VWAP breakdown
    orb_trades = [t for t in trades if t.signal_type == 'ORB']
    vwap_trades = [t for t in trades if t.signal_type == 'VWAP_PB']
    orb_wr = sum(1 for t in orb_trades if t.pnl_pts > 0) / max(len(orb_trades), 1) * 100
    vwap_wr = sum(1 for t in vwap_trades if t.pnl_pts > 0) / max(len(vwap_trades), 1) * 100

    return StockScore(
        symbol=symbol, total_candles=len(df),
        trading_days=df['Time'].dt.date.nunique(),
        avg_volume=volume.mean(), avg_atr=atr_val.mean(),
        atr_pct=round(atr_pct, 3),
        trend_score=round(trend_score, 1), volume_score=round(volume_score, 1),
        signal_score=round(signal_score, 1), liquidity_score=round(liquidity_score, 1),
        total_score=round(total_score, 1),
        total_trades=len(trades), win_rate=round(win_rate, 1),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 3) if trades else 0,
        profit_factor=round(pf, 2),
        orb_trades=len(orb_trades), orb_win_rate=round(orb_wr, 1),
        vwap_trades=len(vwap_trades), vwap_win_rate=round(vwap_wr, 1),
    )


# ─────────── Reports ───────────

def print_stock_report(symbol, trades, total_candles, trading_days, date_range):
    if not trades:
        print(f"  {symbol}: No trades")
        return

    pnl_pts = [t.pnl_pts for t in trades]
    pnl_pct = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]
    wr = len(wins) / len(trades) * 100

    orb = [t for t in trades if t.signal_type == 'ORB']
    vwap = [t for t in trades if t.signal_type == 'VWAP_PB']

    daily = {}
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d")
        daily[d] = daily.get(d, 0) + t.pnl_pts
    prof_days = sum(1 for v in daily.values() if v > 0)

    gp = sum(t.pnl_pts for t in wins) if wins else 0
    gl = abs(sum(t.pnl_pts for t in losses)) if losses else 1
    pf = gp / gl if gl > 0 else 0

    print(f"\n  {'─'*55}")
    print(f"  📈 {symbol} | {date_range}")
    print(f"  {trading_days} days | {total_candles} candles | {len(trades)} trades")
    print(f"  P&L: {sum(pnl_pts):+.2f} pts ({sum(pnl_pct):+.3f}%)")
    print(f"  Win: {wr:.1f}% | PF: {pf:.2f}")
    print(f"  Prof days: {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    if orb:
        orb_wr = sum(1 for t in orb if t.pnl_pts > 0) / len(orb) * 100
        orb_pnl = sum(t.pnl_pct for t in orb)
        print(f"  ORB:     {len(orb):3d} trades | Win: {orb_wr:.0f}% | P&L: {orb_pnl:+.3f}%")
    if vwap:
        vwap_wr = sum(1 for t in vwap if t.pnl_pts > 0) / len(vwap) * 100
        vwap_pnl = sum(t.pnl_pct for t in vwap)
        print(f"  VWAP PB: {len(vwap):3d} trades | Win: {vwap_wr:.0f}% | P&L: {vwap_pnl:+.3f}%")

    reasons = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
    print(f"  Reasons: {' | '.join(f'{r}:{c}' for r, c in sorted(reasons.items(), key=lambda x: -x[1]))}")


def print_ranking(scores, top_n=10):
    sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)

    print(f"\n{'='*110}")
    print(f"  STOCK SCANNER v2 — ORB + VWAP Pullback Strategy | Top {min(top_n, len(sorted_scores))}")
    print(f"{'='*110}")
    print(f"  {'Rank':>4} {'Symbol':>12} {'Score':>6} {'Trades':>7} {'Win%':>6} {'P&L%':>8} {'PF':>5} {'ORB':>5} {'ORB%':>5} {'VWAP':>5} {'VW%':>5} {'Trend':>6} {'Vol':>5}")
    print(f"  {'-'*105}")

    for rank, s in enumerate(sorted_scores[:top_n], 1):
        star = "⭐" if rank <= 5 else "  "
        print(f"  {rank:>3}. {s.symbol:>12} {s.total_score:>6.1f} {s.total_trades:>7} {s.win_rate:>5.1f}% {s.total_pnl_pct:>+7.2f}% {s.profit_factor:>5.2f} {s.orb_trades:>5} {s.orb_win_rate:>4.0f}% {s.vwap_trades:>5} {s.vwap_win_rate:>4.0f}% {s.trend_score:>5.1f} {s.volume_score:>4.1f} {star}")

    print(f"\n{'='*110}")

    # Best per category
    print(f"\n  🏆 HIGHLIGHTS")
    best_wr = max(sorted_scores, key=lambda s: s.win_rate)
    best_pnl = max(sorted_scores, key=lambda s: s.total_pnl_pct)
    best_pf = max(sorted_scores, key=lambda s: s.profit_factor)
    best_orb = max(sorted_scores, key=lambda s: s.orb_win_rate if s.orb_trades >= 5 else 0)
    best_vwap = max(sorted_scores, key=lambda s: s.vwap_win_rate if s.vwap_trades >= 5 else 0)
    print(f"  Best Win Rate: {best_wr.symbol:>12} ({best_wr.win_rate:.1f}%)")
    print(f"  Best P&L:      {best_pnl.symbol:>12} ({best_pnl.total_pnl_pct:+.2f}%)")
    print(f"  Best PF:       {best_pf.symbol:>12} ({best_pf.profit_factor:.2f})")
    print(f"  Best ORB:      {best_orb.symbol:>12} ({best_orb.orb_win_rate:.0f}% on {best_orb.orb_trades} trades)")
    print(f"  Best VWAP PB:  {best_vwap.symbol:>12} ({best_vwap.vwap_win_rate:.0f}% on {best_vwap.vwap_trades} trades)")

    return sorted_scores


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(description="Stock Scanner v2: ORB + VWAP Pullback")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--stock", default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find files
    if args.stock:
        files = glob.glob(f"{args.data_dir}/{args.stock}*_*min.csv") + \
                glob.glob(f"{args.data_dir}/{args.stock.upper()}*_*min.csv")
    else:
        files = glob.glob(f"{args.data_dir}/*_*min.csv")

    if not files:
        print(f"❌ No stock data in {args.data_dir}/. Run fetch_stock_data.py first.")
        return

    print(f"Found {len(files)} stock data files")
    print(f"Strategy: ORB (morning) + VWAP Pullback (afternoon)")
    print(f"{'='*60}")

    scores = []
    all_trades = []

    for idx, fp in enumerate(sorted(files)):
        symbol = os.path.basename(fp).split("_")[0]
        print(f"[{idx+1}/{len(files)}] {symbol}...", end=" ", flush=True)

        try:
            df = pd.read_csv(fp)
            if len(df) < 200:
                print(f"⚠️ Too few candles")
                continue
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            if 'Volume' not in df.columns:
                print(f"⚠️ No Volume")
                continue

            days = df['Time'].dt.date.nunique()
            dr = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"

            buy, sell, stype, atr_v, orh, orl = generate_signals(df)
            trades = run_backtest(df, buy, sell, stype, atr_v, orh, orl)
            score = score_stock(symbol, df, trades)
            scores.append(score)

            orb_t = [t for t in trades if t.signal_type == 'ORB']
            vwap_t = [t for t in trades if t.signal_type == 'VWAP_PB']
            print(f"{len(trades)} trades (ORB:{len(orb_t)} VWAP:{len(vwap_t)}) | Win: {score.win_rate:.1f}% | P&L: {score.total_pnl_pct:+.2f}%")

            if args.detailed or args.stock:
                print_stock_report(symbol, trades, len(df), days, dr)

            for t in trades:
                all_trades.append({
                    'symbol': symbol, 'direction': t.direction,
                    'entry_price': t.entry_price, 'exit_price': t.exit_price,
                    'pnl_pts': t.pnl_pts, 'pnl_pct': t.pnl_pct,
                    'entry_time': t.entry_time, 'exit_time': t.exit_time,
                    'close_reason': t.close_reason, 'signal_type': t.signal_type,
                })
        except Exception as e:
            print(f"❌ {e}")

    if not scores:
        print("❌ No scores")
        return

    sorted_scores = print_ranking(scores, args.top)

    # Save
    pd.DataFrame([{
        'rank': i+1, 'symbol': s.symbol, 'score': s.total_score,
        'trades': s.total_trades, 'win_rate': s.win_rate,
        'pnl_pct': s.total_pnl_pct, 'pf': s.profit_factor,
        'orb_trades': s.orb_trades, 'orb_wr': s.orb_win_rate,
        'vwap_trades': s.vwap_trades, 'vwap_wr': s.vwap_win_rate,
    } for i, s in enumerate(sorted_scores)]).to_csv(f"{RESULTS_DIR}/stock_rankings.csv", index=False)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"{RESULTS_DIR}/all_stock_trades.csv", index=False)

    top5 = sorted_scores[:5]
    print(f"\n🎯 RECOMMENDED TOP 5:")
    for s in top5:
        print(f"  ⭐ {s.symbol}: Score {s.total_score:.1f} | Win {s.win_rate:.1f}% | PF {s.profit_factor:.2f} | ORB:{s.orb_win_rate:.0f}% VWAP:{s.vwap_win_rate:.0f}%")


if __name__ == "__main__":
    main()
