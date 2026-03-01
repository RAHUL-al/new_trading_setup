"""
best_strategy.py — Multi-Indicator Confluence Strategy with Strict Filtering

DESIGN PHILOSOPHY:
  The key to high win rate is NOT predicting every move —
  it's ONLY trading when multiple independent signals ALL agree.
  Most signals are skipped (low confidence). The few taken have high conviction.

STRATEGY: "Triple Confluence Filter"
  3 independent signal layers must ALL agree before entering:

  Layer 1 — TREND: EMA(20) vs EMA(50) + price position
  Layer 2 — MOMENTUM: RSI + Stochastic RSI + MACD histogram direction
  Layer 3 — TRIGGER: UT Bot Alert or price breakout from Bollinger Band squeeze

  + ATR filter (volatility must be in sweet spot: not too low, not too high)
  + Time-of-day filter (certain hours historically more profitable)
  + Candle strength filter (strong body, small wicks)

EXIT:
  - ATR-based trailing stop (tightens in profit)
  - Opposite triple confluence signal
  - Breakeven stop after 1:1 R:R reached
  - Time-based close at 3:24 PM

Usage:
    python best_strategy.py
    python best_strategy.py --file nifty_2min_data.csv
    python best_strategy.py --file nifty_3min_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, time as dt_time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
# Two trading windows (reduces trades, avoids mid-day chop)
WINDOW_1_START = dt_time(9, 15)    # Morning window start
WINDOW_1_END = dt_time(10, 30)     # Morning window end (no new entries after this)
WINDOW_1_CLOSE = dt_time(10, 30)   # Force close morning positions

WINDOW_2_START = dt_time(13, 0)    # Afternoon window start
WINDOW_2_END = dt_time(15, 15)     # No new entries after 3:15 PM
WINDOW_2_CLOSE = dt_time(15, 24)   # Force close at 3:24 PM

SQUARE_OFF_TIME = dt_time(15, 24)
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 25)

MAX_TRADES_PER_WINDOW = 999         # Unlimited trades per window

# ATR sweet spot — too low = no movement, too high = choppy
ATR_MIN = 8.0
ATR_MAX = 80.0

# Risk management
INITIAL_SL_MULT = 1.5      # Initial SL = ATR × 1.5
TRAIL_SL_MULT = 1.2        # Trail SL = ATR × 1.2 (tighter as profit grows)
BREAKEVEN_RR = 1.0          # Move SL to breakeven after 1:1 risk-reward
MIN_HOLD = 3                # Min candles before exit on opposite signal


# ─────────── Data Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    initial_risk: float      # Distance from entry to initial SL
    entry_idx: int
    entry_atr: float
    breakeven_hit: bool = False

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    close_reason: str
    entry_atr: float


# ─────────── Indicators ───────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi_val = rsi(close, rsi_period)
    rsi_low = rsi_val.rolling(stoch_period).min()
    rsi_high = rsi_val.rolling(stoch_period).max()
    stoch = (rsi_val - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan) * 100
    k = stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def macd(close, fast=12, slow=26, signal=9):
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close, period=20, std_mult=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma * 100  # Normalized width
    return upper, sma, lower, width


def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr.ewm(alpha=1/period, adjust=False).mean()


def ut_bot(close_arr, a=2, c=100):
    """UT Bot Alert — returns buy/sell signal arrays."""
    close = close_arr.values if hasattr(close_arr, 'values') else close_arr
    xATR = pd.Series(close).diff().abs().ewm(span=c, adjust=False).mean().values
    nLoss = a * xATR

    trail = np.zeros(len(close))
    trail[0] = close[0]
    for i in range(1, len(close)):
        if close[i] > trail[i-1] and close[i-1] > trail[i-1]:
            trail[i] = max(trail[i-1], close[i] - nLoss[i])
        elif close[i] < trail[i-1] and close[i-1] < trail[i-1]:
            trail[i] = min(trail[i-1], close[i] + nLoss[i])
        elif close[i] > trail[i-1]:
            trail[i] = close[i] - nLoss[i]
        else:
            trail[i] = close[i] + nLoss[i]

    pos = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i-1] < trail[i-1] and close[i] > trail[i-1]:
            pos[i] = 1
        elif close[i-1] > trail[i-1] and close[i] < trail[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]

    buy = np.zeros(len(close), dtype=bool)
    sell = np.zeros(len(close), dtype=bool)
    for i in range(1, len(close)):
        if pos[i] == 1 and pos[i-1] != 1:
            buy[i] = True
        elif pos[i] == -1 and pos[i-1] != -1:
            sell[i] = True

    return buy, sell, pos


# ─────────── Signal Generation ───────────

def generate_signals(df):
    """
    Generate triple-confluence signals.

    Returns DataFrame with columns: buy, sell, confidence, trend, momentum, trigger
    """
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    open_ = df['Open'].astype(float)
    n = len(df)

    # ─── Layer 1: TREND ───
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200)

    trend_bullish = np.zeros(n, dtype=bool)
    trend_bearish = np.zeros(n, dtype=bool)

    for i in range(n):
        bull_score = 0
        bear_score = 0

        # EMA alignment
        if ema_20.iloc[i] > ema_50.iloc[i]:
            bull_score += 1
        else:
            bear_score += 1

        # Price above/below EMAs
        if close.iloc[i] > ema_20.iloc[i]:
            bull_score += 1
        else:
            bear_score += 1

        if close.iloc[i] > ema_50.iloc[i]:
            bull_score += 1
        else:
            bear_score += 1

        trend_bullish[i] = bull_score >= 3  # ALL must agree
        trend_bearish[i] = bear_score >= 3

    # ─── Layer 2: MOMENTUM ───
    rsi_14 = rsi(close, 14)
    stoch_k, stoch_d = stochastic_rsi(close)
    macd_line, macd_signal, macd_hist = macd(close)

    momentum_bullish = np.zeros(n, dtype=bool)
    momentum_bearish = np.zeros(n, dtype=bool)

    for i in range(1, n):
        bull_score = 0
        bear_score = 0

        # RSI: between 40-70 for buy (not overbought), 30-60 for sell
        r = rsi_14.iloc[i]
        if 40 <= r <= 70:
            bull_score += 1
        if 30 <= r <= 60:
            bear_score += 1

        # Stochastic RSI: K > D for buy, K < D for sell
        if not np.isnan(stoch_k.iloc[i]) and not np.isnan(stoch_d.iloc[i]):
            if stoch_k.iloc[i] > stoch_d.iloc[i] and stoch_k.iloc[i] < 80:
                bull_score += 1
            if stoch_k.iloc[i] < stoch_d.iloc[i] and stoch_k.iloc[i] > 20:
                bear_score += 1

        # MACD: histogram positive and growing for buy
        if not np.isnan(macd_hist.iloc[i]) and not np.isnan(macd_hist.iloc[i-1]):
            if macd_hist.iloc[i] > 0 and macd_hist.iloc[i] > macd_hist.iloc[i-1]:
                bull_score += 1
            if macd_hist.iloc[i] < 0 and macd_hist.iloc[i] < macd_hist.iloc[i-1]:
                bear_score += 1

        momentum_bullish[i] = bull_score >= 2  # At least 2 of 3
        momentum_bearish[i] = bear_score >= 2

    # ─── Layer 3: TRIGGER ───
    ut_buy, ut_sell, ut_pos = ut_bot(close, a=2, c=100)
    bb_upper, bb_mid, bb_lower, bb_width = bollinger_bands(close, 20, 2)

    trigger_buy = np.zeros(n, dtype=bool)
    trigger_sell = np.zeros(n, dtype=bool)

    for i in range(2, n):
        # UT Bot fire OR Bollinger breakout
        # UT Bot buy signal
        if ut_buy[i]:
            trigger_buy[i] = True
        # Price breaks above upper Bollinger after squeeze (width was low)
        elif not np.isnan(bb_width.iloc[i]) and not np.isnan(bb_width.iloc[i-1]):
            if close.iloc[i] > bb_upper.iloc[i] and bb_width.iloc[i-1] < bb_width.rolling(50).mean().iloc[i-1]:
                trigger_buy[i] = True

        # UT Bot sell signal
        if ut_sell[i]:
            trigger_sell[i] = True
        elif not np.isnan(bb_width.iloc[i]) and not np.isnan(bb_width.iloc[i-1]):
            if close.iloc[i] < bb_lower.iloc[i] and bb_width.iloc[i-1] < bb_width.rolling(50).mean().iloc[i-1]:
                trigger_sell[i] = True

    # ─── Candle Strength Filter ───
    body = (close - open_).abs()
    full_range = high - low
    body_ratio = body / full_range.replace(0, np.nan)

    strong_candle = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(body_ratio.iloc[i]):
            # Body must be at least 50% of total range (strong conviction)
            strong_candle[i] = body_ratio.iloc[i] >= 0.5

    # ─── ATR Filter ───
    atr_val = atr(high, low, close, ATR_PERIOD if 'ATR_PERIOD' in dir() else 14)

    # ─── COMBINE: Triple Confluence ───
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    confidence = np.zeros(n)

    for i in range(n):
        atr_v = atr_val.iloc[i]
        if np.isnan(atr_v) or atr_v < ATR_MIN or atr_v > ATR_MAX:
            continue

        # BUY: all 3 layers + strong candle + bullish close
        if trend_bullish[i] and momentum_bullish[i] and trigger_buy[i]:
            if strong_candle[i] and close.iloc[i] > open_.iloc[i]:  # Bullish candle
                buy_signal[i] = True
                # Confidence = how many sub-indicators agree (0-1 scale)
                conf = 0
                r = rsi_14.iloc[i]
                if not np.isnan(r): conf += min(r / 100, 1) * 0.2
                if not np.isnan(stoch_k.iloc[i]) and not np.isnan(stoch_d.iloc[i]):
                    conf += (stoch_k.iloc[i] - stoch_d.iloc[i]) / 100 * 0.3
                conf += body_ratio.iloc[i] * 0.3 if not np.isnan(body_ratio.iloc[i]) else 0
                conf += 0.2  # Base for triple confluence
                confidence[i] = min(conf, 1.0)

        # SELL: all 3 layers + strong candle + bearish close
        if trend_bearish[i] and momentum_bearish[i] and trigger_sell[i]:
            if strong_candle[i] and close.iloc[i] < open_.iloc[i]:  # Bearish candle
                sell_signal[i] = True
                conf = 0
                r = rsi_14.iloc[i]
                if not np.isnan(r): conf += (1 - r / 100) * 0.2
                if not np.isnan(stoch_k.iloc[i]) and not np.isnan(stoch_d.iloc[i]):
                    conf += (stoch_d.iloc[i] - stoch_k.iloc[i]) / 100 * 0.3
                conf += body_ratio.iloc[i] * 0.3 if not np.isnan(body_ratio.iloc[i]) else 0
                conf += 0.2
                confidence[i] = min(conf, 1.0)

    result = pd.DataFrame({
        'buy': buy_signal,
        'sell': sell_signal,
        'confidence': confidence,
        'atr': atr_val.values,
        'trend_bull': trend_bullish,
        'trend_bear': trend_bearish,
        'momentum_bull': momentum_bullish,
        'momentum_bear': momentum_bearish,
        'trigger_buy': trigger_buy,
        'trigger_sell': trigger_sell,
    })

    return result


# ─────────── Backtest Engine ───────────

def run_backtest(df, signals):
    """Backtest with dual trading windows and strict trade limits."""
    trades = []
    open_pos = None
    prev_date = None
    window_trades_today = {1: 0, 2: 0}  # Track trades per window per day

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time() if hasattr(row['Time'], 'time') else row['Time']
        curr_date = row['Time'].date() if hasattr(row['Time'], 'date') else None
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        curr_atr = float(signals.iloc[i]['atr']) if i < len(signals) else 0

        # Day boundary reset
        if prev_date and curr_date != prev_date:
            if open_pos:
                pnl = _pnl(open_pos, float(df.iloc[i-1]['Close']))
                trades.append(Trade(open_pos.direction, open_pos.entry_price,
                                    float(df.iloc[i-1]['Close']), open_pos.entry_time,
                                    df.iloc[i-1]['Time'], round(pnl, 2), "DAY_END", open_pos.entry_atr))
                open_pos = None
            window_trades_today = {1: 0, 2: 0}  # Reset daily counters
        prev_date = curr_date

        if not (MARKET_OPEN <= t <= MARKET_CLOSE):
            continue

        # Determine current window
        in_window_1 = WINDOW_1_START <= t <= WINDOW_1_CLOSE
        in_window_2 = WINDOW_2_START <= t <= WINDOW_2_CLOSE
        can_enter_w1 = WINDOW_1_START <= t <= WINDOW_1_END and window_trades_today[1] < MAX_TRADES_PER_WINDOW
        can_enter_w2 = WINDOW_2_START <= t <= WINDOW_2_END and window_trades_today[2] < MAX_TRADES_PER_WINDOW

        # Window 1 close: force close morning positions at 10:30
        if open_pos and t >= WINDOW_1_CLOSE and open_pos.entry_time.time() < WINDOW_1_CLOSE and t < WINDOW_2_START:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "WINDOW_1_CLOSE", open_pos.entry_atr))
            open_pos = None
            continue

        # Window 2 close: square off at 3:24
        if open_pos and t >= SQUARE_OFF_TIME:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "SQUARE_OFF", open_pos.entry_atr))
            open_pos = None
            continue

        # Skip if not in any active window
        if not in_window_1 and not in_window_2:
            continue

        # Stop loss (using high/low for realistic fills)
        if open_pos:
            if open_pos.direction == "CE" and low <= open_pos.stop_loss:
                pnl = _pnl(open_pos, open_pos.stop_loss)
                trades.append(Trade(open_pos.direction, open_pos.entry_price,
                                    open_pos.stop_loss, open_pos.entry_time,
                                    row['Time'], round(pnl, 2), "STOP_LOSS", open_pos.entry_atr))
                open_pos = None
            elif open_pos and open_pos.direction == "PE" and high >= open_pos.stop_loss:
                pnl = _pnl(open_pos, open_pos.stop_loss)
                trades.append(Trade(open_pos.direction, open_pos.entry_price,
                                    open_pos.stop_loss, open_pos.entry_time,
                                    row['Time'], round(pnl, 2), "STOP_LOSS", open_pos.entry_atr))
                open_pos = None

        # Breakeven stop
        if open_pos and not open_pos.breakeven_hit:
            if open_pos.direction == "CE":
                if close - open_pos.entry_price >= open_pos.initial_risk * BREAKEVEN_RR:
                    open_pos.stop_loss = max(open_pos.stop_loss, open_pos.entry_price + 1)
                    open_pos.breakeven_hit = True
            elif open_pos.direction == "PE":
                if open_pos.entry_price - close >= open_pos.initial_risk * BREAKEVEN_RR:
                    open_pos.stop_loss = min(open_pos.stop_loss, open_pos.entry_price - 1)
                    open_pos.breakeven_hit = True

        # Trailing SL (tighter after breakeven)
        if open_pos and curr_atr > 0:
            mult = TRAIL_SL_MULT if open_pos.breakeven_hit else INITIAL_SL_MULT
            if open_pos.direction == "CE":
                new_sl = high - (curr_atr * mult)
                if new_sl > open_pos.stop_loss:
                    open_pos.stop_loss = new_sl
            elif open_pos.direction == "PE":
                new_sl = low + (curr_atr * mult)
                if new_sl < open_pos.stop_loss:
                    open_pos.stop_loss = new_sl

        # Signals
        if i >= len(signals):
            continue
        sig = signals.iloc[i]
        is_buy = bool(sig['buy'])
        is_sell = bool(sig['sell'])

        # Opposite signal close
        if is_buy and open_pos and open_pos.direction == "PE" and (i - open_pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "OPPOSITE_SIGNAL", open_pos.entry_atr))
            open_pos = None
        elif is_sell and open_pos and open_pos.direction == "CE" and (i - open_pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "OPPOSITE_SIGNAL", open_pos.entry_atr))
            open_pos = None

        # New entry — only in allowed windows with trade limit
        if not open_pos and (can_enter_w1 or can_enter_w2):
            if is_buy and curr_atr > 0:
                sl = close - (curr_atr * INITIAL_SL_MULT)
                risk = close - sl
                open_pos = Position("CE", close, row['Time'], sl, risk, i, curr_atr)
                if can_enter_w1: window_trades_today[1] += 1
                if can_enter_w2: window_trades_today[2] += 1
            elif is_sell and curr_atr > 0:
                sl = close + (curr_atr * INITIAL_SL_MULT)
                risk = sl - close
                open_pos = Position("PE", close, row['Time'], sl, risk, i, curr_atr)
                if can_enter_w1: window_trades_today[1] += 1
                if can_enter_w2: window_trades_today[2] += 1

    return trades


def _pnl(pos, exit_price):
    return (exit_price - pos.entry_price) if pos.direction == "CE" else (pos.entry_price - exit_price)


# ─────────── Report ───────────

def print_report(trades, total_candles, total_days, date_range):
    if not trades:
        print("\n❌ No trades generated! Strategy too restrictive for this data.")
        print("   Try: lowering ATR_MIN or relaxing confluence requirements.")
        return

    pnl_list = [t.pnl for t in trades]
    total = sum(pnl_list)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100

    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (cumulative - peak).min()

    daily = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + t.pnl
    prof_days = sum(1 for v in daily.values() if v > 0)

    reasons = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    ce = [t for t in trades if t.direction == "CE"]
    pe = [t for t in trades if t.direction == "PE"]

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy
    expectancy = total / len(trades)

    # Avg RR
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
    avg_rr = avg_win / avg_loss if avg_loss > 0 else float('inf')

    print(f"\n{'='*75}")
    print(f"  TRIPLE CONFLUENCE STRATEGY — BACKTEST RESULTS")
    print(f"  Data: {date_range} | {total_days} days | {total_candles} candles")
    print(f"{'='*75}")

    print(f"\n📊 OVERVIEW")
    print(f"  Total trades:       {len(trades)}")
    print(f"  Avg trades/day:     {len(trades)/max(len(daily),1):.1f}")
    print(f"  Days with trades:   {len(daily)}/{total_days}")

    print(f"\n💰 PERFORMANCE")
    print(f"  Total P&L:          {total:+.2f} pts")
    print(f"  Win rate:           {win_rate:.1f}% {'✅' if win_rate >= 60 else '⚠️'}")
    print(f"  Profit factor:      {profit_factor:.2f} {'✅' if profit_factor >= 1.5 else '⚠️'}")
    print(f"  Expectancy:         {expectancy:+.2f} pts/trade")
    print(f"  Avg R:R:            {avg_rr:.2f}")
    print(f"  Avg win:            {avg_win:+.2f} pts")
    print(f"  Avg loss:           {-avg_loss:+.2f} pts")
    print(f"  Best trade:         {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:        {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:       {max_dd:.2f} pts")
    print(f"  Avg P&L/day:        {total/max(len(daily),1):+.2f} pts")
    print(f"  Profitable days:    {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    print(f"\n🎯 BY DIRECTION")
    if ce:
        ce_wr = sum(1 for t in ce if t.pnl > 0)/len(ce)*100
        print(f"  CE: {len(ce)} trades | P&L: {sum(t.pnl for t in ce):+.2f} | Win: {ce_wr:.1f}%")
    if pe:
        pe_wr = sum(1 for t in pe if t.pnl > 0)/len(pe)*100
        print(f"  PE: {len(pe)} trades | P&L: {sum(t.pnl for t in pe):+.2f} | Win: {pe_wr:.1f}%")

    print(f"\n🔍 CLOSE REASONS")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        r_trades = [t for t in trades if t.close_reason == reason]
        r_pnl = sum(t.pnl for t in r_trades)
        r_wr = sum(1 for t in r_trades if t.pnl > 0)/len(r_trades)*100
        print(f"  {reason:20s}: {count:4d} | P&L: {r_pnl:+8.2f} | Win: {r_wr:.0f}%")

    # Streaks
    ws, ls, cs = 0, 0, 0
    for p in pnl_list:
        if p > 0:
            cs = cs + 1 if cs > 0 else 1
            ws = max(ws, cs)
        else:
            cs = cs - 1 if cs < 0 else -1
            ls = max(ls, abs(cs))

    print(f"\n📈 STREAKS")
    print(f"  Max win streak:     {ws}")
    print(f"  Max loss streak:    {ls}")

    # Monthly breakdown
    monthly = {}
    for t in trades:
        month = t.entry_time.strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"pnl": 0, "trades": 0, "wins": 0}
        monthly[month]["pnl"] += t.pnl
        monthly[month]["trades"] += 1
        if t.pnl > 0:
            monthly[month]["wins"] += 1

    print(f"\n📅 MONTHLY BREAKDOWN")
    print(f"  {'Month':>8} {'P&L':>10} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*35}")
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins']/m['trades']*100 if m['trades'] > 0 else 0
        emoji = "✅" if m['pnl'] > 0 else "❌"
        print(f"  {month:>8} {m['pnl']:>+10.2f} {m['trades']:>7} {wr:>5.1f}% {emoji}")

    print(f"\n{'='*75}")

    # Save
    trades_df = pd.DataFrame([{
        'direction': t.direction, 'entry_price': t.entry_price,
        'exit_price': t.exit_price, 'pnl': t.pnl,
        'entry_time': t.entry_time, 'exit_time': t.exit_time,
        'close_reason': t.close_reason, 'entry_atr': round(t.entry_atr, 2),
    } for t in trades])
    trades_df.to_csv("best_strategy_trades.csv", index=False)
    print(f"  💾 Trades: best_strategy_trades.csv")


# ─────────── Main ───────────

def main():
    global ATR_MIN, ATR_MAX, INITIAL_SL_MULT

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="nifty_2min_data.csv")
    parser.add_argument("--atr-min", type=float, default=ATR_MIN)
    parser.add_argument("--atr-max", type=float, default=ATR_MAX)
    parser.add_argument("--sl-mult", type=float, default=INITIAL_SL_MULT)
    args = parser.parse_args()

    ATR_MIN = args.atr_min
    ATR_MAX = args.atr_max
    INITIAL_SL_MULT = args.sl_mult

    # Find data
    data_file = args.file
    if not os.path.exists(data_file):
        for alt in ["nifty_2min_data.csv", "nifty_3min_data.csv", "nifty_1min_data.csv", "nifty_5min_data.csv"]:
            if os.path.exists(alt):
                data_file = alt
                break

    if not os.path.exists(data_file):
        print("❌ No data file found. Run fetch_nifty_data.py first.")
        return

    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    total_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"
    print(f"Loaded {len(df)} candles | {total_days} days")

    # Generate signals
    print("\n📊 Generating triple confluence signals...")
    signals = generate_signals(df)

    total_buy = signals['buy'].sum()
    total_sell = signals['sell'].sum()
    print(f"  Buy signals:  {total_buy}")
    print(f"  Sell signals: {total_sell}")
    print(f"  Total:        {total_buy + total_sell} ({(total_buy+total_sell)/len(df)*100:.2f}% of candles)")

    # Layer stats
    print(f"\n  Layer breakdown:")
    print(f"    Trend bullish:    {signals['trend_bull'].sum()} candles ({signals['trend_bull'].mean()*100:.1f}%)")
    print(f"    Trend bearish:    {signals['trend_bear'].sum()} candles ({signals['trend_bear'].mean()*100:.1f}%)")
    print(f"    Momentum bull:    {signals['momentum_bull'].sum()} candles")
    print(f"    Momentum bear:    {signals['momentum_bear'].sum()} candles")
    print(f"    Trigger buy:      {signals['trigger_buy'].sum()} candles")
    print(f"    Trigger sell:     {signals['trigger_sell'].sum()} candles")

    # Backtest
    print(f"\n🚀 Running backtest...")
    print(f"  Window 1: {WINDOW_1_START.strftime('%H:%M')} - {WINDOW_1_END.strftime('%H:%M')} (close at {WINDOW_1_CLOSE.strftime('%H:%M')})")
    print(f"  Window 2: {WINDOW_2_START.strftime('%H:%M')} - {WINDOW_2_END.strftime('%H:%M')} (close at {WINDOW_2_CLOSE.strftime('%H:%M')})")
    print(f"  Max trades/window:  {MAX_TRADES_PER_WINDOW}")
    print(f"  ATR range: {ATR_MIN} - {ATR_MAX}")
    print(f"  SL: {INITIAL_SL_MULT}x ATR | Trail: {TRAIL_SL_MULT}x ATR")
    print(f"  Breakeven at {BREAKEVEN_RR}:1 R:R")

    trades = run_backtest(df, signals)
    print_report(trades, len(df), total_days, date_range)


if __name__ == "__main__":
    main()
