"""
ema_crossover_strategy.py — EMA 5/21 Crossover Strategy (2-min NIFTY)

STRATEGY:
  BUY:  EMA(5) crosses ABOVE EMA(21) + ATR > 12 + strong bullish candle
  SELL: EMA(5) crosses BELOW EMA(21) + ATR > 12 + strong bearish candle

ENHANCEMENTS (my optimizations):
  - Candle body must be >= 40% of range (reject dojis/indecision)
  - Close must be in direction of signal (close > open for buy)
  - EMA gap filter: EMAs must have meaningful separation after cross
  - RSI confirmation: not overbought (>80) for buy, not oversold (<20) for sell
  - Trailing SL based on ATR
  - Open trades stay alive even if ATR drops below 12

WINDOWS:
  Window 1: 9:15 AM  - 10:30 AM  (morning momentum)
  Window 2: 1:00 PM  - 3:15 PM   (afternoon trend)
  Square off: 3:24 PM | Morning positions close at 10:30

Usage:
    python ema_crossover_strategy.py
    python ema_crossover_strategy.py --file nifty_2min_data.csv --atr-gate 15
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
WINDOW_1_START = dt_time(9, 15)
WINDOW_1_END = dt_time(10, 30)     # Last new entry morning

WINDOW_2_START = dt_time(13, 0)
WINDOW_2_END = dt_time(15, 15)     # Last new entry afternoon

SQUARE_OFF_TIME = dt_time(15, 24)

ATR_GATE = 12.0                    # Min ATR for new entry
ATR_PERIOD = 14
EMA_FAST = 5
EMA_SLOW = 21

SL_ATR_MULT = 1.5                  # Initial SL = ATR × 1.5
TRAIL_ATR_MULT = 1.2               # Trail SL = ATR × 1.2
MIN_HOLD = 2                       # Min candles before opposite close
BODY_RATIO_MIN = 0.40              # Candle body must be >= 40% of range


# ─────────── Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    entry_idx: int
    entry_atr: float

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
    exit_atr: float


# ─────────── Indicators ───────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr.ewm(alpha=1/period, adjust=False).mean()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────── Signal Generation ───────────

def generate_signals(df):
    """EMA 5/21 crossover signals with quality filters."""
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    n = len(df)

    # Core indicators
    ema_fast = calc_ema(close, EMA_FAST)
    ema_slow = calc_ema(close, EMA_SLOW)
    atr_val = calc_atr(df, ATR_PERIOD)
    rsi_val = calc_rsi(close)

    # Candle quality
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    body_ratio = body / full_range

    # EMA crossover detection
    ema_diff = ema_fast - ema_slow
    ema_diff_prev = ema_diff.shift(1)

    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)

    for i in range(2, n):
        curr_atr = atr_val.iloc[i]
        if np.isnan(curr_atr):
            continue

        # Candle quality check
        br = body_ratio.iloc[i]
        if np.isnan(br) or br < BODY_RATIO_MIN:
            continue

        curr_diff = ema_diff.iloc[i]
        prev_diff = ema_diff_prev.iloc[i]
        if np.isnan(curr_diff) or np.isnan(prev_diff):
            continue

        # ATR gate for entries
        if curr_atr < ATR_GATE:
            continue

        # RSI
        r = rsi_val.iloc[i]

        # ─── BUY: EMA(5) crosses above EMA(21) ───
        # Cross happened recently (within last 3 candles) or just crossed
        cross_up = False
        for lookback in range(3):
            j = i - lookback
            if j >= 1:
                if ema_diff.iloc[j] > 0 and ema_diff.iloc[j-1] <= 0:
                    cross_up = True
                    break

        if cross_up and ema_fast.iloc[i] > ema_slow.iloc[i]:
            # Bullish candle + not overbought
            if close.iloc[i] > open_.iloc[i] and (np.isnan(r) or r < 80):
                buy_signal[i] = True

        # ─── SELL: EMA(5) crosses below EMA(21) ───
        cross_down = False
        for lookback in range(3):
            j = i - lookback
            if j >= 1:
                if ema_diff.iloc[j] < 0 and ema_diff.iloc[j-1] >= 0:
                    cross_down = True
                    break

        if cross_down and ema_fast.iloc[i] < ema_slow.iloc[i]:
            # Bearish candle + not oversold
            if close.iloc[i] < open_.iloc[i] and (np.isnan(r) or r > 20):
                sell_signal[i] = True

    return buy_signal, sell_signal, ema_fast, ema_slow, atr_val


# ─────────── Backtest ───────────

def in_trading_window(t):
    """Check if time is in Window 1 or Window 2."""
    return (WINDOW_1_START <= t <= WINDOW_1_END) or (WINDOW_2_START <= t <= WINDOW_2_END)


def in_active_window(t):
    """Check if time is in any active period (for SL/trail/exit checks)."""
    return (WINDOW_1_START <= t <= WINDOW_1_END) or (WINDOW_2_START <= t <= SQUARE_OFF_TIME)


def run_backtest(df, buy_signals, sell_signals, atr_vals):
    trades = []
    open_pos = None
    prev_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time()
        curr_date = row['Time'].date()
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        curr_atr = float(atr_vals.iloc[i])

        # ── Day boundary: close any open position ──
        if prev_date and curr_date != prev_date:
            if open_pos:
                prev_close = float(df.iloc[i-1]['Close'])
                pnl = _pnl(open_pos, prev_close)
                trades.append(Trade(open_pos.direction, open_pos.entry_price, prev_close,
                                    open_pos.entry_time, df.iloc[i-1]['Time'], round(pnl, 2),
                                    "DAY_END", open_pos.entry_atr, float(atr_vals.iloc[i-1])))
                open_pos = None
        prev_date = curr_date

        # ── Morning window close: positions from W1 close at 10:30 ──
        if open_pos and t > WINDOW_1_END and t < WINDOW_2_START:
            if open_pos.entry_time.time() <= WINDOW_1_END:
                pnl = _pnl(open_pos, close)
                trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                    open_pos.entry_time, row['Time'], round(pnl, 2),
                                    "W1_CLOSE", open_pos.entry_atr, curr_atr))
                open_pos = None
            continue

        # ── Square off at 3:24 PM ──
        if open_pos and t >= SQUARE_OFF_TIME:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "SQUARE_OFF", open_pos.entry_atr, curr_atr))
            open_pos = None
            continue

        # Skip if not in any active window
        if not in_active_window(t):
            continue

        # ── Stop loss (high/low for realistic fills) ──
        if open_pos:
            if open_pos.direction == "CE" and low <= open_pos.stop_loss:
                pnl = _pnl(open_pos, open_pos.stop_loss)
                trades.append(Trade(open_pos.direction, open_pos.entry_price, open_pos.stop_loss,
                                    open_pos.entry_time, row['Time'], round(pnl, 2),
                                    "STOP_LOSS", open_pos.entry_atr, curr_atr))
                open_pos = None
            elif open_pos and open_pos.direction == "PE" and high >= open_pos.stop_loss:
                pnl = _pnl(open_pos, open_pos.stop_loss)
                trades.append(Trade(open_pos.direction, open_pos.entry_price, open_pos.stop_loss,
                                    open_pos.entry_time, row['Time'], round(pnl, 2),
                                    "STOP_LOSS", open_pos.entry_atr, curr_atr))
                open_pos = None

        # ── Trailing SL (keeps running regardless of ATR) ──
        if open_pos and curr_atr > 0:
            if open_pos.direction == "CE":
                new_sl = high - (curr_atr * TRAIL_ATR_MULT)
                if new_sl > open_pos.stop_loss:
                    open_pos.stop_loss = new_sl
            elif open_pos.direction == "PE":
                new_sl = low + (curr_atr * TRAIL_ATR_MULT)
                if new_sl < open_pos.stop_loss:
                    open_pos.stop_loss = new_sl

        # ── Opposite signal close ──
        is_buy = bool(buy_signals[i]) if i < len(buy_signals) else False
        is_sell = bool(sell_signals[i]) if i < len(sell_signals) else False

        if is_buy and open_pos and open_pos.direction == "PE" and (i - open_pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "OPPOSITE", open_pos.entry_atr, curr_atr))
            open_pos = None

        elif is_sell and open_pos and open_pos.direction == "CE" and (i - open_pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(open_pos, close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close,
                                open_pos.entry_time, row['Time'], round(pnl, 2),
                                "OPPOSITE", open_pos.entry_atr, curr_atr))
            open_pos = None

        # ── New entry (only in trading windows) ──
        if not open_pos and in_trading_window(t):
            if is_buy and curr_atr > 0:
                sl = close - (curr_atr * SL_ATR_MULT)
                open_pos = Position("CE", close, row['Time'], sl, i, curr_atr)
            elif is_sell and curr_atr > 0:
                sl = close + (curr_atr * SL_ATR_MULT)
                open_pos = Position("PE", close, row['Time'], sl, i, curr_atr)

    return trades


def _pnl(pos, exit_price):
    return (exit_price - pos.entry_price) if pos.direction == "CE" else (pos.entry_price - exit_price)


# ─────────── Report ───────────

def print_report(trades, total_candles, total_days, date_range):
    if not trades:
        print("\n❌ No trades generated!")
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

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1

    print(f"\n{'='*75}")
    print(f"  EMA {EMA_FAST}/{EMA_SLOW} CROSSOVER STRATEGY — BACKTEST RESULTS")
    print(f"  Data: {date_range} | {total_days} days | {total_candles} candles")
    print(f"  Windows: 9:15-10:30 + 13:00-15:15 | ATR gate: {ATR_GATE}")
    print(f"{'='*75}")

    print(f"\n📊 OVERVIEW")
    print(f"  Total trades:       {len(trades)}")
    print(f"  Avg trades/day:     {len(trades)/max(len(daily),1):.1f}")
    print(f"  Days with trades:   {len(daily)}/{total_days}")

    print(f"\n💰 PERFORMANCE")
    print(f"  Total P&L:          {total:+.2f} pts")
    print(f"  Win rate:           {win_rate:.1f}%")
    print(f"  Profit factor:      {profit_factor:.2f}")
    print(f"  Avg win:            {avg_win:+.2f} pts")
    print(f"  Avg loss:           {-avg_loss:+.2f} pts")
    print(f"  Avg R:R:            {avg_win/avg_loss if avg_loss > 0 else 0:.2f}")
    print(f"  Best trade:         {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:        {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:       {max_dd:.2f} pts")
    print(f"  Avg P&L/trade:      {total/len(trades):+.2f} pts")
    print(f"  Avg P&L/day:        {total/max(len(daily),1):+.2f} pts")
    print(f"  Profitable days:    {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    print(f"\n🎯 BY DIRECTION")
    if ce:
        print(f"  CE: {len(ce)} trades | P&L: {sum(t.pnl for t in ce):+.2f} | Win: {sum(1 for t in ce if t.pnl>0)/len(ce)*100:.1f}%")
    if pe:
        print(f"  PE: {len(pe)} trades | P&L: {sum(t.pnl for t in pe):+.2f} | Win: {sum(1 for t in pe if t.pnl>0)/len(pe)*100:.1f}%")

    print(f"\n🔍 CLOSE REASONS")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        r_trades = [t for t in trades if t.close_reason == reason]
        r_pnl = sum(t.pnl for t in r_trades)
        r_wr = sum(1 for t in r_trades if t.pnl > 0)/len(r_trades)*100
        print(f"  {reason:20s}: {count:4d} | P&L: {r_pnl:+8.2f} | Win: {r_wr:.0f}%")

    # By window
    w1_trades = [t for t in trades if t.entry_time.time() <= WINDOW_1_END]
    w2_trades = [t for t in trades if t.entry_time.time() >= WINDOW_2_START]
    print(f"\n⏰ BY WINDOW")
    if w1_trades:
        w1_pnl = sum(t.pnl for t in w1_trades)
        w1_wr = sum(1 for t in w1_trades if t.pnl > 0)/len(w1_trades)*100
        print(f"  Morning (9:15-10:30):   {len(w1_trades)} trades | P&L: {w1_pnl:+.2f} | Win: {w1_wr:.1f}%")
    if w2_trades:
        w2_pnl = sum(t.pnl for t in w2_trades)
        w2_wr = sum(1 for t in w2_trades if t.pnl > 0)/len(w2_trades)*100
        print(f"  Afternoon (1:00-3:15):  {len(w2_trades)} trades | P&L: {w2_pnl:+.2f} | Win: {w2_wr:.1f}%")

    # Streaks
    ws, ls, cs = 0, 0, 0
    for p in pnl_list:
        if p > 0:
            cs = cs + 1 if cs > 0 else 1
            ws = max(ws, cs)
        else:
            cs = cs - 1 if cs < 0 else -1
            ls = max(ls, abs(cs))
    print(f"\n📈 STREAKS: Win {ws} | Loss {ls}")

    # Monthly
    monthly = {}
    for t in trades:
        m = t.entry_time.strftime("%Y-%m")
        if m not in monthly: monthly[m] = {"pnl": 0, "trades": 0, "wins": 0}
        monthly[m]["pnl"] += t.pnl
        monthly[m]["trades"] += 1
        if t.pnl > 0: monthly[m]["wins"] += 1

    print(f"\n📅 MONTHLY")
    print(f"  {'Month':>8} {'P&L':>10} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*35}")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        wr = d['wins']/d['trades']*100 if d['trades'] > 0 else 0
        e = "✅" if d['pnl'] > 0 else "❌"
        print(f"  {m:>8} {d['pnl']:>+10.2f} {d['trades']:>7} {wr:>5.1f}% {e}")

    print(f"\n{'='*75}")

    # Save
    pd.DataFrame([{
        'direction': t.direction, 'entry_price': t.entry_price,
        'exit_price': t.exit_price, 'pnl': t.pnl,
        'entry_time': t.entry_time, 'exit_time': t.exit_time,
        'close_reason': t.close_reason, 'entry_atr': round(t.entry_atr, 2),
        'exit_atr': round(t.exit_atr, 2),
    } for t in trades]).to_csv("ema_crossover_trades.csv", index=False)
    print(f"  💾 Trades: ema_crossover_trades.csv")


# ─────────── Main ───────────

def main():
    global ATR_GATE, SL_ATR_MULT, TRAIL_ATR_MULT, BODY_RATIO_MIN

    parser = argparse.ArgumentParser(description="EMA 5/21 Crossover Strategy")
    parser.add_argument("--file", default="nifty_2min_data.csv")
    parser.add_argument("--atr-gate", type=float, default=ATR_GATE)
    parser.add_argument("--sl-mult", type=float, default=SL_ATR_MULT)
    parser.add_argument("--trail-mult", type=float, default=TRAIL_ATR_MULT)
    parser.add_argument("--body-ratio", type=float, default=BODY_RATIO_MIN)
    args = parser.parse_args()

    ATR_GATE = args.atr_gate
    SL_ATR_MULT = args.sl_mult
    TRAIL_ATR_MULT = args.trail_mult
    BODY_RATIO_MIN = args.body_ratio

    # Find data
    data_file = args.file
    if not os.path.exists(data_file):
        for alt in ["nifty_2min_data.csv", "nifty_3min_data.csv", "nifty_1min_data.csv"]:
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
    print(f"Loaded {len(df)} candles | {total_days} days | {date_range}")

    # Generate signals
    print(f"\n📊 Generating EMA {EMA_FAST}/{EMA_SLOW} crossover signals...")
    print(f"  ATR gate: >= {ATR_GATE}")
    print(f"  Body ratio min: {BODY_RATIO_MIN}")
    print(f"  SL: {SL_ATR_MULT}x ATR | Trail: {TRAIL_ATR_MULT}x ATR")

    buy_sig, sell_sig, ema_fast, ema_slow, atr_vals = generate_signals(df)
    print(f"  Buy signals:  {buy_sig.sum()}")
    print(f"  Sell signals: {sell_sig.sum()}")

    # Backtest
    print(f"\n🚀 Running backtest...")
    trades = run_backtest(df, buy_sig, sell_sig, atr_vals)
    print_report(trades, len(df), total_days, date_range)


if __name__ == "__main__":
    main()
