"""
strategy_optimizer.py — Test multiple strategies + parameters on NIFTY 1-min data.

Strategies tested:
  1. UT Bot on 5-min candles (reduces noise dramatically)
  2. Supertrend (ATR-based trend follower, proven for index trading)
  3. EMA Crossover on 5-min candles
  4. UT Bot parameter sweep (best a, c combination)

All strategies use:
  - ATR-based stop loss (wider, more adaptive than candle high/low)
  - Minimum hold time (prevent immediate whipsaw closes)
  - Trading window 12:30 - 15:10, square off 15:24

Usage:
    python strategy_optimizer.py
    python strategy_optimizer.py --file nifty_1min_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────── Common Config ───────────
TRADING_START = dt_time(12, 30)
TRADING_END = dt_time(15, 10)
SQUARE_OFF_TIME = dt_time(15, 24)
MARKET_OPEN = dt_time(9, 16)
MARKET_CLOSE = dt_time(15, 30)


# ─────────── Data Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    entry_idx: int  # candle index at entry (for min hold time)

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    close_reason: str


# ─────────── Indicator Functions ───────────

def resample_candles(df_1min: pd.DataFrame, interval_min: int = 3) -> pd.DataFrame:
    """Resample 1-min candles to any interval (3-min, 5-min, etc.)."""
    df = df_1min.copy()
    df = df.set_index('Time')
    resampled = df.resample(f'{interval_min}min', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna()
    resampled = resampled.reset_index()
    # Filter to market hours only
    resampled = resampled[
        (resampled['Time'].dt.time >= dt_time(9, 15)) &
        (resampled['Time'].dt.time <= dt_time(15, 25))
    ]
    return resampled.reset_index(drop=True)


def calculate_true_range(data):
    high = data['High']
    low = data['Low']
    prev_close = data['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    return true_range


def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()


def calculate_ut_bot(data, a=2, c=100):
    """UT Bot Alert — user's exact EWM-based code."""
    xATR = data['Close'].diff().abs().ewm(span=c, adjust=False).mean()
    nLoss = a * xATR
    src = data['Close']
    trail = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        if i == 0:
            trail.iloc[i] = src.iloc[i]
        elif src.iloc[i] > trail.iloc[i-1] and src.iloc[i-1] > trail.iloc[i-1]:
            trail.iloc[i] = max(trail.iloc[i-1], src.iloc[i] - nLoss.iloc[i])
        elif src.iloc[i] < trail.iloc[i-1] and src.iloc[i-1] < trail.iloc[i-1]:
            trail.iloc[i] = min(trail.iloc[i-1], src.iloc[i] + nLoss.iloc[i])
        elif src.iloc[i] > trail.iloc[i-1]:
            trail.iloc[i] = src.iloc[i] - nLoss.iloc[i]
        else:
            trail.iloc[i] = src.iloc[i] + nLoss.iloc[i]

    pos = np.zeros(len(data))
    for i in range(len(data)):
        if i == 0:
            pos[i] = 0
        elif src.iloc[i-1] < trail.iloc[i-1] and src.iloc[i] > trail.iloc[i-1]:
            pos[i] = 1
        elif src.iloc[i-1] > trail.iloc[i-1] and src.iloc[i] < trail.iloc[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]

    return (pos == 1).astype(bool), (pos == -1).astype(bool)


def calculate_supertrend(data, period=10, multiplier=3.0):
    """Supertrend indicator — ATR-based trend follower."""
    tr = calculate_true_range(data)
    atr = rma(tr, period)

    hl2 = (data['High'] + data['Low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=data.index, dtype=float)
    direction = pd.Series(index=data.index, dtype=int)  # 1=up, -1=down

    for i in range(len(data)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        # Adjust bands based on previous values
        if lower_band.iloc[i] > lower_band.iloc[i-1] or data['Close'].iloc[i-1] < lower_band.iloc[i-1]:
            pass  # keep current lower_band
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]

        if upper_band.iloc[i] < upper_band.iloc[i-1] or data['Close'].iloc[i-1] > upper_band.iloc[i-1]:
            pass  # keep current upper_band
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]

        # Direction
        if direction.iloc[i-1] == -1:  # was down
            if data['Close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1  # flip to up
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
        else:  # was up
            if data['Close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1  # flip to down
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]

    # Generate signals on direction change
    buy_signal = (direction == 1) & (direction.shift(1) == -1)
    sell_signal = (direction == -1) & (direction.shift(1) == 1)

    return buy_signal, sell_signal, supertrend, direction


def calculate_ema_crossover(data, fast=9, slow=21):
    """EMA crossover signals."""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()

    buy_signal = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    sell_signal = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    return buy_signal, sell_signal


def calculate_atr(data, period=14):
    tr = calculate_true_range(data)
    return rma(tr, period)


# ─────────── Generic Backtester ───────────

def run_backtest(df, buy_signals, sell_signals, atr_series,
                 atr_gate=0.0, sl_atr_mult=1.5, min_hold=3,
                 use_trailing=True, trail_atr_mult=2.0):
    """
    Generic backtest engine.

    Args:
        sl_atr_mult: Stop loss = entry ± (ATR × sl_atr_mult)
        min_hold: Minimum candles to hold before closing on opposite signal
        use_trailing: Enable ATR-based trailing stop loss
        trail_atr_mult: Trailing SL distance = ATR × trail_atr_mult
    """
    # Convert signals/ATR to numpy arrays for consistent indexing
    buy_arr = np.asarray(buy_signals)
    sell_arr = np.asarray(sell_signals)
    atr_arr = np.asarray(atr_series)

    trades = []
    open_pos = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time() if isinstance(row['Time'], datetime) else row['Time']
        close = row['Close']
        high = row['High']
        low = row['Low']
        atr = float(atr_arr[i]) if i < len(atr_arr) else 0

        if not (MARKET_OPEN <= t <= MARKET_CLOSE):
            continue

        # Square off
        if open_pos and t >= SQUARE_OFF_TIME:
            pnl = (close - open_pos.entry_price) if open_pos.direction == "CE" else (open_pos.entry_price - close)
            trades.append(Trade(open_pos.direction, open_pos.entry_price, close, open_pos.entry_time, row['Time'], round(pnl, 2), "SQUARE_OFF"))
            open_pos = None
            continue

        # Check stop loss
        if open_pos:
            sl_hit = False
            if open_pos.direction == "CE" and close <= open_pos.stop_loss:
                sl_hit = True
            elif open_pos.direction == "PE" and close >= open_pos.stop_loss:
                sl_hit = True

            if sl_hit:
                pnl = (close - open_pos.entry_price) if open_pos.direction == "CE" else (open_pos.entry_price - close)
                trades.append(Trade(open_pos.direction, open_pos.entry_price, close, open_pos.entry_time, row['Time'], round(pnl, 2), "STOP_LOSS"))
                open_pos = None

        # Trailing SL update
        if open_pos and use_trailing and atr > 0:
            if open_pos.direction == "CE":
                new_sl = close - (atr * trail_atr_mult)
                if new_sl > open_pos.stop_loss:
                    open_pos.stop_loss = new_sl
            elif open_pos.direction == "PE":
                new_sl = close + (atr * trail_atr_mult)
                if new_sl < open_pos.stop_loss:
                    open_pos.stop_loss = new_sl

        # Signal processing
        curr_buy = bool(buy_arr[i]) if i < len(buy_arr) else False
        curr_sell = bool(sell_arr[i]) if i < len(sell_arr) else False

        # Opposite signal close (with min hold time)
        if curr_buy and open_pos and open_pos.direction == "PE":
            if (i - open_pos.entry_idx) >= min_hold:
                pnl = open_pos.entry_price - close
                trades.append(Trade(open_pos.direction, open_pos.entry_price, close, open_pos.entry_time, row['Time'], round(pnl, 2), "OPPOSITE_SIGNAL"))
                open_pos = None

        elif curr_sell and open_pos and open_pos.direction == "CE":
            if (i - open_pos.entry_idx) >= min_hold:
                pnl = close - open_pos.entry_price
                trades.append(Trade(open_pos.direction, open_pos.entry_price, close, open_pos.entry_time, row['Time'], round(pnl, 2), "OPPOSITE_SIGNAL"))
                open_pos = None

        # New entry — requires: no position + trading window + ATR gate
        if not open_pos and TRADING_START <= t <= TRADING_END:
            if atr_gate > 0 and atr < atr_gate:
                continue

            if curr_buy:
                sl = close - (atr * sl_atr_mult) if atr > 0 else low
                open_pos = Position("CE", close, row['Time'], sl, i)

            elif curr_sell:
                sl = close + (atr * sl_atr_mult) if atr > 0 else high
                open_pos = Position("PE", close, row['Time'], sl, i)

    # Close any remaining position
    if open_pos and len(df) > 0:
        last = df.iloc[-1]
        pnl = (last['Close'] - open_pos.entry_price) if open_pos.direction == "CE" else (open_pos.entry_price - last['Close'])
        trades.append(Trade(open_pos.direction, open_pos.entry_price, last['Close'], open_pos.entry_time, last['Time'], round(pnl, 2), "END_OF_DATA"))

    return trades


def calc_metrics(trades):
    """Calculate key performance metrics."""
    if not trades:
        return {"total_pnl": 0, "trades": 0, "win_rate": 0, "max_dd": 0, "avg_pnl": 0, "profitable_days_pct": 0}

    pnl_list = [t.pnl for t in trades]
    total = sum(pnl_list)
    wins = [p for p in pnl_list if p > 0]
    win_rate = len(wins) / len(pnl_list) * 100

    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (cumulative - peak).min()

    # Daily P&L
    daily = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + t.pnl
    profitable_days = sum(1 for v in daily.values() if v > 0)
    pct = profitable_days / max(len(daily), 1) * 100

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean([p for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    return {
        "total_pnl": round(total, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_dd, 2),
        "avg_pnl": round(total / len(trades), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profitable_days_pct": round(pct, 0),
        "daily_count": len(daily),
    }


# ─────────── Strategy Runners ───────────

def test_strategy_resampled(df_1min, strategy_fn, interval_min=3, min_candles=10, **kwargs):
    """Generic runner: resample to N-min, apply strategy, backtest."""
    results = []
    for day, day_1min in df_1min.groupby(df_1min['Time'].dt.date):
        day_resampled = resample_candles(day_1min, interval_min=interval_min)
        if len(day_resampled) < min_candles:
            continue

        # Get signals from strategy function
        signals = strategy_fn(day_resampled, **{k: v for k, v in kwargs.items()
                              if k not in ('atr_gate', 'sl_mult', 'min_hold', 'trail_mult')})

        if len(signals) == 2:
            buy, sell = signals
        else:
            buy, sell = signals[0], signals[1]

        atr = calculate_atr(day_resampled, 14)
        trades = run_backtest(
            day_resampled, buy, sell, atr,
            atr_gate=kwargs.get('atr_gate', 0),
            sl_atr_mult=kwargs.get('sl_mult', 2.0),
            min_hold=kwargs.get('min_hold', 2),
            trail_atr_mult=kwargs.get('trail_mult', 2.0),
        )
        results.extend(trades)
    return results


def test_ut_bot_1min(df_1min, a=2, c=100, atr_gate=6.9, sl_mult=1.5, min_hold=5):
    """UT Bot on 1-minute candles (original but with improved SL)."""
    results = []
    for day, day_df in df_1min.groupby(df_1min['Time'].dt.date):
        day_df = day_df.reset_index(drop=True)
        if len(day_df) < 10:
            continue
        buy, sell = calculate_ut_bot(day_df, a=a, c=c)
        atr = calculate_atr(day_df, 14)
        trades = run_backtest(day_df, buy, sell, atr,
                              atr_gate=atr_gate, sl_atr_mult=sl_mult,
                              min_hold=min_hold, trail_atr_mult=2.0)
        results.extend(trades)
    return results


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="nifty_1min_data.csv")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}. Run fetch_nifty_data.py first.")
        return

    print("Loading data...")
    df_1min = pd.read_csv(args.file)
    df_1min['Time'] = pd.to_datetime(df_1min['Time'])
    df_1min = df_1min.sort_values('Time').reset_index(drop=True)
    print(f"  1-min: {len(df_1min)} candles | {df_1min['Time'].dt.date.nunique()} days")

    # Load native 3-min and 5-min CSVs if available (more historical data)
    tf_data = {}
    for tf, csv_file in [(3, "nifty_3min_data.csv"), (5, "nifty_5min_data.csv")]:
        if os.path.exists(csv_file):
            df_tf = pd.read_csv(csv_file)
            df_tf['Time'] = pd.to_datetime(df_tf['Time'])
            df_tf = df_tf.sort_values('Time').reset_index(drop=True)
            tf_data[tf] = df_tf
            print(f"  {tf}-min: {len(df_tf)} candles | {df_tf['Time'].dt.date.nunique()} days (native CSV ✅)")
        else:
            print(f"  {tf}-min: will resample from 1-min data")

    print("\n" + "=" * 100)
    print("  STRATEGY OPTIMIZER — Testing multiple strategies and parameters")
    print("=" * 100)

    all_results = []

    # ─── Helper: run strategy on native CSV data (per day) ───
    def run_on_native(native_df, strategy_fn, sl_mult=2.0, min_hold=2, min_candles=10):
        results = []
        for day, day_df in native_df.groupby(native_df['Time'].dt.date):
            day_df = day_df.reset_index(drop=True)
            if len(day_df) < min_candles:
                continue
            signals = strategy_fn(day_df)
            buy, sell = signals[0], signals[1]
            atr = calculate_atr(day_df, 14)
            trades = run_backtest(day_df, buy, sell, atr,
                                  sl_atr_mult=sl_mult, min_hold=min_hold, trail_atr_mult=2.0)
            results.extend(trades)
        return results

    # ─── 1. Supertrend on 3-min and 5-min ───
    for tf in [3, 5]:
        print(f"\n🔵 Testing Supertrend on {tf}-min candles...")
        for period in [7, 10, 14]:
            for mult in [2.0, 3.0, 4.0]:
                for sl in [1.5, 2.0, 3.0]:
                    label = f"Supertrend {tf}min (P={period}, M={mult}, SL={sl}x)"
                    if tf in tf_data:
                        trades = run_on_native(tf_data[tf], lambda d, p=period, m=mult: calculate_supertrend(d, period=p, multiplier=m)[:2], sl_mult=sl, min_hold=2)
                    else:
                        trades = test_strategy_resampled(
                            df_1min, lambda d, p=period, m=mult: calculate_supertrend(d, period=p, multiplier=m)[:2],
                            interval_min=tf, sl_mult=sl, min_hold=2,
                        )
                    m = calc_metrics(trades)
                    all_results.append({"strategy": label, **m})

    # ─── 2. UT Bot on 3-min and 5-min ───
    for tf in [3, 5]:
        print(f"🟢 Testing UT Bot on {tf}-min candles...")
        for a in [1, 2, 3, 4]:
            for c in [14, 21, 50, 100]:
                for sl in [1.5, 2.0, 3.0]:
                    label = f"UT Bot {tf}min (a={a}, c={c}, SL={sl}x)"
                    if tf in tf_data:
                        trades = run_on_native(tf_data[tf], lambda d, a_=a, c_=c: calculate_ut_bot(d, a=a_, c=c_), sl_mult=sl, min_hold=2)
                    else:
                        trades = test_strategy_resampled(
                            df_1min, lambda d, a_=a, c_=c: calculate_ut_bot(d, a=a_, c=c_),
                            interval_min=tf, sl_mult=sl, min_hold=2,
                        )
                    m = calc_metrics(trades)
                    all_results.append({"strategy": label, **m})

    # ─── 3. EMA Crossover on 3-min and 5-min ───
    for tf in [3, 5]:
        print(f"🟡 Testing EMA Crossover on {tf}-min candles...")
        for fast in [5, 9, 13]:
            for slow in [21, 34, 50]:
                if fast >= slow:
                    continue
                for sl in [1.5, 2.0, 3.0]:
                    label = f"EMA Cross {tf}min ({fast}/{slow}, SL={sl}x)"
                    if tf in tf_data:
                        trades = run_on_native(tf_data[tf], lambda d, f_=fast, s_=slow: calculate_ema_crossover(d, fast=f_, slow=s_), sl_mult=sl, min_hold=2)
                    else:
                        trades = test_strategy_resampled(
                            df_1min, lambda d, f_=fast, s_=slow: calculate_ema_crossover(d, fast=f_, slow=s_),
                            interval_min=tf, sl_mult=sl, min_hold=2,
                        )
                    m = calc_metrics(trades)
                    all_results.append({"strategy": label, **m})

    # ─── 4. UT Bot on 1-min with improved SL ───
    print("🔴 Testing UT Bot on 1-min with ATR-based SL...")
    for a in [2, 3, 4, 5]:
        for c in [50, 100, 150]:
            for sl in [2.0, 3.0, 4.0]:
                label = f"UT Bot 1min (a={a}, c={c}, SL={sl}x, hold=5)"
                trades = test_ut_bot_1min(df_1min, a=a, c=c, atr_gate=6.9, sl_mult=sl, min_hold=5)
                m = calc_metrics(trades)
                all_results.append({"strategy": label, **m})

    # ─── Results ───
    results_df = pd.DataFrame(all_results)

    # Filter: must have at least 50 trades
    results_df = results_df[results_df['trades'] >= 50]

    # Sort by total P&L
    results_df = results_df.sort_values('total_pnl', ascending=False).reset_index(drop=True)

    # Save full results
    results_df.to_csv("optimizer_results.csv", index=False)

    # Print top 20
    print("\n" + "=" * 100)
    print("  TOP 20 STRATEGIES (by Total P&L)")
    print("=" * 100)
    print(f"\n{'#':>3} {'Strategy':<45} {'P&L':>10} {'Trades':>7} {'Win%':>6} {'AvgW':>7} {'AvgL':>7} {'MaxDD':>10} {'ProfDays':>9}")
    print("-" * 100)

    for i, row in results_df.head(20).iterrows():
        pnl_emoji = "✅" if row['total_pnl'] > 0 else "❌"
        print(f"{i+1:3d} {row['strategy']:<45} {row['total_pnl']:>+10.2f} {row['trades']:>7d} {row['win_rate']:>5.1f}% {row['avg_win']:>+7.2f} {row['avg_loss']:>+7.2f} {row['max_dd']:>10.2f} {row['profitable_days_pct']:>8.0f}% {pnl_emoji}")

    # Print worst 5 for comparison
    print(f"\n{'':>3} {'--- WORST 5 ---':<45}")
    for i, row in results_df.tail(5).iterrows():
        print(f"{'':>3} {row['strategy']:<45} {row['total_pnl']:>+10.2f} {row['trades']:>7d} {row['win_rate']:>5.1f}% {row['avg_win']:>+7.2f} {row['avg_loss']:>+7.2f} {row['max_dd']:>10.2f} {row['profitable_days_pct']:>8.0f}% ❌")

    print(f"\n💾 Full results ({len(results_df)} strategies) saved to: optimizer_results.csv")

    # Best strategy details
    if len(results_df) > 0 and results_df.iloc[0]['total_pnl'] > 0:
        best = results_df.iloc[0]
        print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
        print(f"   Total P&L: {best['total_pnl']:+.2f} pts | Win rate: {best['win_rate']:.1f}% | Max DD: {best['max_dd']:.2f} pts")
        print(f"   Avg win: {best['avg_win']:+.2f} | Avg loss: {best['avg_loss']:+.2f} | Trades: {best['trades']}")
        print(f"   Profitable days: {best['profitable_days_pct']:.0f}%")
    else:
        print("\n⚠️  No profitable strategy found with current parameters.")
        print("   Consider: expanding trading window, adjusting SL, or using different timeframes.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
