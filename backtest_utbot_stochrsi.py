"""
backtest_utbot_stochrsi.py — Backtest: UT Bot Alerts + Stochastic RSI + ATR Gating

Strategy Rules:
  ENTRY:
    - UT Bot fires BUY/SELL signal
    - Stochastic RSI confirms (K crosses above D = buy, K crosses below D = sell)
    - ATR(14) must be >= 13 to enter new trade
    - Time must be between 12:50 PM and 3:10 PM

  EXIT:
    - Trailing stop loss (ATR-based)
    - Opposite signal (UT Bot fires reverse + StochRSI confirms)
    - Square off at 3:24 PM
    - If ATR drops below 13 while in trade → keep position, let SL/opposite handle it

Usage:
    python backtest_utbot_stochrsi.py
    python backtest_utbot_stochrsi.py --file nifty_3min_data.csv
    python backtest_utbot_stochrsi.py --file nifty_1min_data.csv --atr-gate 15
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, time as dt_time
from dataclasses import dataclass, field
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
TRADING_START = dt_time(9, 15)     # Signal checking starts
TRADING_END = dt_time(15, 15)      # No new trades after 3:15 PM
SQUARE_OFF_TIME = dt_time(15, 24)  # Force close all at 3:24 PM
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 25)
ATR_GATE = 13.0              # Minimum ATR to enter new trade
ATR_PERIOD = 14               # ATR calculation period
SL_ATR_MULT = 2.0             # Stop loss = entry ± (ATR × multiplier)
TRAIL_ATR_MULT = 1.5          # Trailing SL distance = ATR × multiplier
MIN_HOLD_CANDLES = 3          # Min candles before opposite signal can close

# Stochastic RSI params
STOCH_RSI_PERIOD = 14
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3


# ─────────── Data Models ───────────
@dataclass
class Position:
    direction: str            # "CE" or "PE"
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


# ─────────── Indicator: UT Bot Alerts ───────────

def calculate_ut_bot(data, a=2, c=100):
    """
    UT Bot Alert — EWM-based trailing stop signal.
    Returns: buy_signal (bool array), sell_signal (bool array)
    """
    close = data['Close'].values
    xATR = pd.Series(close).diff().abs().ewm(span=c, adjust=False).mean().values
    nLoss = a * xATR

    trail = np.zeros(len(data))
    trail[0] = close[0]

    for i in range(1, len(data)):
        if close[i] > trail[i-1] and close[i-1] > trail[i-1]:
            trail[i] = max(trail[i-1], close[i] - nLoss[i])
        elif close[i] < trail[i-1] and close[i-1] < trail[i-1]:
            trail[i] = min(trail[i-1], close[i] + nLoss[i])
        elif close[i] > trail[i-1]:
            trail[i] = close[i] - nLoss[i]
        else:
            trail[i] = close[i] + nLoss[i]

    # Position: 1=above trail (bullish), -1=below trail (bearish)
    pos = np.zeros(len(data))
    for i in range(1, len(data)):
        if close[i-1] < trail[i-1] and close[i] > trail[i-1]:
            pos[i] = 1
        elif close[i-1] > trail[i-1] and close[i] < trail[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]

    # Signals: fire on position CHANGE only
    buy_signal = np.zeros(len(data), dtype=bool)
    sell_signal = np.zeros(len(data), dtype=bool)
    for i in range(1, len(data)):
        if pos[i] == 1 and pos[i-1] != 1:
            buy_signal[i] = True
        elif pos[i] == -1 and pos[i-1] != -1:
            sell_signal[i] = True

    return buy_signal, sell_signal, pos


# ─────────── Indicator: Stochastic RSI ───────────

def calculate_stochastic_rsi(data, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """
    Stochastic RSI = Stochastic oscillator applied to RSI values.
    Returns: stoch_k, stoch_d, k_cross_above_d (buy), k_cross_below_d (sell)
    """
    close = data['Close']

    # Step 1: Calculate RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Step 2: Apply Stochastic to RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan) * 100

    # Step 3: Smooth K and D
    stoch_k = stoch_rsi.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()

    # Step 4: Crossover signals
    k_cross_up = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))    # K crosses above D = bullish
    k_cross_down = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))  # K crosses below D = bearish

    return stoch_k.values, stoch_d.values, k_cross_up.values, k_cross_down.values


# ─────────── Indicator: ATR (RMA) ───────────

def calculate_atr(data, period=14):
    """ATR using True Range + RMA (Wilder's smoothing)."""
    high = data['High']
    low = data['Low']
    prev_close = data['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]

    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr.values


# ─────────── Backtest Engine ───────────

def run_backtest(df, ut_buy, ut_sell, stoch_buy, stoch_sell, stoch_k, stoch_d, atr,
                 atr_gate=13.0, sl_mult=2.0, trail_mult=1.5, min_hold=3):
    """
    Combined strategy backtest.

    ENTRY requires ALL of:
      1. UT Bot signal (buy or sell)
      2. Stochastic RSI confirmation (K cross D in same direction)
      3. ATR >= atr_gate
      4. Time between TRADING_START and TRADING_END
      5. No existing position

    OPEN POSITION management:
      - If ATR drops below atr_gate → keep position, don't close
      - Trailing SL continues regardless of ATR
      - Opposite signal can close regardless of ATR
      - Square off at SQUARE_OFF_TIME
    """
    trades = []
    open_pos = None
    prev_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time() if hasattr(row['Time'], 'time') else row['Time']
        curr_date = row['Time'].date() if hasattr(row['Time'], 'date') else None
        close = float(row['Close'])
        high = float(row['High'])

        # ── Day boundary: force close any open position from previous day ──
        if prev_date is not None and curr_date != prev_date and open_pos:
            # Use previous candle's close to square off
            prev_row = df.iloc[i-1]
            pnl = _calc_pnl(open_pos, float(prev_row['Close']))
            trades.append(Trade(
                open_pos.direction, open_pos.entry_price, float(prev_row['Close']),
                open_pos.entry_time, prev_row['Time'], round(pnl, 2),
                "DAY_END_CLOSE", open_pos.entry_atr, float(atr[i-1])
            ))
            open_pos = None
        prev_date = curr_date
        low = float(row['Low'])
        curr_atr = float(atr[i])

        if not (MARKET_OPEN <= t <= MARKET_CLOSE):
            continue

        # ── Square off at 3:24 PM ──
        if open_pos and t >= SQUARE_OFF_TIME:
            pnl = _calc_pnl(open_pos, close)
            trades.append(Trade(
                open_pos.direction, open_pos.entry_price, close,
                open_pos.entry_time, row['Time'], round(pnl, 2),
                "SQUARE_OFF", open_pos.entry_atr, curr_atr
            ))
            open_pos = None
            continue

        # ── Check stop loss (using high/low for more realistic fills) ──
        if open_pos:
            sl_hit = False
            sl_price = close

            if open_pos.direction == "CE":
                if low <= open_pos.stop_loss:  # SL would have been hit at SL level
                    sl_hit = True
                    sl_price = open_pos.stop_loss
            elif open_pos.direction == "PE":
                if high >= open_pos.stop_loss:
                    sl_hit = True
                    sl_price = open_pos.stop_loss

            if sl_hit:
                pnl = _calc_pnl(open_pos, sl_price)
                trades.append(Trade(
                    open_pos.direction, open_pos.entry_price, sl_price,
                    open_pos.entry_time, row['Time'], round(pnl, 2),
                    "STOP_LOSS", open_pos.entry_atr, curr_atr
                ))
                open_pos = None

        # ── Update trailing SL (regardless of ATR level) ──
        if open_pos and curr_atr > 0:
            if open_pos.direction == "CE":
                # Trail upward: new SL = current high - ATR * multiplier
                new_sl = high - (curr_atr * trail_mult)
                if new_sl > open_pos.stop_loss:
                    open_pos.stop_loss = new_sl
            elif open_pos.direction == "PE":
                # Trail downward: new SL = current low + ATR * multiplier
                new_sl = low + (curr_atr * trail_mult)
                if new_sl < open_pos.stop_loss:
                    open_pos.stop_loss = new_sl

        # ── Check for combined signals ──
        is_ut_buy = bool(ut_buy[i])
        is_ut_sell = bool(ut_sell[i])
        is_stoch_buy = bool(stoch_buy[i])
        is_stoch_sell = bool(stoch_sell[i])

        # Combined signal: UT Bot + StochRSI must agree
        combined_buy = is_ut_buy and is_stoch_buy
        combined_sell = is_ut_sell and is_stoch_sell

        # Also allow: UT Bot signal + StochRSI already in favorable zone
        # (K > D for buy, K < D for sell — within last 2 candles)
        if is_ut_buy and not combined_buy:
            # Check if StochRSI K is above D (bullish zone)
            if i >= 1 and not np.isnan(stoch_k[i]) and not np.isnan(stoch_d[i]):
                if stoch_k[i] > stoch_d[i] and stoch_k[i] < 80:  # Not overbought
                    combined_buy = True

        if is_ut_sell and not combined_sell:
            if i >= 1 and not np.isnan(stoch_k[i]) and not np.isnan(stoch_d[i]):
                if stoch_k[i] < stoch_d[i] and stoch_k[i] > 20:  # Not oversold
                    combined_sell = True

        # ── Opposite signal close (with min hold time) ──
        if combined_buy and open_pos and open_pos.direction == "PE":
            if (i - open_pos.entry_idx) >= min_hold:
                pnl = _calc_pnl(open_pos, close)
                trades.append(Trade(
                    open_pos.direction, open_pos.entry_price, close,
                    open_pos.entry_time, row['Time'], round(pnl, 2),
                    "OPPOSITE_SIGNAL", open_pos.entry_atr, curr_atr
                ))
                open_pos = None

        elif combined_sell and open_pos and open_pos.direction == "CE":
            if (i - open_pos.entry_idx) >= min_hold:
                pnl = _calc_pnl(open_pos, close)
                trades.append(Trade(
                    open_pos.direction, open_pos.entry_price, close,
                    open_pos.entry_time, row['Time'], round(pnl, 2),
                    "OPPOSITE_SIGNAL", open_pos.entry_atr, curr_atr
                ))
                open_pos = None

        # ── New entry: requires no position + time window + ATR gate ──
        if not open_pos and TRADING_START <= t <= TRADING_END:
            # ATR must be >= gate for NEW entries only
            if curr_atr < atr_gate:
                continue

            if combined_buy and curr_atr > 0:
                sl = close - (curr_atr * sl_mult)
                open_pos = Position("CE", close, row['Time'], sl, i, curr_atr)

            elif combined_sell and curr_atr > 0:
                sl = close + (curr_atr * sl_mult)
                open_pos = Position("PE", close, row['Time'], sl, i, curr_atr)

    # Close remaining position
    if open_pos and len(df) > 0:
        last = df.iloc[-1]
        pnl = _calc_pnl(open_pos, float(last['Close']))
        trades.append(Trade(
            open_pos.direction, open_pos.entry_price, float(last['Close']),
            open_pos.entry_time, last['Time'], round(pnl, 2),
            "END_OF_DATA", open_pos.entry_atr, float(atr[-1])
        ))

    return trades


def _calc_pnl(pos: Position, exit_price: float) -> float:
    if pos.direction == "CE":
        return exit_price - pos.entry_price
    else:
        return pos.entry_price - exit_price


# ─────────── Report ───────────

def print_report(trades, strategy_name, data_info=""):
    if not trades:
        print(f"\n❌ No trades generated!")
        return

    pnl_list = [t.pnl for t in trades]
    total = sum(pnl_list)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100

    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (cumulative - peak).min()

    # Daily stats
    daily = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + t.pnl
    prof_days = sum(1 for v in daily.values() if v > 0)

    # Close reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    # Direction breakdown
    ce_trades = [t for t in trades if t.direction == "CE"]
    pe_trades = [t for t in trades if t.direction == "PE"]

    # Win/loss streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for p in pnl_list:
        if p > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS")
    print(f"  Strategy: {strategy_name}")
    if data_info:
        print(f"  {data_info}")
    print(f"{'='*70}")

    print(f"\n📊 OVERVIEW")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Trading days:     {len(daily)}")
    print(f"  Avg trades/day:   {len(trades)/max(len(daily),1):.1f}")

    print(f"\n💰 PERFORMANCE")
    print(f"  Total P&L:        {total:+.2f} pts")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Avg win:          {np.mean([t.pnl for t in wins]):+.2f} pts" if wins else "  Avg win: N/A")
    print(f"  Avg loss:         {np.mean([t.pnl for t in losses]):+.2f} pts" if losses else "  Avg loss: N/A")
    print(f"  Best trade:       {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:      {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:     {max_dd:.2f} pts")
    print(f"  Avg P&L/trade:    {total/len(trades):+.2f} pts")
    print(f"  Avg P&L/day:      {total/max(len(daily),1):+.2f} pts")
    print(f"  Profitable days:  {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    print(f"\n📈 STREAKS")
    print(f"  Max win streak:   {max_win_streak}")
    print(f"  Max loss streak:  {max_loss_streak}")

    print(f"\n🎯 BY DIRECTION")
    if ce_trades:
        ce_pnl = sum(t.pnl for t in ce_trades)
        ce_wr = sum(1 for t in ce_trades if t.pnl > 0) / len(ce_trades) * 100
        print(f"  CE trades: {len(ce_trades)} | P&L: {ce_pnl:+.2f} pts | Win rate: {ce_wr:.1f}%")
    if pe_trades:
        pe_pnl = sum(t.pnl for t in pe_trades)
        pe_wr = sum(1 for t in pe_trades if t.pnl > 0) / len(pe_trades) * 100
        print(f"  PE trades: {len(pe_trades)} | P&L: {pe_pnl:+.2f} pts | Win rate: {pe_wr:.1f}%")

    print(f"\n🔍 CLOSE REASONS")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        reason_trades = [t for t in trades if t.close_reason == reason]
        reason_pnl = sum(t.pnl for t in reason_trades)
        print(f"  {reason:20s}: {count:4d} trades | P&L: {reason_pnl:+.2f} pts")

    # Top 5 best/worst days
    sorted_days = sorted(daily.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_days) >= 5:
        print(f"\n📅 TOP 5 BEST DAYS")
        for day, pnl in sorted_days[:5]:
            print(f"  {day}: {pnl:+.2f} pts")
        print(f"\n📅 TOP 5 WORST DAYS")
        for day, pnl in sorted_days[-5:]:
            print(f"  {day}: {pnl:+.2f} pts")

    print(f"\n{'='*70}")

    # Save trade log
    trades_df = pd.DataFrame([{
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'pnl': t.pnl,
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'close_reason': t.close_reason,
        'entry_atr': round(t.entry_atr, 2),
        'exit_atr': round(t.exit_atr, 2),
    } for t in trades])
    output = "backtest_utbot_stochrsi_trades.csv"
    trades_df.to_csv(output, index=False)
    print(f"  💾 Trade log saved to: {output}")


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(description="Backtest UT Bot + Stochastic RSI + ATR")
    parser.add_argument("--file", default="nifty_3min_data.csv", help="Data file (CSV)")
    parser.add_argument("--atr-gate", type=float, default=ATR_GATE, help=f"ATR gate for new entries (default: {ATR_GATE})")
    parser.add_argument("--sl-mult", type=float, default=SL_ATR_MULT, help=f"SL = ATR × multiplier (default: {SL_ATR_MULT})")
    parser.add_argument("--trail-mult", type=float, default=TRAIL_ATR_MULT, help=f"Trail SL = ATR × multiplier (default: {TRAIL_ATR_MULT})")
    parser.add_argument("--ut-a", type=float, default=2, help="UT Bot sensitivity (default: 2)")
    parser.add_argument("--ut-c", type=int, default=100, help="UT Bot ATR period (default: 100)")
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD_CANDLES, help=f"Min candles before opposite close (default: {MIN_HOLD_CANDLES})")
    args = parser.parse_args()

    # Find data file
    data_file = args.file
    if not os.path.exists(data_file):
        for alt in ["nifty_3min_data.csv", "nifty_5min_data.csv", "nifty_1min_data.csv"]:
            if os.path.exists(alt):
                data_file = alt
                break

    if not os.path.exists(data_file):
        print("❌ No data file found. Run fetch_nifty_data.py first.")
        return

    # Load data
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    total_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"
    print(f"Loaded {len(df)} candles | {total_days} days | {date_range}")

    # Calculate indicators
    print("\n📊 Calculating indicators...")

    # UT Bot
    ut_buy, ut_sell, ut_pos = calculate_ut_bot(df, a=args.ut_a, c=args.ut_c)
    print(f"  UT Bot (a={args.ut_a}, c={args.ut_c}): {ut_buy.sum()} buy signals, {ut_sell.sum()} sell signals")

    # Stochastic RSI
    stoch_k, stoch_d, stoch_buy, stoch_sell = calculate_stochastic_rsi(
        df, rsi_period=STOCH_RSI_PERIOD, stoch_period=STOCH_RSI_PERIOD,
        k_smooth=STOCH_K_PERIOD, d_smooth=STOCH_D_PERIOD
    )
    print(f"  Stochastic RSI (14,3,3): {stoch_buy.sum()} K↑D signals, {stoch_sell.sum()} K↓D signals")

    # ATR
    atr = calculate_atr(df, period=ATR_PERIOD)
    atr_above_gate = (atr >= args.atr_gate).sum()
    print(f"  ATR({ATR_PERIOD}): avg={np.nanmean(atr):.2f}, candles >= {args.atr_gate}: {atr_above_gate}/{len(atr)} ({atr_above_gate/len(atr)*100:.1f}%)")

    # Run backtest
    print(f"\n🚀 Running backtest...")
    print(f"  Trading window: {TRADING_START.strftime('%H:%M')} - {TRADING_END.strftime('%H:%M')}")
    print(f"  ATR gate: >= {args.atr_gate} (new entries only)")
    print(f"  SL: {args.sl_mult}x ATR | Trail: {args.trail_mult}x ATR")
    print(f"  Min hold: {args.min_hold} candles")

    trades = run_backtest(
        df, ut_buy, ut_sell, stoch_buy, stoch_sell, stoch_k, stoch_d, atr,
        atr_gate=args.atr_gate, sl_mult=args.sl_mult,
        trail_mult=args.trail_mult, min_hold=args.min_hold,
    )

    strategy_name = f"UT Bot (a={args.ut_a},c={args.ut_c}) + StochRSI(14,3,3) + ATR({ATR_PERIOD})>={args.atr_gate}"
    data_info = f"Data: {date_range} | {total_days} days | {len(df)} candles"
    print_report(trades, strategy_name, data_info)


if __name__ == "__main__":
    main()
