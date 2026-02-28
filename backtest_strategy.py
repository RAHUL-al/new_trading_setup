"""
backtest_strategy.py — Backtest the NIFTY trading strategy on historical 1-min data.

Strategy:
  - UT Bot Alerts (a=2, c=100, EWM-based) for buy/sell signals
  - ATR RMA(14) > 6.9 gate for new entries
  - EMA(9)/EMA(21) trend filter (BUY only in uptrend, SELL only in downtrend)
  - Trailing stop loss using candle high/low
  - Trading window: 12:30 to 15:10 (new entries only)
  - Square off at 15:24

Usage:
    python backtest_strategy.py
    python backtest_strategy.py --file nifty_1min_data.csv --start 2025-06-01 --end 2025-12-31

Output: Prints detailed trade log + performance metrics.
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, time as dt_time
from dataclasses import dataclass, field
from typing import Optional, List


# ─────────── Strategy Parameters (match live config) ───────────
UT_BOT_A = 2          # Key value (ATR multiplier for trailing stop)
UT_BOT_C = 100        # ATR period for UT Bot (EWM span)
ATR_RMA_PERIOD = 14   # Separate ATR period for gating
ATR_MIN_THRESHOLD = 6.9

EMA_FAST = 9
EMA_SLOW = 21

TRADING_START = dt_time(12, 30)
TRADING_END = dt_time(15, 10)
SQUARE_OFF_TIME = dt_time(15, 24)
MARKET_OPEN = dt_time(9, 16)      # Signal engine starts at 9:16
MARKET_CLOSE = dt_time(15, 30)

MIN_CANDLES_FOR_SIGNAL = 5  # Need at least 5 candles before generating signals


# ─────────── Indicator Functions (exact copies from live code) ───────────

def calculate_true_range(data):
    """True Range = max(High-Low, abs(High-prev_Close), abs(Low-prev_Close))"""
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
    """RMA (Wilder's Smoothing) — EWM with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calculate_indicators(data, a=2, c=100, h=False):
    """UT Bot Alert indicator — user's exact EWM-based code."""
    xATR = data['Close'].diff().abs().ewm(span=c, adjust=False).mean()
    nLoss = a * xATR
    src = data['Close']
    xATRTrailingStop = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        if i == 0:
            xATRTrailingStop.iloc[i] = src.iloc[i]
        elif src.iloc[i] > xATRTrailingStop.iloc[i - 1] and src.iloc[i - 1] > xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = max(xATRTrailingStop.iloc[i - 1], src.iloc[i] - nLoss.iloc[i])
        elif src.iloc[i] < xATRTrailingStop.iloc[i - 1] and src.iloc[i - 1] < xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = min(xATRTrailingStop.iloc[i - 1], src.iloc[i] + nLoss.iloc[i])
        elif src.iloc[i] > xATRTrailingStop.iloc[i - 1]:
            xATRTrailingStop.iloc[i] = src.iloc[i] - nLoss.iloc[i]
        else:
            xATRTrailingStop.iloc[i] = src.iloc[i] + nLoss.iloc[i]

    pos = np.zeros(len(data))
    for i in range(len(data)):
        if i == 0:
            pos[i] = 0
        elif src.iloc[i - 1] < xATRTrailingStop.iloc[i - 1] and src.iloc[i] > xATRTrailingStop.iloc[i - 1]:
            pos[i] = 1
        elif src.iloc[i - 1] > xATRTrailingStop.iloc[i - 1] and src.iloc[i] < xATRTrailingStop.iloc[i - 1]:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    signals = pd.DataFrame(index=data.index)
    signals['buy'] = (pos == 1)
    signals['sell'] = (pos == -1)
    return signals


def calculate_atr_rma(data, period=14):
    """Separate ATR using True Range + RMA."""
    true_range = calculate_true_range(data)
    return rma(true_range, period)


def calculate_ema_trend(data, fast=9, slow=21):
    """EMA trend filter — returns trend direction per row."""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    return ema_fast, ema_slow


# ─────────── Position / Trade Models ───────────

@dataclass
class Position:
    direction: str          # "CE" (buy/bullish) or "PE" (sell/bearish)
    entry_price: float      # NIFTY index price at entry
    entry_time: datetime
    stop_loss: float        # Trailing SL (NIFTY index level)
    highest_high: float     # For CE trailing
    lowest_low: float       # For PE trailing

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    close_reason: str


# ─────────── Backtester ───────────

class Backtester:
    def __init__(self):
        self.trades: List[Trade] = []
        self.open_pos: Optional[Position] = None
        self.daily_trades: dict = {}

    def run(self, df: pd.DataFrame):
        """Run backtest on preprocessed DataFrame with all indicators."""
        print("\n" + "=" * 70)
        print("  BACKTEST RUNNING")
        print(f"  Strategy: UT Bot (a={UT_BOT_A}, c={UT_BOT_C}) + ATR RMA({ATR_RMA_PERIOD}) > {ATR_MIN_THRESHOLD} + EMA({EMA_FAST}/{EMA_SLOW})")
        print(f"  Trading window: {TRADING_START.strftime('%H:%M')} - {TRADING_END.strftime('%H:%M')} | Square off: {SQUARE_OFF_TIME.strftime('%H:%M')}")
        print("=" * 70)

        prev_buy = False
        prev_sell = False

        for i in range(len(df)):
            row = df.iloc[i]
            candle_time = row['Time']
            t = candle_time.time()
            nifty_close = row['Close']
            nifty_high = row['High']
            nifty_low = row['Low']

            # ── Check stop loss (every candle, during market hours) ──
            if self.open_pos and MARKET_OPEN <= t <= MARKET_CLOSE:
                if self.open_pos.direction == "CE" and nifty_close <= self.open_pos.stop_loss:
                    self._close_position(nifty_close, candle_time, "STOP_LOSS")
                elif self.open_pos.direction == "PE" and nifty_close >= self.open_pos.stop_loss:
                    self._close_position(nifty_close, candle_time, "STOP_LOSS")

            # ── Square off at 15:24 ──
            if self.open_pos and t >= SQUARE_OFF_TIME:
                self._close_position(nifty_close, candle_time, "SQUARE_OFF")

            # ── Update trailing SL on every candle ──
            if self.open_pos and MARKET_OPEN <= t <= MARKET_CLOSE:
                self._update_trailing_sl(row)

            # ── Signal edge detection ──
            curr_buy = bool(row.get('buy_signal', False))
            curr_sell = bool(row.get('sell_signal', False))
            atr_val = row.get('atr_rma', 0)
            ema9_val = row.get('ema9', 0)
            ema21_val = row.get('ema21', 0)
            trend = "up" if ema9_val > ema21_val else "down"
            atr_ok = atr_val > ATR_MIN_THRESHOLD

            # New BUY edge
            if curr_buy and not prev_buy and MARKET_OPEN <= t <= MARKET_CLOSE:
                # Close opposite PE position (ALWAYS)
                if self.open_pos and self.open_pos.direction == "PE":
                    self._close_position(nifty_close, candle_time, "OPPOSITE_SIGNAL")

                # New CE entry — requires: no position + trading window + trend up + ATR ok
                if not self.open_pos and TRADING_START <= t <= TRADING_END:
                    if trend == "up" and atr_ok:
                        sl = nifty_low  # SL = low of signal candle
                        self.open_pos = Position(
                            direction="CE",
                            entry_price=nifty_close,
                            entry_time=candle_time,
                            stop_loss=sl,
                            highest_high=nifty_high,
                            lowest_low=nifty_low,
                        )

            # New SELL edge
            elif curr_sell and not prev_sell and MARKET_OPEN <= t <= MARKET_CLOSE:
                # Close opposite CE position (ALWAYS)
                if self.open_pos and self.open_pos.direction == "CE":
                    self._close_position(nifty_close, candle_time, "OPPOSITE_SIGNAL")

                # New PE entry — requires: no position + trading window + trend down + ATR ok
                if not self.open_pos and TRADING_START <= t <= TRADING_END:
                    if trend == "down" and atr_ok:
                        sl = nifty_high  # SL = high of signal candle
                        self.open_pos = Position(
                            direction="PE",
                            entry_price=nifty_close,
                            entry_time=candle_time,
                            stop_loss=sl,
                            highest_high=nifty_high,
                            lowest_low=nifty_low,
                        )

            prev_buy = curr_buy
            prev_sell = curr_sell

        # Close any remaining position
        if self.open_pos:
            last = df.iloc[-1]
            self._close_position(last['Close'], last['Time'], "END_OF_DATA")

        return self.trades

    def _update_trailing_sl(self, row):
        """Trailing stop loss on candle high/low (same as live code)."""
        if not self.open_pos:
            return

        pos = self.open_pos
        close = row['Close']
        high = row['High']
        low = row['Low']

        if pos.direction == "CE":
            # Trail when close exceeds previous highest high
            if close > pos.highest_high:
                pos.highest_high = high
                new_sl = low
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl

        elif pos.direction == "PE":
            # Trail when close drops below previous lowest low
            if close < pos.lowest_low or pos.lowest_low == 0:
                pos.lowest_low = low
                new_sl = high
                if new_sl < pos.stop_loss or pos.stop_loss == 0:
                    pos.stop_loss = new_sl

    def _close_position(self, exit_price, exit_time, reason):
        """Close position and record trade."""
        if not self.open_pos:
            return

        pos = self.open_pos

        # P&L: For CE, profit when NIFTY goes up. For PE, profit when NIFTY goes down.
        if pos.direction == "CE":
            pnl = exit_price - pos.entry_price
        else:  # PE
            pnl = pos.entry_price - exit_price

        trade = Trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=round(pnl, 2),
            close_reason=reason,
        )
        self.trades.append(trade)
        self.open_pos = None


# ─────────── Report ───────────

def print_report(trades: List[Trade], df: pd.DataFrame):
    """Print backtest performance report."""
    if not trades:
        print("\n❌ No trades generated! Check your data or parameters.")
        return

    pnl_list = [t.pnl for t in trades]
    total_pnl = sum(pnl_list)
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    win_rate = len(wins) / len(pnl_list) * 100

    # Daily P&L
    daily_pnl = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d")
        daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl

    # Drawdown
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = drawdown.min()

    # Streaks
    max_win_streak = max_loss_streak = current_streak = 0
    streak_type = None
    for p in pnl_list:
        if p > 0:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # By direction
    ce_trades = [t for t in trades if t.direction == "CE"]
    pe_trades = [t for t in trades if t.direction == "PE"]
    ce_pnl = sum(t.pnl for t in ce_trades)
    pe_pnl = sum(t.pnl for t in pe_trades)

    # By close reason
    reasons = {}
    for t in trades:
        r = t.close_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl

    data_days = df['Time'].dt.date.nunique()
    trading_days = len(daily_pnl)

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n📊 OVERVIEW")
    print(f"  Data period:      {df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Total data days:  {data_days}")
    print(f"  Days with trades: {trading_days}")
    print(f"  Total candles:    {len(df)}")

    print(f"\n💰 PERFORMANCE")
    print(f"  Total P&L:        {total_pnl:+.2f} pts")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Avg win:          {np.mean(wins):+.2f} pts" if wins else "  Avg win:          N/A")
    print(f"  Avg loss:         {np.mean(losses):+.2f} pts" if losses else "  Avg loss:         N/A")
    print(f"  Best trade:       {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:      {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:     {max_drawdown:.2f} pts")
    print(f"  Avg P&L/trade:    {np.mean(pnl_list):+.2f} pts")
    if trading_days > 0:
        print(f"  Avg P&L/day:      {total_pnl/trading_days:+.2f} pts")

    print(f"\n📈 STREAKS")
    print(f"  Max win streak:   {max_win_streak}")
    print(f"  Max loss streak:  {max_loss_streak}")

    print(f"\n🎯 BY DIRECTION")
    print(f"  CE trades: {len(ce_trades)} | P&L: {ce_pnl:+.2f} pts | Win rate: {len([t for t in ce_trades if t.pnl > 0])/max(len(ce_trades),1)*100:.1f}%")
    print(f"  PE trades: {len(pe_trades)} | P&L: {pe_pnl:+.2f} pts | Win rate: {len([t for t in pe_trades if t.pnl > 0])/max(len(pe_trades),1)*100:.1f}%")

    print(f"\n🔍 BY CLOSE REASON")
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        print(f"  {reason:20s}: {data['count']:3d} trades | P&L: {data['pnl']:+.2f} pts")

    # Daily P&L summary
    profitable_days = sum(1 for v in daily_pnl.values() if v > 0)
    losing_days = sum(1 for v in daily_pnl.values() if v <= 0)
    print(f"\n📅 DAILY SUMMARY")
    print(f"  Profitable days:  {profitable_days}/{trading_days} ({profitable_days/max(trading_days,1)*100:.0f}%)")
    print(f"  Losing days:      {losing_days}/{trading_days}")
    if daily_pnl:
        best_day = max(daily_pnl.items(), key=lambda x: x[1])
        worst_day = min(daily_pnl.items(), key=lambda x: x[1])
        print(f"  Best day:         {best_day[0]} ({best_day[1]:+.2f} pts)")
        print(f"  Worst day:        {worst_day[0]} ({worst_day[1]:+.2f} pts)")

    print("\n" + "=" * 70)

    # Trade log
    print("\n📝 TRADE LOG (last 30 trades)")
    print(f"  {'#':>3} {'Dir':>3} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Reason':>18} {'Entry Time':>20} {'Exit Time':>20}")
    print("  " + "-" * 95)
    for i, t in enumerate(trades[-30:], 1):
        emoji = "✅" if t.pnl > 0 else "❌"
        print(f"  {i:3d} {t.direction:>3} {t.entry_price:10.2f} {t.exit_price:10.2f} {t.pnl:+8.2f} {t.close_reason:>18} {t.entry_time.strftime('%m-%d %H:%M'):>20} {t.exit_time.strftime('%m-%d %H:%M'):>20} {emoji}")

    # Save trades to CSV
    trades_df = pd.DataFrame([{
        "direction": t.direction,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "pnl": t.pnl,
        "close_reason": t.close_reason,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
    } for t in trades])
    trades_df.to_csv("backtest_trades.csv", index=False)
    print(f"\n💾 Full trade log saved to: backtest_trades.csv")


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(description="Backtest NIFTY trading strategy")
    parser.add_argument("--file", default="nifty_1min_data.csv", help="Path to NIFTY 1-min CSV")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Data file not found: {args.file}")
        print(f"   Run 'python fetch_nifty_data.py' first to download data.")
        return

    # Load data
    print(f"Loading data from {args.file}...")
    df = pd.read_csv(args.file)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    # Date filter
    if args.start:
        df = df[df['Time'] >= args.start]
    if args.end:
        df = df[df['Time'] <= args.end + " 23:59:59"]

    print(f"Loaded {len(df)} candles | {df['Time'].dt.date.nunique()} trading days")

    # ── Process day by day (reset indicators per day like live) ──
    df['date'] = df['Time'].dt.date
    all_days = sorted(df['date'].unique())

    processed_frames = []
    for day in all_days:
        day_df = df[df['date'] == day].copy().reset_index(drop=True)

        if len(day_df) < MIN_CANDLES_FOR_SIGNAL:
            continue

        # Rename for indicator functions
        day_df = day_df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"
        })

        # Calculate signals
        signals = calculate_indicators(day_df, a=UT_BOT_A, c=UT_BOT_C)
        day_df['buy_signal'] = signals['buy'].values
        day_df['sell_signal'] = signals['sell'].values

        # Calculate ATR RMA(14)
        atr = calculate_atr_rma(day_df, period=ATR_RMA_PERIOD)
        day_df['atr_rma'] = atr.values

        # Calculate EMA trend
        ema_fast, ema_slow = calculate_ema_trend(day_df, fast=EMA_FAST, slow=EMA_SLOW)
        day_df['ema9'] = ema_fast.values
        day_df['ema21'] = ema_slow.values

        processed_frames.append(day_df)

    if not processed_frames:
        print("❌ No valid trading days found in data!")
        return

    full_df = pd.concat(processed_frames, ignore_index=True)
    print(f"Processed {len(all_days)} days | {len(full_df)} candles with indicators")

    # Run backtest
    backtester = Backtester()
    trades = backtester.run(full_df)

    # Print report
    print_report(trades, full_df)


if __name__ == "__main__":
    main()
