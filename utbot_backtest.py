"""
utbot_backtest.py — UT Bot Alert Strategy Backtest (NIFTY 1-min)

STRATEGY: UT Bot Alert (Pure ATR + Trailing Stop)
  - ATR calculated with RMA (Wilder's moving average), period 14
  - Trailing stop: ATR * key_value (1.0) below/above price
  - BUY signal: price crosses ABOVE trailing stop
  - SELL signal: price crosses BELOW trailing stop

RULES:
  - New trades only between 1:00 PM and 3:15 PM
  - Square off all positions at 3:24 PM
  - Trailing SL: if candle closes beyond previous SL → trail
  - Close on: opposite signal OR trailing SL hit OR 3:24 PM
  - Show EACH DAY result + overall summary

DATA: NIFTY 1-minute, 2 years

Usage:
    python utbot_backtest.py
    python utbot_backtest.py --atr-key 1.5
    python utbot_backtest.py --min-atr 6.9
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 10

ENTRY_START = dt_time(9, 15)      # 9:15 AM
ENTRY_END = dt_time(15, 15)       # 3:15 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM


# ─────────── UT Bot Indicators ───────────

def calc_rma(series, period):
    """RMA (Wilder's Moving Average) — same as EMA with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_atr_rma(df, period=14):
    """ATR using RMA (Wilder's method)."""
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)
    
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    
    return calc_rma(tr, period)


def compute_ut_bot_signals(df, atr_period=14, key_value=1.0, min_atr=10):
    """
    UT Bot Alert signal generation.
    
    Trailing stop logic:
      Long: trail_stop = close - ATR * key_value
      Short: trail_stop = close + ATR * key_value
      Trail only moves in favorable direction (up for long, down for short)
    
    Signal:
      BUY = close crosses above trailing stop
      SELL = close crosses below trailing stop
    """
    close = df['Close'].astype(float).values
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    n = len(close)
    
    # ATR with RMA
    atr = calc_atr_rma(df, atr_period).values
    
    # UT Bot trailing stop
    trail_stop = np.zeros(n)
    direction = np.zeros(n)  # 1 = long, -1 = short
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    
    trail_stop[0] = close[0]
    direction[0] = 1
    
    for i in range(1, n):
        nloss = atr[i] * key_value
        
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]
        
        if prev_dir == 1:  # Was long
            new_long_ts = close[i] - nloss
            trail_stop[i] = max(new_long_ts, prev_ts)
            
            if close[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close[i] + nloss
                if atr[i] >= min_atr:
                    sell_signal[i] = True
            else:
                direction[i] = 1
        else:  # Was short
            new_short_ts = close[i] + nloss
            trail_stop[i] = min(new_short_ts, prev_ts)
            
            if close[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
                if atr[i] >= min_atr:
                    buy_signal[i] = True
            else:
                direction[i] = -1
    
    return buy_signal, sell_signal, trail_stop, atr, direction


# ─────────── Backtest ───────────

def run_backtest(df, buy_sig, sell_sig, trail_stop, atr_vals):
    """
    Simple backtest:
    - New trades only 1:00 PM - 3:15 PM
    - Square off at 3:24 PM
    - Trail SL: if candle closes beyond previous SL → update SL
    - Close on: opposite signal, SL hit, or square off
    """
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)
    
    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None
    
    for i in range(len(df)):
        t = df.iloc[i]['Time'].time()
        curr_date = df.iloc[i]['Time'].date()
        c = float(close.iloc[i])
        h = float(high_v.iloc[i])
        l = float(low_v.iloc[i])
        curr_atr = float(atr_vals[i])

        # ── Day boundary reset ──
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                trade = _make_trade(pos, prev_close, df.iloc[i-1]['Time'], "DAY_END")
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                pos = None
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0}
        prev_date = curr_date
        
        if curr_date not in daily_results:
            daily_results[curr_date] = {'trades': [], 'pnl': 0}
        
        # ── Square off at 3:24 PM ──
        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF")
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue
        
        in_window = ENTRY_START <= t <= ENTRY_END
        
        # ── SL check (trailing stop hit) ──
        if pos:
            sl_hit = False
            if pos['dir'] == "LONG" and l <= pos['sl']:
                sl_hit = True
                exit_price = pos['sl']
            elif pos['dir'] == "SHORT" and h >= pos['sl']:
                sl_hit = True
                exit_price = pos['sl']
            
            if sl_hit:
                trade = _make_trade(pos, exit_price, df.iloc[i]['Time'], "TRAIL_SL")
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
        
        # ── Trailing SL update ──
        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl
        
        # ── Signal handling ──
        is_buy = bool(buy_sig[i])
        is_sell = bool(sell_sig[i])
        
        # Opposite signal → close position
        if is_buy and pos and pos['dir'] == "SHORT":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE")
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        elif is_sell and pos and pos['dir'] == "LONG":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE")
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        
        # ── New entry (only within window) ──
        if not pos and in_window and t <= ENTRY_END:
            if is_buy and curr_atr >= MIN_ATR:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
            elif is_sell and curr_atr >= MIN_ATR:
                sl = c + curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'SHORT', 'entry': c, 'sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
    
    return all_trades, daily_results


def _pnl(pos, exit_price):
    if pos['dir'] == "LONG":
        return exit_price - pos['entry']
    else:
        return pos['entry'] - exit_price


def _make_trade(pos, exit_price, exit_time, reason):
    pnl = _pnl(pos, exit_price)
    return {
        'dir': pos['dir'],
        'entry': pos['entry'],
        'exit': round(exit_price, 2),
        'entry_time': pos['entry_time'],
        'exit_time': exit_time,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl / pos['entry'] * 100, 4),
        'reason': reason,
    }


def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ─────────── Reports ───────────

def print_daily_results(daily_results):
    """Print each day's result."""
    sorted_days = sorted(daily_results.keys())
    
    print(f"\n{'='*90}")
    print(f"  {'Date':>12} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'Win%':>6} {'P&L':>10} {'Status':>8}")
    print(f"  {'-'*85}")
    
    cumulative_pnl = 0
    win_days = 0
    loss_days = 0
    
    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        
        if len(trades) == 0:
            continue
        
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] <= 0)
        wr = wins / len(trades) * 100 if len(trades) > 0 else 0
        cumulative_pnl += day_pnl
        
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        if day_pnl > 0:
            win_days += 1
        elif day_pnl < 0:
            loss_days += 1
        
        print(f"  {str(day):>12} {len(trades):>7} {wins:>5} {losses:>5} {wr:>5.0f}% {day_pnl:>+9.2f} {icon:>8}")
    
    print(f"  {'-'*85}")
    print(f"  {'TOTAL':>12} {'':>7} {'':>5} {'':>5} {'':>6} {cumulative_pnl:>+9.2f}")
    total_days = win_days + loss_days
    if total_days > 0:
        print(f"  Win days: {win_days} | Loss days: {loss_days} | Win%: {win_days/total_days*100:.1f}%")
    print(f"{'='*90}")


def print_summary(all_trades, daily_results):
    """Print overall summary."""
    if not all_trades:
        print("❌ No trades")
        return
    
    n = len(all_trades)
    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    wr = len(wins) / n * 100
    
    pnl_list = [t['pnl'] for t in all_trades]
    total_pnl = sum(pnl_list)
    
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 1
    pf = gp / gl if gl > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Streak analysis
    max_win_streak = 0
    max_loss_streak = 0
    curr_streak = 0
    for t in all_trades:
        if t['pnl'] > 0:
            if curr_streak > 0:
                curr_streak += 1
            else:
                curr_streak = 1
            max_win_streak = max(max_win_streak, curr_streak)
        else:
            if curr_streak < 0:
                curr_streak -= 1
            else:
                curr_streak = -1
            max_loss_streak = max(max_loss_streak, abs(curr_streak))
    
    # Drawdown
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = drawdown.max()
    
    # Reason breakdown
    reasons = {}
    for t in all_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    
    # Daily
    trading_days = sum(1 for d in daily_results.values() if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)
    
    # Monthly
    monthly = {}
    for t in all_trades:
        m = t['entry_time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'pnl': 0, 'trades': 0, 'wins': 0}
        monthly[m]['pnl'] += t['pnl']
        monthly[m]['trades'] += 1
        if t['pnl'] > 0:
            monthly[m]['wins'] += 1
    
    print(f"\n{'='*65}")
    print(f"  🤖 UT BOT ALERT — OVERALL RESULTS")
    print(f"{'='*65}")
    
    print(f"\n  📊 TRADE STATS")
    print(f"  Total trades:      {n}")
    print(f"  Wins:              {len(wins)} ({wr:.1f}%)")
    print(f"  Losses:            {len(losses)}")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Risk/Reward:       {rr:.2f}")
    print(f"  Avg win:           {avg_win:+.2f} pts")
    print(f"  Avg loss:          {avg_loss:+.2f} pts")
    
    print(f"\n  💰 P&L")
    print(f"  Total P&L:         {total_pnl:+.2f} pts")
    print(f"  Best trade:        {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:       {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:      {max_dd:.2f} pts")
    
    print(f"\n  📅 DAILY")
    print(f"  Trading days:      {trading_days}")
    print(f"  Profitable days:   {win_days} ({win_days/max(trading_days,1)*100:.0f}%)")
    print(f"  Avg P&L/day:       {total_pnl/max(trading_days,1):+.2f} pts")
    
    print(f"\n  🔄 STREAKS")
    print(f"  Max win streak:    {max_win_streak}")
    print(f"  Max loss streak:   {max_loss_streak}")
    
    print(f"\n  🔍 CLOSE REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_trades = [t for t in all_trades if t['reason'] == r]
        r_wr = sum(1 for t in r_trades if t['pnl'] > 0) / len(r_trades) * 100
        print(f"    {r:15s}: {c:4d} trades | Win: {r_wr:.0f}%")
    
    print(f"\n  📅 MONTHLY BREAKDOWN")
    print(f"  {'Month':>10} {'P&L':>10} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*35}")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        m_wr = d['wins'] / max(d['trades'], 1) * 100
        icon = "✅" if d['pnl'] > 0 else "❌"
        print(f"  {m:>10} {d['pnl']:>+9.2f} {d['trades']:>7} {m_wr:>5.0f}% {icon}")
    
    print(f"\n{'='*65}")


# ─────────── Main ───────────

def main():
    global ATR_KEY_VALUE, MIN_ATR
    
    parser = argparse.ArgumentParser(description="UT Bot Alert Backtest (NIFTY 1-min)")
    parser.add_argument("--file", default="nifty_1min_data.csv", help="Data file")
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE, help="ATR multiplier (default: 1.0)")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help="Min ATR threshold (default: 10)")
    args = parser.parse_args()
    
    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    
    # Load data
    print(f"Loading {args.file}...")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        print(f"Run: python fetch_nifty_data.py")
        return
    
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    total_candles = len(df)
    total_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"
    
    print(f"Loaded {total_candles:,} candles | {total_days} days")
    print(f"Date range: {date_range}")
    
    print(f"\n🤖 UT BOT ALERT BACKTEST")
    print(f"ATR: RMA({args.atr_period}) × {ATR_KEY_VALUE} | Min ATR: {MIN_ATR}")
    print(f"Window: {ENTRY_START.strftime('%H:%M')} - {ENTRY_END.strftime('%H:%M')} | Square off: {SQUARE_OFF.strftime('%H:%M')}")
    print(f"{'='*60}")
    
    # Generate UT Bot signals
    print(f"\nComputing UT Bot signals...")
    buy_sig, sell_sig, trail_stop, atr_vals, direction = compute_ut_bot_signals(
        df, args.atr_period, ATR_KEY_VALUE, MIN_ATR
    )
    
    total_buy = buy_sig.sum()
    total_sell = sell_sig.sum()
    
    print(f"  UT Bot buy signals:  {total_buy}")
    print(f"  UT Bot sell signals: {total_sell}")
    print(f"  ATR range:           {atr_vals[~np.isnan(atr_vals)].min():.1f} - {atr_vals[~np.isnan(atr_vals)].max():.1f}")
    
    # Backtest
    print(f"\n🚀 Running backtest...")
    all_trades, daily_results = run_backtest(
        df, buy_sig, sell_sig, trail_stop, atr_vals
    )
    
    print(f"  Total trades: {len(all_trades)}")
    
    # Daily results
    print_daily_results(daily_results)
    
    # Overall results
    print_summary(all_trades, daily_results)
    
    # Save trades
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("utbot_trades.csv", index=False)
        print(f"\n💾 Trades saved: utbot_trades.csv")


if __name__ == "__main__":
    main()
