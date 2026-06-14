"""
utbot_2min_backtest.py — UT Bot Alert Strategy Backtest (NIFTY 2-min)

STRATEGY: UT Bot Alert (Pure ATR + Trailing Stop) on 2-MINUTE timeframe
  - ATR calculated with RMA (Wilder's moving average)
  - Trailing stop: ATR × key_value below/above price
  - BUY signal: price crosses ABOVE trailing stop (direction flips long)
  - SELL signal: price crosses BELOW trailing stop (direction flips short)

NO ML, NO LOOK-AHEAD — pure indicator signals only.

RECOVERY LOT MANAGEMENT:
  - Start with qty = 1
  - After 2 consecutive losses → qty += 1
  - Recovery target = average of consecutive loss points
  - Extra lot P&L tracked; once recovery_target met → qty resets to 1
  - Every additional 2 consecutive losses → qty += 1 again

RULES:
  - Entry windows configurable (default: 09:20 - 15:15)
  - Square off all positions at 15:24
  - Trailing SL: ATR-based, only moves in favorable direction
  - Close on: opposite signal OR trailing SL hit OR square-off

DATA: NIFTY 2-minute OHLCV CSV

Usage:
    python utbot_2min_backtest.py
    python utbot_2min_backtest.py --file nifty_2min_data.csv
    python utbot_2min_backtest.py --atr-key 1.5 --min-atr 8
    python utbot_2min_backtest.py --atr-period 10 --atr-key 1.2
    python utbot_2min_backtest.py --window-start 13:00 --window-end 15:03
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')


# ─────────── Default Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5

ENTRY_START = dt_time(9, 20)
ENTRY_END = dt_time(15, 15)
SQUARE_OFF = dt_time(15, 24)

LOT_SIZE = 75


# ─────────── UT Bot Indicators ───────────

def calc_rma(series, period):
    """RMA (Wilder's Moving Average) — same as EMA with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_atr(df, period=14):
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


def compute_ut_bot_signals(df, atr_period=14, key_value=1.0, min_atr=6.5):
    """
    UT Bot Alert signal generation on 2-min candles.

    Trailing stop logic:
      Long mode:  trail_stop = close - ATR × key_value  (only moves UP)
      Short mode: trail_stop = close + ATR × key_value  (only moves DOWN)

    Signal triggers:
      BUY  = close crosses above trailing stop → direction flips to LONG
      SELL = close crosses below trailing stop → direction flips to SHORT

    Returns: buy_signal, sell_signal, trail_stop, atr, direction arrays
    """
    close = df['Close'].astype(float).values
    n = len(close)

    atr = calc_atr(df, atr_period).values

    trail_stop = np.zeros(n)
    direction = np.zeros(n)     # 1 = long, -1 = short
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
                # Flip to short
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
                # Flip to long
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
                if atr[i] >= min_atr:
                    buy_signal[i] = True
            else:
                direction[i] = -1

    return buy_signal, sell_signal, trail_stop, atr, direction


# ─────────── Backtest Engine ───────────

def run_backtest(df, buy_sig, sell_sig, atr_vals,
                 entry_start, entry_end, square_off):
    """
    Backtest with Recovery Lot Management:
    - Start qty=1
    - 2 consecutive losses → qty += 1 (losses use total P&L: qty × raw_pnl)
    - Cumulative total losses tracked until fully recovered → qty=1
    - Wins during recovery subtract total P&L from unrecovered losses
    - Additional consecutive losses → qty += 1 again
    """
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)

    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None

    # ── Recovery Lot Management State ──
    qty = 1                      # current lot count
    consecutive_losses = 0       # back-to-back loss counter
    loss_streak_pts = []         # total P&L (qty×raw) of each loss in current streak
    recovering = False           # are we in recovery mode?
    total_unrecovered_loss = 0.0 # cumulative unrecovered loss (total P&L)

    def _close_trade(pos, exit_price, exit_time, reason, curr_date):
        """Close a position and update recovery state."""
        nonlocal qty, consecutive_losses, loss_streak_pts
        nonlocal recovering, total_unrecovered_loss

        raw_pnl = _pnl(pos, exit_price)
        trade_qty = pos.get('qty', 1)  # qty THIS trade was entered with
        actual_pnl = raw_pnl * trade_qty  # Total P&L for this trade

        status = ""

        # ── Update recovery state based on this trade's result ──
        if raw_pnl < 0:
            actual_loss = abs(actual_pnl)  # Total loss (qty × raw loss)
            consecutive_losses += 1
            loss_streak_pts.append(actual_loss)
            total_unrecovered_loss += actual_loss

            # Every 2 consecutive losses → qty increases
            if consecutive_losses >= 2 and consecutive_losses % 2 == 0:
                qty += 1
                recovering = True
                avg_loss = np.mean(loss_streak_pts)
                status = (f"L{consecutive_losses}→QTY↑{qty} "
                          f"avg:{avg_loss:.1f} rec:{total_unrecovered_loss:.1f}")
            else:
                if recovering:
                    status = (f"L{consecutive_losses} "
                              f"(rec:{total_unrecovered_loss:.1f})")
                else:
                    status = f"L{consecutive_losses}"

        else:  # win or breakeven
            if recovering:
                total_unrecovered_loss -= actual_pnl

                if total_unrecovered_loss <= 0:
                    status = f"RECOVERED ✅"
                    qty = 1
                    recovering = False
                    total_unrecovered_loss = 0.0
                else:
                    status = (f"REC +{actual_pnl:.1f} "
                              f"left:{total_unrecovered_loss:.1f}")

            consecutive_losses = 0
            loss_streak_pts = []

        trade = {
            'dir': pos['dir'],
            'entry': pos['entry'],
            'exit': round(exit_price, 2),
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'raw_pnl': round(raw_pnl, 2),
            'qty': trade_qty,
            'pnl': round(actual_pnl, 2),
            'pnl_pct': round(raw_pnl / pos['entry'] * 100, 4),
            'reason': reason,
            'status': status,
        }

        all_trades.append(trade)
        _add_daily(daily_results, curr_date, trade)
        return trade

    for i in range(len(df)):
        t = df.iloc[i]['Time'].time()
        curr_date = df.iloc[i]['Time'].date()
        c = float(close.iloc[i])
        h = float(high_v.iloc[i])
        l = float(low_v.iloc[i])
        curr_atr = float(atr_vals[i])

        # ── Day boundary ──
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                _close_trade(pos, prev_close, df.iloc[i-1]['Time'],
                             "DAY_END", prev_date)
                pos = None

            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0}

        prev_date = curr_date

        if curr_date not in daily_results:
            daily_results[curr_date] = {'trades': [], 'pnl': 0}

        # ── Square off ──
        if pos and t >= square_off:
            _close_trade(pos, c, df.iloc[i]['Time'],
                         "SQUARE_OFF", curr_date)
            pos = None
            continue

        in_window = entry_start <= t <= entry_end

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
                _close_trade(pos, exit_price, df.iloc[i]['Time'],
                             "TRAIL_SL", curr_date)
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
            _close_trade(pos, c, df.iloc[i]['Time'],
                         "OPPOSITE", curr_date)
            pos = None
        elif is_sell and pos and pos['dir'] == "LONG":
            _close_trade(pos, c, df.iloc[i]['Time'],
                         "OPPOSITE", curr_date)
            pos = None

        # ── New entry (with current qty) ──
        if not pos and in_window:
            if is_buy and curr_atr >= MIN_ATR:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {
                    'dir': 'LONG', 'entry': c, 'sl': sl,
                    'entry_time': df.iloc[i]['Time'], 'entry_idx': i,
                    'qty': qty,  # ← enters with current qty
                }
            elif is_sell and curr_atr >= MIN_ATR:
                sl = c + curr_atr * ATR_KEY_VALUE
                pos = {
                    'dir': 'SHORT', 'entry': c, 'sl': sl,
                    'entry_time': df.iloc[i]['Time'], 'entry_idx': i,
                    'qty': qty,  # ← enters with current qty
                }

    return all_trades, daily_results


def _pnl(pos, exit_price):
    if pos['dir'] == "LONG":
        return exit_price - pos['entry']
    else:
        return pos['entry'] - exit_price


def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ─────────── Reports ───────────

def print_daily_results(daily_results):
    """Print each day's result."""
    sorted_days = sorted(daily_results.keys())

    print(f"\n{'='*110}")
    print(f"  📅 DAILY RESULTS")
    print(f"{'='*110}")
    print(f"  {'Date':<12} {'Day':<4} {'Trades':>7} "
          f"{'Wins':>5} {'Loss':>5} {'Win%':>6} {'P&L (Pts)':>11} {'Cum P&L':>11} {'':>4}")
    print(f"  {'-'*90}")

    cumulative_pnl = 0
    win_days = 0
    loss_days = 0

    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']

        if len(trades) == 0:
            continue

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_name = day_names[day.weekday()]

        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] <= 0)
        wr = wins / len(trades) * 100 if len(trades) > 0 else 0
        cumulative_pnl += day_pnl

        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        if day_pnl > 0:
            win_days += 1
        elif day_pnl < 0:
            loss_days += 1

        print(f"  {str(day):<12} {day_name:<4} {len(trades):>7} "
              f"{wins:>5} {losses:>5} {wr:>5.0f}% {day_pnl:>+11.2f} "
              f"{cumulative_pnl:>+11.2f} {icon:>4}")

    print(f"  {'-'*90}")
    print(f"  {'TOTAL':<12} {'':<4} {'':<7} "
          f"{'':<5} {'':<5} {'':<6} {cumulative_pnl:>+11.2f}")
    total_days = win_days + loss_days
    if total_days > 0:
        print(f"  Win days: {win_days} | Loss days: {loss_days} | "
              f"Day Win%: {win_days/total_days*100:.1f}%")
    print(f"{'='*110}")


def print_detailed_trades(all_trades, daily_results,
                          log_file="utbot_2min_detailed_log.txt"):
    """Print AND save detailed per-trade log with qty & recovery columns."""
    sorted_days = sorted(daily_results.keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*130}")
    out(f"  📋 DETAILED TRADE LOG — WITH RECOVERY LOT MANAGEMENT")
    out(f"{'='*130}")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    cum_pnl = 0

    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']

        if len(trades) == 0:
            continue

        cum_pnl += day_pnl
        day_name = day_names[day.weekday()]
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = len(trades) - wins
        wr = wins / len(trades) * 100
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"

        out(f"\n  ┌{'─'*128}┐")
        out(f"  │ 📅 {day} ({day_name}) | Trades: {len(trades)} "
            f"(W:{wins} L:{losses} {wr:.0f}%) | Day P&L: "
            f"{day_pnl:+.2f} {icon} | Cum: {cum_pnl:+.2f}")
        out(f"  ├───┬──────┬─────────┬─────────┬───────────"
            f"┬───────────┬────────┬─────┬──────────┬────────────┬──────────────────────────────┤")
        out(f"  │ # │ Dir  │ Entry T │ Exit T  │ Entry Pr  "
            f"│ Exit Pr   │ Pts    │ Qty │ Total P&L│ Reason     │ Recovery Status               │")
        out(f"  ├───┼──────┼─────────┼─────────┼───────────"
            f"┼───────────┼────────┼─────┼──────────┼────────────┼──────────────────────────────┤")

        for j, t in enumerate(trades, 1):
            entry_t = (t['entry_time'].strftime('%H:%M')
                       if hasattr(t['entry_time'], 'strftime')
                       else str(t['entry_time'])[-8:-3])
            exit_t = (t['exit_time'].strftime('%H:%M')
                      if hasattr(t['exit_time'], 'strftime')
                      else str(t['exit_time'])[-8:-3])

            dir_icon = "🟢" if t['dir'] == 'LONG' else "🔴"
            pnl_icon = "✅" if t['pnl'] > 0 else "❌" if t['pnl'] < 0 else "➖"

            status = t.get('status', '')
            if len(status) > 30:
                status = status[:28] + ".."

            out(f"  │{j:>2} │ {dir_icon}{t['dir']:>4} │ {entry_t:>7} │ "
                f"{exit_t:>7} │ {t['entry']:>9.2f} │ {t['exit']:>9.2f} │ "
                f"{t['raw_pnl']:>+6.2f} │ {t['qty']:>3} │ "
                f"{t['pnl']:>+8.2f}{pnl_icon}│ "
                f"{t['reason']:<10} │ {status:<30}│")

        out(f"  └───┴──────┴─────────┴─────────┴───────────"
            f"┴───────────┴────────┴─────┴──────────┴────────────┴──────────────────────────────┘")

    out(f"\n{'='*130}")
    out(f"  GRAND TOTAL: {cum_pnl:+.2f} pts")
    out(f"{'='*130}")

    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n💾 Detailed log saved: {log_file}")


def print_summary(all_trades, daily_results):
    """Print overall summary with monthly + day-of-week breakdown."""
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
            curr_streak = curr_streak + 1 if curr_streak > 0 else 1
            max_win_streak = max(max_win_streak, curr_streak)
        else:
            curr_streak = curr_streak - 1 if curr_streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(curr_streak))

    # Drawdown
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = drawdown.max()

    # Daily stats
    trading_days = sum(1 for d in daily_results.values()
                       if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)

    # Close reasons
    reasons = {}
    for t in all_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1

    # Monthly breakdown
    monthly = {}
    for t in all_trades:
        m = t['entry_time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'pnl': 0, 'trades': 0, 'wins': 0}
        monthly[m]['pnl'] += t['pnl']
        monthly[m]['trades'] += 1
        if t['pnl'] > 0:
            monthly[m]['wins'] += 1

    # Day-of-week breakdown
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dow_stats = {i: {'pnl': 0, 'trades': 0, 'wins': 0} for i in range(5)}
    for t in all_trades:
        dow = t['entry_time'].weekday()
        if dow < 5:
            dow_stats[dow]['pnl'] += t['pnl']
            dow_stats[dow]['trades'] += 1
            if t['pnl'] > 0:
                dow_stats[dow]['wins'] += 1

    # Qty usage stats
    qty_stats = {}
    for t in all_trades:
        q = t['qty']
        if q not in qty_stats:
            qty_stats[q] = {'count': 0, 'pnl': 0, 'wins': 0}
        qty_stats[q]['count'] += 1
        qty_stats[q]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            qty_stats[q]['wins'] += 1

    max_qty = max(t['qty'] for t in all_trades)
    recovery_trades = sum(1 for t in all_trades if t['qty'] > 1)

    # ── Print ──
    print(f"\n{'='*70}")
    print(f"  🤖 UT BOT ALERT (2-MIN) — OVERALL RESULTS")
    print(f"{'='*70}")

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
    print(f"  Total ₹ P&L:       ₹{total_pnl * LOT_SIZE:+,.0f} (with qty {LOT_SIZE}/lot)")
    print(f"  Best trade:        {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:       {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:      {max_dd:.2f} pts")

    print(f"\n  📅 DAILY")
    print(f"  Trading days:      {trading_days}")
    print(f"  Profitable days:   {win_days} "
          f"({win_days/max(trading_days,1)*100:.0f}%)")
    print(f"  Avg P&L/day:       "
          f"{total_pnl/max(trading_days,1):+.2f} pts")

    print(f"\n  🔄 STREAKS")
    print(f"  Max win streak:    {max_win_streak}")
    print(f"  Max loss streak:   {max_loss_streak}")

    print(f"\n  📦 QTY / RECOVERY STATS")
    print(f"  Max qty used:      {max_qty}")
    print(f"  Recovery trades:   {recovery_trades} ({recovery_trades/n*100:.1f}% of all)")
    print(f"  {'Qty':>5} {'Trades':>8} {'Win%':>6} {'P&L':>11}")
    print(f"  {'-'*35}")
    for q in sorted(qty_stats.keys()):
        d = qty_stats[q]
        q_wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
        print(f"  {q:>5} {d['count']:>8} {q_wr:>5.0f}% {d['pnl']:>+10.2f}")

    print(f"\n  🔍 CLOSE REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_trades = [t for t in all_trades if t['reason'] == r]
        r_wr = sum(1 for t in r_trades if t['pnl'] > 0) / len(r_trades) * 100
        r_pnl = sum(t['pnl'] for t in r_trades)
        print(f"    {r:15s}: {c:4d} trades | Win: {r_wr:.0f}% | "
              f"P&L: {r_pnl:+.2f}")

    print(f"\n  📅 MONTHLY BREAKDOWN")
    print(f"  {'Month':>10} {'P&L':>11} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*38}")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        m_wr = d['wins'] / max(d['trades'], 1) * 100
        icon = "✅" if d['pnl'] > 0 else "❌"
        print(f"  {m:>10} {d['pnl']:>+10.2f} {d['trades']:>7} "
              f"{m_wr:>5.0f}% {icon}")

    print(f"\n  📅 DAY-OF-WEEK PERFORMANCE")
    print(f"  {'Day':<12} {'P&L':>11} {'Trades':>7} {'Win%':>6} "
          f"{'Avg P&L':>10}")
    print(f"  {'-'*50}")
    for i in range(5):
        d = dow_stats[i]
        if d['trades'] > 0:
            d_wr = d['wins'] / d['trades'] * 100
            avg = d['pnl'] / d['trades']
            icon = "✅" if d['pnl'] > 0 else "❌"
            print(f"  {day_names[i]:<12} {d['pnl']:>+10.2f} "
                  f"{d['trades']:>7} {d_wr:>5.0f}% {avg:>+9.2f} {icon}")
        else:
            print(f"  {day_names[i]:<12} {'---':>11} {'0':>7} "
                  f"{'---':>6} {'---':>10}")

    print(f"\n{'='*70}")


# ─────────── Main ───────────

def main():
    global ATR_KEY_VALUE, MIN_ATR, ATR_PERIOD, LOT_SIZE

    parser = argparse.ArgumentParser(
        description="UT Bot Alert Backtest (NIFTY 2-min) with Recovery Lot Management"
    )
    parser.add_argument("--file", default="nifty_2min_data.csv",
                        help="2-min data CSV file (default: nifty_2min_data.csv)")
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD,
                        help="ATR period (default: 14)")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE,
                        help="ATR multiplier for trailing stop (default: 1.0)")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR,
                        help="Min ATR threshold for entry (default: 6.5)")
    parser.add_argument("--window-start", type=str, default="09:20",
                        help="Entry window start HH:MM (default: 09:20)")
    parser.add_argument("--window-end", type=str, default="15:15",
                        help="Entry window end HH:MM (default: 15:15)")
    parser.add_argument("--square-off", type=str, default="15:24",
                        help="Square off time HH:MM (default: 15:24)")
    parser.add_argument("--lot-size", type=int, default=LOT_SIZE,
                        help="Qty per lot (default: 75)")
    args = parser.parse_args()

    ATR_PERIOD = args.atr_period
    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    LOT_SIZE = args.lot_size

    # Parse time windows
    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    entry_start = parse_time(args.window_start)
    entry_end = parse_time(args.window_end)
    square_off = parse_time(args.square_off)

    # ── Load data ──
    print(f"📂 Loading {args.file}...")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        print(f"Make sure you have the 2-min NIFTY data CSV.")
        return

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    total_candles = len(df)
    total_days = df['Time'].dt.date.nunique()
    date_range = (f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → "
                  f"{df['Time'].iloc[-1].strftime('%Y-%m-%d')}")

    print(f"  Loaded {total_candles:,} candles | {total_days} days")
    print(f"  Date range: {date_range}")

    print(f"\n{'='*70}")
    print(f"  🤖 UT BOT ALERT BACKTEST — 2-MINUTE TIMEFRAME")
    print(f"  📦 Recovery Lot Management: ON")
    print(f"{'='*70}")
    print(f"  ATR: RMA({ATR_PERIOD}) × {ATR_KEY_VALUE} | Min ATR: {MIN_ATR}")
    print(f"  Window: {entry_start.strftime('%H:%M')} - "
          f"{entry_end.strftime('%H:%M')} | "
          f"Square off: {square_off.strftime('%H:%M')}")
    print(f"  Qty per lot: {LOT_SIZE} | Start qty: 1")
    print(f"  Rule: 2 consecutive losses → qty += 1")
    print(f"  Recovery: cumulative total losses tracked → fully covered → qty = 1")
    print(f"{'='*70}")

    # ── Compute UT Bot signals ──
    print(f"\n⚡ Computing UT Bot signals on 2-min candles...")
    buy_sig, sell_sig, trail_stop, atr_vals, direction = \
        compute_ut_bot_signals(df, ATR_PERIOD, ATR_KEY_VALUE, MIN_ATR)

    total_buy = buy_sig.sum()
    total_sell = sell_sig.sum()
    valid_atr = atr_vals[~np.isnan(atr_vals)]

    print(f"  UT Bot BUY signals:  {total_buy}")
    print(f"  UT Bot SELL signals: {total_sell}")
    print(f"  ATR range:           {valid_atr.min():.2f} - "
          f"{valid_atr.max():.2f}")
    print(f"  ATR mean:            {valid_atr.mean():.2f}")

    # ── Run backtest ──
    print(f"\n🚀 Running backtest with Recovery Lot Management...")
    all_trades, daily_results = run_backtest(
        df, buy_sig, sell_sig, atr_vals,
        entry_start, entry_end, square_off
    )

    print(f"  Total trades: {len(all_trades)}")

    if not all_trades:
        print("\n❌ No trades generated. Try adjusting parameters:")
        print(f"  - Lower --min-atr (current: {MIN_ATR})")
        print("  - Wider window (--window-start / --window-end)")
        return

    # ── Reports ──
    print_daily_results(daily_results)
    print_summary(all_trades, daily_results)
    print_detailed_trades(all_trades, daily_results)

    # ── Save trades CSV ──
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv("utbot_2min_trades.csv", index=False)
    print(f"\n💾 Trades saved: utbot_2min_trades.csv")


if __name__ == "__main__":
    main()
