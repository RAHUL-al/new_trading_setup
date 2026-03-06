"""
utbot_backtest.py — UT Bot Alert Strategy Backtest (NIFTY 1-min) [OPTIMIZED]

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

OPTIMIZATIONS:
  1. MARTINGALE RECOVERY: Double qty after >15pt loss (max 3x), reset after recovery (7pt buffer)
  2. COOLDOWN: Skip 3 candles after SL hit (anti-whipsaw)
  3. MAX DAILY LOSS: Stop trading at -80pts/day
  4. MAX TRADES/DAY: Cap at 8 trades
  5. DIRECTION BIAS: Only trade in UT Bot trailing stop direction

DATA: NIFTY 1-minute, 2 years

Usage:
    python utbot_backtest.py                            # Default: nifty_1min_data.csv
    python utbot_backtest.py --atr-key 1.5              # Different ATR multiplier
    python utbot_backtest.py --min-atr 6.9              # Lower ATR threshold
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0               # UT Bot: trailing stop = ATR * key_value
MIN_ATR = 10                      # ATR must be > 10

ENTRY_START = dt_time(13, 0)      # 1:00 PM
ENTRY_END = dt_time(15, 15)       # 3:15 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM


# ─────────── UT Bot Indicators ───────────

def calc_rma(series, period):
    """RMA (Wilder's Moving Average) — same as EMA with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_ema(series, period):
    """Standard EMA."""
    return series.ewm(span=period, adjust=False).mean()


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
        
        # Previous trail stop
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]
        
        # Calculate new trail stop based on previous direction
        if prev_dir == 1:  # Was long
            new_long_ts = close[i] - nloss
            # Trail up only (never lower the stop)
            trail_stop[i] = max(new_long_ts, prev_ts)
            
            # Check if price crossed below → switch to short
            if close[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close[i] + nloss
                if atr[i] >= min_atr:
                    sell_signal[i] = True
            else:
                direction[i] = 1
        else:  # Was short
            new_short_ts = close[i] + nloss
            # Trail down only (never raise the stop)
            trail_stop[i] = min(new_short_ts, prev_ts)
            
            # Check if price crossed above → switch to long
            if close[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
                if atr[i] >= min_atr:
                    buy_signal[i] = True
            else:
                direction[i] = -1
    
    return buy_signal, sell_signal, trail_stop, atr, direction


# ─────────── Backtest ───────────

def run_backtest(df, buy_sig, sell_sig, trail_stop, atr_vals, direction):
    """
    Optimized Backtest with:
    - New trades only 1:00 PM - 3:15 PM
    - Square off at 3:24 PM
    - Trail SL: if candle closes beyond previous SL → update SL
    - Close on: opposite signal, SL hit, or square off
    
    OPTIMIZATIONS:
    1. MARTINGALE RECOVERY: If trade loses > LOSS_THRESHOLD pts → double qty (max 3x)
       Reset to 1x when accumulated loss recovered (within RECOVERY_BUFFER pts)
    2. COOLDOWN: Skip COOLDOWN_CANDLES candles after SL hit (avoid whipsaw)
    3. MAX DAILY LOSS: Stop trading for the day if daily loss > MAX_DAILY_LOSS
    4. MAX TRADES/DAY: Cap at MAX_TRADES_PER_DAY
    5. DIRECTION BIAS: Only trade in direction of UT Bot trailing stop
    
    Returns all_trades, daily_results, stats dict.
    """
    # ── Optimization parameters ──
    LOSS_THRESHOLD = 15.0       # Double qty after losing > this many pts
    MAX_QTY = 3                 # Max qty multiplier (cap martingale)
    RECOVERY_BUFFER = 7.0      # Reset qty when loss recovered within this buffer
    COOLDOWN_CANDLES = 3        # Skip N candles after SL hit
    MAX_DAILY_LOSS = -80.0      # Stop trading for day if daily P&L hits this
    MAX_TRADES_PER_DAY = 8      # Max trades per day
    
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)
    
    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None
    
    # ── Martingale state ──
    qty = 1                     # Current lot multiplier
    accumulated_loss = 0.0      # Tracks loss to recover
    recovering = False          # True when in recovery mode
    
    # ── Cooldown state ──
    cooldown_until = -1         # Skip entries until this candle index
    
    # ── Daily counters ──
    daily_trade_count = 0
    daily_pnl_running = 0.0
    daily_stopped = False       # True if daily loss limit hit
    
    # ── Stats ──
    stats = {
        'martingale_activations': 0,
        'martingale_recoveries': 0,
        'cooldown_skips': 0,
        'daily_stops': 0,
        'max_trades_stops': 0,
        'direction_filtered': 0,
    }
    
    for i in range(len(df)):
        t = df.iloc[i]['Time'].time()
        curr_date = df.iloc[i]['Time'].date()
        c = float(close.iloc[i])
        h = float(high_v.iloc[i])
        l = float(low_v.iloc[i])
        curr_atr = float(atr_vals[i])
        curr_dir = int(direction[i])  # 1 = bullish, -1 = bearish

        # ── Day boundary reset ──
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                pnl_raw = _pnl_raw(pos, prev_close)
                trade = _make_trade_qty(pos, prev_close, df.iloc[i-1]['Time'], "DAY_END", qty)
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                # Update martingale on day end
                if pnl_raw < -LOSS_THRESHOLD and not recovering:
                    qty = min(qty * 2, MAX_QTY)
                    accumulated_loss += pnl_raw
                    recovering = True
                    stats['martingale_activations'] += 1
                elif recovering:
                    accumulated_loss += pnl_raw
                    if accumulated_loss >= -RECOVERY_BUFFER:
                        qty = 1
                        accumulated_loss = 0.0
                        recovering = False
                        stats['martingale_recoveries'] += 1
                pos = None
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0}
            # Reset daily counters
            daily_trade_count = 0
            daily_pnl_running = 0.0
            daily_stopped = False
            cooldown_until = -1
        prev_date = curr_date
        
        if curr_date not in daily_results:
            daily_results[curr_date] = {'trades': [], 'pnl': 0}
        
        # ── Square off at 3:24 PM ──
        if pos and t >= SQUARE_OFF:
            pnl_raw = _pnl_raw(pos, c)
            trade = _make_trade_qty(pos, c, df.iloc[i]['Time'], "SQUARE_OFF", qty)
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            daily_pnl_running += trade['pnl']
            daily_trade_count += 1
            # Update martingale
            if pnl_raw < -LOSS_THRESHOLD and not recovering:
                qty = min(qty * 2, MAX_QTY)
                accumulated_loss += pnl_raw
                recovering = True
                stats['martingale_activations'] += 1
            elif recovering:
                accumulated_loss += pnl_raw
                if accumulated_loss >= -RECOVERY_BUFFER:
                    qty = 1
                    accumulated_loss = 0.0
                    recovering = False
                    stats['martingale_recoveries'] += 1
            pos = None
            continue
        
        in_window = ENTRY_START <= t <= ENTRY_END
        
        # ── Skip if daily stopped ──
        if daily_stopped:
            continue
        
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
                pnl_raw = _pnl_raw(pos, exit_price)
                trade = _make_trade_qty(pos, exit_price, df.iloc[i]['Time'], "TRAIL_SL", qty)
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                daily_pnl_running += trade['pnl']
                daily_trade_count += 1
                
                # ── Martingale: check if loss > threshold ──
                if pnl_raw < -LOSS_THRESHOLD and not recovering:
                    qty = min(qty * 2, MAX_QTY)
                    accumulated_loss += pnl_raw
                    recovering = True
                    stats['martingale_activations'] += 1
                elif recovering:
                    accumulated_loss += pnl_raw
                    if accumulated_loss >= -RECOVERY_BUFFER:
                        # Recovered! Reset to 1x
                        qty = 1
                        accumulated_loss = 0.0
                        recovering = False
                        stats['martingale_recoveries'] += 1
                
                # ── Cooldown: set N candle skip ──
                cooldown_until = i + COOLDOWN_CANDLES
                
                # ── Check daily loss limit ──
                if daily_pnl_running <= MAX_DAILY_LOSS:
                    daily_stopped = True
                    stats['daily_stops'] += 1
                
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
            pnl_raw = _pnl_raw(pos, c)
            trade = _make_trade_qty(pos, c, df.iloc[i]['Time'], "OPPOSITE", qty)
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            daily_pnl_running += trade['pnl']
            daily_trade_count += 1
            if pnl_raw < -LOSS_THRESHOLD and not recovering:
                qty = min(qty * 2, MAX_QTY)
                accumulated_loss += pnl_raw
                recovering = True
                stats['martingale_activations'] += 1
            elif recovering:
                accumulated_loss += pnl_raw
                if accumulated_loss >= -RECOVERY_BUFFER:
                    qty = 1
                    accumulated_loss = 0.0
                    recovering = False
                    stats['martingale_recoveries'] += 1
            pos = None
        elif is_sell and pos and pos['dir'] == "LONG":
            pnl_raw = _pnl_raw(pos, c)
            trade = _make_trade_qty(pos, c, df.iloc[i]['Time'], "OPPOSITE", qty)
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            daily_pnl_running += trade['pnl']
            daily_trade_count += 1
            if pnl_raw < -LOSS_THRESHOLD and not recovering:
                qty = min(qty * 2, MAX_QTY)
                accumulated_loss += pnl_raw
                recovering = True
                stats['martingale_activations'] += 1
            elif recovering:
                accumulated_loss += pnl_raw
                if accumulated_loss >= -RECOVERY_BUFFER:
                    qty = 1
                    accumulated_loss = 0.0
                    recovering = False
                    stats['martingale_recoveries'] += 1
            pos = None
        
        # ── New entry (only within window + all checks pass) ──
        if not pos and in_window and t <= ENTRY_END:
            # Check cooldown
            if i < cooldown_until:
                stats['cooldown_skips'] += 1
                continue
            
            # Check max trades per day
            if daily_trade_count >= MAX_TRADES_PER_DAY:
                stats['max_trades_stops'] += 1
                continue
            
            # Check daily loss limit
            if daily_pnl_running <= MAX_DAILY_LOSS:
                daily_stopped = True
                stats['daily_stops'] += 1
                continue
            
            if is_buy and curr_atr >= MIN_ATR:
                # DIRECTION BIAS: only buy if UT Bot direction is bullish
                if curr_dir != 1:
                    stats['direction_filtered'] += 1
                else:
                    sl = c - curr_atr * ATR_KEY_VALUE
                    pos = {'dir': 'LONG', 'entry': c, 'sl': sl,
                           'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
            elif is_sell and curr_atr >= MIN_ATR:
                # DIRECTION BIAS: only sell if UT Bot direction is bearish
                if curr_dir != -1:
                    stats['direction_filtered'] += 1
                else:
                    sl = c + curr_atr * ATR_KEY_VALUE
                    pos = {'dir': 'SHORT', 'entry': c, 'sl': sl,
                           'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
    
    return all_trades, daily_results, stats


def _pnl_raw(pos, exit_price):
    """Raw P&L (1 qty) for internal calculations."""
    if pos['dir'] == "LONG":
        return exit_price - pos['entry']
    else:
        return pos['entry'] - exit_price


def _make_trade_qty(pos, exit_price, exit_time, reason, qty=1):
    """Create trade record with qty-adjusted P&L."""
    raw_pnl = _pnl_raw(pos, exit_price)
    adj_pnl = raw_pnl * qty
    return {
        'dir': pos['dir'],
        'entry': pos['entry'],
        'exit': round(exit_price, 2),
        'entry_time': pos['entry_time'],
        'exit_time': exit_time,
        'pnl': round(adj_pnl, 2),
        'raw_pnl': round(raw_pnl, 2),
        'qty': qty,
        'pnl_pct': round(raw_pnl / pos['entry'] * 100, 4),
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
    print(f"  Win days: {win_days} | Loss days: {loss_days} | Win%: {win_days/(win_days+loss_days)*100:.1f}%")
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
    streak = 0
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
        print(f"Run: python fetch_nifty_data.py --interval ONE_MINUTE")
        return
    
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    total_candles = len(df)
    total_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"
    
    print(f"Loaded {total_candles:,} candles | {total_days} days")
    print(f"Date range: {date_range}")
    
    print(f"\n🤖 UT BOT ALERT BACKTEST (Optimized)")
    print(f"ATR: RMA({args.atr_period}) × {ATR_KEY_VALUE} | Min ATR: {MIN_ATR}")
    print(f"Window: {ENTRY_START.strftime('%H:%M')} - {ENTRY_END.strftime('%H:%M')} | Square off: {SQUARE_OFF.strftime('%H:%M')}")
    print(f"Optimizations: Martingale recovery | Cooldown | Max daily loss | Direction bias")
    print(f"{'='*70}")
    
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
    print(f"\n🚀 Running optimized backtest...")
    all_trades, daily_results, stats = run_backtest(
        df, buy_sig, sell_sig, trail_stop, atr_vals, direction
    )
    
    print(f"  Total trades: {len(all_trades)}")
    
    # Print optimization stats
    print(f"\n  ⚡ OPTIMIZATION STATS")
    print(f"  Martingale activations:  {stats['martingale_activations']} (qty doubled)")
    print(f"  Martingale recoveries:   {stats['martingale_recoveries']} (loss recovered, qty reset)")
    print(f"  Cooldown skips:          {stats['cooldown_skips']} (entries skipped after SL)")
    print(f"  Daily loss stops:        {stats['daily_stops']} (day stopped early)")
    print(f"  Max trades/day stops:    {stats['max_trades_stops']} (trade limit hit)")
    print(f"  Direction filtered:      {stats['direction_filtered']} (bias filter blocked)")
    
    # Check how many trades used elevated qty
    if all_trades:
        elevated = sum(1 for t in all_trades if t.get('qty', 1) > 1)
        print(f"  Elevated qty trades:     {elevated} ({elevated/len(all_trades)*100:.1f}%)")
    
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
