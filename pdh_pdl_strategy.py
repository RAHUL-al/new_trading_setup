"""
pdh_pdl_strategy.py — Previous Day High/Low Breakout + Retest Strategy

THIS IS DIFFERENT FROM ALL PREVIOUS STRATEGIES.
  Previous strategies used computed indicators (EMA, RSI, VWAP) that are noisy on 1-min.
  This uses REAL institutional levels (PDH/PDL) that large traders actually watch.

CONCEPT:
  1. Calculate Previous Day High (PDH) and Previous Day Low (PDL)
  2. Resample 1-min → 5-min candles (less noise)
  3. Wait for 5-min candle to CLOSE above PDH (breakout confirmed)
  4. Wait for price to pull back near PDH (retest)
  5. Enter LONG when retest holds + new 5-min candle closes above PDH
  6. SL = just below PDH (very tight, context-based)
  7. Target = 2x SL distance
  8. Same logic in reverse for SELL below PDL

WHY THIS WORKS:
  - PDH/PDL are real support/resistance — institutions place orders here
  - Breakout + retest = confirmation pattern (not a fakeout)
  - SL is tight because it's at a known level
  - 0-1 trade per day, very selective
  - Works on any instrument

Usage:
    python pdh_pdl_strategy.py
    python pdh_pdl_strategy.py --target-mult 2.5
    python pdh_pdl_strategy.py --max-trades 2
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
RESAMPLE_MINUTES = 5              # Resample 1-min → 5-min
ATR_PERIOD = 14

# Entry rules
RETEST_BUFFER_ATR = 0.3           # Retest must come within 0.3*ATR of PDH/PDL
SL_BUFFER_PTS = 5                 # SL = PDH/PDL ± 5 pts buffer
TARGET_MULT = 2.0                 # Target = 2x SL distance (1:2 risk:reward)
MAX_TRADES_PER_DAY = 1
BREAKOUT_CONFIRM_CANDLES = 2      # Need 2 consecutive 5-min closes beyond level

# Time window
ENTRY_START = dt_time(13, 0)      # 1:00 PM
ENTRY_END = dt_time(15, 0)        # 3:00 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM


# ─────────── Data Prep ───────────

def resample_to_5min(df):
    """Resample 1-min candles to 5-min."""
    df = df.set_index('Time')
    resampled = df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    resampled = resampled.reset_index()
    resampled.rename(columns={'index': 'Time'}, inplace=True)
    return resampled


def calc_atr_5min(df, period=14):
    """ATR on 5-min data."""
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def get_previous_day_levels(df):
    """For each day, get previous day's high and low."""
    df['date'] = df['Time'].dt.date
    
    # Daily highs and lows
    daily_hl = df.groupby('date').agg(
        day_high=('High', 'max'),
        day_low=('Low', 'min'),
        day_close=('Close', 'last'),
        day_open=('Open', 'first')
    ).reset_index()
    
    # Shift to get previous day
    daily_hl['pdh'] = daily_hl['day_high'].shift(1)
    daily_hl['pdl'] = daily_hl['day_low'].shift(1)
    daily_hl['pd_close'] = daily_hl['day_close'].shift(1)
    daily_hl['pd_range'] = daily_hl['pdh'] - daily_hl['pdl']
    
    return daily_hl.set_index('date')


# ─────────── Backtest ───────────

def run_backtest(df_5m, daily_levels, target_mult, max_trades):
    """
    Backtest PDH/PDL breakout + retest strategy.
    
    State machine per day:
      IDLE → BREAKOUT_DETECTED → RETEST_WAIT → POSITION_OPEN → (EXIT)
    """
    close = df_5m['Close'].astype(float)
    high = df_5m['High'].astype(float)
    low = df_5m['Low'].astype(float)
    open_ = df_5m['Open'].astype(float)
    
    atr = calc_atr_5min(df_5m, ATR_PERIOD)
    
    all_trades = []
    daily_results = {}
    
    # State
    state = 'IDLE'
    breakout_dir = None   # 'LONG' or 'SHORT'
    breakout_level = 0    # PDH or PDL
    pos = None
    trades_today = 0
    current_date = None
    pdh = pdl = None
    
    for i in range(len(df_5m)):
        t = df_5m.iloc[i]['Time']
        t_time = t.time()
        t_date = t.date()
        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        o = float(open_.iloc[i])
        curr_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 20
        
        # ── New day ──
        if current_date != t_date:
            # Close any open position at previous day end
            if pos and current_date:
                prev_c = float(close.iloc[i-1])
                trade = _make_trade(pos, prev_c, df_5m.iloc[i-1]['Time'], "DAY_END")
                all_trades.append(trade)
                _add_daily(daily_results, current_date, trade)
                pos = None
            
            current_date = t_date
            trades_today = 0
            state = 'IDLE'
            breakout_dir = None
            
            if t_date not in daily_results:
                daily_results[t_date] = {'trades': [], 'pnl': 0}
            
            # Get PDH/PDL for today
            if t_date in daily_levels.index:
                pdh = daily_levels.loc[t_date, 'pdh']
                pdl = daily_levels.loc[t_date, 'pdl']
                pd_range = daily_levels.loc[t_date, 'pd_range']
                if np.isnan(pdh) or np.isnan(pdl) or pd_range < 10:
                    pdh = pdl = None
            else:
                pdh = pdl = None
            continue
        
        if pdh is None:
            continue
        
        # ── Square off at 3:24 PM ──
        if pos and t_time >= SQUARE_OFF:
            trade = _make_trade(pos, c, t, "SQUARE_OFF")
            all_trades.append(trade)
            _add_daily(daily_results, current_date, trade)
            pos = None
            state = 'IDLE'
            continue
        
        # ── Manage open position ──
        if pos:
            # Target hit
            if pos['dir'] == 'LONG' and h >= pos['target']:
                trade = _make_trade(pos, pos['target'], t, "TARGET")
                all_trades.append(trade)
                _add_daily(daily_results, current_date, trade)
                pos = None
                state = 'IDLE'
                continue
            elif pos['dir'] == 'SHORT' and l <= pos['target']:
                trade = _make_trade(pos, pos['target'], t, "TARGET")
                all_trades.append(trade)
                _add_daily(daily_results, current_date, trade)
                pos = None
                state = 'IDLE'
                continue
            
            # SL hit
            if pos['dir'] == 'LONG' and l <= pos['sl']:
                trade = _make_trade(pos, pos['sl'], t, "STOP_LOSS")
                all_trades.append(trade)
                _add_daily(daily_results, current_date, trade)
                pos = None
                state = 'IDLE'
                continue
            elif pos['dir'] == 'SHORT' and h >= pos['sl']:
                trade = _make_trade(pos, pos['sl'], t, "STOP_LOSS")
                all_trades.append(trade)
                _add_daily(daily_results, current_date, trade)
                pos = None
                state = 'IDLE'
                continue
            
            continue  # Position open, nothing to do
        
        # ── Only look for entries in window ──
        if not (ENTRY_START <= t_time <= ENTRY_END):
            continue
        
        if trades_today >= max_trades:
            continue
        
        # ═══ STATE MACHINE ═══
        
        if state == 'IDLE':
            # Look for breakout: 5-min candle CLOSES above PDH or below PDL
            if c > pdh:
                breakout_dir = 'LONG'
                breakout_level = pdh
                state = 'BREAKOUT_DETECTED'
            elif c < pdl:
                breakout_dir = 'SHORT'
                breakout_level = pdl
                state = 'BREAKOUT_DETECTED'
        
        elif state == 'BREAKOUT_DETECTED':
            # Confirm breakout: another candle still beyond the level
            if breakout_dir == 'LONG':
                if c > pdh:
                    state = 'RETEST_WAIT'  # Breakout confirmed, wait for retest
                else:
                    state = 'IDLE'  # Fakeout, reset
            elif breakout_dir == 'SHORT':
                if c < pdl:
                    state = 'RETEST_WAIT'
                else:
                    state = 'IDLE'
        
        elif state == 'RETEST_WAIT':
            # Wait for price to come back near the level (retest)
            retest_zone = curr_atr * RETEST_BUFFER_ATR
            
            if breakout_dir == 'LONG':
                # Price pulls back to near PDH
                if l <= pdh + retest_zone:
                    # Retest happened! Now check if it holds
                    if c > pdh:
                        # Retest held → ENTER LONG
                        entry = c
                        sl = pdh - SL_BUFFER_PTS
                        sl_dist = entry - sl
                        if sl_dist > 3:  # Min SL distance
                            target = entry + sl_dist * target_mult
                            pos = {'dir': 'LONG', 'entry': entry, 'sl': sl,
                                   'target': round(target, 2),
                                   'entry_time': t, 'level': f"PDH={pdh:.2f}"}
                            trades_today += 1
                            state = 'POSITION_OPEN'
                        else:
                            state = 'IDLE'
                    else:
                        # Retest broke down → fakeout
                        state = 'IDLE'
                elif c < pdh - retest_zone:
                    # Price fell too far below, fakeout
                    state = 'IDLE'
            
            elif breakout_dir == 'SHORT':
                # Price pulls back up to near PDL
                if h >= pdl - retest_zone:
                    if c < pdl:
                        # Retest held → ENTER SHORT
                        entry = c
                        sl = pdl + SL_BUFFER_PTS
                        sl_dist = sl - entry
                        if sl_dist > 3:
                            target = entry - sl_dist * target_mult
                            pos = {'dir': 'SHORT', 'entry': entry, 'sl': sl,
                                   'target': round(target, 2),
                                   'entry_time': t, 'level': f"PDL={pdl:.2f}"}
                            trades_today += 1
                            state = 'POSITION_OPEN'
                        else:
                            state = 'IDLE'
                    else:
                        state = 'IDLE'
                elif c > pdl + retest_zone:
                    state = 'IDLE'
    
    # Close any remaining position
    if pos:
        c = float(close.iloc[-1])
        trade = _make_trade(pos, c, df_5m.iloc[-1]['Time'], "DAY_END")
        all_trades.append(trade)
        _add_daily(daily_results, current_date, trade)
    
    return all_trades, daily_results


def _pnl(pos, exit_price):
    return (exit_price - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - exit_price)

def _make_trade(pos, exit_price, exit_time, reason):
    pnl = _pnl(pos, exit_price)
    return {
        'dir': pos['dir'], 'entry': round(pos['entry'], 2),
        'exit': round(exit_price, 2),
        'sl': round(pos.get('sl', 0), 2),
        'target': round(pos.get('target', 0), 2),
        'entry_time': pos['entry_time'], 'exit_time': exit_time,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl / pos['entry'] * 100, 4),
        'reason': reason,
        'level': pos.get('level', ''),
    }

def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ─────────── Reports ───────────

def print_daily(daily_results):
    sorted_days = sorted(daily_results.keys())
    
    print(f"\n{'='*115}")
    print(f"  {'Date':>12} {'Dir':>5} {'Entry':>9} {'Exit':>9} {'SL':>9} {'Tgt':>9} "
          f"{'P&L':>9} {'Reason':>10} {'Level'}")
    print(f"  {'-'*110}")
    
    cum = 0
    wd = 0
    ld = 0
    
    for day in sorted_days:
        trades = daily_results[day]['trades']
        if not trades:
            continue
        
        day_pnl = daily_results[day]['pnl']
        cum += day_pnl
        icon = "✅" if day_pnl > 0 else "❌"
        if day_pnl > 0: wd += 1
        else: ld += 1
        
        for t in trades:
            print(f"  {str(day):>12} {t['dir']:>5} {t['entry']:>9.2f} {t['exit']:>9.2f} "
                  f"{t['sl']:>9.2f} {t['target']:>9.2f} {t['pnl']:>+8.2f} "
                  f"{t['reason']:>10} {t['level']} {icon}")
    
    tot = wd + ld
    print(f"  {'-'*110}")
    print(f"  Cumulative: {cum:+.2f} pts | Win days: {wd}/{tot} ({wd/max(tot,1)*100:.0f}%)")
    print(f"{'='*115}")


def print_summary(all_trades, daily_results):
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
    
    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()
    
    # Direction
    longs = [t for t in all_trades if t['dir'] == 'LONG']
    shorts = [t for t in all_trades if t['dir'] == 'SHORT']
    long_wr = sum(1 for t in longs if t['pnl'] > 0) / max(len(longs), 1) * 100
    short_wr = sum(1 for t in shorts if t['pnl'] > 0) / max(len(shorts), 1) * 100
    
    # Reasons
    reasons = {}
    for t in all_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    
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
    
    total_days = len(daily_results)
    trading_days = sum(1 for d in daily_results.values() if d['trades'])
    
    print(f"\n{'='*65}")
    print(f"  📊 PDH/PDL BREAKOUT + RETEST — RESULTS")
    print(f"  5-min candles | Target: {TARGET_MULT}x SL | Max {MAX_TRADES_PER_DAY}/day")
    print(f"{'='*65}")
    
    print(f"\n  🎯 ACCURACY")
    print(f"  Total trades:      {n}")
    print(f"  Win rate:          {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Risk:Reward:       1:{rr:.2f}")
    print(f"  Avg win:           {avg_win:+.2f} pts")
    print(f"  Avg loss:          {avg_loss:+.2f} pts")
    
    print(f"\n  💰 P&L")
    print(f"  Total P&L:         {total_pnl:+.2f} pts")
    print(f"  Best trade:        {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:       {min(pnl_list):+.2f} pts")
    print(f"  Max drawdown:      {max_dd:.2f} pts")
    
    print(f"\n  📅 FREQUENCY")
    print(f"  Total days:        {total_days}")
    print(f"  Trading days:      {trading_days} ({trading_days/max(total_days,1)*100:.0f}%)")
    print(f"  Avg trades/week:   {n/(max(total_days,1)/5):.1f}")
    
    print(f"\n  📈 DIRECTION")
    print(f"  LONG:  {len(longs):4d} | Win: {long_wr:.0f}% | P&L: {sum(t['pnl'] for t in longs):+.2f}")
    print(f"  SHORT: {len(shorts):4d} | Win: {short_wr:.0f}% | P&L: {sum(t['pnl'] for t in shorts):+.2f}")
    
    print(f"\n  🔍 EXIT REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_t = [t for t in all_trades if t['reason'] == r]
        r_wr = sum(1 for t in r_t if t['pnl'] > 0) / len(r_t) * 100
        print(f"    {r:12s}: {c:4d} | Win: {r_wr:4.0f}% | P&L: {sum(t['pnl'] for t in r_t):+.2f}")
    
    print(f"\n  📅 MONTHLY")
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
    global TARGET_MULT, MAX_TRADES_PER_DAY
    
    parser = argparse.ArgumentParser(description="PDH/PDL Breakout + Retest Strategy")
    parser.add_argument("--file", default="nifty_1min_data.csv")
    parser.add_argument("--target-mult", type=float, default=TARGET_MULT)
    parser.add_argument("--max-trades", type=int, default=MAX_TRADES_PER_DAY)
    parser.add_argument("--sl-buffer", type=float, default=SL_BUFFER_PTS)
    args = parser.parse_args()
    
    TARGET_MULT = args.target_mult
    MAX_TRADES_PER_DAY = args.max_trades
    
    # Load 1-min data
    print(f"Loading {args.file}...")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        print("Run: python fetch_nifty_data.py --interval ONE_MINUTE")
        return
    
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} 1-min candles | {df['Time'].dt.date.nunique()} days")
    print(f"Range: {df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Resample to 5-min
    print(f"\nResampling 1-min → 5-min...")
    df_5m = resample_to_5min(df)
    print(f"5-min candles: {len(df_5m):,}")
    
    # Get PDH/PDL levels
    daily_levels = get_previous_day_levels(df_5m)
    valid_days = daily_levels.dropna(subset=['pdh', 'pdl'])
    print(f"Days with valid PDH/PDL: {len(valid_days)}")
    
    # Show some sample levels
    print(f"\nSample PDH/PDL levels:")
    for _, row in valid_days.head(5).iterrows():
        print(f"  {row.name}: PDH={row['pdh']:.2f} PDL={row['pdl']:.2f} Range={row['pd_range']:.2f}")
    
    print(f"\n📊 PDH/PDL BREAKOUT + RETEST STRATEGY")
    print(f"5-min candles | Entry: {ENTRY_START.strftime('%H:%M')}-{ENTRY_END.strftime('%H:%M')}")
    print(f"SL buffer: {args.sl_buffer} pts | Target: {TARGET_MULT}x SL | Max: {MAX_TRADES_PER_DAY}/day")
    print(f"Logic: Break above PDH → wait retest → enter LONG (vice versa for PDL)")
    print(f"{'='*65}")
    
    # Backtest
    print(f"\n🚀 Running backtest...")
    all_trades, daily_results = run_backtest(df_5m, daily_levels, TARGET_MULT, MAX_TRADES_PER_DAY)
    
    print(f"  Trades taken: {len(all_trades)}")
    
    if all_trades:
        print_daily(daily_results)
        print_summary(all_trades, daily_results)
        
        pd.DataFrame(all_trades).to_csv("pdh_pdl_trades.csv", index=False)
        print(f"\n💾 Saved: pdh_pdl_trades.csv")
    else:
        print("\n❌ No trades. Try:")
        print("  python pdh_pdl_strategy.py --max-trades 2")
        print("  python pdh_pdl_strategy.py --sl-buffer 3")


if __name__ == "__main__":
    main()
