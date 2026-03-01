"""
sniper_strategy.py — Ultra-Selective "Sniper Entry" Backtest (NIFTY 1-min)

CONCEPT: Take VERY FEW trades but with VERY HIGH accuracy
  Only enter when 6+ conditions ALL align simultaneously.
  Max 1 trade per day. Small risk, defined target.

6 CONFLUENCE CONDITIONS (ALL must be true):
  1. VWAP bias     — Price above VWAP = only BUY, below = only SELL
  2. EMA 20 trend  — Close above EMA20 = bullish, below = bearish
  3. RSI recovery  — BUY: RSI was <40, now >45 (recovery from oversold)
                     SELL: RSI was >60, now <55 (recovery from overbought)
  4. Volume spike  — Current volume > 2x 20-period average
  5. Strong candle — Body > 60% of range, direction matches signal
  6. Pullback done — At least 1 of last 3 candles was opposite color (confirms pullback happened)

RISK MANAGEMENT:
  - SL: Below/above the signal candle low/high (tight, context-based)
  - Target: 1.5x SL (risk:reward = 1:1.5)
  - Max 1 trade per day (only the BEST setup)
  - Window: 1:00 PM - 3:00 PM (most liquid, trending period)
  - Square off: 3:20 PM (buffer before close)

WHY THIS WORKS:
  - 6 conditions = very rare alignment = high probability
  - VWAP + EMA = trend confirmation (no counter-trend trades)
  - RSI recovery = momentum shift confirmed
  - Volume spike = institutional participation
  - Strong candle = conviction
  - Pullback = you're buying the dip, not the top

Usage:
    python sniper_strategy.py                          # Default
    python sniper_strategy.py --file nifty_1min_data.csv
    python sniper_strategy.py --target-mult 2.0        # 1:2 risk:reward
    python sniper_strategy.py --min-confluences 5      # Slightly less strict
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
EMA_PERIOD = 20
RSI_PERIOD = 14
VOL_AVG_PERIOD = 20
ATR_PERIOD = 14

# Confluence thresholds
VOL_SPIKE_MULT = 1.5              # Volume must be > 1.5x average (was 2x, too strict)
MIN_BODY_RATIO = 0.55             # Candle body > 55% of range
RSI_BUY_MIN = 40                  # BUY: RSI between 40-65 (healthy momentum, not overbought)
RSI_BUY_MAX = 65
RSI_SELL_MIN = 35                 # SELL: RSI between 35-60 (healthy momentum, not oversold)
RSI_SELL_MAX = 60

# Risk management
TARGET_MULT = 1.5                 # Target = 1.5x SL distance
MAX_TRADES_PER_DAY = 1            # Only the BEST setup
MIN_CONFLUENCES = 5               # Need 5/6 (achievable now)

# Window
ENTRY_START = dt_time(13, 0)      # 1:00 PM
ENTRY_END = dt_time(15, 0)        # 3:00 PM (stop new entries)
SQUARE_OFF = dt_time(15, 20)      # 3:20 PM


# ─────────── Indicators ───────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

def calc_atr(df, period=14):
    h, l, c = df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_vwap(df):
    """Intraday VWAP (resets each day)."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    
    tp = (high + low + close) / 3  # Typical price
    
    # Reset at each day boundary
    dates = df['Time'].dt.date
    cum_tp_vol = (tp * volume).groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()
    
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


# ─────────── Confluence Scanner ───────────

def scan_confluences(df):
    """
    For each candle, compute all 6 confluence conditions.
    Returns a DataFrame with confluence scores and signal direction.
    """
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    
    # Indicators
    ema20 = calc_ema(close, EMA_PERIOD)
    rsi = calc_rsi(close, RSI_PERIOD)
    atr = calc_atr(df, ATR_PERIOD)
    vwap = calc_vwap(df)
    vol_avg = volume.rolling(VOL_AVG_PERIOD).mean()
    
    # Candle body analysis
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    body_ratio = body / full_range
    bullish_candle = close > open_
    bearish_candle = close < open_
    
    # Pullback check: 2 of last 4 candles were opposite color (stronger pullback)
    had_red_in_4 = bearish_candle.rolling(4).sum() >= 2
    had_green_in_4 = bullish_candle.rolling(4).sum() >= 2
    
    n = len(df)
    results = []
    
    for i in range(20, n):  # Start after warmup
        t = df.iloc[i]['Time'].time()
        c = float(close.iloc[i])
        v = float(volume.iloc[i])
        
        c_ema = float(ema20.iloc[i])
        c_rsi = float(rsi.iloc[i]) if not np.isnan(rsi.iloc[i]) else 50
        c_vwap = float(vwap.iloc[i]) if not np.isnan(vwap.iloc[i]) else c
        c_vol_avg = float(vol_avg.iloc[i]) if not np.isnan(vol_avg.iloc[i]) else v
        c_atr = float(atr.iloc[i])
        c_body_ratio = float(body_ratio.iloc[i]) if not np.isnan(body_ratio.iloc[i]) else 0
        
        is_bullish = bool(bullish_candle.iloc[i])
        is_bearish = bool(bearish_candle.iloc[i])
        had_red = bool(had_red_in_4.iloc[i]) if not np.isnan(had_red_in_4.iloc[i]) else False
        had_green = bool(had_green_in_4.iloc[i]) if not np.isnan(had_green_in_4.iloc[i]) else False
        
        # ═══ BUY CONFLUENCES ═══
        buy_score = 0
        buy_reasons = []
        
        # 1. VWAP bias: price above VWAP
        if c > c_vwap:
            buy_score += 1
            buy_reasons.append("VWAP↑")
        
        # 2. EMA trend: close above EMA20 AND close > open (current candle bullish)
        if c > c_ema:
            buy_score += 1
            buy_reasons.append("EMA↑")
        
        # 3. RSI healthy zone: between 40-65 (not overbought, has room to run)
        if RSI_BUY_MIN <= c_rsi <= RSI_BUY_MAX:
            buy_score += 1
            buy_reasons.append("RSI↑")
        
        # 4. Volume spike
        if c_vol_avg > 0 and v > c_vol_avg * VOL_SPIKE_MULT:
            buy_score += 1
            buy_reasons.append("VOL↑")
        
        # 5. Strong bullish candle
        if is_bullish and c_body_ratio >= MIN_BODY_RATIO:
            buy_score += 1
            buy_reasons.append("CANDLE↑")
        
        # 6. Pullback happened (had at least 1 red candle in last 3)
        if had_red:
            buy_score += 1
            buy_reasons.append("PULL↑")
        
        # ═══ SELL CONFLUENCES ═══
        sell_score = 0
        sell_reasons = []
        
        # 1. VWAP bias: price below VWAP
        if c < c_vwap:
            sell_score += 1
            sell_reasons.append("VWAP↓")
        
        # 2. EMA trend: close below EMA20
        if c < c_ema:
            sell_score += 1
            sell_reasons.append("EMA↓")
        
        # 3. RSI healthy zone: between 35-60 (not oversold, has room to fall)
        if RSI_SELL_MIN <= c_rsi <= RSI_SELL_MAX:
            sell_score += 1
            sell_reasons.append("RSI↓")
        
        # 4. Volume spike
        if c_vol_avg > 0 and v > c_vol_avg * VOL_SPIKE_MULT:
            sell_score += 1
            sell_reasons.append("VOL↓")
        
        # 5. Strong bearish candle
        if is_bearish and c_body_ratio >= MIN_BODY_RATIO:
            sell_score += 1
            sell_reasons.append("CANDLE↓")
        
        # 6. Pullback happened (had at least 1 green candle in last 3)
        if had_green:
            sell_score += 1
            sell_reasons.append("PULL↓")
        
        results.append({
            'idx': i,
            'time': df.iloc[i]['Time'],
            'close': c,
            'high': float(high.iloc[i]),
            'low': float(low.iloc[i]),
            'atr': c_atr,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'buy_reasons': buy_reasons,
            'sell_reasons': sell_reasons,
            'rsi': c_rsi,
            'vol_ratio': round(v / c_vol_avg, 1) if c_vol_avg > 0 else 0,
        })
    
    return results


# ─────────── Backtest ───────────

def run_backtest(df, confluences, min_conf, target_mult):
    """
    Backtest: enter when confluences >= min_conf,
    SL at signal candle low/high, target = SL × target_mult.
    Max 1 trade per day.
    """
    all_trades = []
    daily_results = {}
    
    pos = None  # {'dir','entry','sl','target','entry_time','date'}
    prev_date = None
    trades_today = 0
    
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    
    # Build confluence lookup
    conf_by_idx = {c['idx']: c for c in confluences}
    
    for i in range(len(df)):
        t = df.iloc[i]['Time'].time()
        curr_date = df.iloc[i]['Time'].date()
        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        
        # Day reset
        if prev_date and curr_date != prev_date:
            if pos:
                prev_c = float(close.iloc[i-1])
                trade = _make_trade(pos, prev_c, df.iloc[i-1]['Time'], "DAY_END")
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                pos = None
            trades_today = 0
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0}
        prev_date = curr_date
        
        if curr_date not in daily_results:
            daily_results[curr_date] = {'trades': [], 'pnl': 0}
        
        # Square off
        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF")
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue
        
        # Manage open position
        if pos:
            # Check target hit
            if pos['dir'] == 'LONG' and h >= pos['target']:
                trade = _make_trade(pos, pos['target'], df.iloc[i]['Time'], "TARGET")
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
                continue
            elif pos['dir'] == 'SHORT' and l <= pos['target']:
                trade = _make_trade(pos, pos['target'], df.iloc[i]['Time'], "TARGET")
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
                continue
            
            # Check SL hit
            if pos['dir'] == 'LONG' and l <= pos['sl']:
                trade = _make_trade(pos, pos['sl'], df.iloc[i]['Time'], "STOP_LOSS")
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
                continue
            elif pos['dir'] == 'SHORT' and h >= pos['sl']:
                trade = _make_trade(pos, pos['sl'], df.iloc[i]['Time'], "STOP_LOSS")
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
                continue
        
        # New entry — only in window, max 1/day
        in_window = ENTRY_START <= t <= ENTRY_END
        if pos or not in_window or trades_today >= MAX_TRADES_PER_DAY:
            continue
        
        conf = conf_by_idx.get(i)
        if not conf:
            continue
        
        # BUY setup
        if conf['buy_score'] >= min_conf:
            entry = c
            sl = conf['low'] - 1  # SL just below signal candle low
            sl_dist = entry - sl
            if sl_dist > 0:
                target = entry + sl_dist * target_mult
                pos = {'dir': 'LONG', 'entry': entry, 'sl': sl, 'target': target,
                       'entry_time': conf['time'], 'date': curr_date,
                       'score': conf['buy_score'], 'reasons': conf['buy_reasons']}
                trades_today += 1
        
        # SELL setup (only if didn't just enter BUY)
        elif conf['sell_score'] >= min_conf:
            entry = c
            sl = conf['high'] + 1  # SL just above signal candle high
            sl_dist = sl - entry
            if sl_dist > 0:
                target = entry - sl_dist * target_mult
                pos = {'dir': 'SHORT', 'entry': entry, 'sl': sl, 'target': target,
                       'entry_time': conf['time'], 'date': curr_date,
                       'score': conf['sell_score'], 'reasons': conf['sell_reasons']}
                trades_today += 1
    
    return all_trades, daily_results


def _pnl(pos, exit_price):
    return (exit_price - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - exit_price)

def _make_trade(pos, exit_price, exit_time, reason):
    pnl = _pnl(pos, exit_price)
    return {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': round(exit_price, 2),
        'sl': pos.get('sl', 0), 'target': pos.get('target', 0),
        'entry_time': pos['entry_time'], 'exit_time': exit_time,
        'pnl': round(pnl, 2), 'pnl_pct': round(pnl / pos['entry'] * 100, 4),
        'reason': reason, 'score': pos.get('score', 0),
        'confluences': ' '.join(pos.get('reasons', [])),
    }

def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ─────────── Reports ───────────

def print_daily(daily_results):
    sorted_days = sorted(daily_results.keys())
    
    print(f"\n{'='*105}")
    print(f"  {'Date':>12} {'Dir':>5} {'Entry':>9} {'Exit':>9} {'SL':>9} {'Tgt':>9} {'P&L':>9} {'Reason':>10} {'Confluences'}")
    print(f"  {'-'*100}")
    
    cum_pnl = 0
    win_days = 0
    loss_days = 0
    no_trade_days = 0
    
    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        
        if not trades:
            no_trade_days += 1
            continue
        
        cum_pnl += day_pnl
        icon = "✅" if day_pnl > 0 else "❌"
        if day_pnl > 0:
            win_days += 1
        else:
            loss_days += 1
        
        for t in trades:
            print(f"  {str(day):>12} {t['dir']:>5} {t['entry']:>9.2f} {t['exit']:>9.2f} "
                  f"{t['sl']:>9.2f} {t['target']:>9.2f} {t['pnl']:>+8.2f} {t['reason']:>10} "
                  f"{t['confluences']} {icon}")
    
    total_trading = win_days + loss_days
    print(f"  {'-'*100}")
    print(f"  Cumulative P&L: {cum_pnl:+.2f} pts")
    print(f"  Win days: {win_days}/{total_trading} ({win_days/max(total_trading,1)*100:.0f}%) "
          f"| No-trade days: {no_trade_days}")
    print(f"{'='*105}")


def print_summary(all_trades, daily_results):
    if not all_trades:
        print("❌ No trades taken. Try lowering --min-confluences to 5")
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
    
    # Close reason breakdown
    reasons = {}
    for t in all_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    
    # Direction
    longs = [t for t in all_trades if t['dir'] == 'LONG']
    shorts = [t for t in all_trades if t['dir'] == 'SHORT']
    long_wr = sum(1 for t in longs if t['pnl'] > 0) / max(len(longs), 1) * 100
    short_wr = sum(1 for t in shorts if t['pnl'] > 0) / max(len(shorts), 1) * 100
    
    # Drawdown
    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()
    
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
    
    trading_days = sum(1 for d in daily_results.values() if d['trades'])
    total_days = len(daily_results)
    
    print(f"\n{'='*65}")
    print(f"  🎯 SNIPER STRATEGY — OVERALL RESULTS")
    print(f"  {MIN_CONFLUENCES} confluences | Target: {TARGET_MULT}x SL | Max {MAX_TRADES_PER_DAY}/day")
    print(f"{'='*65}")
    
    print(f"\n  📊 ACCURACY")
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
    print(f"  No-trade days:     {total_days - trading_days}")
    
    print(f"\n  📈 DIRECTION")
    print(f"  LONG:  {len(longs):4d} trades | Win: {long_wr:.0f}% | P&L: {sum(t['pnl'] for t in longs):+.2f}")
    print(f"  SHORT: {len(shorts):4d} trades | Win: {short_wr:.0f}% | P&L: {sum(t['pnl'] for t in shorts):+.2f}")
    
    print(f"\n  🔍 EXIT REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_t = [t for t in all_trades if t['reason'] == r]
        r_wr = sum(1 for t in r_t if t['pnl'] > 0) / len(r_t) * 100
        r_pnl = sum(t['pnl'] for t in r_t)
        print(f"    {r:12s}: {c:4d} | Win: {r_wr:4.0f}% | P&L: {r_pnl:+.2f}")
    
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
    global TARGET_MULT, MIN_CONFLUENCES, MAX_TRADES_PER_DAY
    
    parser = argparse.ArgumentParser(description="Sniper Entry Strategy (NIFTY 1-min)")
    parser.add_argument("--file", default="nifty_1min_data.csv")
    parser.add_argument("--target-mult", type=float, default=TARGET_MULT, help="Target multiplier (default: 1.5x SL)")
    parser.add_argument("--min-confluences", type=int, default=MIN_CONFLUENCES, help="Min confluences (default: 6)")
    parser.add_argument("--max-trades", type=int, default=MAX_TRADES_PER_DAY, help="Max trades/day (default: 1)")
    args = parser.parse_args()
    
    TARGET_MULT = args.target_mult
    MIN_CONFLUENCES = args.min_confluences
    MAX_TRADES_PER_DAY = args.max_trades
    
    # Load
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
    
    print(f"Loaded {total_candles:,} candles | {total_days} days")
    print(f"Range: {df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}")
    
    print(f"\n🎯 SNIPER ENTRY STRATEGY")
    print(f"Confluences required: {MIN_CONFLUENCES}/6 | Target: {TARGET_MULT}x SL | Max: {MAX_TRADES_PER_DAY}/day")
    print(f"Window: {ENTRY_START.strftime('%H:%M')} - {ENTRY_END.strftime('%H:%M')} | Square off: {SQUARE_OFF.strftime('%H:%M')}")
    print(f"Conditions: VWAP + EMA20 + RSI recovery + Volume 2x + Strong candle + Pullback")
    print(f"{'='*65}")
    
    # Scan confluences
    print(f"\nScanning confluences...")
    confluences = scan_confluences(df)
    
    buy_qualified = sum(1 for c in confluences if c['buy_score'] >= MIN_CONFLUENCES)
    sell_qualified = sum(1 for c in confluences if c['sell_score'] >= MIN_CONFLUENCES)
    
    print(f"  Total candles scanned: {len(confluences)}")
    print(f"  BUY setups (≥{MIN_CONFLUENCES}): {buy_qualified}")
    print(f"  SELL setups (≥{MIN_CONFLUENCES}): {sell_qualified}")
    print(f"  Total qualified:       {buy_qualified + sell_qualified} "
          f"({(buy_qualified + sell_qualified)/max(len(confluences),1)*100:.2f}% of candles)")
    
    # Score distribution
    all_buy = [c['buy_score'] for c in confluences]
    all_sell = [c['sell_score'] for c in confluences]
    for score in range(6, 0, -1):
        nb = sum(1 for s in all_buy if s >= score)
        ns = sum(1 for s in all_sell if s >= score)
        print(f"    Score ≥{score}: {nb+ns:6d} ({nb} buy + {ns} sell)")
    
    # Backtest
    print(f"\n🚀 Running backtest...")
    all_trades, daily_results = run_backtest(df, confluences, MIN_CONFLUENCES, TARGET_MULT)
    
    print(f"  Trades taken: {len(all_trades)}")
    
    if all_trades:
        # Daily
        print_daily(daily_results)
        # Summary
        print_summary(all_trades, daily_results)
        
        # Save
        trades_df = pd.DataFrame([{k: v for k, v in t.items()} for t in all_trades])
        trades_df.to_csv("sniper_trades.csv", index=False)
        print(f"\n💾 Saved: sniper_trades.csv")
    else:
        print(f"\n❌ No trades qualified. Try:")
        print(f"   python sniper_strategy.py --min-confluences 5")
        print(f"   python sniper_strategy.py --min-confluences 4")


if __name__ == "__main__":
    main()
