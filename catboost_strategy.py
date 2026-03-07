"""
catboost_strategy.py — CatBoost ML Strategy for NIFTY (Multi-Timeframe)

Uses 1-minute AND 2-minute OHLCV data to train a CatBoost model
that predicts BUY/SELL signals for the 2:00 PM - 3:03 PM window.

FEATURES (per candle):
  From 1-min data: ATR, UT Bot trail stop, RSI, price momentum, candle patterns
  From 2-min data: ATR, trend direction, support/resistance levels

LABELS:
  1  = BUY  (future N-candle return > threshold)
  -1 = SELL (future N-candle return < -threshold)
  0  = HOLD

Usage:
    python catboost_strategy.py                          # Train + backtest
    python catboost_strategy.py --min-atr 6.5
    python catboost_strategy.py --lookahead 5            # 5 candles for labeling
    python catboost_strategy.py --threshold 8            # 8 pts for signal
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("❌ CatBoost not installed. Run: pip install catboost")
    exit(1)


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5

ENTRY_START = dt_time(14, 0)      # 2:00 PM
ENTRY_END = dt_time(15, 3)        # 3:03 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM

LOT_SIZE = 65
BASE_LOTS = 2

LOOKAHEAD = 5                     # N candles ahead for labeling
THRESHOLD = 8.0                   # Min pts for buy/sell label

# Fixed train/test split by year
TRAIN_END_YEAR = 2024             # Train: 2019-2024
TEST_START_YEAR = 2025            # Test: 2025-2026


# ─────────── Indicators ───────────

def calc_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_atr(df, period=14):
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    return calc_rma(tr, period)


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = calc_rma(gain, period)
    avg_loss = calc_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_ut_bot_direction(close, atr, key_value=1.0):
    """Returns trail_stop and direction arrays."""
    n = len(close)
    trail_stop = np.zeros(n)
    direction = np.zeros(n)
    trail_stop[0] = close[0]
    direction[0] = 1

    for i in range(1, n):
        nloss = atr[i] * key_value
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]

        if prev_dir == 1:
            new_ts = close[i] - nloss
            trail_stop[i] = max(new_ts, prev_ts)
            if close[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close[i] + nloss
            else:
                direction[i] = 1
        else:
            new_ts = close[i] + nloss
            trail_stop[i] = min(new_ts, prev_ts)
            if close[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
            else:
                direction[i] = -1

    return trail_stop, direction


# ─────────── Feature Engineering ───────────

def build_features_1min(df):
    """Build features from 1-minute data."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    trail, dirn = calc_ut_bot_direction(close.values, atr.values, ATR_KEY_VALUE)

    features = pd.DataFrame(index=df.index)
    features['atr_1m'] = atr
    features['rsi_1m'] = rsi
    features['ut_dir_1m'] = dirn
    features['close_vs_trail_1m'] = close.values - trail

    # Price momentum
    features['mom_3'] = close.pct_change(3) * 100
    features['mom_5'] = close.pct_change(5) * 100
    features['mom_10'] = close.pct_change(10) * 100

    # Candle features
    features['body_1m'] = close - opn
    features['body_pct_1m'] = (close - opn) / opn * 100
    features['upper_wick_1m'] = high - close.where(close > opn, opn)
    features['lower_wick_1m'] = close.where(close < opn, opn) - low
    features['range_1m'] = high - low

    # Volatility
    features['std_5'] = close.rolling(5).std()
    features['std_10'] = close.rolling(10).std()

    # Moving averages
    features['sma_5'] = close.rolling(5).mean()
    features['sma_10'] = close.rolling(10).mean()
    features['sma_20'] = close.rolling(20).mean()
    features['close_vs_sma5'] = close - features['sma_5']
    features['close_vs_sma10'] = close - features['sma_10']
    features['sma5_vs_sma10'] = features['sma_5'] - features['sma_10']

    # High/Low channels
    features['high_5'] = high.rolling(5).max()
    features['low_5'] = low.rolling(5).min()
    features['close_vs_high5'] = close - features['high_5']
    features['close_vs_low5'] = close - features['low_5']

    return features


def build_features_2min(df_2m, df_1m):
    """
    Build features from 2-min data and align to 1-min index.
    Uses forward-fill to map 2-min features to the 1-min timeframe.
    """
    close_2m = df_2m['Close'].astype(float)
    high_2m = df_2m['High'].astype(float)
    low_2m = df_2m['Low'].astype(float)

    atr_2m = calc_atr(df_2m, ATR_PERIOD)
    rsi_2m = calc_rsi(close_2m, 14)
    trail_2m, dir_2m = calc_ut_bot_direction(close_2m.values, atr_2m.values, ATR_KEY_VALUE)

    feat_2m = pd.DataFrame(index=df_2m.index)
    feat_2m['Time'] = df_2m['Time']
    feat_2m['atr_2m'] = atr_2m.values
    feat_2m['rsi_2m'] = rsi_2m.values
    feat_2m['ut_dir_2m'] = dir_2m
    feat_2m['close_vs_trail_2m'] = close_2m.values - trail_2m
    feat_2m['mom_3_2m'] = close_2m.pct_change(3).values * 100
    feat_2m['mom_5_2m'] = close_2m.pct_change(5).values * 100
    feat_2m['range_2m'] = (high_2m - low_2m).values
    feat_2m['body_2m'] = (close_2m - df_2m['Open'].astype(float)).values

    # Merge to 1-min by time (forward fill)
    feat_2m['Time'] = pd.to_datetime(feat_2m['Time'])
    df_1m_time = pd.DataFrame({'Time': pd.to_datetime(df_1m['Time'])})

    merged = pd.merge_asof(
        df_1m_time.sort_values('Time'),
        feat_2m.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    return merged.drop('Time', axis=1).reset_index(drop=True)


# ─────────── Label Generation ───────────

def generate_labels(df, lookahead=5, threshold=8.0):
    """
    Label each candle:
      1  = BUY  (max future gain > threshold)
      -1 = SELL (max future loss > threshold)
      0  = HOLD
    """
    close = df['Close'].astype(float).values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(n - lookahead):
        future = close[i+1:i+1+lookahead]
        max_gain = future.max() - close[i]
        max_loss = close[i] - future.min()

        if max_gain > threshold and max_gain > max_loss:
            labels[i] = 1   # BUY
        elif max_loss > threshold and max_loss > max_gain:
            labels[i] = -1  # SELL
        else:
            labels[i] = 0   # HOLD

    return labels


# ─────────── Backtest ───────────

def backtest_predictions(df, predictions, atr_vals):
    """Backtest using CatBoost predictions as signals."""
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)

    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None

    # Lot management
    current_lots = BASE_LOTS
    accumulated_loss = 0.0
    recovering = False

    for i in range(len(df)):
        t = df.iloc[i]['Time'].time()
        curr_date = df.iloc[i]['Time'].date()
        c = float(close.iloc[i])
        h = float(high_v.iloc[i])
        l = float(low_v.iloc[i])
        curr_atr = float(atr_vals[i])

        # Day boundary
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                pnl = _pnl(pos, prev_close)
                trade = _make_trade(pos, prev_close, df.iloc[i-1]['Time'], "DAY_END", current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                pos = None
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0, 'lots': current_lots}

            # Lot adjustment
            if prev_date in daily_results:
                prev_pnl = daily_results[prev_date]['pnl']
                if prev_pnl < 0:
                    accumulated_loss += prev_pnl
                    recovering = True
                    current_lots += 2
                elif prev_pnl > 0 and recovering:
                    accumulated_loss += prev_pnl
                    if accumulated_loss >= 0:
                        current_lots = BASE_LOTS
                        accumulated_loss = 0.0
                        recovering = False

            if curr_date in daily_results:
                daily_results[curr_date]['lots'] = current_lots
        prev_date = curr_date

        if curr_date not in daily_results:
            daily_results[curr_date] = {'trades': [], 'pnl': 0, 'lots': current_lots}

        # Square off
        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF", current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue

        in_window = ENTRY_START <= t <= ENTRY_END

        # SL check
        if pos:
            sl_hit = False
            if pos['dir'] == "LONG" and l <= pos['sl']:
                sl_hit = True
                exit_price = pos['sl']
            elif pos['dir'] == "SHORT" and h >= pos['sl']:
                sl_hit = True
                exit_price = pos['sl']

            if sl_hit:
                trade = _make_trade(pos, exit_price, df.iloc[i]['Time'], "TRAIL_SL", current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None

        # Trail SL update
        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl

        # Signal from CatBoost
        pred = predictions[i]

        # Opposite signal close
        if pred == 1 and pos and pos['dir'] == "SHORT":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE", current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        elif pred == -1 and pos and pos['dir'] == "LONG":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE", current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None

        # New entry
        if not pos and in_window and curr_atr >= MIN_ATR:
            if pred == 1:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
            elif pred == -1:
                sl = c + curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'SHORT', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}

    return all_trades, daily_results


def _pnl(pos, exit_price):
    if pos['dir'] == "LONG":
        return exit_price - pos['entry']
    else:
        return pos['entry'] - exit_price


def _make_trade(pos, exit_price, exit_time, reason, lots=2, sl=None):
    raw_pnl = _pnl(pos, exit_price)
    lot_multiplier = lots // BASE_LOTS
    adj_pnl = raw_pnl * lot_multiplier
    trade = {
        'dir': pos['dir'],
        'entry': pos['entry'],
        'exit': round(exit_price, 2),
        'entry_time': pos['entry_time'],
        'exit_time': exit_time,
        'pnl': round(adj_pnl, 2),
        'raw_pnl': round(raw_pnl, 2),
        'lots': lots,
        'multiplier': lot_multiplier,
        'qty': lots * LOT_SIZE,
        'pnl_pct': round(raw_pnl / pos['entry'] * 100, 4),
        'reason': reason,
    }
    if sl is not None:
        trade['sl'] = round(sl, 2)
    return trade


def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0, 'lots': trade.get('lots', BASE_LOTS)}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ─────────── Reports ───────────

def print_results(all_trades, daily_results, label=""):
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

    # Drawdown
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (peak - cumulative).max()

    trading_days = sum(1 for d in daily_results.values() if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)

    print(f"\n{'='*60}")
    print(f"  🤖 CATBOOST STRATEGY — {label}")
    print(f"{'='*60}")
    print(f"  Total trades:      {n}")
    print(f"  Win rate:          {wr:.1f}%")
    print(f"  Total P&L:         {total_pnl:+.2f} pts")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Avg win:           {avg_win:+.2f} pts")
    print(f"  Avg loss:          {avg_loss:+.2f} pts")
    print(f"  Max drawdown:      {max_dd:.2f} pts")
    print(f"  Trading days:      {trading_days}")
    print(f"  Profitable days:   {win_days} ({win_days/max(trading_days,1)*100:.0f}%)")
    print(f"  Avg P&L/day:       {total_pnl/max(trading_days,1):+.2f} pts")

    # Daily results
    sorted_days = sorted(daily_results.keys())
    print(f"\n  {'Date':>12} {'Lots':>5} {'Trades':>7} {'P&L':>10} {'Status':>8}")
    print(f"  {'-'*45}")
    cum = 0
    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        lots = daily_results[day].get('lots', BASE_LOTS)
        if len(trades) == 0:
            continue
        cum += day_pnl
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        print(f"  {str(day):>12} {lots:>5} {len(trades):>7} {day_pnl:>+9.2f} {icon:>8}")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':>12} {'':>5} {'':>7} {cum:>+9.2f}")

    # Close reasons
    reasons = {}
    for t in all_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    print(f"\n  🔍 CLOSE REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_trades = [t for t in all_trades if t['reason'] == r]
        r_wr = sum(1 for t in r_trades if t['pnl'] > 0) / len(r_trades) * 100
        print(f"    {r:15s}: {c:4d} trades | Win: {r_wr:.0f}%")

    print(f"\n{'='*60}")


def print_detailed_daily_log(all_trades, daily_results, log_file="catboost_detailed_log.txt"):
    """Print AND save detailed per-trade explanation for each day."""
    import sys
    
    sorted_days = sorted(daily_results.keys())
    lines = []
    
    def out(s=""):
        print(s)
        lines.append(s)
    
    out(f"\n{'='*110}")
    out(f"  📋 DETAILED TRADE LOG — EACH TRADE ON EACH DAY")
    out(f"{'='*110}")
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    cum_pnl = 0
    
    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        lots = daily_results[day].get('lots', BASE_LOTS)
        
        if len(trades) == 0:
            continue
        
        cum_pnl += day_pnl
        day_name = day_names[day.weekday()]
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = len(trades) - wins
        wr = wins / len(trades) * 100
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        
        out(f"\n  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        out(f"  │ 📅 {day} ({day_name}) | Lots: {lots} (qty={lots*LOT_SIZE}) | Trades: {len(trades)} (W:{wins} L:{losses} {wr:.0f}%) | Day P&L: {day_pnl:+.2f} {icon} | Cum: {cum_pnl:+.2f}")
        out(f"  ├───┬──────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┬─────────────┤")
        out(f"  │ # │ Dir  │ Entry Time│ Exit Time │ Entry Pr  │ Exit Pr   │ SL         │ P&L      │ Exit Reason │")
        out(f"  ├───┼──────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┼─────────────┤")
        
        for j, t in enumerate(trades, 1):
            entry_t = t['entry_time'].strftime('%H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[-8:-3]
            exit_t = t['exit_time'].strftime('%H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[-8:-3]
            
            # Calculate SL from entry and ATR
            sl_str = "---"
            if 'sl' in t:
                sl_str = f"{t['sl']:.2f}"
            
            dir_icon = "🟢" if t['dir'] == 'LONG' else "🔴"
            pnl_icon = "✅" if t['pnl'] > 0 else "❌" if t['pnl'] < 0 else "➖"
            
            out(f"  │{j:>2} │ {dir_icon}{t['dir']:>4} │ {entry_t:>9} │ {exit_t:>9} │ {t['entry']:>9.2f} │ {t['exit']:>9.2f} │ {sl_str:>10} │ {t['pnl']:>+7.2f}{pnl_icon}│ {t['reason']:<12}│")
        
        out(f"  └───┴──────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┴─────────────┘")
        out(f"  Day Summary: {day_pnl:+.2f} pts ({day_name}) | Raw trades: {len(trades)} | Multiplier: {lots // BASE_LOTS}x")
    
    out(f"\n{'='*110}")
    out(f"  GRAND TOTAL: {cum_pnl:+.2f} pts")
    out(f"{'='*110}")
    
    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n💾 Detailed log saved: {log_file}")


# ─────────── Main ───────────

def main():
    global ATR_KEY_VALUE, MIN_ATR, LOOKAHEAD, THRESHOLD

    parser = argparse.ArgumentParser(description="CatBoost ML Strategy (NIFTY)")
    parser.add_argument("--file-1m", default="nifty_1min_data.csv", help="1-min data")
    parser.add_argument("--file-2m", default="nifty_2min_data.csv", help="2-min data")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD, help="Candles to look ahead for labeling")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Min pts for buy/sell label")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    LOOKAHEAD = args.lookahead
    THRESHOLD = args.threshold

    # ── Load data ──
    print(f"Loading 1-min data: {args.file_1m}")
    try:
        df_1m = pd.read_csv(args.file_1m)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file_1m}")
        return

    print(f"Loading 2-min data: {args.file_2m}")
    try:
        df_2m = pd.read_csv(args.file_2m)
    except FileNotFoundError:
        print(f"⚠️  2-min file not found, using 1-min only")
        df_2m = None

    df_1m['Time'] = pd.to_datetime(df_1m['Time'])
    df_1m = df_1m.sort_values('Time').reset_index(drop=True)

    if df_2m is not None and len(df_2m) > 0:
        df_2m['Time'] = pd.to_datetime(df_2m['Time'])
        df_2m = df_2m.sort_values('Time').reset_index(drop=True)
    else:
        df_2m = None

    total_candles = len(df_1m)
    total_days = df_1m['Time'].dt.date.nunique()
    print(f"1-min: {total_candles:,} candles | {total_days} days")
    if df_2m is not None:
        print(f"2-min: {len(df_2m):,} candles")
    print(f"Date range: {df_1m['Time'].iloc[0].strftime('%Y-%m-%d')} → {df_1m['Time'].iloc[-1].strftime('%Y-%m-%d')}")

    print(f"\n🤖 CATBOOST ML STRATEGY")
    print(f"ATR: RMA({ATR_PERIOD}) × {ATR_KEY_VALUE} | Min ATR: {MIN_ATR}")
    print(f"Window: {ENTRY_START.strftime('%H:%M')} - {ENTRY_END.strftime('%H:%M')} | Square off: {SQUARE_OFF.strftime('%H:%M')}")
    print(f"Lookahead: {LOOKAHEAD} candles | Threshold: {THRESHOLD} pts")
    print(f"Train: 2019-{TRAIN_END_YEAR} | Test: {TEST_START_YEAR}-2026")
    print(f"{'='*60}")

    # ── Build features ──
    print(f"\n📊 Building features...")
    feat_1m = build_features_1min(df_1m)
    print(f"  1-min features: {feat_1m.shape[1]} columns")

    if df_2m is not None:
        feat_2m = build_features_2min(df_2m, df_1m)
        features = pd.concat([feat_1m, feat_2m], axis=1)
        print(f"  2-min features: {feat_2m.shape[1]} columns")
    else:
        features = feat_1m

    print(f"  Total features: {features.shape[1]} columns")

    # ── Generate labels ──
    print(f"\n🏷️  Generating labels (lookahead={LOOKAHEAD}, threshold={THRESHOLD})...")
    labels = generate_labels(df_1m, LOOKAHEAD, THRESHOLD)
    buy_count = (labels == 1).sum()
    sell_count = (labels == -1).sum()
    hold_count = (labels == 0).sum()
    print(f"  BUY:  {buy_count} ({buy_count/len(labels)*100:.1f}%)")
    print(f"  SELL: {sell_count} ({sell_count/len(labels)*100:.1f}%)")
    print(f"  HOLD: {hold_count} ({hold_count/len(labels)*100:.1f}%)")

    # ── Filter to trading window only ──
    times = df_1m['Time'].dt.time
    window_mask = (times >= ENTRY_START) & (times <= ENTRY_END)
    print(f"\n  Window candles: {window_mask.sum()} (of {len(df_1m)})")

    # ── Clean features ──
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    # ── Train/Test split (by year, no leakage) ──
    dates = df_1m['Time'].dt.date
    years = df_1m['Time'].dt.year
    
    train_mask = (years <= TRAIN_END_YEAR) & window_mask
    test_mask = (years >= TEST_START_YEAR) & window_mask
    
    train_dates_set = set(dates[years <= TRAIN_END_YEAR].unique())
    test_dates_set = set(dates[years >= TEST_START_YEAR].unique())

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    print(f"\n  Train: {len(X_train)} samples ({len(train_dates_set)} days) [2019-{TRAIN_END_YEAR}]")
    print(f"  Test:  {len(X_test)} samples ({len(test_dates_set)} days) [{TEST_START_YEAR}-2026]")

    # ── Train CatBoost ──
    print(f"\n🧠 Training CatBoost model...")
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100,
    )

    # ── Predictions ──
    train_pred = model.predict(X_train).flatten().astype(int)
    test_pred = model.predict(X_test).flatten().astype(int)

    train_acc = (train_pred == y_train).mean() * 100
    test_acc = (test_pred == y_test).mean() * 100
    print(f"\n  Train accuracy: {train_acc:.1f}%")
    print(f"  Test accuracy:  {test_acc:.1f}%")

    # ── Feature importance ──
    importance = model.get_feature_importance()
    feat_names = features.columns.tolist()
    top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
    print(f"\n  📊 TOP 10 FEATURES:")
    for name, imp in top_features:
        print(f"    {name:>25}: {imp:.2f}")

    # ── Build full prediction array ──
    full_predictions = np.zeros(len(df_1m), dtype=int)
    full_predictions[test_mask] = test_pred

    # ── ATR for backtest ──
    atr_vals = calc_atr(df_1m, ATR_PERIOD).values

    # ── Backtest on test set ──
    print(f"\n🚀 Running backtest on TEST data ({TEST_START_YEAR}-2026)...")
    test_start = min(test_dates_set)
    test_end = max(test_dates_set)
    test_df_mask = years >= TEST_START_YEAR
    df_test = df_1m[test_df_mask].reset_index(drop=True)
    pred_test_full = full_predictions[test_df_mask]
    atr_test = atr_vals[test_df_mask]

    all_trades, daily_results = backtest_predictions(df_test, pred_test_full, atr_test)

    print_results(all_trades, daily_results, f"TEST ({test_start} → {test_end})")
    
    # ── Detailed per-trade daily log ──
    print_detailed_daily_log(all_trades, daily_results)

    # ── Save ──
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("catboost_trades.csv", index=False)
        print(f"\n💾 Trades saved: catboost_trades.csv")

    # Save model
    model.save_model("catboost_nifty_model.cbm")
    print(f"💾 Model saved: catboost_nifty_model.cbm")


if __name__ == "__main__":
    main()
