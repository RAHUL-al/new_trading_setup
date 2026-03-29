"""
catboost_strategy.py — CatBoost ML Strategy for NIFTY (Pure 2-Minute Timeframe)

Uses strictly 2-minute OHLCV data to train a CatBoost model
that predicts BUY/SELL signals for the 2:00 PM - 3:03 PM window.
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

ENTRY_START = dt_time(9, 20)      # 9:20 AM
ENTRY_END = dt_time(15, 15)       # 3:15 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM

LOT_SIZE = 65
BASE_LOTS = 2

LOOKAHEAD = 3                     # N candles ahead for labeling (3 candles x 2 = 6 mins)
THRESHOLD = 8.0                   # Min pts for buy/sell label

TRAIN_END_YEAR = 2024
TEST_START_YEAR = 2025

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

def build_features_2m(df):
    """Build all features exclusively from 2-minute data."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    trail, dirn = calc_ut_bot_direction(close.values, atr.values, ATR_KEY_VALUE)

    features = pd.DataFrame(index=df.index)
    
    # Core Indicators
    features['atr_2m'] = atr
    features['rsi_2m'] = rsi
    features['ut_dir_2m'] = dirn
    features['close_vs_trail_2m'] = close.values - trail

    # Price momentum
    features['mom_3_2m'] = close.pct_change(3) * 100
    features['mom_5_2m'] = close.pct_change(5) * 100
    features['mom_10_2m'] = close.pct_change(10) * 100

    # Candle features
    features['body_2m'] = close - opn
    features['body_pct_2m'] = (close - opn) / opn.replace(0, 1e-10) * 100
    features['upper_wick_2m'] = high - close.where(close > opn, opn)
    features['lower_wick_2m'] = close.where(close < opn, opn) - low
    features['range_2m'] = high - low

    # Volatility
    features['std_5_2m'] = close.rolling(5).std()
    features['std_10_2m'] = close.rolling(10).std()

    # Moving averages
    features['sma_5_2m'] = close.rolling(5).mean()
    features['sma_10_2m'] = close.rolling(10).mean()
    features['sma_20_2m'] = close.rolling(20).mean()
    features['close_vs_sma5_2m'] = close - features['sma_5_2m']
    features['close_vs_sma10_2m'] = close - features['sma_10_2m']
    features['sma5_vs_sma10_2m'] = features['sma_5_2m'] - features['sma_10_2m']

    # High/Low channels
    features['high_5_2m'] = high.rolling(5).max()
    features['low_5_2m'] = low.rolling(5).min()
    features['close_vs_high5_2m'] = close - features['high_5_2m']
    features['close_vs_low5_2m'] = close - features['low_5_2m']

    return features


# ─────────── Label Generation ───────────

def generate_labels(df, lookahead=3, threshold=8.0):
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
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)

    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None

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

        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                trade = _make_trade(pos, prev_close, df.iloc[i-1]['Time'], "DAY_END", current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                pos = None
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0, 'lots': current_lots}

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

        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF", current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue

        in_window = ENTRY_START <= t <= ENTRY_END

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

        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl

        pred = predictions[i]

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
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (peak - cumulative).max()
    trading_days = sum(1 for d in daily_results.values() if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)

    print(f"\n{'='*60}")
    print(f"  🤖 PURE 2-MIN CATBOOST STRATEGY — {label}")
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

def main():
    global ATR_KEY_VALUE, MIN_ATR, LOOKAHEAD, THRESHOLD, ENTRY_START, ENTRY_END, SQUARE_OFF

    parser = argparse.ArgumentParser(description="Pure 2-Min CatBoost ML Strategy")
    parser.add_argument("--file-2m", default="nifty_2min_data.csv", help="2-min data CSV file")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--window-start", type=str, default="09:20")
    parser.add_argument("--window-end", type=str, default="15:15")
    parser.add_argument("--square-off", type=str, default="15:24")
    parser.add_argument("--test-from", type=str, default="2025-01-01")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    LOOKAHEAD = args.lookahead
    THRESHOLD = args.threshold

    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    ENTRY_START = parse_time(args.window_start)
    ENTRY_END = parse_time(args.window_end)
    SQUARE_OFF = parse_time(args.square_off)
    test_from_date = datetime.strptime(args.test_from, "%Y-%m-%d").date()

    print(f"Loading 2-min data: {args.file_2m}")
    try:
        df_2m = pd.read_csv(args.file_2m)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file_2m}")
        return

    df_2m['Time'] = pd.to_datetime(df_2m['Time'])
    df_2m = df_2m.sort_values('Time').reset_index(drop=True)
    
    total_candles = len(df_2m)
    total_days = df_2m['Time'].dt.date.nunique()
    print(f"2-min: {total_candles:,} candles | {total_days} days")

    # ── Build features ──
    print(f"\n📊 Building 2-min features natively...")
    features = build_features_2m(df_2m)
    
    # ── Generate labels ──
    print(f"\n🏷️  Generating labels (lookahead={LOOKAHEAD}, threshold={THRESHOLD})...")
    labels = generate_labels(df_2m, LOOKAHEAD, THRESHOLD)
    
    times = df_2m['Time'].dt.time
    window_mask = (times >= ENTRY_START) & (times <= ENTRY_END)

    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    dates = df_2m['Time'].dt.date
    train_mask = (dates < test_from_date) & window_mask
    test_mask = (dates >= test_from_date) & window_mask

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    print(f"\n🧠 Training Pure 2-Minute CatBoost model...")
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

    test_pred = model.predict(X_test).flatten().astype(int)
    full_predictions = np.zeros(len(df_2m), dtype=int)
    full_predictions[test_mask] = test_pred

    atr_vals = calc_atr(df_2m, ATR_PERIOD).values

    print(f"\n🚀 Running TRUTHFUL backtest on TEST data...")
    test_df_mask = dates >= test_from_date
    df_test = df_2m[test_df_mask].reset_index(drop=True)
    pred_test_full = full_predictions[test_df_mask]
    atr_test = atr_vals[test_df_mask]

    all_trades, daily_results = backtest_predictions(df_test, pred_test_full, atr_test)
    print_results(all_trades, daily_results, "PURE 2-MIN (REALITY CHECK)")

    model.save_model("catboost_nifty_model.cbm")
    print(f"\n💾 Look-Ahead-Free Model saved: catboost_nifty_model.cbm")

if __name__ == "__main__":
    main()
