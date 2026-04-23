"""
rf_strategy.py — Random Forest + UT Bot + StochRSI Strategy (NIFTY 2-MIN ONLY)

STRATEGY:
  - Timeframe: 2-minute candles ONLY
  - ATR gate: >= 12 (configurable via --min-atr)
  - Phase 1 (09:15-13:00): OBSERVATION — analyze market direction, no trades
  - Phase 2 (13:00-15:12): ACTIVE TRADING
      LONG:  RF=BULL + StochRSI K < 10 + UT Bot bullish + ATR >= 12
      SHORT: RF=BEAR + StochRSI K > 90 + UT Bot bearish + ATR >= 12
  - Test period: March + April 2026 (2 months)

Usage:
    python rf_strategy.py                                # Default: test Mar+Apr 2026
    python rf_strategy.py --min-atr 15                   # Change ATR gate
    python rf_strategy.py --stoch-buy 15 --stoch-sell 85 # Adjust StochRSI thresholds
    python rf_strategy.py --test-from 2026-03-01         # Custom test start
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 12.0                          # ATR gate (configurable)

OBSERVATION_END = dt_time(13, 0)        # Analyze until 1:00 PM
ENTRY_START = dt_time(13, 0)            # Trading starts 1:00 PM
ENTRY_END = dt_time(15, 12)             # No new trades after 3:12 PM
SQUARE_OFF = dt_time(15, 20)            # Square off 3:20 PM

STOCH_BUY_THRESHOLD = 10
STOCH_SELL_THRESHOLD = 90
STOCH_RSI_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3

REGIME_FORWARD_CANDLES = 30
REGIME_SLOPE_THRESHOLD = 0.02

LOT_SIZE = 65
BASE_LOTS = 2

HIGH_POINTS_THRESHOLD = 30
MID_POINTS_THRESHOLD = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════

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


def calc_stochastic_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calc_rsi(close, rsi_period)
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    denom = (rsi_high - rsi_low).replace(0, 1e-10)
    stoch_rsi = (rsi - rsi_low) / denom * 100
    stoch_k = stoch_rsi.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()
    return stoch_k, stoch_d


def calc_ut_bot_direction(close_arr, atr_arr, key_value=1.0):
    if hasattr(close_arr, 'values'):
        close_arr = close_arr.values
    if hasattr(atr_arr, 'values'):
        atr_arr = atr_arr.values
    n = len(close_arr)
    trail_stop = np.zeros(n)
    direction = np.zeros(n)
    trail_stop[0] = close_arr[0]
    direction[0] = 1

    for i in range(1, n):
        nloss = atr_arr[i] * key_value
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]

        if prev_dir == 1:
            new_ts = close_arr[i] - nloss
            trail_stop[i] = max(new_ts, prev_ts)
            if close_arr[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close_arr[i] + nloss
            else:
                direction[i] = 1
        else:
            new_ts = close_arr[i] + nloss
            trail_stop[i] = min(new_ts, prev_ts)
            if close_arr[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close_arr[i] - nloss
            else:
                direction[i] = -1

    return trail_stop, direction


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════
#  FEATURE ENGINEERING (2-MIN ONLY)
# ═══════════════════════════════════════════

def build_features(df):
    """Build features from 2-min candle data."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    stoch_k, stoch_d = calc_stochastic_rsi(close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
                                            STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    trail, dirn = calc_ut_bot_direction(close, atr, ATR_KEY_VALUE)

    ema_5 = calc_ema(close, 5)
    ema_10 = calc_ema(close, 10)
    ema_20 = calc_ema(close, 20)
    ema_50 = calc_ema(close, 50)

    features = pd.DataFrame(index=df.index)

    # Core indicators
    features['atr'] = atr
    features['rsi'] = rsi
    features['stoch_k'] = stoch_k
    features['stoch_d'] = stoch_d
    features['stoch_kd_diff'] = stoch_k - stoch_d
    features['ut_dir'] = dirn
    features['close_vs_trail'] = close.values - trail

    # UT Bot signals
    ut_change = np.diff(dirn, prepend=dirn[0])
    features['ut_change'] = ut_change
    features['ut_buy_signal'] = (ut_change > 0).astype(int)
    features['ut_sell_signal'] = (ut_change < 0).astype(int)

    # StochRSI zones
    features['stoch_oversold'] = (stoch_k < STOCH_BUY_THRESHOLD).astype(int)
    features['stoch_overbought'] = (stoch_k > STOCH_SELL_THRESHOLD).astype(int)
    features['stoch_k_cross_up'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    features['stoch_k_cross_dn'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # Price momentum
    features['mom_3'] = close.pct_change(3) * 100
    features['mom_5'] = close.pct_change(5) * 100
    features['mom_10'] = close.pct_change(10) * 100
    features['mom_20'] = close.pct_change(20) * 100

    # Candle patterns
    features['body'] = close - opn
    features['body_pct'] = (close - opn) / opn * 100
    features['upper_wick'] = high - close.where(close > opn, opn)
    features['lower_wick'] = close.where(close < opn, opn) - low
    features['range'] = high - low
    features['body_vs_range'] = (close - opn).abs() / (high - low).replace(0, 1e-10)

    # Volatility
    features['std_5'] = close.rolling(5).std()
    features['std_10'] = close.rolling(10).std()
    features['std_20'] = close.rolling(20).std()

    # EMA relationships
    features['close_vs_ema5'] = close - ema_5
    features['close_vs_ema10'] = close - ema_10
    features['close_vs_ema20'] = close - ema_20
    features['close_vs_ema50'] = close - ema_50
    features['ema5_vs_ema10'] = ema_5 - ema_10
    features['ema10_vs_ema20'] = ema_10 - ema_20
    features['ema20_vs_ema50'] = ema_20 - ema_50
    features['ema20_slope'] = ema_20.diff(3) / 3

    # High/Low channels
    features['close_vs_high5'] = close - high.rolling(5).max()
    features['close_vs_low5'] = close - low.rolling(5).min()
    features['close_vs_high10'] = close - high.rolling(10).max()
    features['close_vs_low10'] = close - low.rolling(10).min()

    # Morning session patterns
    bullish_candle = (close > opn).astype(int)
    features['bullish_count_10'] = bullish_candle.rolling(10).sum()
    features['bullish_count_20'] = bullish_candle.rolling(20).sum()
    features['bullish_ratio_10'] = features['bullish_count_10'] / 10
    features['bullish_ratio_20'] = features['bullish_count_20'] / 20

    # Day return (filled later)
    features['day_return'] = 0.0

    return features


# ═══════════════════════════════════════════
#  REGIME LABEL GENERATION
# ═══════════════════════════════════════════

def generate_regime_labels(df, forward_candles=30, slope_threshold=0.02):
    close = df['Close'].astype(float)
    ema_20 = calc_ema(close, 20)
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - forward_candles):
        current_ema = ema_20.iloc[i]
        future_ema = ema_20.iloc[i + forward_candles]
        if current_ema == 0:
            continue
        slope = (future_ema - current_ema) / current_ema * 100
        if slope > slope_threshold:
            labels[i] = 1
        elif slope < -slope_threshold:
            labels[i] = -1

    return labels


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=500):
    y_train_m = y_train + 1
    y_test_m = y_test + 1

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    print(f"\n  Training Random Forest ({n_estimators} trees)...")
    model.fit(X_train, y_train_m)

    train_pred = model.predict(X_train) - 1
    test_pred = model.predict(X_test) - 1

    train_acc = accuracy_score(y_train, train_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    print(f"  Train acc: {train_acc:.1f}% | Test acc: {test_acc:.1f}%")
    print(classification_report(y_test + 1, test_pred + 1,
                                target_names=['BEAR', 'CHOP', 'BULL'], zero_division=0))
    return model


# ═══════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════

def backtest_strategy(df, regime_predictions, stoch_k_vals, ut_dir_vals, atr_vals,
                      stoch_buy_thresh=10, stoch_sell_thresh=90, min_atr=12.0):
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

        # Day boundary
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(close.iloc[i-1])
                trade = _make_trade(pos, prev_close, df.iloc[i-1]['Time'], "DAY_END",
                                    current_lots, pos.get('initial_sl'))
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

        # Square off
        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue

        in_window = ENTRY_START <= t <= ENTRY_END

        # SL check
        if pos:
            if pos['dir'] == "LONG" and l <= pos['sl']:
                trade = _make_trade(pos, pos['sl'], df.iloc[i]['Time'], "TRAIL_SL",
                                    current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None
            elif pos['dir'] == "SHORT" and h >= pos['sl']:
                trade = _make_trade(pos, pos['sl'], df.iloc[i]['Time'], "TRAIL_SL",
                                    current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, curr_date, trade)
                pos = None

        # Trailing SL update
        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl

        regime = regime_predictions[i]
        sk = stoch_k_vals[i] if not np.isnan(stoch_k_vals[i]) else 50
        ut_d = ut_dir_vals[i]

        # Regime flip → close
        if regime == 1 and pos and pos['dir'] == "SHORT":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "REGIME_FLIP",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        elif regime == -1 and pos and pos['dir'] == "LONG":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "REGIME_FLIP",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None

        # New entry: ATR >= min_atr required
        if not pos and in_window and curr_atr >= min_atr:
            if regime == 1 and sk < stoch_buy_thresh and ut_d == 1:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
            elif regime == -1 and sk > stoch_sell_thresh and ut_d == -1:
                sl = c + curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'SHORT', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}

    return all_trades, daily_results


def _pnl(pos, exit_price):
    return (exit_price - pos['entry']) if pos['dir'] == "LONG" else (pos['entry'] - exit_price)


def _make_trade(pos, exit_price, exit_time, reason, lots=2, sl=None):
    raw_pnl = _pnl(pos, exit_price)
    lot_multiplier = lots // BASE_LOTS
    adj_pnl = raw_pnl * lot_multiplier
    trade = {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': round(exit_price, 2),
        'entry_time': pos['entry_time'], 'exit_time': exit_time,
        'pnl': round(adj_pnl, 2), 'raw_pnl': round(raw_pnl, 2),
        'lots': lots, 'multiplier': lot_multiplier, 'qty': lots * LOT_SIZE,
        'pnl_pct': round(raw_pnl / pos['entry'] * 100, 4), 'reason': reason,
    }
    if sl is not None:
        trade['sl'] = round(sl, 2)
    return trade


def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0, 'lots': trade.get('lots', BASE_LOTS)}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ═══════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════

def categorize_trades(trades):
    high_pts = [t for t in trades if abs(t['raw_pnl']) >= HIGH_POINTS_THRESHOLD]
    mid_pts = [t for t in trades if MID_POINTS_THRESHOLD <= abs(t['raw_pnl']) < HIGH_POINTS_THRESHOLD]
    low_pts = [t for t in trades if abs(t['raw_pnl']) < MID_POINTS_THRESHOLD]
    return high_pts, mid_pts, low_pts


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

    print(f"\n{'='*70}")
    print(f"  🌲 RANDOM FOREST [2-MIN] — {label}")
    print(f"{'='*70}")
    print(f"  Total trades:      {n}")
    print(f"  Win rate:          {wr:.1f}%")
    print(f"  Total P&L:         {total_pnl:+.2f} pts")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Avg win:           {avg_win:+.2f} pts")
    print(f"  Avg loss:          {avg_loss:+.2f} pts")
    print(f"  Max drawdown:      {max_dd:.2f} pts")
    print(f"  Trading days:      {trading_days}")
    print(f"  Win days:          {win_days} ({win_days/max(trading_days,1)*100:.0f}%)")
    print(f"  Avg P&L/day:       {total_pnl/max(trading_days,1):+.2f} pts")

    # Categories
    high_pts, mid_pts, low_pts = categorize_trades(all_trades)
    print(f"\n  📊 TRADE CATEGORIES:")
    for cat_name, cat_trades, thresh in [
        (f"HIGH (>={HIGH_POINTS_THRESHOLD}pts)", high_pts, None),
        (f"MID ({MID_POINTS_THRESHOLD}-{HIGH_POINTS_THRESHOLD}pts)", mid_pts, None),
        (f"LOW (<{MID_POINTS_THRESHOLD}pts)", low_pts, None),
    ]:
        if not cat_trades:
            print(f"    {cat_name}: 0 trades")
            continue
        cn = len(cat_trades)
        cw = sum(1 for t in cat_trades if t['pnl'] > 0)
        cp = sum(t['pnl'] for t in cat_trades)
        print(f"    {cat_name}: {cn} trades | Win: {cw}/{cn} ({cw/cn*100:.0f}%) | P&L: {cp:+.2f}")

    # Daily
    sorted_days = sorted(daily_results.keys())
    print(f"\n  {'Date':>12} {'Lots':>5} {'Trades':>7} {'P&L':>10} {'Status':>8}")
    print(f"  {'-'*45}")
    cum = 0
    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        lots = daily_results[day].get('lots', BASE_LOTS)
        if not trades:
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

    print(f"\n{'='*70}")


def print_detailed_log(all_trades, daily_results, log_file="rf_detailed_log.txt"):
    sorted_days = sorted(daily_results.keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*110}")
    out(f"  📋 DETAILED TRADE LOG — 2-MIN RANDOM FOREST")
    out(f"{'='*110}")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    cum_pnl = 0

    for day in sorted_days:
        trades = daily_results[day]['trades']
        day_pnl = daily_results[day]['pnl']
        lots = daily_results[day].get('lots', BASE_LOTS)
        if not trades:
            continue

        cum_pnl += day_pnl
        dn = day_names[day.weekday()]
        w = sum(1 for t in trades if t['pnl'] > 0)
        l = len(trades) - w
        wr = w / len(trades) * 100
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"

        out(f"\n  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        out(f"  │ 📅 {day} ({dn}) | Lots: {lots} | Trades: {len(trades)} (W:{w} L:{l} {wr:.0f}%) | P&L: {day_pnl:+.2f} {icon} | Cum: {cum_pnl:+.2f}")
        out(f"  ├───┬──────┬───────────┬───────────┬───────────┬───────────┬──────────┬─────────────┤")
        out(f"  │ # │ Dir  │ Entry     │ Exit      │ Entry Pr  │ Exit Pr   │ P&L      │ Reason      │")
        out(f"  ├───┼──────┼───────────┼───────────┼───────────┼───────────┼──────────┼─────────────┤")

        for j, t in enumerate(trades, 1):
            et = t['entry_time'].strftime('%H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[-8:-3]
            xt = t['exit_time'].strftime('%H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[-8:-3]
            di = "🟢" if t['dir'] == 'LONG' else "🔴"
            pi = "✅" if t['pnl'] > 0 else "❌" if t['pnl'] < 0 else "➖"
            out(f"  │{j:>2} │ {di}{t['dir']:>4} │ {et:>9} │ {xt:>9} │ {t['entry']:>9.2f} │ {t['exit']:>9.2f} │ {t['pnl']:>+7.2f}{pi}│ {t['reason']:<12}│")

        out(f"  └───┴──────┴───────────┴───────────┴───────────┴───────────┴──────────┴─────────────┘")

    out(f"\n{'='*110}")
    out(f"  GRAND TOTAL: {cum_pnl:+.2f} pts")
    out(f"{'='*110}")

    log_path = os.path.join(SCRIPT_DIR, log_file)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n💾 Log saved: {log_path}")


# ═══════════════════════════════════════════
#  EXPORT FOR FRONTEND
# ═══════════════════════════════════════════

def export_frontend_data(df_test, all_trades, stoch_k_vals, stoch_d_vals,
                          atr_vals, ut_dir_vals, regime_preds):
    """Export candle data + trade markers as JSON for frontend chart."""

    # Candle data
    candles = []
    for i, row in df_test.iterrows():
        candles.append({
            'time': int(row['Time'].timestamp()),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
        })

    # Trade markers
    markers = []
    for t in all_trades:
        entry_ts = int(t['entry_time'].timestamp()) if hasattr(t['entry_time'], 'timestamp') else 0
        exit_ts = int(t['exit_time'].timestamp()) if hasattr(t['exit_time'], 'timestamp') else 0

        markers.append({
            'type': 'entry',
            'time': entry_ts,
            'price': t['entry'],
            'dir': t['dir'],
            'pnl': t['pnl'],
            'reason': t['reason'],
        })
        markers.append({
            'type': 'exit',
            'time': exit_ts,
            'price': t['exit'],
            'dir': t['dir'],
            'pnl': t['pnl'],
            'reason': t['reason'],
        })

    # Indicator lines (sampled to reduce JSON size)
    stoch_data = []
    atr_data = []
    for i, row in df_test.iterrows():
        ts = int(row['Time'].timestamp())
        sk = float(stoch_k_vals[i]) if not np.isnan(stoch_k_vals[i]) else None
        sd = float(stoch_d_vals[i]) if not np.isnan(stoch_d_vals[i]) else None
        stoch_data.append({'time': ts, 'k': sk, 'd': sd})
        atr_data.append({'time': ts, 'value': round(float(atr_vals[i]), 2)})

    data = {
        'candles': candles,
        'markers': markers,
        'stochRSI': stoch_data,
        'atr': atr_data,
        'config': {
            'stochBuy': STOCH_BUY_THRESHOLD,
            'stochSell': STOCH_SELL_THRESHOLD,
            'minATR': MIN_ATR,
            'entryStart': str(ENTRY_START),
            'entryEnd': str(ENTRY_END),
            'squareOff': str(SQUARE_OFF),
        },
        'summary': {
            'total_trades': len(all_trades),
            'wins': sum(1 for t in all_trades if t['pnl'] > 0),
            'losses': sum(1 for t in all_trades if t['pnl'] <= 0),
            'total_pnl': round(sum(t['pnl'] for t in all_trades), 2),
            'win_rate': round(sum(1 for t in all_trades if t['pnl'] > 0) / max(len(all_trades), 1) * 100, 1),
        },
        'trades': [{
            'dir': t['dir'],
            'entry': t['entry'],
            'exit': t['exit'],
            'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
            'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
            'pnl': t['pnl'],
            'raw_pnl': t['raw_pnl'],
            'reason': t['reason'],
        } for t in all_trades],
    }

    out_path = os.path.join(SCRIPT_DIR, "frontend_data.json")
    with open(out_path, 'w') as f:
        json.dump(data, f)
    print(f"💾 Frontend data exported: {out_path} ({len(candles)} candles, {len(markers)} markers)")
    return out_path


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    global ATR_KEY_VALUE, MIN_ATR, ENTRY_START, ENTRY_END, SQUARE_OFF
    global STOCH_BUY_THRESHOLD, STOCH_SELL_THRESHOLD, OBSERVATION_END

    parser = argparse.ArgumentParser(description="RF + StochRSI + UT Bot (2-MIN ONLY)")
    parser.add_argument("--file", default=os.path.join(SCRIPT_DIR, "nifty_2min_data.csv"))
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help="Min ATR to enter (default: 12)")
    parser.add_argument("--stoch-buy", type=float, default=STOCH_BUY_THRESHOLD)
    parser.add_argument("--stoch-sell", type=float, default=STOCH_SELL_THRESHOLD)
    parser.add_argument("--regime-thresh", type=float, default=REGIME_SLOPE_THRESHOLD)
    parser.add_argument("--regime-forward", type=int, default=REGIME_FORWARD_CANDLES)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--obs-end", type=str, default="13:00", help="Observation end (default: 13:00)")
    parser.add_argument("--window-end", type=str, default="15:12", help="Entry window end (default: 15:12)")
    parser.add_argument("--square-off", type=str, default="15:20", help="Square off (default: 15:20)")
    parser.add_argument("--test-from", type=str, default="2026-03-01", help="Test start (default: 2026-03-01)")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    STOCH_BUY_THRESHOLD = args.stoch_buy
    STOCH_SELL_THRESHOLD = args.stoch_sell

    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    OBSERVATION_END = parse_time(args.obs_end)
    ENTRY_START = OBSERVATION_END
    ENTRY_END = parse_time(args.window_end)
    SQUARE_OFF = parse_time(args.square_off)

    test_from_date = datetime.strptime(args.test_from, "%Y-%m-%d").date()

    print(f"{'='*70}")
    print(f"  🌲 RANDOM FOREST + STOCHRSI + UT BOT [2-MIN ONLY]")
    print(f"{'='*70}")
    print(f"  Observation: 09:15 - {args.obs_end}")
    print(f"  Trading:     {args.obs_end} - {args.window_end}")
    print(f"  Square off:  {args.square_off}")
    print(f"  ATR gate:    >= {MIN_ATR}")
    print(f"  StochRSI:    BUY < {STOCH_BUY_THRESHOLD} | SELL > {STOCH_SELL_THRESHOLD}")
    print(f"  Test from:   {args.test_from} (March + April 2026)")

    # Load data
    print(f"\n📊 Loading: {args.file}")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        return

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    total_candles = len(df)
    total_days = df['Time'].dt.date.nunique()
    print(f"  {total_candles:,} candles | {total_days} days")
    print(f"  Range: {df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}")

    # Build features
    print(f"\n📊 Building features...")
    features = build_features(df)
    print(f"  Features: {features.shape[1]} columns")

    # Fill day_return
    close = df['Close'].astype(float)
    day_groups = df.groupby(df['Time'].dt.date)
    day_returns = pd.Series(0.0, index=df.index)
    for day, group in day_groups:
        day_open = float(group.iloc[0]['Open'])
        if day_open > 0:
            day_returns.loc[group.index] = (close.loc[group.index] - day_open) / day_open * 100
    features['day_return'] = day_returns.values

    # Labels
    print(f"  Generating regime labels...")
    labels = generate_regime_labels(df, args.regime_forward, args.regime_thresh)
    bull = (labels == 1).sum()
    bear = (labels == -1).sum()
    chop = (labels == 0).sum()
    print(f"  BULL={bull} ({bull/len(labels)*100:.1f}%) | "
          f"BEAR={bear} ({bear/len(labels)*100:.1f}%) | "
          f"CHOP={chop} ({chop/len(labels)*100:.1f}%)")

    # Train/test split
    times = df['Time'].dt.time
    window_mask = (times >= dt_time(9, 20)) & (times <= ENTRY_END)
    dates = df['Time'].dt.date

    train_mask = (dates < test_from_date) & window_mask
    test_mask = (dates >= test_from_date) & window_mask

    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    X_train = features[train_mask].values
    y_train = labels[train_mask]
    X_test = features[test_mask].values
    y_test = labels[test_mask]

    train_days = dates[dates < test_from_date].nunique()
    test_days = dates[dates >= test_from_date].nunique()

    print(f"\n  Train: {len(X_train)} samples ({train_days} days) [up to {args.test_from}]")
    print(f"  Test:  {len(X_test)} samples ({test_days} days) [{args.test_from} onwards]")

    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ Insufficient data!")
        return

    # Train
    rf_model = train_random_forest(X_train, y_train, X_test, y_test, args.rf_trees)

    # Feature importance
    importance = rf_model.feature_importances_
    feat_names = features.columns.tolist()
    top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
    print(f"\n  📊 TOP 10 FEATURES:")
    for name, imp in top_features:
        print(f"    {name:>25}: {imp:.4f}")

    # Predictions
    print(f"\n  🔮 Generating predictions...")
    rf_full_pred = np.zeros(len(df), dtype=int)
    rf_test_pred = rf_model.predict(X_test) - 1
    rf_full_pred[test_mask] = rf_test_pred

    # Indicators
    atr_vals = calc_atr(df, ATR_PERIOD).values
    stoch_k_vals, stoch_d_vals = calc_stochastic_rsi(
        close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH
    )
    stoch_k_vals = stoch_k_vals.values
    stoch_d_vals = stoch_d_vals.values
    _, ut_dir_vals = calc_ut_bot_direction(close, calc_atr(df, ATR_PERIOD), ATR_KEY_VALUE)

    # Backtest
    print(f"\n  🚀 Running backtest (test period)...")
    test_df_mask = dates >= test_from_date
    df_test = df[test_df_mask].reset_index(drop=True)
    pred_test = rf_full_pred[test_df_mask]
    atr_test = atr_vals[test_df_mask]
    stoch_k_test = stoch_k_vals[test_df_mask]
    stoch_d_test = stoch_d_vals[test_df_mask]
    ut_dir_test = ut_dir_vals[test_df_mask]

    all_trades, daily_results = backtest_strategy(
        df_test, pred_test, stoch_k_test, ut_dir_test, atr_test,
        stoch_buy_thresh=STOCH_BUY_THRESHOLD,
        stoch_sell_thresh=STOCH_SELL_THRESHOLD,
        min_atr=MIN_ATR,
    )

    test_start = df_test['Time'].dt.date.min()
    test_end = df_test['Time'].dt.date.max()
    print_results(all_trades, daily_results, f"TEST ({test_start} → {test_end})")
    print_detailed_log(all_trades, daily_results)

    # Export for frontend
    export_frontend_data(df_test, all_trades, stoch_k_test, stoch_d_test,
                          atr_test, ut_dir_test, pred_test)

    # Save model
    model_path = os.path.join(SCRIPT_DIR, "rf_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"💾 Model saved: {model_path}")

    # Save trades CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        tp = os.path.join(SCRIPT_DIR, "rf_trades.csv")
        trades_df.to_csv(tp, index=False)
        print(f"💾 Trades CSV: {tp}")

    # Metadata
    metadata = {
        "strategy": "Random Forest + StochRSI + UT Bot (2-MIN)",
        "min_atr": MIN_ATR,
        "stoch_buy": STOCH_BUY_THRESHOLD,
        "stoch_sell": STOCH_SELL_THRESHOLD,
        "observation_end": args.obs_end,
        "entry_end": args.window_end,
        "test_from": args.test_from,
        "rf_trees": args.rf_trees,
        "total_trades": len(all_trades),
        "total_pnl": round(sum(t['pnl'] for t in all_trades), 2) if all_trades else 0,
        "last_trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = os.path.join(SCRIPT_DIR, "model_metadata.json")
    with open(mp, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Metadata: {mp}")


if __name__ == "__main__":
    main()
