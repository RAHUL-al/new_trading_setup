"""
rf_strategy.py — Random Forest + UT Bot + StochRSI Strategy (NIFTY)

ARCHITECTURE:
  Two SEPARATE models trained and evaluated INDEPENDENTLY:
    Model A: Trained on 2-minute candle data
    Model B: Trained on 3-minute candle data

  Each model gets its OWN:
    - Feature set (UT Bot + StochRSI + ATR + momentum + patterns)
    - Random Forest classifier
    - Backtest results
    - Trade categorization (HIGH / MID / LOW points)

  Results are printed side-by-side for comparison.

STRATEGY LOGIC (per timeframe):
  Phase 1 (09:15-11:00): OBSERVATION — analyze market direction
    → RF builds regime confidence from morning price action patterns
    → Determine if market is BULLISH or BEARISH

  Phase 2 (11:00-15:15): ACTIVE TRADING
    LONG:  RF=BULL + StochRSI K < 10 (oversold) + UT Bot bullish
    SHORT: RF=BEAR + StochRSI K > 90 (overbought) + UT Bot bearish

  Regime Change: If RF flips direction → close + reverse

ZERO LOOK-AHEAD:
  Regime labels use forward EMА slope — only for training.
  At inference: model uses current/past data only.

Usage:
    python rf_strategy.py                                    # Both timeframes
    python rf_strategy.py --test-from 2026-04-01             # April test
    python rf_strategy.py --stoch-buy 15 --stoch-sell 85     # Adjust thresholds
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
MIN_ATR = 6.5

OBSERVATION_END = dt_time(11, 0)
ENTRY_START = dt_time(11, 0)
ENTRY_END = dt_time(15, 15)
SQUARE_OFF = dt_time(15, 24)

STOCH_BUY_THRESHOLD = 10
STOCH_SELL_THRESHOLD = 90
STOCH_RSI_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3

REGIME_FORWARD_CANDLES = 30
REGIME_SLOPE_THRESHOLD = 0.02

LOT_SIZE = 65
BASE_LOTS = 2

# Trade categorization thresholds (in points)
HIGH_POINTS_THRESHOLD = 30   # > 30 pts = HIGH
MID_POINTS_THRESHOLD = 10    # 10-30 pts = MID
                              # < 10 pts = LOW

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
#  FEATURE ENGINEERING (SINGLE TIMEFRAME)
# ═══════════════════════════════════════════

def build_features(df, tf_label=""):
    """Build features from a single timeframe dataframe.
    tf_label: suffix like '2m' or '3m' for column naming.
    """
    sfx = f"_{tf_label}" if tf_label else ""
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

    # ── Core Indicators ──
    features[f'atr{sfx}'] = atr
    features[f'rsi{sfx}'] = rsi
    features[f'stoch_k{sfx}'] = stoch_k
    features[f'stoch_d{sfx}'] = stoch_d
    features[f'stoch_kd_diff{sfx}'] = stoch_k - stoch_d
    features[f'ut_dir{sfx}'] = dirn
    features[f'close_vs_trail{sfx}'] = close.values - trail

    # ── UT Bot change detection ──
    ut_change = np.diff(dirn, prepend=dirn[0])
    features[f'ut_change{sfx}'] = ut_change
    features[f'ut_buy_signal{sfx}'] = (ut_change > 0).astype(int)
    features[f'ut_sell_signal{sfx}'] = (ut_change < 0).astype(int)

    # ── StochRSI zone flags ──
    features[f'stoch_oversold{sfx}'] = (stoch_k < STOCH_BUY_THRESHOLD).astype(int)
    features[f'stoch_overbought{sfx}'] = (stoch_k > STOCH_SELL_THRESHOLD).astype(int)
    features[f'stoch_k_cross_up_d{sfx}'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    features[f'stoch_k_cross_dn_d{sfx}'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # ── Price Momentum ──
    features[f'mom_3{sfx}'] = close.pct_change(3) * 100
    features[f'mom_5{sfx}'] = close.pct_change(5) * 100
    features[f'mom_10{sfx}'] = close.pct_change(10) * 100
    features[f'mom_20{sfx}'] = close.pct_change(20) * 100

    # ── Candle Patterns ──
    features[f'body{sfx}'] = close - opn
    features[f'body_pct{sfx}'] = (close - opn) / opn * 100
    features[f'upper_wick{sfx}'] = high - close.where(close > opn, opn)
    features[f'lower_wick{sfx}'] = close.where(close < opn, opn) - low
    features[f'range{sfx}'] = high - low
    features[f'body_vs_range{sfx}'] = (close - opn).abs() / (high - low).replace(0, 1e-10)

    # ── Volatility ──
    features[f'std_5{sfx}'] = close.rolling(5).std()
    features[f'std_10{sfx}'] = close.rolling(10).std()
    features[f'std_20{sfx}'] = close.rolling(20).std()

    # ── EMA relationships ──
    features[f'close_vs_ema5{sfx}'] = close - ema_5
    features[f'close_vs_ema10{sfx}'] = close - ema_10
    features[f'close_vs_ema20{sfx}'] = close - ema_20
    features[f'close_vs_ema50{sfx}'] = close - ema_50
    features[f'ema5_vs_ema10{sfx}'] = ema_5 - ema_10
    features[f'ema10_vs_ema20{sfx}'] = ema_10 - ema_20
    features[f'ema20_vs_ema50{sfx}'] = ema_20 - ema_50
    features[f'ema20_slope{sfx}'] = ema_20.diff(3) / 3

    # ── High/Low channels ──
    features[f'close_vs_high5{sfx}'] = close - high.rolling(5).max()
    features[f'close_vs_low5{sfx}'] = close - low.rolling(5).min()
    features[f'close_vs_high10{sfx}'] = close - high.rolling(10).max()
    features[f'close_vs_low10{sfx}'] = close - low.rolling(10).min()

    # ── Morning session pattern (for regime detection before 11:00) ──
    # Count of bullish/bearish candles in last N bars
    bullish_candle = (close > opn).astype(int)
    features[f'bullish_count_10{sfx}'] = bullish_candle.rolling(10).sum()
    features[f'bullish_count_20{sfx}'] = bullish_candle.rolling(20).sum()
    features[f'bullish_ratio_10{sfx}'] = features[f'bullish_count_10{sfx}'] / 10
    features[f'bullish_ratio_20{sfx}'] = features[f'bullish_count_20{sfx}'] / 20

    # ── Cumulative return from day start (intraday bias) ──
    # This requires knowing the day's open — compute per day later in main
    features[f'day_return{sfx}'] = 0.0  # placeholder, filled in main

    return features


# ═══════════════════════════════════════════
#  REGIME LABEL GENERATION
# ═══════════════════════════════════════════

def generate_regime_labels(df, forward_candles=30, slope_threshold=0.02):
    """
    Regime labels based on forward EMA20 slope.
    BULL (+1) / BEAR (-1) / CHOP (0)
    Only used for training — no look-ahead at inference.
    """
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
            labels[i] = 1    # BULL
        elif slope < -slope_threshold:
            labels[i] = -1   # BEAR
        else:
            labels[i] = 0    # CHOP

    return labels


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=500, label=""):
    y_train_mapped = y_train + 1
    y_test_mapped = y_test + 1

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

    print(f"\n  Training Random Forest ({n_estimators} trees) [{label}]...")
    model.fit(X_train, y_train_mapped)

    train_pred = model.predict(X_train) - 1
    test_pred = model.predict(X_test) - 1

    train_acc = accuracy_score(y_train, train_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    print(f"  Train acc: {train_acc:.1f}% | Test acc: {test_acc:.1f}%")

    target_names = ['BEAR', 'CHOP', 'BULL']
    print(classification_report(y_test + 1, test_pred + 1,
                                target_names=target_names, zero_division=0))

    return model


# ═══════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════

def backtest_strategy(df, regime_predictions, stoch_k_vals, ut_dir_vals, atr_vals,
                      stoch_buy_thresh=10, stoch_sell_thresh=90):
    """
    Backtest with:
      - No trades before ENTRY_START (11:00)
      - LONG:  regime=BULL + StochRSI < buy_thresh + UT_Bot bullish
      - SHORT: regime=BEAR + StochRSI > sell_thresh + UT_Bot bearish
      - Regime flip → close position
      - Trailing SL + Square off
    """
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

        # Current indicators
        regime = regime_predictions[i]
        sk = stoch_k_vals[i] if not np.isnan(stoch_k_vals[i]) else 50
        ut_d = ut_dir_vals[i]

        # Regime change → close existing
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

        # New entry
        if not pos and in_window and curr_atr >= MIN_ATR:
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
    if pos['dir'] == "LONG":
        return exit_price - pos['entry']
    else:
        return pos['entry'] - exit_price


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
    """Categorize trades into HIGH / MID / LOW point buckets."""
    high_pts = [t for t in trades if abs(t['raw_pnl']) >= HIGH_POINTS_THRESHOLD]
    mid_pts = [t for t in trades if MID_POINTS_THRESHOLD <= abs(t['raw_pnl']) < HIGH_POINTS_THRESHOLD]
    low_pts = [t for t in trades if abs(t['raw_pnl']) < MID_POINTS_THRESHOLD]
    return high_pts, mid_pts, low_pts


def print_category_stats(trades, category_name):
    """Print stats for a trade category."""
    if not trades:
        print(f"    {category_name}: No trades")
        return

    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / n * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / n

    print(f"    {category_name}: {n} trades | Win: {wr:.0f}% | "
          f"P&L: {total_pnl:+.2f} pts | Avg: {avg_pnl:+.2f} pts")


def print_results(all_trades, daily_results, label="", tf_name=""):
    if not all_trades:
        print(f"\n❌ No trades [{tf_name}]")
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
    print(f"  🌲 RANDOM FOREST [{tf_name}] — {label}")
    print(f"{'='*70}")
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

    # ── Trade Categories ──
    high_pts, mid_pts, low_pts = categorize_trades(all_trades)
    print(f"\n  📊 TRADE CATEGORIES (by absolute points):")
    print(f"    HIGH (>={HIGH_POINTS_THRESHOLD} pts):")
    print_category_stats(high_pts, "     HIGH")
    print(f"    MID ({MID_POINTS_THRESHOLD}-{HIGH_POINTS_THRESHOLD} pts):")
    print_category_stats(mid_pts, "      MID")
    print(f"    LOW (<{MID_POINTS_THRESHOLD} pts):")
    print_category_stats(low_pts, "      LOW")

    # ── Winning trades by category ──
    high_wins = [t for t in high_pts if t['pnl'] > 0]
    mid_wins = [t for t in mid_pts if t['pnl'] > 0]
    low_wins = [t for t in low_pts if t['pnl'] > 0]
    print(f"\n  ✅ WINNING TRADES by category:")
    print(f"    HIGH wins: {len(high_wins)} | P&L: {sum(t['pnl'] for t in high_wins):+.2f} pts")
    print(f"    MID wins:  {len(mid_wins)} | P&L: {sum(t['pnl'] for t in mid_wins):+.2f} pts")
    print(f"    LOW wins:  {len(low_wins)} | P&L: {sum(t['pnl'] for t in low_wins):+.2f} pts")

    # ── Daily results ──
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

    print(f"\n{'='*70}")


def print_detailed_daily_log(all_trades, daily_results, tf_name="", log_file=None):
    sorted_days = sorted(daily_results.keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*110}")
    out(f"  📋 DETAILED TRADE LOG — [{tf_name}]")
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
        losses_count = len(trades) - wins
        wr = wins / len(trades) * 100
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"

        out(f"\n  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        out(f"  │ 📅 {day} ({day_name}) | Lots: {lots} | Trades: {len(trades)} (W:{wins} L:{losses_count} {wr:.0f}%) | Day P&L: {day_pnl:+.2f} {icon} | Cum: {cum_pnl:+.2f}")
        out(f"  ├───┬──────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┬─────────────┤")
        out(f"  │ # │ Dir  │ Entry Time│ Exit Time │ Entry Pr  │ Exit Pr   │ SL         │ P&L      │ Exit Reason │")
        out(f"  ├───┼──────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┼─────────────┤")

        for j, t in enumerate(trades, 1):
            entry_t = t['entry_time'].strftime('%H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[-8:-3]
            exit_t = t['exit_time'].strftime('%H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[-8:-3]
            sl_str = f"{t['sl']:.2f}" if 'sl' in t else "---"
            dir_icon = "🟢" if t['dir'] == 'LONG' else "🔴"
            pnl_icon = "✅" if t['pnl'] > 0 else "❌" if t['pnl'] < 0 else "➖"
            out(f"  │{j:>2} │ {dir_icon}{t['dir']:>4} │ {entry_t:>9} │ {exit_t:>9} │ {t['entry']:>9.2f} │ {t['exit']:>9.2f} │ {sl_str:>10} │ {t['pnl']:>+7.2f}{pnl_icon}│ {t['reason']:<12}│")

        out(f"  └───┴──────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┴─────────────┘")

    out(f"\n{'='*110}")
    out(f"  GRAND TOTAL [{tf_name}]: {cum_pnl:+.2f} pts")
    out(f"{'='*110}")

    if log_file:
        log_path = os.path.join(SCRIPT_DIR, log_file)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n💾 Detailed log saved: {log_path}")


def print_comparison(results_2m, results_3m):
    """Print side-by-side comparison of 2-min vs 3-min results."""
    print(f"\n{'='*70}")
    print(f"  📊 COMPARISON: 2-MINUTE vs 3-MINUTE")
    print(f"{'='*70}")

    def _stats(trades, daily):
        if not trades:
            return {'n': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'dd': 0, 'days': 0, 'win_days': 0}
        n = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / n * 100
        pnl = sum(t['pnl'] for t in trades)
        gp = sum(t['pnl'] for t in wins) if wins else 0
        gl = abs(sum(t['pnl'] for t in losses)) if losses else 1
        pf = gp / gl if gl > 0 else 0
        pnl_list = [t['pnl'] for t in trades]
        cumulative = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cumulative)
        dd = (peak - cumulative).max()
        trading_days = sum(1 for d in daily.values() if len(d['trades']) > 0)
        win_days = sum(1 for d in daily.values() if d['pnl'] > 0)
        high, mid, low = categorize_trades(trades)
        return {'n': n, 'wr': wr, 'pnl': pnl, 'pf': pf, 'dd': dd,
                'days': trading_days, 'win_days': win_days,
                'high': len(high), 'mid': len(mid), 'low': len(low)}

    s2 = _stats(*results_2m) if results_2m[0] else _stats([], {})
    s3 = _stats(*results_3m) if results_3m[0] else _stats([], {})

    print(f"\n  {'Metric':<22} {'2-MIN':>12} {'3-MIN':>12} {'BETTER':>10}")
    print(f"  {'-'*58}")

    def _row(label, v2, v3, fmt="+.2f", higher_better=True):
        s2_str = f"{v2:{fmt}}"
        s3_str = f"{v3:{fmt}}"
        if v2 == v3:
            better = "TIE"
        elif (v2 > v3) == higher_better:
            better = "← 2-MIN"
        else:
            better = "3-MIN →"
        print(f"  {label:<22} {s2_str:>12} {s3_str:>12} {better:>10}")

    _row("Total Trades", s2['n'], s3['n'], "d", False)
    _row("Win Rate %", s2['wr'], s3['wr'], ".1f")
    _row("Total P&L", s2['pnl'], s3['pnl'])
    _row("Profit Factor", s2['pf'], s3['pf'], ".2f")
    _row("Max Drawdown", s2['dd'], s3['dd'], ".2f", False)
    _row("Trading Days", s2['days'], s3['days'], "d")
    _row("Win Days", s2['win_days'], s3['win_days'], "d")
    _row("HIGH pt trades", s2.get('high',0), s3.get('high',0), "d")
    _row("MID pt trades", s2.get('mid',0), s3.get('mid',0), "d")
    _row("LOW pt trades", s2.get('low',0), s3.get('low',0), "d")

    print(f"  {'-'*58}")
    winner = "2-MINUTE" if s2['pnl'] > s3['pnl'] else "3-MINUTE" if s3['pnl'] > s2['pnl'] else "TIE"
    print(f"\n  🏆 OVERALL WINNER: {winner}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════
#  RUN PIPELINE FOR ONE TIMEFRAME
# ═══════════════════════════════════════════

def run_single_timeframe(df, tf_name, test_from_date, n_estimators=500,
                          stoch_buy=10, stoch_sell=90, regime_thresh=0.02,
                          regime_forward=30):
    """Run the full pipeline for a single timeframe. Returns (trades, daily_results, model)."""
    print(f"\n{'#'*70}")
    print(f"  RUNNING PIPELINE: {tf_name}")
    print(f"{'#'*70}")

    # Build features
    print(f"\n  📊 Building features [{tf_name}]...")
    features = build_features(df, tf_name.replace('-', '').replace(' ', ''))
    print(f"    Features: {features.shape[1]} columns")

    # Fill day_return
    close = df['Close'].astype(float)
    suffix = f"_{tf_name.replace('-', '').replace(' ', '')}"
    day_groups = df.groupby(df['Time'].dt.date)
    day_returns = pd.Series(0.0, index=df.index)
    for day, group in day_groups:
        day_open = float(group.iloc[0]['Open'])
        if day_open > 0:
            day_returns.loc[group.index] = (close.loc[group.index] - day_open) / day_open * 100
    features[f'day_return{suffix}'] = day_returns.values

    # Generate regime labels
    print(f"  🏷️  Generating regime labels [{tf_name}]...")
    labels = generate_regime_labels(df, regime_forward, regime_thresh)
    bull = (labels == 1).sum()
    bear = (labels == -1).sum()
    chop = (labels == 0).sum()
    print(f"    BULL={bull} ({bull/len(labels)*100:.1f}%) | "
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

    print(f"    Train: {len(X_train)} samples ({train_days} days)")
    print(f"    Test:  {len(X_test)} samples ({test_days} days)")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ❌ Insufficient data for [{tf_name}]. Skipping.")
        return [], {}, None, features

    # Train RF
    rf_model = train_random_forest(X_train, y_train, X_test, y_test,
                                    n_estimators, label=tf_name)

    # Feature importance
    importance = rf_model.feature_importances_
    feat_names = features.columns.tolist()
    top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
    print(f"\n  📊 TOP 10 FEATURES [{tf_name}]:")
    for name, imp in top_features:
        print(f"    {name:>35}: {imp:.4f}")

    # Generate predictions
    print(f"\n  🔮 Generating predictions [{tf_name}]...")
    rf_full_pred = np.zeros(len(df), dtype=int)
    rf_test_pred = rf_model.predict(X_test) - 1
    rf_full_pred[test_mask] = rf_test_pred
    print(f"    BULL={sum(rf_test_pred==1)} BEAR={sum(rf_test_pred==-1)} CHOP={sum(rf_test_pred==0)}")

    # Prepare indicators for backtest
    atr_vals = calc_atr(df, ATR_PERIOD).values
    stoch_k_vals, stoch_d_vals = calc_stochastic_rsi(
        close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH
    )
    stoch_k_vals = stoch_k_vals.values
    _, ut_dir_vals = calc_ut_bot_direction(close, calc_atr(df, ATR_PERIOD), ATR_KEY_VALUE)

    # Backtest on test set
    print(f"\n  🚀 Running backtest [{tf_name}]...")
    test_df_mask = dates >= test_from_date
    df_test = df[test_df_mask].reset_index(drop=True)
    pred_test = rf_full_pred[test_df_mask]
    atr_test = atr_vals[test_df_mask]
    stoch_k_test = stoch_k_vals[test_df_mask]
    ut_dir_test = ut_dir_vals[test_df_mask]

    all_trades, daily_results = backtest_strategy(
        df_test, pred_test, stoch_k_test, ut_dir_test, atr_test,
        stoch_buy_thresh=stoch_buy, stoch_sell_thresh=stoch_sell,
    )

    test_start = df_test['Time'].dt.date.min()
    test_end = df_test['Time'].dt.date.max()
    print_results(all_trades, daily_results,
                  f"TEST ({test_start} → {test_end})", tf_name)
    print_detailed_daily_log(all_trades, daily_results,
                             tf_name, f"rf_detailed_log_{tf_name.lower().replace('-','').replace(' ','')}.txt")

    # Save trades
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_path = os.path.join(SCRIPT_DIR, f"rf_trades_{tf_name.lower().replace('-','').replace(' ','')}.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"  💾 Trades saved: {trades_path}")

    # Save model
    model_path = os.path.join(SCRIPT_DIR, f"rf_model_{tf_name.lower().replace('-','').replace(' ','')}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"  💾 Model saved: {model_path}")

    return all_trades, daily_results, rf_model, features


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    global ATR_KEY_VALUE, MIN_ATR, ENTRY_START, ENTRY_END, SQUARE_OFF
    global STOCH_BUY_THRESHOLD, STOCH_SELL_THRESHOLD, OBSERVATION_END

    parser = argparse.ArgumentParser(description="RF + StochRSI + UT Bot (2-min & 3-min SEPARATE)")
    parser.add_argument("--file-2m", default=os.path.join(SCRIPT_DIR, "nifty_2min_data.csv"), help="2-min data")
    parser.add_argument("--file-3m", default=os.path.join(SCRIPT_DIR, "nifty_3min_data.csv"), help="3-min data")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--stoch-buy", type=float, default=STOCH_BUY_THRESHOLD)
    parser.add_argument("--stoch-sell", type=float, default=STOCH_SELL_THRESHOLD)
    parser.add_argument("--regime-thresh", type=float, default=REGIME_SLOPE_THRESHOLD)
    parser.add_argument("--regime-forward", type=int, default=REGIME_FORWARD_CANDLES,
                        help="Forward candles for regime label (default: 30)")
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--obs-end", type=str, default="11:00")
    parser.add_argument("--window-end", type=str, default="15:15")
    parser.add_argument("--square-off", type=str, default="15:24")
    parser.add_argument("--test-from", type=str, default="2026-04-01")
    parser.add_argument("--only-2m", action="store_true", help="Run only 2-min")
    parser.add_argument("--only-3m", action="store_true", help="Run only 3-min")
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
    print(f"  🌲 RANDOM FOREST + STOCHRSI + UT BOT")
    print(f"  SEPARATE 2-MIN & 3-MIN MODELS")
    print(f"{'='*70}")
    print(f"  Observation: 09:15 - {args.obs_end}")
    print(f"  Trading:     {args.obs_end} - {args.window_end}")
    print(f"  Square off:  {args.square_off}")
    print(f"  StochRSI BUY: K < {STOCH_BUY_THRESHOLD} | SELL: K > {STOCH_SELL_THRESHOLD}")
    print(f"  Test from:   {args.test_from}")

    results_2m = ([], {})
    results_3m = ([], {})

    # ── 2-MINUTE MODEL ──
    if not args.only_3m:
        print(f"\n📊 Loading 2-min data: {args.file_2m}")
        try:
            df_2m = pd.read_csv(args.file_2m)
            df_2m['Time'] = pd.to_datetime(df_2m['Time'])
            df_2m = df_2m.sort_values('Time').reset_index(drop=True)
            print(f"  {len(df_2m):,} candles | {df_2m['Time'].dt.date.nunique()} days")
            print(f"  Range: {df_2m['Time'].iloc[0].strftime('%Y-%m-%d')} → "
                  f"{df_2m['Time'].iloc[-1].strftime('%Y-%m-%d')}")

            trades_2m, daily_2m, model_2m, feats_2m = run_single_timeframe(
                df_2m, "2-MIN", test_from_date,
                n_estimators=args.rf_trees,
                stoch_buy=STOCH_BUY_THRESHOLD,
                stoch_sell=STOCH_SELL_THRESHOLD,
                regime_thresh=args.regime_thresh,
                regime_forward=args.regime_forward,
            )
            results_2m = (trades_2m, daily_2m)
        except FileNotFoundError:
            print(f"  ❌ File not found: {args.file_2m}")

    # ── 3-MINUTE MODEL ──
    if not args.only_2m:
        print(f"\n📊 Loading 3-min data: {args.file_3m}")
        try:
            df_3m = pd.read_csv(args.file_3m)
            df_3m['Time'] = pd.to_datetime(df_3m['Time'])
            df_3m = df_3m.sort_values('Time').reset_index(drop=True)
            print(f"  {len(df_3m):,} candles | {df_3m['Time'].dt.date.nunique()} days")
            print(f"  Range: {df_3m['Time'].iloc[0].strftime('%Y-%m-%d')} → "
                  f"{df_3m['Time'].iloc[-1].strftime('%Y-%m-%d')}")

            trades_3m, daily_3m, model_3m, feats_3m = run_single_timeframe(
                df_3m, "3-MIN", test_from_date,
                n_estimators=args.rf_trees,
                stoch_buy=STOCH_BUY_THRESHOLD,
                stoch_sell=STOCH_SELL_THRESHOLD,
                regime_thresh=args.regime_thresh,
                regime_forward=args.regime_forward,
            )
            results_3m = (trades_3m, daily_3m)
        except FileNotFoundError:
            print(f"  ❌ File not found: {args.file_3m}")

    # ── COMPARISON ──
    if not args.only_2m and not args.only_3m:
        print_comparison(results_2m, results_3m)

    # ── Save metadata ──
    metadata = {
        "strategy": "Random Forest + StochRSI + UT Bot (Separate 2m/3m)",
        "test_from": args.test_from,
        "stoch_buy": STOCH_BUY_THRESHOLD,
        "stoch_sell": STOCH_SELL_THRESHOLD,
        "observation_end": args.obs_end,
        "rf_trees": args.rf_trees,
        "last_trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(SCRIPT_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n💾 Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
