"""
rf_strategy.py — Random Forest + UT Bot + StochRSI (NIFTY 2-MIN ONLY)

ENHANCEMENTS:
  - RF probability confidence filtering (only enter when model is >55% sure)
  - Multi-ATR backtest: runs at ATR=[6,8,10,12,14,16,18,20] for frontend comparison
  - Skipped signal tracking: records WHY each near-miss wasn't taken
  - Better initial SL: max(ATR * 1.5, 15 pts)
  - IST time export for frontend chart

ZERO LOOK-AHEAD:
  Regime labels use forward EMA slope ONLY for training.
  At inference: model predicts from current/past features only.
  Backtest iterates candle-by-candle, never peeking ahead.

Usage:
    python rf_strategy.py                                # Default
    python rf_strategy.py --min-atr 12                   # ATR gate
    python rf_strategy.py --rf-confidence 0.6            # Stricter confidence
    python rf_strategy.py --test-from 2026-03-01         # March+April test
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
import calendar
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
ATR_SL_MULTIPLIER = 1.5       # Initial SL = max(ATR * 1.5, MIN_SL_POINTS)
MIN_SL_POINTS = 15.0          # Minimum SL in points
MIN_ATR = 12.0

OBSERVATION_END = dt_time(13, 0)
ENTRY_START = dt_time(13, 0)
ENTRY_END = dt_time(15, 12)
SQUARE_OFF = dt_time(15, 20)

STOCH_BUY_THRESHOLD = 10
STOCH_SELL_THRESHOLD = 90
STOCH_RSI_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3

RF_CONFIDENCE = 0.55           # Min RF probability to enter

REGIME_FORWARD_CANDLES = 30
REGIME_SLOPE_THRESHOLD = 0.02

LOT_SIZE = 65
BASE_LOTS = 2

HIGH_POINTS_THRESHOLD = 30
MID_POINTS_THRESHOLD = 10

ATR_LEVELS = [6, 8, 10, 12, 14, 16, 18, 20]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════

def calc_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calc_atr(df, period=14):
    h, l, c = df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float)
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
    if hasattr(close_arr, 'values'): close_arr = close_arr.values
    if hasattr(atr_arr, 'values'): atr_arr = atr_arr.values
    n = len(close_arr)
    trail_stop = np.zeros(n)
    direction = np.zeros(n)
    trail_stop[0] = close_arr[0]
    direction[0] = 1
    for i in range(1, n):
        nloss = atr_arr[i] * key_value
        if direction[i-1] == 1:
            trail_stop[i] = max(close_arr[i] - nloss, trail_stop[i-1])
            if close_arr[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close_arr[i] + nloss
            else:
                direction[i] = 1
        else:
            trail_stop[i] = min(close_arr[i] + nloss, trail_stop[i-1])
            if close_arr[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close_arr[i] - nloss
            else:
                direction[i] = -1
    return trail_stop, direction

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════
#  FEATURES (2-MIN)
# ═══════════════════════════════════════════

def build_features(df):
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

    f = pd.DataFrame(index=df.index)

    # Core
    f['atr'] = atr
    f['rsi'] = rsi
    f['stoch_k'] = stoch_k
    f['stoch_d'] = stoch_d
    f['stoch_kd_diff'] = stoch_k - stoch_d
    f['ut_dir'] = dirn
    f['close_vs_trail'] = close.values - trail

    # UT Bot signals
    ut_change = np.diff(dirn, prepend=dirn[0])
    f['ut_change'] = ut_change
    f['ut_buy_signal'] = (ut_change > 0).astype(int)
    f['ut_sell_signal'] = (ut_change < 0).astype(int)

    # StochRSI zones
    f['stoch_oversold'] = (stoch_k < STOCH_BUY_THRESHOLD).astype(int)
    f['stoch_overbought'] = (stoch_k > STOCH_SELL_THRESHOLD).astype(int)
    f['stoch_k_cross_up'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    f['stoch_k_cross_dn'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # Momentum
    f['mom_3'] = close.pct_change(3) * 100
    f['mom_5'] = close.pct_change(5) * 100
    f['mom_10'] = close.pct_change(10) * 100
    f['mom_20'] = close.pct_change(20) * 100

    # Candle
    f['body'] = close - opn
    f['body_pct'] = (close - opn) / opn * 100
    f['upper_wick'] = high - close.where(close > opn, opn)
    f['lower_wick'] = close.where(close < opn, opn) - low
    f['range'] = high - low
    f['body_vs_range'] = (close - opn).abs() / (high - low).replace(0, 1e-10)

    # Volatility
    f['std_5'] = close.rolling(5).std()
    f['std_10'] = close.rolling(10).std()
    f['std_20'] = close.rolling(20).std()
    f['atr_percentile'] = atr.rolling(50).apply(lambda x: (x.iloc[-1] > x).mean() * 100, raw=False)

    # EMAs
    f['close_vs_ema5'] = close - ema_5
    f['close_vs_ema10'] = close - ema_10
    f['close_vs_ema20'] = close - ema_20
    f['close_vs_ema50'] = close - ema_50
    f['ema5_vs_ema10'] = ema_5 - ema_10
    f['ema10_vs_ema20'] = ema_10 - ema_20
    f['ema20_vs_ema50'] = ema_20 - ema_50
    f['ema20_slope'] = ema_20.diff(3) / 3
    f['ema50_slope'] = ema_50.diff(5) / 5

    # Channels
    f['close_vs_high5'] = close - high.rolling(5).max()
    f['close_vs_low5'] = close - low.rolling(5).min()
    f['close_vs_high10'] = close - high.rolling(10).max()
    f['close_vs_low10'] = close - low.rolling(10).min()

    # Pattern
    bullish = (close > opn).astype(int)
    f['bullish_count_10'] = bullish.rolling(10).sum()
    f['bullish_count_20'] = bullish.rolling(20).sum()
    f['bullish_ratio_10'] = f['bullish_count_10'] / 10
    f['bullish_ratio_20'] = f['bullish_count_20'] / 20

    # RSI divergence (price making new low but RSI not)
    f['rsi_slope_5'] = rsi.diff(5)
    f['price_slope_5'] = close.diff(5)

    f['day_return'] = 0.0
    return f


# ═══════════════════════════════════════════
#  REGIME LABELS
# ═══════════════════════════════════════════

def generate_regime_labels(df, forward_candles=30, slope_threshold=0.02):
    close = df['Close'].astype(float)
    ema_20 = calc_ema(close, 20)
    n = len(df)
    labels = np.zeros(n, dtype=int)
    for i in range(n - forward_candles):
        cur = ema_20.iloc[i]
        fut = ema_20.iloc[i + forward_candles]
        if cur == 0: continue
        slope = (fut - cur) / cur * 100
        if slope > slope_threshold: labels[i] = 1
        elif slope < -slope_threshold: labels[i] = -1
    return labels


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=500):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=0,
    )
    print(f"\n  Training RF ({n_estimators} trees)...")
    model.fit(X_train, y_train + 1)

    train_pred = model.predict(X_train) - 1
    test_pred = model.predict(X_test) - 1
    print(f"  Train: {accuracy_score(y_train, train_pred)*100:.1f}% | "
          f"Test: {accuracy_score(y_test, test_pred)*100:.1f}%")
    print(classification_report(y_test+1, test_pred+1,
                                target_names=['BEAR','CHOP','BULL'], zero_division=0))
    return model


# ═══════════════════════════════════════════
#  BACKTEST (WITH SKIPPED SIGNAL TRACKING)
# ═══════════════════════════════════════════

def backtest_strategy(df, regime_preds, regime_proba, stoch_k_vals, ut_dir_vals, atr_vals,
                      stoch_buy_thresh=10, stoch_sell_thresh=90, min_atr=12.0,
                      rf_confidence=0.55):
    """
    Candle-by-candle backtest with skipped signal tracking.
    regime_proba: array of max probability from RF for each candle (confidence)
    """
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)

    pos = None
    all_trades = []
    daily_results = {}
    skipped_signals = []
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
                trade = _make_trade(pos, float(close.iloc[i-1]), df.iloc[i-1]['Time'],
                                    "DAY_END", current_lots, pos.get('initial_sl'))
                all_trades.append(trade)
                _add_daily(daily_results, prev_date, trade)
                pos = None
            if curr_date not in daily_results:
                daily_results[curr_date] = {'trades': [], 'pnl': 0, 'lots': current_lots}
            if prev_date in daily_results:
                if daily_results[prev_date]['pnl'] < 0:
                    accumulated_loss += daily_results[prev_date]['pnl']
                    recovering = True
                    current_lots += 2
                elif daily_results[prev_date]['pnl'] > 0 and recovering:
                    accumulated_loss += daily_results[prev_date]['pnl']
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

        # Trail SL
        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']: pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']: pos['sl'] = new_sl

        regime = regime_preds[i]
        sk = stoch_k_vals[i] if not np.isnan(stoch_k_vals[i]) else 50
        ut_d = ut_dir_vals[i]
        confidence = regime_proba[i] if i < len(regime_proba) else 0.5

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

        # ── Entry logic with skipped signal tracking ──
        if not pos and in_window:
            # Check BUY conditions
            buy_regime = (regime == 1)
            buy_stoch = (sk < stoch_buy_thresh)
            buy_ut = (ut_d == 1)
            buy_atr = (curr_atr >= min_atr)
            buy_conf = (confidence >= rf_confidence)

            # Check SELL conditions
            sell_regime = (regime == -1)
            sell_stoch = (sk > stoch_sell_thresh)
            sell_ut = (ut_d == -1)
            sell_atr = buy_atr  # same ATR check
            sell_conf = buy_conf  # same confidence check

            entered = False

            # BUY entry
            if buy_regime and buy_stoch and buy_ut and buy_atr and buy_conf:
                initial_sl_dist = max(curr_atr * ATR_SL_MULTIPLIER, MIN_SL_POINTS)
                sl = c - initial_sl_dist
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
                entered = True

            # SELL entry
            elif sell_regime and sell_stoch and sell_ut and sell_atr and sell_conf:
                initial_sl_dist = max(curr_atr * ATR_SL_MULTIPLIER, MIN_SL_POINTS)
                sl = c + initial_sl_dist
                pos = {'dir': 'SHORT', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
                entered = True

            # ── Track skipped signals (near misses) ──
            if not entered:
                skip_reasons = []

                # BUY near-miss: regime is BULL
                if buy_regime:
                    if not buy_stoch: skip_reasons.append(f"STOCH_K={sk:.1f} (need <{stoch_buy_thresh})")
                    if not buy_ut: skip_reasons.append("UT_BOT_BEARISH")
                    if not buy_atr: skip_reasons.append(f"ATR={curr_atr:.1f} (need >={min_atr})")
                    if not buy_conf: skip_reasons.append(f"RF_CONF={confidence:.2f} (need >={rf_confidence})")

                # SELL near-miss: regime is BEAR
                elif sell_regime:
                    if not sell_stoch: skip_reasons.append(f"STOCH_K={sk:.1f} (need >{stoch_sell_thresh})")
                    if not sell_ut: skip_reasons.append("UT_BOT_BULLISH")
                    if not sell_atr: skip_reasons.append(f"ATR={curr_atr:.1f} (need >={min_atr})")
                    if not sell_conf: skip_reasons.append(f"RF_CONF={confidence:.2f} (need >={rf_confidence})")

                # Only record if regime had a direction AND at least one other condition was met
                if skip_reasons and len(skip_reasons) <= 2:
                    potential_dir = "LONG" if buy_regime else "SHORT" if sell_regime else None
                    if potential_dir:
                        ts_val = int(calendar.timegm(df.iloc[i]['Time'].timetuple()))
                        skipped_signals.append({
                            'time': ts_val,
                            'price': round(c, 2),
                            'potential_dir': potential_dir,
                            'reasons': skip_reasons,
                            'atr': round(curr_atr, 2),
                            'stoch_k': round(sk, 1),
                            'ut_dir': int(ut_d),
                            'rf_conf': round(confidence, 3),
                        })

        elif pos and in_window:
            # Already in position — check if opposite signal was available
            buy_regime = (regime == 1)
            sell_regime = (regime == -1)
            if (buy_regime and pos['dir'] == 'LONG') or (sell_regime and pos['dir'] == 'SHORT'):
                pass  # same direction, fine
            elif (buy_regime and sk < stoch_buy_thresh and ut_d == 1 and pos['dir'] == 'SHORT'):
                pass  # regime flip will handle
            elif (sell_regime and sk > stoch_sell_thresh and ut_d == -1 and pos['dir'] == 'LONG'):
                pass  # regime flip will handle

    return all_trades, daily_results, skipped_signals


def _pnl(pos, exit_price):
    return (exit_price - pos['entry']) if pos['dir'] == "LONG" else (pos['entry'] - exit_price)

def _make_trade(pos, exit_price, exit_time, reason, lots=2, sl=None):
    raw_pnl = _pnl(pos, exit_price)
    lot_m = lots // BASE_LOTS
    trade = {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': round(exit_price, 2),
        'entry_time': pos['entry_time'], 'exit_time': exit_time,
        'pnl': round(raw_pnl * lot_m, 2), 'raw_pnl': round(raw_pnl, 2),
        'lots': lots, 'multiplier': lot_m, 'qty': lots * LOT_SIZE,
        'pnl_pct': round(raw_pnl / pos['entry'] * 100, 4), 'reason': reason,
    }
    if sl is not None: trade['sl'] = round(sl, 2)
    return trade

def _add_daily(daily_results, date, trade):
    if date not in daily_results:
        daily_results[date] = {'trades': [], 'pnl': 0, 'lots': trade.get('lots', BASE_LOTS)}
    daily_results[date]['trades'].append(trade)
    daily_results[date]['pnl'] += trade['pnl']


# ═══════════════════════════════════════════
#  MULTI-ATR RUNNER
# ═══════════════════════════════════════════

def run_multi_atr_backtest(df_test, regime_preds, regime_proba, stoch_k, ut_dir, atr,
                           atr_levels, stoch_buy, stoch_sell, rf_conf):
    """Run independent backtests at each ATR level."""
    results = {}
    for level in atr_levels:
        trades, daily, skipped = backtest_strategy(
            df_test, regime_preds, regime_proba, stoch_k, ut_dir, atr,
            stoch_buy_thresh=stoch_buy, stoch_sell_thresh=stoch_sell,
            min_atr=float(level), rf_confidence=rf_conf,
        )
        results[str(level)] = {
            'trades': trades, 'daily': daily, 'skipped': skipped,
        }
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"    ATR>={level:>2}: {len(trades):>3} trades | "
              f"Win: {wins}/{len(trades)} ({wins/max(len(trades),1)*100:.0f}%) | "
              f"P&L: {total_pnl:+.2f}")
    return results


# ═══════════════════════════════════════════
#  EXPORT FOR FRONTEND
# ═══════════════════════════════════════════

def export_frontend_data(df_test, multi_atr_results, stoch_k_vals, stoch_d_vals,
                          atr_vals, ut_dir_vals, regime_preds, regime_proba):
    """Export comprehensive JSON: candles + multi-ATR results + skipped signals."""

    # Candles with IST timestamps (calendar.timegm treats naive as UTC → correct IST display)
    candles = []
    for _, row in df_test.iterrows():
        ts = int(calendar.timegm(row['Time'].timetuple()))
        candles.append({
            'time': ts,
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
        })

    # StochRSI + ATR overlay data
    stoch_data = []
    atr_data = []
    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        ts = int(calendar.timegm(row['Time'].timetuple()))
        sk = float(stoch_k_vals[idx]) if not np.isnan(stoch_k_vals[idx]) else None
        sd = float(stoch_d_vals[idx]) if not np.isnan(stoch_d_vals[idx]) else None
        stoch_data.append({'time': ts, 'k': round(sk, 2) if sk else None, 'd': round(sd, 2) if sd else None})
        atr_data.append({'time': ts, 'value': round(float(atr_vals[idx]), 2)})

    # Build per-ATR-level results
    atr_results = {}
    for level_str, res in multi_atr_results.items():
        trades = res['trades']
        skipped = res['skipped']

        # Trade markers
        markers = []
        for t in trades:
            ets = int(calendar.timegm(t['entry_time'].timetuple())) if hasattr(t['entry_time'], 'timetuple') else 0
            xts = int(calendar.timegm(t['exit_time'].timetuple())) if hasattr(t['exit_time'], 'timetuple') else 0
            markers.append({'type': 'entry', 'time': ets, 'price': t['entry'],
                            'dir': t['dir'], 'pnl': t['pnl'], 'reason': t['reason']})
            markers.append({'type': 'exit', 'time': xts, 'price': t['exit'],
                            'dir': t['dir'], 'pnl': t['pnl'], 'reason': t['reason']})

        # Summary
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses_count = sum(1 for t in trades if t['pnl'] <= 0)
        total_pnl = sum(t['pnl'] for t in trades)
        gp = sum(t['pnl'] for t in trades if t['pnl'] > 0) or 0
        gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0)) or 1

        pnl_list = [t['pnl'] for t in trades]
        cum = 0; peak = 0; max_dd = 0
        for p in pnl_list:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        high_pts = [t for t in trades if abs(t['raw_pnl']) >= HIGH_POINTS_THRESHOLD]
        mid_pts = [t for t in trades if MID_POINTS_THRESHOLD <= abs(t['raw_pnl']) < HIGH_POINTS_THRESHOLD]
        low_pts = [t for t in trades if abs(t['raw_pnl']) < MID_POINTS_THRESHOLD]

        summary = {
            'total_trades': len(trades), 'wins': wins, 'losses': losses_count,
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(wins / max(len(trades), 1) * 100, 1),
            'profit_factor': round(gp / gl, 2),
            'max_drawdown': round(max_dd, 2),
            'avg_win': round(np.mean([t['pnl'] for t in trades if t['pnl'] > 0]), 2) if wins else 0,
            'avg_loss': round(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]), 2) if losses_count else 0,
            'high_pts': len(high_pts), 'mid_pts': len(mid_pts), 'low_pts': len(low_pts),
        }

        # Serialize trades
        trade_list = [{
            'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
            'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
            'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
            'pnl': t['pnl'], 'raw_pnl': t['raw_pnl'], 'reason': t['reason'],
        } for t in trades]

        atr_results[level_str] = {
            'markers': markers, 'summary': summary,
            'trades': trade_list, 'skipped': skipped[:200],  # cap at 200
        }

    data = {
        'candles': candles,
        'stochRSI': stoch_data,
        'atr': atr_data,
        'atr_results': atr_results,
        'atr_levels': [str(l) for l in ATR_LEVELS],
        'config': {
            'stochBuy': STOCH_BUY_THRESHOLD, 'stochSell': STOCH_SELL_THRESHOLD,
            'defaultATR': MIN_ATR, 'entryStart': str(ENTRY_START),
            'entryEnd': str(ENTRY_END), 'squareOff': str(SQUARE_OFF),
            'rfConfidence': RF_CONFIDENCE,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "frontend_data.json")
    with open(out_path, 'w') as f:
        json.dump(data, f)
    size_mb = os.path.getsize(out_path) / (1024*1024)
    print(f"💾 Frontend data: {out_path} ({size_mb:.1f} MB)")
    return out_path


# ═══════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════

def print_results(all_trades, daily_results, label=""):
    if not all_trades:
        print("❌ No trades"); return

    n = len(all_trades)
    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    wr = len(wins) / n * 100
    total_pnl = sum(t['pnl'] for t in all_trades)
    gp = sum(t['pnl'] for t in wins) or 0
    gl = abs(sum(t['pnl'] for t in losses)) or 1

    pnl_list = [t['pnl'] for t in all_trades]
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (peak - cumulative).max()
    trading_days = sum(1 for d in daily_results.values() if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)

    print(f"\n{'='*70}")
    print(f"  🌲 RF [2-MIN] — {label}")
    print(f"{'='*70}")
    print(f"  Trades: {n} | Win: {wr:.1f}% | P&L: {total_pnl:+.2f} pts")
    print(f"  PF: {gp/gl:.2f} | Avg win: {np.mean([t['pnl'] for t in wins]) if wins else 0:+.2f} | "
          f"Avg loss: {np.mean([t['pnl'] for t in losses]) if losses else 0:+.2f}")
    print(f"  Max DD: {max_dd:.2f} | Days: {trading_days} | Win days: {win_days}")

    high = [t for t in all_trades if abs(t['raw_pnl']) >= HIGH_POINTS_THRESHOLD]
    mid = [t for t in all_trades if MID_POINTS_THRESHOLD <= abs(t['raw_pnl']) < HIGH_POINTS_THRESHOLD]
    low = [t for t in all_trades if abs(t['raw_pnl']) < MID_POINTS_THRESHOLD]
    print(f"\n  Categories: HIGH={len(high)} MID={len(mid)} LOW={len(low)}")

    sorted_days = sorted(daily_results.keys())
    print(f"\n  {'Date':>12} {'Lots':>5} {'#':>3} {'P&L':>10}")
    print(f"  {'-'*35}")
    cum = 0
    for day in sorted_days:
        trades = daily_results[day]['trades']
        if not trades: continue
        day_pnl = daily_results[day]['pnl']
        cum += day_pnl
        icon = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        print(f"  {str(day):>12} {daily_results[day].get('lots',2):>5} {len(trades):>3} {day_pnl:>+9.2f} {icon}")
    print(f"  {'-'*35}")
    print(f"  {'TOTAL':>12} {'':>5} {'':>3} {cum:>+9.2f}")

    reasons = {}
    for t in all_trades: reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    print(f"\n  Reasons: {reasons}")
    print(f"{'='*70}")


def print_detailed_log(all_trades, daily_results, log_file="rf_detailed_log.txt"):
    sorted_days = sorted(daily_results.keys())
    lines = []
    def out(s=""): print(s); lines.append(s)

    out(f"\n{'='*100}")
    out(f"  📋 DETAILED TRADE LOG — 2-MIN RF")
    out(f"{'='*100}")

    cum_pnl = 0
    for day in sorted_days:
        trades = daily_results[day]['trades']
        if not trades: continue
        day_pnl = daily_results[day]['pnl']
        cum_pnl += day_pnl
        w = sum(1 for t in trades if t['pnl'] > 0)
        icon = "✅" if day_pnl > 0 else "❌"
        out(f"\n  📅 {day} | T:{len(trades)} W:{w} | P&L:{day_pnl:+.2f} {icon} | Cum:{cum_pnl:+.2f}")
        for j, t in enumerate(trades, 1):
            et = t['entry_time'].strftime('%H:%M') if hasattr(t['entry_time'], 'strftime') else "?"
            xt = t['exit_time'].strftime('%H:%M') if hasattr(t['exit_time'], 'strftime') else "?"
            di = "🟢" if t['dir'] == 'LONG' else "🔴"
            pi = "✅" if t['pnl'] > 0 else "❌"
            out(f"    {j}. {di}{t['dir']:>5} {et}→{xt} | {t['entry']:.2f}→{t['exit']:.2f} | {t['pnl']:+.2f}{pi} | {t['reason']}")

    out(f"\n  TOTAL: {cum_pnl:+.2f}")
    log_path = os.path.join(SCRIPT_DIR, log_file)
    with open(log_path, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
    print(f"💾 Log: {log_path}")


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    global ATR_KEY_VALUE, MIN_ATR, ENTRY_START, ENTRY_END, SQUARE_OFF
    global STOCH_BUY_THRESHOLD, STOCH_SELL_THRESHOLD, OBSERVATION_END, RF_CONFIDENCE

    parser = argparse.ArgumentParser(description="RF + StochRSI + UT Bot (2-MIN)")
    parser.add_argument("--file", default=os.path.join(SCRIPT_DIR, "nifty_2min_data.csv"))
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--stoch-buy", type=float, default=STOCH_BUY_THRESHOLD)
    parser.add_argument("--stoch-sell", type=float, default=STOCH_SELL_THRESHOLD)
    parser.add_argument("--regime-thresh", type=float, default=REGIME_SLOPE_THRESHOLD)
    parser.add_argument("--regime-forward", type=int, default=REGIME_FORWARD_CANDLES)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-confidence", type=float, default=RF_CONFIDENCE,
                        help="Min RF probability to enter (default: 0.55)")
    parser.add_argument("--obs-end", type=str, default="13:00")
    parser.add_argument("--window-end", type=str, default="15:12")
    parser.add_argument("--square-off", type=str, default="15:20")
    parser.add_argument("--test-from", type=str, default="2026-03-01")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    STOCH_BUY_THRESHOLD = args.stoch_buy
    STOCH_SELL_THRESHOLD = args.stoch_sell
    RF_CONFIDENCE = args.rf_confidence

    def pt(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)
    OBSERVATION_END = pt(args.obs_end)
    ENTRY_START = OBSERVATION_END
    ENTRY_END = pt(args.window_end)
    SQUARE_OFF = pt(args.square_off)

    test_from_date = datetime.strptime(args.test_from, "%Y-%m-%d").date()

    print(f"{'='*70}")
    print(f"  🌲 RF + STOCHRSI + UT BOT [2-MIN]")
    print(f"{'='*70}")
    print(f"  Window: {args.obs_end} → {args.window_end} | SqOff: {args.square_off}")
    print(f"  ATR ≥ {MIN_ATR} | StochRSI: <{STOCH_BUY_THRESHOLD} / >{STOCH_SELL_THRESHOLD}")
    print(f"  RF Confidence ≥ {RF_CONFIDENCE} | Test: {args.test_from}")

    # Load
    print(f"\n📊 Loading: {args.file}")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"❌ Not found: {args.file}"); return

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    close = df['Close'].astype(float)
    print(f"  {len(df):,} candles | {df['Time'].dt.date.nunique()} days")

    # Features
    print(f"\n📊 Features...")
    features = build_features(df)
    # Fill day_return
    for day, grp in df.groupby(df['Time'].dt.date):
        day_open = float(grp.iloc[0]['Open'])
        if day_open > 0:
            features.loc[grp.index, 'day_return'] = (close.loc[grp.index] - day_open) / day_open * 100
    print(f"  {features.shape[1]} columns")

    # Labels
    labels = generate_regime_labels(df, args.regime_forward, args.regime_thresh)
    bull = (labels == 1).sum(); bear = (labels == -1).sum()
    print(f"  Labels: BULL={bull} BEAR={bear} CHOP={len(labels)-bull-bear}")

    # Split
    times = df['Time'].dt.time
    window_mask = (times >= dt_time(9, 20)) & (times <= ENTRY_END)
    dates = df['Time'].dt.date

    train_mask = (dates < test_from_date) & window_mask
    test_mask = (dates >= test_from_date) & window_mask

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    X_train, y_train = features[train_mask].values, labels[train_mask]
    X_test, y_test = features[test_mask].values, labels[test_mask]

    print(f"  Train: {len(X_train)} ({dates[dates < test_from_date].nunique()} days)")
    print(f"  Test:  {len(X_test)} ({dates[dates >= test_from_date].nunique()} days)")
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ No data!"); return

    # Train
    rf_model = train_random_forest(X_train, y_train, X_test, y_test, args.rf_trees)

    # Importance
    top = sorted(zip(features.columns, rf_model.feature_importances_), key=lambda x: -x[1])[:10]
    print(f"\n  TOP 10 FEATURES:")
    for name, imp in top:
        print(f"    {name:>25}: {imp:.4f}")

    # Predictions + probabilities
    print(f"\n  🔮 Predictions...")
    rf_full_pred = np.zeros(len(df), dtype=int)
    rf_full_proba = np.full(len(df), 0.33)
    test_pred = rf_model.predict(X_test) - 1
    test_proba = rf_model.predict_proba(X_test).max(axis=1)
    rf_full_pred[test_mask] = test_pred
    rf_full_proba[test_mask] = test_proba

    # Indicators
    atr_vals = calc_atr(df, ATR_PERIOD).values
    stoch_k_vals, stoch_d_vals = calc_stochastic_rsi(close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
                                                      STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    stoch_k_vals, stoch_d_vals = stoch_k_vals.values, stoch_d_vals.values
    _, ut_dir_vals = calc_ut_bot_direction(close, calc_atr(df, ATR_PERIOD), ATR_KEY_VALUE)

    # Test subset
    test_df_mask = dates >= test_from_date
    df_test = df[test_df_mask].reset_index(drop=True)
    pred_test = rf_full_pred[test_df_mask]
    proba_test = rf_full_proba[test_df_mask]
    atr_test = atr_vals[test_df_mask]
    stoch_k_test = stoch_k_vals[test_df_mask]
    stoch_d_test = stoch_d_vals[test_df_mask]
    ut_dir_test = ut_dir_vals[test_df_mask]

    # Multi-ATR backtest
    print(f"\n  🔄 Multi-ATR Backtest:")
    multi_results = run_multi_atr_backtest(
        df_test, pred_test, proba_test, stoch_k_test, ut_dir_test, atr_test,
        ATR_LEVELS, STOCH_BUY_THRESHOLD, STOCH_SELL_THRESHOLD, RF_CONFIDENCE,
    )

    # Print main result
    main_key = str(int(MIN_ATR))
    if main_key in multi_results:
        trades_main = multi_results[main_key]['trades']
        daily_main = multi_results[main_key]['daily']
        skipped_main = multi_results[main_key]['skipped']
        print_results(trades_main, daily_main, f"ATR≥{main_key}")
        print_detailed_log(trades_main, daily_main)
        print(f"\n  ⚠️ Skipped signals (near misses): {len(skipped_main)}")

    # Export
    export_frontend_data(df_test, multi_results, stoch_k_test, stoch_d_test,
                          atr_test, ut_dir_test, pred_test, proba_test)

    # Save model
    with open(os.path.join(SCRIPT_DIR, "rf_model.pkl"), 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"💾 Model saved")

    # Metadata
    metadata = {
        "strategy": "RF + StochRSI + UT Bot (2-MIN)",
        "min_atr": MIN_ATR, "rf_confidence": RF_CONFIDENCE,
        "stoch_buy": STOCH_BUY_THRESHOLD, "stoch_sell": STOCH_SELL_THRESHOLD,
        "obs_end": args.obs_end, "entry_end": args.window_end,
        "test_from": args.test_from, "rf_trees": args.rf_trees,
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(SCRIPT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Metadata saved")


if __name__ == "__main__":
    main()
