"""
live_test_catboost.py — Row-by-Row CatBoost Live Test Simulator

Replays any historical date's NIFTY data candle by candle through the
trained catboost_nifty_model.cbm, simulating exactly what live trading sees.

Features are built using the SAME functions from catboost_strategy.py,
so predictions match the backtest exactly.

Usage:
    python live_test_catboost.py --date 2026-06-16
    python live_test_catboost.py --date 2026-06-10 --step           # Press Enter per candle
    python live_test_catboost.py --date 2026-06-16 --speed 0.5      # Auto-play with delay
    python live_test_catboost.py --date 2026-06-16 --window-start 09:20 --window-end 15:15

Requirements:
    pip install catboost pandas numpy
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import time as time_module
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("❌ CatBoost not installed. Run: pip install catboost")
    sys.exit(1)

# Import backtest-identical feature builders from catboost_strategy.py
try:
    from catboost_strategy import (
        build_features_1min,
        build_features_2min,
        calc_atr,
        ATR_PERIOD,
    )
    print("✅ Imported feature builders from catboost_strategy.py")
except ImportError:
    print("❌ catboost_strategy.py not found in current directory.")
    print("   Make sure you run this from the Trading_setup_code folder.")
    sys.exit(1)


# ─────────── Config ───────────
MODEL_PATH = "catboost_nifty_model.cbm"
CSV_1M = "nifty_1min_data.csv"
CSV_2M = "nifty_2min_data.csv"

ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5
LOT_SIZE = 65
BASE_LOTS = 2

WARMUP_DAYS = 50  # Days of prior data for indicator convergence


# ─────────── Trade Helpers ───────────

def _calc_pnl(pos, exit_price):
    """Calculate points P&L for a position."""
    if pos['dir'] == 'LONG':
        return round(exit_price - pos['entry'], 2)
    else:
        return round(pos['entry'] - exit_price, 2)


def _make_trade(pos, exit_price, exit_time_str, reason, pnl):
    """Create a trade record."""
    return {
        'dir': pos['dir'],
        'entry': pos['entry'],
        'exit': round(exit_price, 2),
        'sl': round(pos.get('initial_sl', pos['sl']), 2),
        'entry_time': pos.get('entry_time', ''),
        'exit_time': exit_time_str,
        'pnl': round(pnl, 2),
        'reason': reason,
    }


# ─────────── Trade Simulation (per candle) ───────────

def simulate_candle(candle_time, time_str, pred, close, high, low, atr,
                    position, trades, daily_pnl, wins, losses,
                    entry_start, entry_end, square_off, min_atr):
    """
    Process one candle through the full trade logic:
      1. Square-off check
      2. Stop-loss check
      3. Trailing SL update
      4. Opposite signal close
      5. New entry

    Returns updated (position, trades, daily_pnl, wins, losses, action_log).
    """
    action_log = []

    # ── 1. Square off ──
    if position and candle_time >= square_off:
        pnl = _calc_pnl(position, close)
        trades.append(_make_trade(position, close, time_str, "SQUARE_OFF", pnl))
        daily_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        icon = "✅" if pnl > 0 else "❌"
        action_log.append(
            f"⏹️  SQUARE OFF {position['dir']} @ {close:.2f} | "
            f"P&L: {pnl:+.2f} {icon}"
        )
        position = None
        return position, trades, daily_pnl, wins, losses, action_log

    # ── 2. Stop-loss check ──
    if position:
        sl_hit = False
        if position['dir'] == 'LONG' and low <= position['sl']:
            sl_hit = True
            exit_price = position['sl']
        elif position['dir'] == 'SHORT' and high >= position['sl']:
            sl_hit = True
            exit_price = position['sl']

        if sl_hit:
            pnl = _calc_pnl(position, exit_price)
            trades.append(_make_trade(position, exit_price, time_str, "TRAIL_SL", pnl))
            daily_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            icon = "✅" if pnl > 0 else "❌"
            action_log.append(
                f"🛑 SL HIT on {position['dir']} @ {exit_price:.2f} | "
                f"P&L: {pnl:+.2f} {icon}"
            )
            position = None

    # ── 3. Trailing SL update ──
    if position:
        if position['dir'] == 'LONG':
            new_sl = close - atr * ATR_KEY_VALUE
            if new_sl > position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                action_log.append(f"📈 Trail SL raised: {old_sl:.2f} → {new_sl:.2f}")
        elif position['dir'] == 'SHORT':
            new_sl = close + atr * ATR_KEY_VALUE
            if new_sl < position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                action_log.append(f"📉 Trail SL lowered: {old_sl:.2f} → {new_sl:.2f}")

    # ── 4. Opposite signal close ──
    if pred == 1 and position and position['dir'] == 'SHORT':
        pnl = _calc_pnl(position, close)
        trades.append(_make_trade(position, close, time_str, "OPPOSITE", pnl))
        daily_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        icon = "✅" if pnl > 0 else "❌"
        action_log.append(
            f"🔄 BUY signal → CLOSED SHORT @ {close:.2f} | P&L: {pnl:+.2f} {icon}"
        )
        position = None
    elif pred == -1 and position and position['dir'] == 'LONG':
        pnl = _calc_pnl(position, close)
        trades.append(_make_trade(position, close, time_str, "OPPOSITE", pnl))
        daily_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        icon = "✅" if pnl > 0 else "❌"
        action_log.append(
            f"🔄 SELL signal → CLOSED LONG @ {close:.2f} | P&L: {pnl:+.2f} {icon}"
        )
        position = None

    # ── 5. New entry ──
    in_window = entry_start <= candle_time <= entry_end
    atr_ok = atr >= min_atr
    if not position and pred != 0 and in_window and atr_ok and candle_time < square_off:
        if pred == 1:
            sl = close - atr * ATR_KEY_VALUE
            position = {
                'dir': 'LONG', 'entry': close, 'sl': sl,
                'initial_sl': sl, 'entry_time': time_str,
            }
            action_log.append(
                f"🟢 ENTERED LONG @ {close:.2f} | SL: {sl:.2f} | ATR: {atr:.2f}"
            )
        elif pred == -1:
            sl = close + atr * ATR_KEY_VALUE
            position = {
                'dir': 'SHORT', 'entry': close, 'sl': sl,
                'initial_sl': sl, 'entry_time': time_str,
            }
            action_log.append(
                f"🔴 ENTERED SHORT @ {close:.2f} | SL: {sl:.2f} | ATR: {atr:.2f}"
            )

    return position, trades, daily_pnl, wins, losses, action_log


# ─────────── Display ───────────

def print_header(test_date, window_start, window_end, square_off, min_atr, step_mode, speed):
    """Print simulation header."""
    mode_str = "Step-by-step (Enter=next)" if step_mode else f"Auto-play ({speed}s delay)"
    print(f"\n{'═' * 82}")
    print(f"  🚀 CATBOOST LIVE TEST SIMULATOR — {test_date}")
    print(f"  Window: {window_start} - {window_end} | Square off: {square_off}")
    print(f"  Min ATR: {min_atr} | Mode: {mode_str}")
    print(f"{'═' * 82}")
    if step_mode:
        print("  ➡️  Press Enter to advance | 'q' to quit | 's' to skip to end\n")


def print_candle_dashboard(candle_num, total_candles, time_str, close, high, low,
                           atr, rsi, ut_dir, pred, position, daily_pnl,
                           wins, losses, action_log):
    """Print rich per-candle dashboard."""
    signal_map = {1: "🟢 BUY ", -1: "🔴 SELL", 0: "⚪ HOLD"}
    signal_str = signal_map.get(pred, "⚪ HOLD")

    ut_str = "BULL 📈" if ut_dir == 1 else "BEAR 📉" if ut_dir == -1 else "FLAT ➖"

    # Position info
    unrealized = 0
    if position:
        if position['dir'] == 'LONG':
            unrealized = round(close - position['entry'], 2)
        else:
            unrealized = round(position['entry'] - close, 2)
        pos_icon = "🟢" if position['dir'] == 'LONG' else "🔴"
        sl_dist = round(abs(close - position['sl']), 2)
        pos_str = (
            f"{pos_icon} {position['dir']} @ {position['entry']:.2f} | "
            f"SL: {position['sl']:.2f} (dist: {sl_dist:.2f}) | "
            f"Unreal: {unrealized:+.2f}"
        )
    else:
        pos_str = "💤 FLAT"

    total_trades = wins + losses
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    live_pnl = daily_pnl + unrealized

    w = 82
    bar = '─' * w

    print(f"  ┌{bar}┐")
    line1 = f"🕐 {time_str}  |  Candle {candle_num}/{total_candles}"
    print(f"  │ {line1:<{w-1}}│")
    print(f"  ├{bar}┤")

    line2 = f"NIFTY = {close:.2f}  |  H: {high:.2f}  L: {low:.2f}"
    print(f"  │ {line2:<{w-1}}│")

    line3 = f"ATR: {atr:.2f}  |  RSI: {rsi:.1f}  |  UT Bot: {ut_str}"
    print(f"  │ {line3:<{w-1}}│")

    print(f"  ├{bar}┤")

    line4 = f"🧠 Prediction: {signal_str}"
    print(f"  │ {line4:<{w-1}}│")

    line5 = f"💼 {pos_str}"
    print(f"  │ {line5:<{w-1}}│")

    pnl_icon = "✅" if live_pnl > 0 else "❌" if live_pnl < 0 else "➖"
    line6 = (
        f"💰 Day P&L: {live_pnl:+.2f} pts {pnl_icon}  |  "
        f"Trades: {total_trades} (W:{wins} L:{losses} {wr:.0f}%)"
    )
    print(f"  │ {line6:<{w-1}}│")

    if action_log:
        print(f"  ├{bar}┤")
        for action in action_log:
            print(f"  │ {action:<{w-1}}│")

    print(f"  └{bar}┘")


def print_daily_summary(trades, daily_pnl, wins, losses, test_date, predictions_log):
    """Print comprehensive end-of-day summary."""
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0) if trades else 0
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0)) if trades else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    if trades:
        pnl_list = [t['pnl'] for t in trades]
        cumulative = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cumulative)
        max_dd = (peak - cumulative).max()
    else:
        max_dd = 0

    # Prediction stats
    buy_count = sum(1 for p in predictions_log if p == 1)
    sell_count = sum(1 for p in predictions_log if p == -1)
    hold_count = sum(1 for p in predictions_log if p == 0)
    total_preds = len(predictions_log)

    print(f"\n{'═' * 82}")
    print(f"  📋 END OF DAY SUMMARY — {test_date}")
    print(f"{'═' * 82}")

    day_icon = "✅" if daily_pnl > 0 else "❌" if daily_pnl < 0 else "➖"
    print(f"  Total P&L:       {daily_pnl:+.2f} pts {day_icon}")
    print(f"  Total trades:    {total}")
    print(f"  Win rate:        {wr:.1f}%")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Max drawdown:    {max_dd:.2f} pts")

    if trades:
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losses > 0 else 0
        best = max(t['pnl'] for t in trades)
        worst = min(t['pnl'] for t in trades)
        print(f"  Avg win:         {avg_win:+.2f} pts")
        print(f"  Avg loss:        {avg_loss:+.2f} pts")
        print(f"  Best trade:      {best:+.2f} pts")
        print(f"  Worst trade:     {worst:+.2f} pts")

    print(f"\n  🧠 Prediction breakdown ({total_preds} candles):")
    print(f"     BUY:  {buy_count:>4} ({buy_count/max(total_preds,1)*100:.1f}%)")
    print(f"     SELL: {sell_count:>4} ({sell_count/max(total_preds,1)*100:.1f}%)")
    print(f"     HOLD: {hold_count:>4} ({hold_count/max(total_preds,1)*100:.1f}%)")

    if trades:
        print(f"\n  {'#':>3} {'Dir':<6} {'Entry':>8} {'Exit':>8} "
              f"{'SL':>8} {'P&L':>8} {'Entry@':>7} {'Exit@':>7} {'Reason':<12}")
        print(f"  {'─' * 72}")
        for i, t in enumerate(trades, 1):
            icon = "✅" if t['pnl'] > 0 else "❌"
            print(
                f"  {i:>3} {t['dir']:<6} {t['entry']:>8.2f} {t['exit']:>8.2f} "
                f"{t['sl']:>8.2f} {t['pnl']:>+7.2f}{icon} "
                f"{t['entry_time']:>7} {t['exit_time']:>7} {t['reason']:<12}"
            )
        print(f"  {'─' * 72}")
        print(f"  {'':>3} {'':>6} {'':>8} {'':>8} "
              f"{'TOTAL':>8} {daily_pnl:>+7.2f}")

    # Close reasons
    if trades:
        reasons = {}
        for t in trades:
            reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
        print(f"\n  🔍 Close reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            r_trades = [t for t in trades if t['reason'] == r]
            r_wr = sum(1 for t in r_trades if t['pnl'] > 0) / len(r_trades) * 100
            print(f"     {r:15s}: {c:3d} trades | Win: {r_wr:.0f}%")

    print(f"{'═' * 82}\n")


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(
        description="CatBoost Live Test Simulator — Row-by-Row Prediction"
    )
    parser.add_argument(
        "--date", required=True,
        help="Test date (YYYY-MM-DD), e.g. 2026-06-16"
    )
    parser.add_argument("--model", default=MODEL_PATH, help="CatBoost model file")
    parser.add_argument("--file-1m", default=CSV_1M, help="1-min NIFTY CSV")
    parser.add_argument("--file-2m", default=CSV_2M, help="2-min NIFTY CSV")
    parser.add_argument(
        "--step", action="store_true",
        help="Step-by-step mode (press Enter per candle)"
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Auto-play delay in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--window-start", type=str, default="09:20",
        help="Entry window start HH:MM (default: 09:20)"
    )
    parser.add_argument(
        "--window-end", type=str, default="15:15",
        help="Entry window end HH:MM (default: 15:15)"
    )
    parser.add_argument(
        "--square-off", type=str, default="15:24",
        help="Square off time HH:MM (default: 15:24)"
    )
    parser.add_argument(
        "--min-atr", type=float, default=MIN_ATR,
        help="Minimum ATR for entry (default: 6.5)"
    )
    args = parser.parse_args()

    # Parse time arguments
    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    entry_start = parse_time(args.window_start)
    entry_end = parse_time(args.window_end)
    square_off = parse_time(args.square_off)
    min_atr = args.min_atr

    test_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    # ════════════════════════════════════════════════════════
    # 1. LOAD MODEL
    # ════════════════════════════════════════════════════════
    print(f"\n🧠 Loading CatBoost model: {args.model}")
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("   Train first: python catboost_strategy.py")
        return

    model = CatBoostClassifier()
    model.load_model(args.model)
    print(f"   ✅ Loaded ({model.tree_count_} trees)")

    # Get feature names from model
    try:
        model_features = model.feature_names_
        if not model_features or len(model_features) == 0:
            raise ValueError("empty feature names")
        print(f"   Features: {len(model_features)} columns (from model)")
    except Exception:
        model_features = None
        print("   ⚠️  Model has no feature names — using column order from data")

    # ════════════════════════════════════════════════════════
    # 2. LOAD DATA
    # ════════════════════════════════════════════════════════
    print(f"\n📂 Loading data...")

    if not os.path.exists(args.file_1m):
        print(f"❌ File not found: {args.file_1m}")
        return

    df_1m = pd.read_csv(args.file_1m)
    df_1m['Time'] = pd.to_datetime(df_1m['Time']).dt.tz_localize(None)
    df_1m = df_1m.sort_values('Time').reset_index(drop=True)
    print(f"   1-min: {len(df_1m):,} candles")

    df_2m = None
    if os.path.exists(args.file_2m):
        df_2m = pd.read_csv(args.file_2m)
        df_2m['Time'] = pd.to_datetime(df_2m['Time']).dt.tz_localize(None)
        df_2m = df_2m.sort_values('Time').reset_index(drop=True)
        print(f"   2-min: {len(df_2m):,} candles")
    else:
        print(f"   ⚠️  2-min CSV not found — using 1-min features only")

    # ════════════════════════════════════════════════════════
    # 3. VALIDATE TEST DATE
    # ════════════════════════════════════════════════════════
    available_dates = sorted(df_1m['Time'].dt.date.unique())

    if test_date not in available_dates:
        print(f"\n❌ Date {test_date} not found in data!")
        print(f"   Available range: {available_dates[0]} → {available_dates[-1]}")
        nearby = [d for d in available_dates if abs((d - test_date).days) <= 7]
        if nearby:
            print(f"   Nearby dates: {', '.join(str(d) for d in nearby)}")
        return

    # ════════════════════════════════════════════════════════
    # 4. SPLIT: WARMUP + TEST DAY
    # ════════════════════════════════════════════════════════
    date_idx = list(available_dates).index(test_date)
    warmup_start_idx = max(0, date_idx - WARMUP_DAYS)
    warmup_dates = set(available_dates[warmup_start_idx:date_idx])

    # 1-min split
    warmup_mask_1m = df_1m['Time'].dt.date.isin(warmup_dates)
    test_mask_1m = df_1m['Time'].dt.date == test_date
    df_warmup_1m = df_1m[warmup_mask_1m].reset_index(drop=True)
    df_test_1m = df_1m[test_mask_1m].reset_index(drop=True)
    df_combined_1m = pd.concat([df_warmup_1m, df_test_1m], ignore_index=True)

    # 2-min split
    df_combined_2m = None
    if df_2m is not None:
        all_needed_dates = warmup_dates | {test_date}
        mask_2m = df_2m['Time'].dt.date.isin(all_needed_dates)
        df_combined_2m = df_2m[mask_2m].reset_index(drop=True)

    warmup_len = len(df_warmup_1m)
    test_candles = len(df_test_1m)

    print(f"\n📅 Test date: {test_date}")
    print(f"   Warmup: {len(warmup_dates)} days ({warmup_len:,} candles)")
    print(f"   Test day: {test_candles} candles")

    if test_candles == 0:
        print("❌ No candles found for this date!")
        return

    # ════════════════════════════════════════════════════════
    # 5. BUILD FEATURES (ONCE — all backward-looking)
    # ════════════════════════════════════════════════════════
    print(f"\n📊 Building features (backtest-identical)...")
    feat_1m = build_features_1min(df_combined_1m)
    print(f"   1-min features: {feat_1m.shape[1]} columns")

    if df_combined_2m is not None and len(df_combined_2m) > 0:
        feat_2m = build_features_2min(df_combined_2m, df_combined_1m)
        all_features = pd.concat([feat_1m, feat_2m], axis=1)
        print(f"   2-min features: {feat_2m.shape[1]} columns")
    else:
        all_features = feat_1m

    all_features = all_features.fillna(0)
    all_features = all_features.replace([np.inf, -np.inf], 0)

    # ATR series
    atr_series = calc_atr(df_combined_1m, ATR_PERIOD)

    print(f"   Total features: {all_features.shape[1]} columns")

    # Determine feature column order for model
    if model_features:
        feature_cols = list(model_features)
        # Validate all features exist
        missing = [f for f in feature_cols if f not in all_features.columns]
        if missing:
            print(f"   ⚠️  Missing features in data: {missing}")
            print(f"       These will be filled with 0")
    else:
        feature_cols = all_features.columns.tolist()

    # ════════════════════════════════════════════════════════
    # 6. ROW-BY-ROW SIMULATION
    # ════════════════════════════════════════════════════════
    print_header(
        test_date, args.window_start, args.window_end,
        args.square_off, min_atr, args.step, args.speed
    )

    position = None
    trades = []
    daily_pnl = 0.0
    wins = 0
    losses = 0
    predictions_log = []
    step_mode = args.step

    for i in range(test_candles):
        # Index in the combined DataFrame
        combined_idx = warmup_len + i

        row = df_combined_1m.iloc[combined_idx]
        candle_time = row['Time'].time()
        time_str = row['Time'].strftime('%H:%M')
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        atr_val = float(atr_series.iloc[combined_idx])

        # Get feature vector for this candle
        feat_row = all_features.iloc[combined_idx]
        feature_vector = []
        for f in feature_cols:
            v = float(feat_row.get(f, 0)) if f in feat_row.index else 0.0
            if v != v or abs(v) == float('inf'):  # NaN or inf check
                v = 0.0
            feature_vector.append(v)

        # Display features: RSI and UT Bot direction
        rsi = float(feat_row.get('rsi_1m', 50))
        ut_dir = float(feat_row.get('ut_dir_1m', 0))

        # ── Predict ──
        pred = model.predict([feature_vector]).flatten()[0]
        pred = int(pred)
        predictions_log.append(pred)

        # ── Simulate trade logic ──
        position, trades, daily_pnl, wins, losses, action_log = simulate_candle(
            candle_time, time_str, pred, close, high, low, atr_val,
            position, trades, daily_pnl, wins, losses,
            entry_start, entry_end, square_off, min_atr
        )

        # ── Display dashboard ──
        print_candle_dashboard(
            i + 1, test_candles, time_str, close, high, low,
            atr_val, rsi, ut_dir, pred, position, daily_pnl,
            wins, losses, action_log
        )

        # ── Pacing ──
        if step_mode:
            try:
                user_input = input("  ➡️  [Enter=next | q=quit | s=skip to end] → ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n⏹️  Simulation stopped.")
                break
            if user_input == 'q':
                print("\n⏹️  Simulation stopped by user.")
                break
            elif user_input == 's':
                step_mode = False
                print("  ⏩ Switching to auto-play (no delay)...\n")
        else:
            if args.speed > 0:
                time_module.sleep(args.speed)

    # ── Close any open position at end of day ──
    if position:
        last_close = float(df_combined_1m.iloc[warmup_len + test_candles - 1]['Close'])
        last_time = df_combined_1m.iloc[warmup_len + test_candles - 1]['Time'].strftime('%H:%M')
        pnl = _calc_pnl(position, last_close)
        trades.append(_make_trade(position, last_close, last_time, "DAY_END", pnl))
        daily_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        icon = "✅" if pnl > 0 else "❌"
        print(f"\n  ⏹️  DAY END — Closed {position['dir']} @ {last_close:.2f} | "
              f"P&L: {pnl:+.2f} {icon}")
        position = None

    # ── Final summary ──
    print_daily_summary(trades, daily_pnl, wins, losses, test_date, predictions_log)


if __name__ == "__main__":
    main()
