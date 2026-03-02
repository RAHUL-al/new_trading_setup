"""
overnight_strategy.py — Overnight Gap Trading Strategy (v2)

CONCEPT:
  1. Resample stock data to 5-min candles
  2. At 3:15 PM, analyze last few 5-min candles for strongest momentum
  3. Score each stock (EMA trend, close position, volume, RSI, ATR, body strength)
  4. BUY strongest / SELL weakest at ~3:20 PM
  5. EXIT at next day 9:15 AM open
  6. SKIP FRIDAYS — no position going into weekend

CHANGES FROM v1:
  - Uses 5-min candles (less noise than 1-min/3-min)
  - Skips Fridays (no weekend risk)
  - Analyzes last 3 five-minute candles for consistency
  - Stronger scoring: consecutive bullish/bearish 5-min candles = higher score

Usage:
    python overnight_strategy.py                              # All stocks
    python overnight_strategy.py --stock RELIANCE --detailed  # One stock
    python overnight_strategy.py --top 5                      # Top 5 picks
    python overnight_strategy.py --min-score 60               # Adjust threshold
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime, time as dt_time, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
DATA_DIR = "stock_data"
RESULTS_DIR = "scan_results"

SCAN_TIME = dt_time(15, 10)       # Scan 5-min candle at 3:10 PM
ENTRY_TIME = dt_time(15, 20)      # Enter position ~3:20 PM
EXIT_TIME = dt_time(9, 15)        # Exit at 9:15 AM next day open

EMA_FAST = 5
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20

# Score thresholds
MIN_SCORE = 60
TOP_PICKS = 5


# ─────────── Indicators ───────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))


# ─────────── Resample to 5-min ───────────

def resample_to_5min(df):
    """Resample 1-min or 3-min candles to 5-min."""
    df_indexed = df.set_index('Time')
    resampled = df_indexed.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return resampled.reset_index()


# ─────────── Day-Level Signal Builder (5-min) ───────────

def build_daily_signals(df):
    """
    Resample to 5-min, then for each day:
    Analyze last few 5-min candles before close for momentum.
    Skip FRIDAYS (weekday 4).
    """
    # Resample to 5-min
    df_5m = resample_to_5min(df)
    if len(df_5m) < 50:
        return pd.DataFrame()

    close = df_5m['Close'].astype(float)
    open_ = df_5m['Open'].astype(float)
    high = df_5m['High'].astype(float)
    low = df_5m['Low'].astype(float)
    volume = df_5m['Volume'].astype(float)

    # Compute indicators on 5-min data
    ema_fast = calc_ema(close, EMA_FAST)
    ema_slow = calc_ema(close, EMA_SLOW)
    atr_val = calc_atr(df_5m, ATR_PERIOD)
    rsi_val = calc_rsi(close, RSI_PERIOD)
    vol_avg = volume.rolling(VOLUME_AVG_PERIOD).mean()

    df_5m['date'] = df_5m['Time'].dt.date
    days = sorted(df_5m['date'].unique())

    # Also need 1-min data for entry/exit prices
    df['date'] = df['Time'].dt.date
    df_close = df['Close'].astype(float)
    df_open = df['Open'].astype(float)

    signals = []

    for i, day in enumerate(days):
        # ── SKIP FRIDAYS ──
        day_dt = pd.Timestamp(day)
        if day_dt.weekday() == 4:  # Friday = 4
            continue

        day_5m = df_5m[df_5m['date'] == day]
        if len(day_5m) < 6:
            continue

        # Find 5-min candle at/after SCAN_TIME (3:10 PM)
        scan_candles = day_5m[day_5m['Time'].dt.time >= SCAN_TIME]
        if len(scan_candles) == 0:
            continue
        scan_idx = scan_candles.index[0]

        # Get last 3 five-minute candles up to scan
        scan_pos = day_5m.index.get_loc(scan_idx)
        if scan_pos < 2:
            continue
        last3_idx = day_5m.index[scan_pos-2:scan_pos+1]

        # Entry price from original data (~3:20 PM)
        day_1m = df[df['date'] == day]
        entry_candles = day_1m[day_1m['Time'].dt.time >= ENTRY_TIME]
        if len(entry_candles) == 0:
            entry_idx = day_1m.index[-1]
        else:
            entry_idx = entry_candles.index[0]
        entry_price = float(df_close.loc[entry_idx])
        entry_time = df.loc[entry_idx, 'Time']

        # Exit: next trading day's first candle (9:15 AM open)
        if i + 1 >= len(days):
            continue
        next_day = days[i + 1]
        next_day_1m = df[df['date'] == next_day]
        if len(next_day_1m) == 0:
            continue
        exit_idx = next_day_1m.index[0]
        exit_price = float(df_open.loc[exit_idx])
        exit_time = df.loc[exit_idx, 'Time']

        # ── Compute signals from 5-min data at scan time ──
        s_close = float(close.loc[scan_idx])
        s_ema_fast = float(ema_fast.loc[scan_idx])
        s_ema_slow = float(ema_slow.loc[scan_idx])
        s_atr = float(atr_val.loc[scan_idx])
        s_rsi = float(rsi_val.loc[scan_idx]) if not np.isnan(rsi_val.loc[scan_idx]) else 50
        s_vol = float(volume.loc[scan_idx])
        s_vol_avg = float(vol_avg.loc[scan_idx]) if not np.isnan(vol_avg.loc[scan_idx]) else s_vol

        # Day's range from 5-min data
        day_high = float(day_5m['High'].max())
        day_low = float(day_5m['Low'].min())
        day_range = day_high - day_low if day_high > day_low else 0.01

        # ── SCORING (0-100) ──

        # 1. EMA alignment (0-20) — trend direction
        ema_gap_pct = (s_ema_fast - s_ema_slow) / s_close * 100
        ema_score = min(abs(ema_gap_pct) * 40, 20)

        # 2. Close position in day range (0-20)
        close_position = (s_close - day_low) / day_range
        if ema_gap_pct > 0:
            position_score = close_position * 20
        else:
            position_score = (1 - close_position) * 20

        # 3. Volume surge (0-15)
        vol_ratio = s_vol / s_vol_avg if s_vol_avg > 0 else 1
        vol_score = min(vol_ratio * 7.5, 15)

        # 4. RSI momentum (0-15) — healthy zone, not extreme
        if ema_gap_pct > 0:  # BUY
            if 45 <= s_rsi <= 70:
                rsi_score = 15
            elif 35 <= s_rsi <= 80:
                rsi_score = 8
            else:
                rsi_score = 0
        else:  # SELL
            if 30 <= s_rsi <= 55:
                rsi_score = 15
            elif 20 <= s_rsi <= 65:
                rsi_score = 8
            else:
                rsi_score = 0

        # 5. ATR (enough volatility for gap) (0-10)
        atr_pct = s_atr / s_close * 100
        atr_score = min(atr_pct * 20, 10)

        # 6. Last 3 five-min candles consistency (0-20) ← NEW
        #    All 3 bullish or all 3 bearish = strong conviction
        last3_bullish = 0
        last3_bearish = 0
        last3_body_pct = 0
        for idx in last3_idx:
            c_c = float(close.loc[idx])
            c_o = float(open_.loc[idx])
            c_h = float(high.loc[idx])
            c_l = float(low.loc[idx])
            rng = c_h - c_l if c_h > c_l else 0.01
            body_pct = abs(c_c - c_o) / rng
            last3_body_pct += body_pct
            if c_c > c_o:
                last3_bullish += 1
            else:
                last3_bearish += 1

        avg_body = last3_body_pct / 3
        if ema_gap_pct > 0:  # BUY direction
            consistency = last3_bullish / 3  # 1.0 = all 3 bullish
        else:  # SELL direction
            consistency = last3_bearish / 3
        consistency_score = consistency * 15 + avg_body * 5  # 0-20
        consistency_score = min(consistency_score, 20)

        total_score = ema_score + position_score + vol_score + rsi_score + atr_score + consistency_score

        # Direction
        direction = "BUY" if ema_gap_pct > 0 else "SELL"

        # P&L
        if direction == "BUY":
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        pnl_pct = pnl_pts / entry_price * 100

        signals.append({
            'date': day,
            'weekday': day_dt.strftime('%A'),
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl_pts': round(pnl_pts, 2),
            'pnl_pct': round(pnl_pct, 4),
            'score': round(total_score, 1),
            'ema_gap': round(ema_gap_pct, 3),
            'close_pos': round(close_position, 2),
            'vol_ratio': round(vol_ratio, 2),
            'rsi': round(s_rsi, 1),
            'atr_pct': round(atr_pct, 3),
            'consistency': round(consistency, 2),
            'avg_body': round(avg_body, 2),
        })

    return pd.DataFrame(signals)


# ─────────── Backtest ───────────

def backtest_stock(symbol, file_path, min_score=60, detailed=False):
    """Backtest overnight strategy for one stock."""
    df = pd.read_csv(file_path)
    if len(df) < 200:
        return None

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    if 'Volume' not in df.columns:
        return None

    trading_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"

    signals = build_daily_signals(df)
    if len(signals) == 0:
        return None

    # Count Fridays skipped
    all_days = df['Time'].dt.date.unique()
    friday_count = sum(1 for d in all_days if pd.Timestamp(d).weekday() == 4)

    qualified = signals[signals['score'] >= min_score].copy()

    if len(qualified) == 0:
        if detailed:
            print(f"\n  {symbol}: {len(signals)} days (excl {friday_count} Fri), 0 qualified (score≥{min_score})")
            if len(signals) > 0:
                print(f"  Score range: {signals['score'].min():.1f} - {signals['score'].max():.1f}")
        return None

    # Metrics
    wins = qualified[qualified['pnl_pts'] > 0]
    losses = qualified[qualified['pnl_pts'] <= 0]
    win_rate = len(wins) / len(qualified) * 100
    total_pnl = qualified['pnl_pct'].sum()
    avg_pnl = qualified['pnl_pct'].mean()

    gp = wins['pnl_pts'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_pts'].sum()) if len(losses) > 0 else 1
    pf = gp / gl if gl > 0 else 0

    buys = qualified[qualified['direction'] == 'BUY']
    sells = qualified[qualified['direction'] == 'SELL']
    buy_wr = len(buys[buys['pnl_pts'] > 0]) / max(len(buys), 1) * 100
    sell_wr = len(sells[sells['pnl_pts'] > 0]) / max(len(sells), 1) * 100

    high_score = qualified[qualified['score'] >= 75]
    high_wr = len(high_score[high_score['pnl_pts'] > 0]) / max(len(high_score), 1) * 100

    if detailed:
        print(f"\n  {'='*70}")
        print(f"  🌙 {symbol} — OVERNIGHT STRATEGY (5-min, no Fridays)")
        print(f"  {date_range} | {trading_days} days | {friday_count} Fridays skipped")
        print(f"  {'='*70}")

        print(f"\n  📊 SIGNALS")
        print(f"  Total days scanned: {len(signals)} (excl Fridays)")
        print(f"  Qualified (≥{min_score}): {len(qualified)} ({len(qualified)/max(len(signals),1)*100:.0f}%)")
        if len(qualified) > 0:
            print(f"  Score range:        {qualified['score'].min():.1f} - {qualified['score'].max():.1f}")
            print(f"  Avg score:          {qualified['score'].mean():.1f}")

        print(f"\n  💰 PERFORMANCE")
        print(f"  Total trades:       {len(qualified)}")
        print(f"  Win rate:           {win_rate:.1f}%")
        print(f"  Profit factor:      {pf:.2f}")
        print(f"  Total P&L:          {total_pnl:+.3f}%")
        print(f"  Avg P&L/trade:      {avg_pnl:+.4f}%")
        print(f"  Best trade:         {qualified['pnl_pct'].max():+.4f}%")
        print(f"  Worst trade:        {qualified['pnl_pct'].min():+.4f}%")

        print(f"\n  📈 DIRECTION BREAKDOWN")
        print(f"  BUY trades:   {len(buys):4d} | Win: {buy_wr:.1f}% | P&L: {buys['pnl_pct'].sum():+.3f}%")
        print(f"  SELL trades:  {len(sells):4d} | Win: {sell_wr:.1f}% | P&L: {sells['pnl_pct'].sum():+.3f}%")

        if len(high_score) > 0:
            print(f"\n  ⭐ HIGH SCORE (≥75)")
            print(f"  Trades: {len(high_score)} | Win: {high_wr:.1f}% | P&L: {high_score['pnl_pct'].sum():+.3f}%")

        # Monthly
        qualified['month'] = pd.to_datetime(qualified['date']).dt.to_period('M')
        monthly = qualified.groupby('month').agg(
            trades=('pnl_pts', 'count'),
            pnl=('pnl_pct', 'sum'),
            wr=('pnl_pts', lambda x: (x > 0).mean() * 100)
        )
        print(f"\n  📅 MONTHLY")
        for month, row in monthly.iterrows():
            icon = "✅" if row['pnl'] > 0 else "❌"
            print(f"    {str(month):10s}: {row['trades']:3.0f} trades | Win: {row['wr']:4.0f}% | P&L: {row['pnl']:+.3f}% {icon}")

        # Day-of-week analysis
        qualified['dow'] = pd.to_datetime(qualified['date']).dt.day_name()
        dow_stats = qualified.groupby('dow').agg(
            trades=('pnl_pts', 'count'),
            pnl=('pnl_pct', 'sum'),
            wr=('pnl_pts', lambda x: (x > 0).mean() * 100)
        )
        print(f"\n  📆 DAY-OF-WEEK (entry day)")
        for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday']:
            if dow in dow_stats.index:
                r = dow_stats.loc[dow]
                icon = "✅" if r['pnl'] > 0 else "❌"
                print(f"    {dow:12s}: {r['trades']:3.0f} trades | Win: {r['wr']:4.0f}% | P&L: {r['pnl']:+.3f}% {icon}")

        # Last 5 trades
        print(f"\n  📋 LAST 5 TRADES")
        for _, t in qualified.tail(5).iterrows():
            icon = "✅" if t['pnl_pts'] > 0 else "❌"
            print(f"    {t['date']} ({t['weekday'][:3]}) {t['direction']:4s} Score:{t['score']:5.1f} | "
                  f"Entry:{t['entry_price']:8.2f} → Exit:{t['exit_price']:8.2f} | "
                  f"P&L: {t['pnl_pct']:+.4f}% {icon}")

        print(f"  {'='*70}")

    return {
        'symbol': symbol,
        'days_scanned': len(signals),
        'fridays_skipped': friday_count,
        'trades': len(qualified),
        'win_rate': round(win_rate, 1),
        'pnl_pct': round(total_pnl, 3),
        'avg_pnl': round(avg_pnl, 4),
        'profit_factor': round(pf, 2),
        'avg_score': round(qualified['score'].mean(), 1),
        'buy_trades': len(buys),
        'buy_wr': round(buy_wr, 1),
        'sell_trades': len(sells),
        'sell_wr': round(sell_wr, 1),
        'high_score_trades': len(high_score),
        'high_score_wr': round(high_wr, 1),
        'all_trades': qualified.to_dict('records'),
    }


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(description="Overnight Strategy v2 (5-min, no Fridays)")
    parser.add_argument("--stock", default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=MIN_SCORE)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find files
    if args.stock:
        stock_upper = args.stock.upper()
        all_files = glob.glob(f"{args.data_dir}/*_*min.csv")
        files = []
        for f in all_files:
            sym = os.path.basename(f).split('_')[0].upper()
            if sym == stock_upper:
                files.append(f)
                break
    else:
        files = sorted(set(glob.glob(f"{args.data_dir}/*_*min.csv")))

    if not files:
        print(f"❌ No data in {args.data_dir}/. Run fetch_stock_data.py first.")
        return

    print(f"🌙 OVERNIGHT STRATEGY v2 — 5-min candles, no Fridays")
    print(f"Entry: ~3:20 PM → Exit: next day 9:15 AM open")
    print(f"5-min candle analysis | Last 3 candle consistency check")
    print(f"Fridays SKIPPED (no weekend risk)")
    print(f"Min score: {args.min_score} | Stocks: {len(files)}")
    print(f"{'='*60}")

    results = []
    all_trades = []

    for idx, fp in enumerate(sorted(files)):
        symbol = os.path.basename(fp).split("_")[0]
        print(f"[{idx+1}/{len(files)}] {symbol}...", end=" ", flush=True)

        try:
            result = backtest_stock(symbol, fp, args.min_score,
                                     detailed=args.detailed or bool(args.stock))
            if result:
                results.append(result)
                print(f"{result['trades']} trades | Win: {result['win_rate']:.1f}% | "
                      f"P&L: {result['pnl_pct']:+.3f}% | PF: {result['profit_factor']:.2f}")

                for t in result['all_trades']:
                    t_copy = t.copy()
                    t_copy['symbol'] = symbol
                    all_trades.append(t_copy)
            else:
                print("⚠️ No qualified signals")
        except Exception as e:
            print(f"❌ {e}")

    if not results:
        print("❌ No results. Try lowering --min-score")
        return

    # Ranking
    sorted_results = sorted(results, key=lambda r: r['pnl_pct'], reverse=True)

    print(f"\n{'='*115}")
    print(f"  🌙 OVERNIGHT v2 RANKINGS | 5-min | No Fridays | Score ≥ {args.min_score}")
    print(f"{'='*115}")
    print(f"  {'Rank':>4} {'Symbol':>12} {'Trades':>7} {'Win%':>6} {'P&L%':>9} {'AvgP&L':>8} {'PF':>5} {'AvgScr':>7} "
          f"{'BuyT':>5} {'Buy%':>5} {'SellT':>5} {'Sel%':>5} {'Hi≥75':>5}")
    print(f"  {'-'*110}")

    for rank, r in enumerate(sorted_results[:args.top], 1):
        star = "⭐" if rank <= 5 else "  "
        print(f"  {rank:>3}. {r['symbol']:>12} {r['trades']:>7} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.3f}% "
              f"{r['avg_pnl']:>+7.4f}% {r['profit_factor']:>5.2f} {r['avg_score']:>6.1f} "
              f"{r['buy_trades']:>5} {r['buy_wr']:>4.0f}% {r['sell_trades']:>5} {r['sell_wr']:>4.0f}% "
              f"{r['high_score_trades']:>5} {star}")

    print(f"\n{'='*115}")

    # Summary
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(int(r['trades'] * r['win_rate'] / 100) for r in results)
    overall_wr = total_wins / max(total_trades, 1) * 100
    profitable = [r for r in results if r['pnl_pct'] > 0]

    print(f"\n  📊 OVERALL SUMMARY")
    print(f"  Stocks tested:     {len(results)}")
    print(f"  Total trades:      {total_trades}")
    print(f"  Overall win rate:  {overall_wr:.1f}%")
    print(f"  Profitable stocks: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.0f}%)")

    # Best picks
    print(f"\n  🏆 HIGHLIGHTS")
    best_wr = max(results, key=lambda r: r['win_rate'])
    best_pnl = max(results, key=lambda r: r['pnl_pct'])
    best_pf = max(results, key=lambda r: r['profit_factor'])
    print(f"  Best Win Rate:  {best_wr['symbol']:>12} ({best_wr['win_rate']:.1f}%)")
    print(f"  Best P&L:       {best_pnl['symbol']:>12} ({best_pnl['pnl_pct']:+.3f}%)")
    print(f"  Best PF:        {best_pf['symbol']:>12} ({best_pf['profit_factor']:.2f})")

    # Save
    save_data = [{k: v for k, v in r.items() if k != 'all_trades'} for r in sorted_results]
    pd.DataFrame(save_data).to_csv(f"{RESULTS_DIR}/overnight_v2_rankings.csv", index=False)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"{RESULTS_DIR}/overnight_v2_all_trades.csv", index=False)

    top5 = sorted_results[:5]
    print(f"\n🎯 TOP 5 FOR OVERNIGHT TRADING (Mon-Thu only):")
    for r in top5:
        print(f"  ⭐ {r['symbol']}: Win {r['win_rate']:.1f}% | P&L {r['pnl_pct']:+.3f}% | PF {r['profit_factor']:.2f} | {r['trades']} trades")
    print(f"\n  💾 {RESULTS_DIR}/overnight_v2_rankings.csv")
    print(f"  💾 {RESULTS_DIR}/overnight_v2_all_trades.csv")


if __name__ == "__main__":
    main()
