"""
overnight_strategy.py — Overnight Gap Trading Strategy

CONCEPT:
  1. At 3:15-3:20 PM, scan all stocks for strongest momentum
  2. Rank by signal strength → pick top N
  3. BUY at ~3:20 PM close price
  4. SELL at next day 9:15 AM open price
  5. Profit from overnight gap

WHY THIS WORKS:
  - Stocks with strong close tend to gap up next morning
  - After-hours news/global markets push in the direction of momentum
  - Only 1 trade per stock per day → minimal brokerage
  - No intraday noise or whipsaws
  - Clean entry/exit: close price → next open price

SIGNAL STRENGTH SCORING (at 3:15 PM):
  - EMA 5 vs EMA 21 alignment (+trend)
  - Close position in day range (near high = bullish)
  - Volume surge (>1.3x avg = institutional interest)
  - RSI momentum (not overbought/oversold)
  - ATR (enough volatility for gap)
  - Candle body strength

Usage:
    python overnight_strategy.py                              # All stocks
    python overnight_strategy.py --stock RELIANCE --detailed  # One stock
    python overnight_strategy.py --top 5                      # Top 5 picks
    python overnight_strategy.py --min-score 70               # Higher threshold
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

SCAN_TIME = dt_time(15, 15)       # Scan at 3:15 PM
ENTRY_TIME = dt_time(15, 20)      # Enter position ~3:20 PM (last candle before close)
EXIT_TIME = dt_time(9, 15)        # Exit at 9:15 AM next day open

EMA_FAST = 5
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20

# Score thresholds
MIN_SCORE = 60                    # Minimum score to take a trade (0-100)
TOP_PICKS = 5                     # Max stocks to trade per day


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


# ─────────── Day-Level Data Builder ───────────

def build_daily_signals(df):
    """
    From intraday data, build daily signals:
    For each trading day, compute indicators at SCAN_TIME (3:15 PM),
    get the ENTRY price (~3:20 PM close), and the EXIT price (next day 9:15 AM open).
    """
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    
    # Compute indicators on full data
    ema_fast = calc_ema(close, EMA_FAST)
    ema_slow = calc_ema(close, EMA_SLOW)
    atr_val = calc_atr(df, ATR_PERIOD)
    rsi_val = calc_rsi(close, RSI_PERIOD)
    vol_avg = volume.rolling(VOLUME_AVG_PERIOD).mean()
    
    # Group by day
    df['date'] = df['Time'].dt.date
    days = sorted(df['date'].unique())
    
    signals = []
    
    for i, day in enumerate(days):
        day_data = df[df['date'] == day]
        
        if len(day_data) < 10:
            continue
        
        # Find scan candle (closest to 3:15 PM)
        scan_candles = day_data[day_data['Time'].dt.time >= SCAN_TIME]
        if len(scan_candles) == 0:
            continue
        scan_idx = scan_candles.index[0]
        
        # Find entry candle (closest to 3:20 PM, last tradeable candle)
        entry_candles = day_data[day_data['Time'].dt.time >= ENTRY_TIME]
        if len(entry_candles) == 0:
            entry_idx = day_data.index[-1]  # Last candle of day
        else:
            entry_idx = entry_candles.index[0]
        
        entry_price = float(close.loc[entry_idx])
        entry_time = df.loc[entry_idx, 'Time']
        
        # Find exit: next day's first candle (9:15 AM open)
        if i + 1 >= len(days):
            continue  # No next day
        next_day = days[i + 1]
        next_day_data = df[df['date'] == next_day]
        if len(next_day_data) == 0:
            continue
        
        exit_idx = next_day_data.index[0]
        exit_price = float(open_.loc[exit_idx])
        exit_time = df.loc[exit_idx, 'Time']
        
        # ── Compute signals at scan time ──
        s_close = float(close.loc[scan_idx])
        s_open = float(open_.loc[scan_idx])
        s_high = float(high.loc[scan_idx])
        s_low = float(low.loc[scan_idx])
        s_volume = float(volume.loc[scan_idx])
        s_ema_fast = float(ema_fast.loc[scan_idx])
        s_ema_slow = float(ema_slow.loc[scan_idx])
        s_atr = float(atr_val.loc[scan_idx])
        s_rsi = float(rsi_val.loc[scan_idx]) if not np.isnan(rsi_val.loc[scan_idx]) else 50
        s_vol_avg = float(vol_avg.loc[scan_idx]) if not np.isnan(vol_avg.loc[scan_idx]) else s_volume
        
        # Day's range
        day_high = float(day_data['High'].max())
        day_low = float(day_data['Low'].min())
        day_range = day_high - day_low if day_high > day_low else 0.01
        
        # ── SCORING (0-100) ──
        
        # 1. EMA alignment (0-25)
        ema_gap_pct = (s_ema_fast - s_ema_slow) / s_close * 100
        ema_score = min(abs(ema_gap_pct) * 50, 25)  # Bigger gap = stronger trend
        
        # 2. Close position in day range (0-25)
        close_position = (s_close - day_low) / day_range  # 0=at low, 1=at high
        if ema_gap_pct > 0:  # Bullish → close near high is good
            position_score = close_position * 25
        else:  # Bearish → close near low is good
            position_score = (1 - close_position) * 25
        
        # 3. Volume surge (0-20)
        vol_ratio = s_volume / s_vol_avg if s_vol_avg > 0 else 1
        vol_score = min(vol_ratio * 10, 20)
        
        # 4. RSI momentum (0-15) — not extreme
        if ema_gap_pct > 0:  # BUY
            if 40 <= s_rsi <= 70:
                rsi_score = 15
            elif 30 <= s_rsi <= 80:
                rsi_score = 8
            else:
                rsi_score = 0
        else:  # SELL
            if 30 <= s_rsi <= 60:
                rsi_score = 15
            elif 20 <= s_rsi <= 70:
                rsi_score = 8
            else:
                rsi_score = 0
        
        # 5. ATR (enough volatility) (0-15)
        atr_pct = s_atr / s_close * 100
        atr_score = min(atr_pct * 30, 15)
        
        total_score = ema_score + position_score + vol_score + rsi_score + atr_score
        
        # Direction: BUY if bullish, SELL if bearish
        direction = "BUY" if ema_gap_pct > 0 else "SELL"
        
        # P&L
        if direction == "BUY":
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        pnl_pct = pnl_pts / entry_price * 100
        
        signals.append({
            'date': day,
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
    
    # Build daily signals
    signals = build_daily_signals(df)
    
    if len(signals) == 0:
        return None
    
    # Filter by minimum score
    qualified = signals[signals['score'] >= min_score].copy()
    all_signals = signals.copy()
    
    if len(qualified) == 0:
        if detailed:
            print(f"\n  {symbol}: {len(signals)} days scanned, 0 qualified (min score: {min_score})")
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
    
    # Buy vs Sell breakdown
    buys = qualified[qualified['direction'] == 'BUY']
    sells = qualified[qualified['direction'] == 'SELL']
    buy_wr = len(buys[buys['pnl_pts'] > 0]) / max(len(buys), 1) * 100
    sell_wr = len(sells[sells['pnl_pts'] > 0]) / max(len(sells), 1) * 100
    
    # Score buckets
    high_score = qualified[qualified['score'] >= 75]
    high_wr = len(high_score[high_score['pnl_pts'] > 0]) / max(len(high_score), 1) * 100
    
    if detailed:
        print(f"\n  {'='*65}")
        print(f"  🌙 {symbol} — OVERNIGHT STRATEGY")
        print(f"  {date_range} | {trading_days} days")
        print(f"  {'='*65}")
        
        print(f"\n  📊 SIGNALS")
        print(f"  Total days scanned: {len(signals)}")
        print(f"  Qualified (≥{min_score}): {len(qualified)} ({len(qualified)/len(signals)*100:.0f}%)")
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
        
        # Monthly breakdown
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
        
        # Last 5 trades
        print(f"\n  📋 LAST 5 TRADES")
        for _, t in qualified.tail(5).iterrows():
            icon = "✅" if t['pnl_pts'] > 0 else "❌"
            print(f"    {t['date']} {t['direction']:4s} Score:{t['score']:5.1f} | "
                  f"Entry:{t['entry_price']:8.2f} → Exit:{t['exit_price']:8.2f} | "
                  f"P&L: {t['pnl_pct']:+.4f}% {icon}")
        
        print(f"  {'='*65}")
    
    return {
        'symbol': symbol,
        'days_scanned': len(signals),
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
    parser = argparse.ArgumentParser(description="Overnight Gap Strategy Backtest")
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
    
    print(f"🌙 OVERNIGHT GAP STRATEGY — Backtest")
    print(f"Entry: ~3:20 PM (close) → Exit: 9:15 AM (next open)")
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
    
    print(f"\n{'='*110}")
    print(f"  🌙 OVERNIGHT STRATEGY RANKINGS | Min Score ≥ {args.min_score}")
    print(f"{'='*110}")
    print(f"  {'Rank':>4} {'Symbol':>12} {'Trades':>7} {'Win%':>6} {'P&L%':>9} {'AvgP&L':>8} {'PF':>5} {'AvgScr':>7} "
          f"{'BuyT':>5} {'Buy%':>5} {'SellT':>5} {'Sel%':>5} {'Hi≥75':>5}")
    print(f"  {'-'*105}")
    
    for rank, r in enumerate(sorted_results[:args.top], 1):
        star = "⭐" if rank <= 5 else "  "
        print(f"  {rank:>3}. {r['symbol']:>12} {r['trades']:>7} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.3f}% "
              f"{r['avg_pnl']:>+7.4f}% {r['profit_factor']:>5.2f} {r['avg_score']:>6.1f} "
              f"{r['buy_trades']:>5} {r['buy_wr']:>4.0f}% {r['sell_trades']:>5} {r['sell_wr']:>4.0f}% "
              f"{r['high_score_trades']:>5} {star}")
    
    print(f"\n{'='*110}")
    
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
    pd.DataFrame(save_data).to_csv(f"{RESULTS_DIR}/overnight_rankings.csv", index=False)
    
    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"{RESULTS_DIR}/overnight_all_trades.csv", index=False)
    
    top5 = sorted_results[:5]
    print(f"\n🎯 TOP 5 FOR OVERNIGHT TRADING:")
    for r in top5:
        print(f"  ⭐ {r['symbol']}: Win {r['win_rate']:.1f}% | P&L {r['pnl_pct']:+.3f}% | PF {r['profit_factor']:.2f} | {r['trades']} trades")
    print(f"\n  💾 {RESULTS_DIR}/overnight_rankings.csv")
    print(f"  💾 {RESULTS_DIR}/overnight_all_trades.csv")


if __name__ == "__main__":
    main()
