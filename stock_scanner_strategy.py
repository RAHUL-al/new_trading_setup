"""
stock_scanner_strategy.py — Scan NIFTY 50 → Pick Top Stocks → Backtest with Volume Strategy

PIPELINE:
  Phase 1: SCAN  — Load all stock CSVs from stock_data/
  Phase 2: SCORE — Rate each stock on trend, volume, signal quality, ATR
  Phase 3: RANK  — Select top 5-10 stocks
  Phase 4: TEST  — Full backtest on selected stocks with volume-enhanced strategy

STRATEGY: EMA 5/21 Crossover + Volume Confirmation
  BUY:
    - EMA(5) crosses above EMA(21)
    - Volume ≥ 1.5× 20-candle average (institutional interest)
    - Close > VWAP (price at premium = demand)
    - OBV rising (accumulation)
    - ATR > dynamic threshold
    - Candle body > 40% of range (conviction)
    - RSI < 75 (not overbought)

  SELL: Mirror conditions

Usage:
    python stock_scanner_strategy.py                       # Full pipeline
    python stock_scanner_strategy.py --top 5               # Show top 5
    python stock_scanner_strategy.py --stock RELIANCE      # Test one stock
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime, time as dt_time
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings('ignore')


# ─────────── Config ───────────
DATA_DIR = "stock_data"
RESULTS_DIR = "scan_results"

WINDOW_1_START = dt_time(9, 15)
WINDOW_1_END = dt_time(10, 30)
WINDOW_2_START = dt_time(13, 0)
WINDOW_2_END = dt_time(15, 15)
SQUARE_OFF_TIME = dt_time(15, 24)

EMA_FAST = 5
EMA_SLOW = 21

SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.2
MIN_HOLD = 2
BODY_RATIO_MIN = 0.40
VOL_SURGE_MULT = 1.5     # Volume must be ≥ 1.5x 20-candle average


# ─────────── Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    entry_idx: int
    entry_atr: float

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_pts: float
    pnl_pct: float
    close_reason: str

@dataclass
class StockScore:
    symbol: str
    total_candles: int
    trading_days: int
    avg_volume: float
    avg_atr: float
    atr_pct: float           # ATR as % of price (normalized)
    trend_score: float       # 0-100
    volume_score: float      # 0-100
    signal_score: float      # 0-100 (backtest win rate)
    liquidity_score: float   # 0-100
    total_score: float       # Weighted average
    total_trades: int
    win_rate: float
    total_pnl_pct: float     # Total P&L as %
    profit_factor: float
    avg_pnl_pct: float       # Avg P&L per trade as %


# ─────────── Indicators ───────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr.ewm(alpha=1/period, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))


def vwap(df):
    """VWAP — Volume Weighted Average Price (intraday, resets daily)."""
    df = df.copy()
    df['date'] = df['Time'].dt.date
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).groupby(df['date']).cumsum()
    cum_vol = df['Volume'].groupby(df['date']).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def obv(close, volume):
    """On Balance Volume — cumulative volume flow."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ─────────── Signal Generation ───────────

def generate_signals(df):
    """EMA crossover + Volume + VWAP + OBV confirmation."""
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    n = len(df)

    # Indicators
    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)
    atr_val = atr(df)
    rsi_val = rsi(close)
    vwap_val = vwap(df)
    obv_val = obv(close, volume)
    obv_ema = ema(obv_val, 10)
    vol_avg = volume.rolling(20).mean()

    # Candle quality
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    body_ratio = body / full_range

    # EMA difference
    ema_diff = ema_fast - ema_slow

    # Dynamic ATR threshold (percentile)
    atr_threshold = atr_val.rolling(100).quantile(0.25)

    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)

    for i in range(3, n):
        # Candle quality
        br = body_ratio.iloc[i]
        if np.isnan(br) or br < BODY_RATIO_MIN:
            continue

        curr_atr = atr_val.iloc[i]
        curr_vol = volume.iloc[i]
        avg_vol = vol_avg.iloc[i]

        if np.isnan(curr_atr) or np.isnan(avg_vol) or avg_vol == 0:
            continue

        # Dynamic ATR gate
        atr_thresh = atr_threshold.iloc[i]
        if np.isnan(atr_thresh):
            atr_thresh = curr_atr * 0.5
        if curr_atr < atr_thresh:
            continue

        # Volume surge check
        vol_surge = curr_vol >= (avg_vol * VOL_SURGE_MULT)

        # VWAP position
        curr_vwap = vwap_val.iloc[i] if i < len(vwap_val) else np.nan

        # OBV trend
        obv_rising = False
        obv_falling = False
        if not np.isnan(obv_val.iloc[i]) and not np.isnan(obv_ema.iloc[i]):
            obv_rising = obv_val.iloc[i] > obv_ema.iloc[i]
            obv_falling = obv_val.iloc[i] < obv_ema.iloc[i]

        # RSI
        r = rsi_val.iloc[i]

        # EMA crossover (within last 3 candles)
        cross_up = False
        cross_down = False
        for lb in range(3):
            j = i - lb
            if j >= 1:
                if ema_diff.iloc[j] > 0 and ema_diff.iloc[j-1] <= 0:
                    cross_up = True
                if ema_diff.iloc[j] < 0 and ema_diff.iloc[j-1] >= 0:
                    cross_down = True

        # ─── BUY SIGNAL ───
        if cross_up and ema_fast.iloc[i] > ema_slow.iloc[i]:
            if close.iloc[i] > open_.iloc[i]:          # Bullish candle
                if vol_surge:                           # Volume confirmation
                    buy_score = 0
                    if not np.isnan(curr_vwap) and close.iloc[i] > curr_vwap:
                        buy_score += 1                  # Above VWAP
                    if obv_rising:
                        buy_score += 1                  # OBV accumulation
                    if not np.isnan(r) and r < 75:
                        buy_score += 1                  # Not overbought

                    if buy_score >= 2:                  # At least 2 of 3 confirmations
                        buy_signal[i] = True

        # ─── SELL SIGNAL ───
        if cross_down and ema_fast.iloc[i] < ema_slow.iloc[i]:
            if close.iloc[i] < open_.iloc[i]:           # Bearish candle
                if vol_surge:                            # Volume confirmation
                    sell_score = 0
                    if not np.isnan(curr_vwap) and close.iloc[i] < curr_vwap:
                        sell_score += 1                  # Below VWAP
                    if obv_falling:
                        sell_score += 1                  # OBV distribution
                    if not np.isnan(r) and r > 25:
                        sell_score += 1                  # Not oversold

                    if sell_score >= 2:
                        sell_signal[i] = True

    return buy_signal, sell_signal, atr_val


# ─────────── Backtest ───────────

def in_window(t):
    return (WINDOW_1_START <= t <= WINDOW_1_END) or (WINDOW_2_START <= t <= WINDOW_2_END)


def run_backtest(df, buy_sig, sell_sig, atr_val):
    trades = []
    pos = None
    prev_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time()
        curr_date = row['Time'].date()
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        curr_atr = float(atr_val.iloc[i])

        # Day reset
        if prev_date and curr_date != prev_date:
            if pos:
                prev_close = float(df.iloc[i-1]['Close'])
                pnl = _pnl(pos, prev_close)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, prev_close,
                                    pos.entry_time, df.iloc[i-1]['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "DAY_END"))
                pos = None
        prev_date = curr_date

        # Morning close
        if pos and t > WINDOW_1_END and t < WINDOW_2_START:
            if pos.entry_time.time() <= WINDOW_1_END:
                pnl = _pnl(pos, close)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, close,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "W1_CLOSE"))
                pos = None
            continue

        # Square off
        if pos and t >= SQUARE_OFF_TIME:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "SQUARE_OFF"))
            pos = None
            continue

        if not (WINDOW_1_START <= t <= SQUARE_OFF_TIME):
            continue

        # SL check
        if pos:
            if pos.direction == "LONG" and low <= pos.stop_loss:
                pnl = _pnl(pos, pos.stop_loss)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, pos.stop_loss,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "STOP_LOSS"))
                pos = None
            elif pos and pos.direction == "SHORT" and high >= pos.stop_loss:
                pnl = _pnl(pos, pos.stop_loss)
                pnl_pct = pnl / pos.entry_price * 100
                trades.append(Trade(pos.direction, pos.entry_price, pos.stop_loss,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl_pct, 3), "STOP_LOSS"))
                pos = None

        # Trail SL
        if pos and curr_atr > 0:
            if pos.direction == "LONG":
                new_sl = high - curr_atr * TRAIL_ATR_MULT
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
            elif pos.direction == "SHORT":
                new_sl = low + curr_atr * TRAIL_ATR_MULT
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl

        # Opposite signal
        is_buy = bool(buy_sig[i])
        is_sell = bool(sell_sig[i])

        if is_buy and pos and pos.direction == "SHORT" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "OPPOSITE"))
            pos = None
        elif is_sell and pos and pos.direction == "LONG" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            pnl_pct = pnl / pos.entry_price * 100
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl_pct, 3), "OPPOSITE"))
            pos = None

        # New entry
        if not pos and in_window(t):
            if is_buy and curr_atr > 0:
                sl = close - curr_atr * SL_ATR_MULT
                pos = Position("LONG", close, row['Time'], sl, i, curr_atr)
            elif is_sell and curr_atr > 0:
                sl = close + curr_atr * SL_ATR_MULT
                pos = Position("SHORT", close, row['Time'], sl, i, curr_atr)

    return trades


def _pnl(pos, exit_price):
    return (exit_price - pos.entry_price) if pos.direction == "LONG" else (pos.entry_price - exit_price)


# ─────────── Scoring ───────────

def score_stock(symbol, df, trades):
    """Score a stock on multiple dimensions."""
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    # Trend score: how often is EMA(20) > EMA(50) or < EMA(50) consistently
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema_aligned = ((ema20 > ema50) | (ema20 < ema50)).sum() / len(df)
    # Strong trend = runs of same direction
    ema_direction = (ema20 > ema50).astype(int)
    runs = (ema_direction != ema_direction.shift(1)).cumsum()
    avg_run = ema_direction.groupby(runs).count().mean()
    trend_score = min(avg_run / 50 * 100, 100)  # Longer runs = better trend

    # Volume score: consistency + magnitude
    vol_cv = volume.std() / volume.mean() if volume.mean() > 0 else 10
    vol_nonzero = (volume > 0).mean()
    volume_score = min(vol_nonzero * 60 + (1 - min(vol_cv, 3) / 3) * 40, 100)

    # Signal score: win rate from backtest
    if trades:
        win_rate = sum(1 for t in trades if t.pnl_pts > 0) / len(trades) * 100
        signal_score = min(win_rate * 1.5, 100)  # Scale up
    else:
        signal_score = 0

    # Liquidity: avg volume × avg price
    avg_turnover = volume.mean() * close.mean()
    liquidity_score = min(avg_turnover / 1e7 * 100, 100)  # Normalize to 10M

    # ATR % of price
    atr_val = atr(df)
    atr_pct = (atr_val / close * 100).mean()

    # Total weighted score
    total_score = (
        trend_score * 0.25 +
        volume_score * 0.25 +
        signal_score * 0.30 +
        liquidity_score * 0.20
    )

    # Backtest metrics
    if trades:
        pnl_list = [t.pnl_pct for t in trades]
        wins = [t for t in trades if t.pnl_pts > 0]
        losses = [t for t in trades if t.pnl_pts <= 0]
        gross_profit = sum(t.pnl_pts for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_pts for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        pnl_list = [0]
        pf = 0
        win_rate = 0

    return StockScore(
        symbol=symbol,
        total_candles=len(df),
        trading_days=df['Time'].dt.date.nunique(),
        avg_volume=volume.mean(),
        avg_atr=atr_val.mean(),
        atr_pct=round(atr_pct, 3),
        trend_score=round(trend_score, 1),
        volume_score=round(volume_score, 1),
        signal_score=round(signal_score, 1),
        liquidity_score=round(liquidity_score, 1),
        total_score=round(total_score, 1),
        total_trades=len(trades),
        win_rate=round(win_rate if trades else 0, 1),
        total_pnl_pct=round(sum(pnl_list), 3),
        profit_factor=round(pf, 2),
        avg_pnl_pct=round(np.mean(pnl_list), 4),
    )


# ─────────── Reports ───────────

def print_stock_report(symbol, trades, total_candles, trading_days, date_range):
    """Detailed report for a single stock."""
    if not trades:
        print(f"  {symbol}: No trades")
        return

    pnl_pts = [t.pnl_pts for t in trades]
    pnl_pct = [t.pnl_pct for t in trades]
    total_pnl = sum(pnl_pts)
    total_pnl_pct = sum(pnl_pct)
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]
    wr = len(wins) / len(trades) * 100

    daily = {}
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d")
        daily[d] = daily.get(d, 0) + t.pnl_pts
    prof_days = sum(1 for v in daily.values() if v > 0)

    gp = sum(t.pnl_pts for t in wins) if wins else 0
    gl = abs(sum(t.pnl_pts for t in losses)) if losses else 1
    pf = gp / gl if gl > 0 else 0

    print(f"\n  {'─'*50}")
    print(f"  📈 {symbol} | {date_range}")
    print(f"  {trading_days} days | {total_candles} candles | {len(trades)} trades")
    print(f"  P&L: {total_pnl:+.2f} pts ({total_pnl_pct:+.3f}%)")
    print(f"  Win: {wr:.1f}% | PF: {pf:.2f} | Avg: {np.mean(pnl_pct):+.4f}%/trade")
    print(f"  Prof days: {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    # Close reasons
    reasons = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
    reason_str = " | ".join(f"{r}:{c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1]))
    print(f"  Reasons: {reason_str}")


def print_ranking(scores, top_n=10):
    """Print ranked stock table."""
    sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)

    print(f"\n{'='*100}")
    print(f"  STOCK SCANNER RESULTS — Top {min(top_n, len(sorted_scores))} Stocks")
    print(f"{'='*100}")
    print(f"  {'Rank':>4} {'Symbol':>12} {'Score':>6} {'Trades':>7} {'Win%':>6} {'P&L%':>8} {'PF':>5} {'Trend':>6} {'Volume':>7} {'Signal':>7} {'Liq':>5} {'ATR%':>6}")
    print(f"  {'-'*95}")

    for rank, s in enumerate(sorted_scores[:top_n], 1):
        star = "⭐" if rank <= 5 else "  "
        print(f"  {rank:>3}. {s.symbol:>12} {s.total_score:>6.1f} {s.total_trades:>7} {s.win_rate:>5.1f}% {s.total_pnl_pct:>+7.2f}% {s.profit_factor:>5.2f} {s.trend_score:>5.1f} {s.volume_score:>6.1f} {s.signal_score:>6.1f} {s.liquidity_score:>4.1f} {s.atr_pct:>5.2f}% {star}")

    print(f"\n{'='*100}")

    # Best for each dimension
    print(f"\n  🏆 BEST PER CATEGORY")
    best_trend = max(sorted_scores, key=lambda s: s.trend_score)
    best_vol = max(sorted_scores, key=lambda s: s.volume_score)
    best_signal = max(sorted_scores, key=lambda s: s.signal_score)
    best_pnl = max(sorted_scores, key=lambda s: s.total_pnl_pct)
    best_pf = max(sorted_scores, key=lambda s: s.profit_factor)
    print(f"  Best Trend:   {best_trend.symbol:>12} ({best_trend.trend_score:.1f})")
    print(f"  Best Volume:  {best_vol.symbol:>12} ({best_vol.volume_score:.1f})")
    print(f"  Best Signal:  {best_signal.symbol:>12} ({best_signal.signal_score:.1f})")
    print(f"  Best P&L:     {best_pnl.symbol:>12} ({best_pnl.total_pnl_pct:+.2f}%)")
    print(f"  Best PF:      {best_pf.symbol:>12} ({best_pf.profit_factor:.2f})")

    return sorted_scores


# ─────────── Main ───────────

def main():
    parser = argparse.ArgumentParser(description="Stock Scanner + Volume Strategy")
    parser.add_argument("--top", type=int, default=10, help="Show top N stocks (default: 10)")
    parser.add_argument("--stock", default=None, help="Test only one stock")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--detailed", action="store_true", help="Show detailed report per stock")
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find stock CSV files
    if args.stock:
        patterns = [
            f"{data_dir}/{args.stock}_*.csv",
            f"{data_dir}/{args.stock.upper()}_*.csv"
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
    else:
        files = glob.glob(f"{data_dir}/*_*min.csv")

    if not files:
        print(f"❌ No stock data in {data_dir}/. Run fetch_stock_data.py first.")
        return

    print(f"Found {len(files)} stock data files in {data_dir}/")
    print(f"{'='*60}")

    scores = []
    all_trades_data = []

    for idx, file_path in enumerate(sorted(files)):
        fname = os.path.basename(file_path)
        symbol = fname.split("_")[0]

        print(f"[{idx+1}/{len(files)}] Scanning {symbol}...", end=" ", flush=True)

        try:
            df = pd.read_csv(file_path)
            if len(df) < 200:
                print(f"⚠️ Too few candles ({len(df)}), skipping")
                continue

            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)

            # Ensure Volume column exists
            if 'Volume' not in df.columns:
                print(f"⚠️ No Volume column, skipping")
                continue

            trading_days = df['Time'].dt.date.nunique()
            date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"

            # Generate signals & backtest
            buy_sig, sell_sig, atr_val = generate_signals(df)
            trades = run_backtest(df, buy_sig, sell_sig, atr_val)

            # Score
            score = score_stock(symbol, df, trades)
            scores.append(score)

            print(f"{len(trades)} trades | Win: {score.win_rate:.1f}% | P&L: {score.total_pnl_pct:+.2f}%")

            if args.detailed or args.stock:
                print_stock_report(symbol, trades, len(df), trading_days, date_range)

            # Save trades
            if trades:
                for t in trades:
                    all_trades_data.append({
                        'symbol': symbol, 'direction': t.direction,
                        'entry_price': t.entry_price, 'exit_price': t.exit_price,
                        'pnl_pts': t.pnl_pts, 'pnl_pct': t.pnl_pct,
                        'entry_time': t.entry_time, 'exit_time': t.exit_time,
                        'close_reason': t.close_reason,
                    })

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    if not scores:
        print("❌ No stocks scored. Check your data.")
        return

    # Rank stocks
    sorted_scores = print_ranking(scores, args.top)

    # Save results
    scores_df = pd.DataFrame([{
        'rank': i+1, 'symbol': s.symbol, 'total_score': s.total_score,
        'trades': s.total_trades, 'win_rate': s.win_rate,
        'pnl_pct': s.total_pnl_pct, 'profit_factor': s.profit_factor,
        'trend': s.trend_score, 'volume': s.volume_score,
        'signal': s.signal_score, 'liquidity': s.liquidity_score,
        'atr_pct': s.atr_pct, 'avg_volume': round(s.avg_volume),
    } for i, s in enumerate(sorted_scores)])
    scores_file = os.path.join(RESULTS_DIR, "stock_rankings.csv")
    scores_df.to_csv(scores_file, index=False)
    print(f"\n💾 Rankings: {scores_file}")

    if all_trades_data:
        trades_file = os.path.join(RESULTS_DIR, "all_stock_trades.csv")
        pd.DataFrame(all_trades_data).to_csv(trades_file, index=False)
        print(f"💾 All trades: {trades_file}")

    # Summary
    top5 = sorted_scores[:5]
    print(f"\n🎯 RECOMMENDED TOP 5 STOCKS FOR TRADING:")
    for s in top5:
        print(f"  ⭐ {s.symbol}: Score {s.total_score:.1f} | {s.total_trades} trades | Win {s.win_rate:.1f}% | PF {s.profit_factor:.2f}")


if __name__ == "__main__":
    main()
