"""
ml_stock_strategy.py — XGBoost ML Model for Ultra-Selective Stock Trading

HOW THIS IS DIFFERENT FROM FIXED-RULE STRATEGIES:
  Fixed rules (EMA cross, ORB) apply the SAME logic every time.
  ML learns WHICH CONDITIONS lead to profitable trades from YOUR data.
  It finds patterns humans can't see — complex interactions between
  volume, price, time, and momentum.

FEATURES (40+ including volume-based):
  - Price: returns, gaps, candle body ratio, wick ratios
  - Trend: EMA slopes, EMA gaps, price vs EMA distance
  - Momentum: RSI, MACD histogram, rate of change
  - Volume: surge ratio, OBV slope, VWAP distance, volume trend
  - Volatility: ATR, Bollinger width, intraday range
  - Time: hour, minute bucket, day of week
  - Context: prev candle direction, consecutive same-direction candles

PIPELINE:
  1. Load stock data (3-min, 2 years)
  2. Engineer 40+ features per candle
  3. Label: next N candles move > threshold → BUY/SELL
  4. Train XGBoost on first ~18 months
  5. Test on last ~6 months
  6. Only trade where confidence > 70%
  7. Backtest with trailing SL

Usage:
    python ml_stock_strategy.py                              # All stocks
    python ml_stock_strategy.py --stock HCLTECH              # One stock
    python ml_stock_strategy.py --confidence 0.75            # Higher filter
    python ml_stock_strategy.py --mode train                 # Train only
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
import joblib
from datetime import datetime, time as dt_time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ Install xgboost: pip install xgboost")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Install scikit-learn: pip install scikit-learn")


# ─────────── Config ───────────
DATA_DIR = "stock_data"
MODEL_DIR = "ml_models"
RESULTS_DIR = "scan_results"

# Window configurations
WINDOW_CONFIGS = {
    'morning':   {'w1': (dt_time(9, 15), dt_time(10, 30)), 'w2': None, 'sqoff': dt_time(10, 30)},
    'afternoon': {'w1': None, 'w2': (dt_time(13, 0), dt_time(15, 15)), 'sqoff': dt_time(15, 24)},
    'both':      {'w1': (dt_time(9, 15), dt_time(10, 30)), 'w2': (dt_time(13, 0), dt_time(15, 15)), 'sqoff': dt_time(15, 24)},
    'full':      {'w1': (dt_time(9, 15), dt_time(15, 15)), 'w2': None, 'sqoff': dt_time(15, 24)},
}

ACTIVE_WINDOW = 'afternoon'       # Default: 1:00 PM - 3:24 PM

CONFIDENCE_THRESHOLD = 0.70       # Only trade when prediction > 70% confident
LOOKAHEAD_CANDLES = 5             # Predict movement over next 5 candles
MOVE_THRESHOLD_PCT = 0.15         # 0.15% move = label as up/down
TRAIN_RATIO = 0.75                # First 75% for training

SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.0
MIN_HOLD = 2


# ─────────── Models ───────────
@dataclass
class Position:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    entry_idx: int
    confidence: float

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
    confidence: float


# ─────────── Feature Engineering ───────────

def engineer_features(df):
    """Create 40+ features from OHLCV data."""
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    features = pd.DataFrame(index=df.index)

    # ── Price Returns ──
    for p in [1, 2, 3, 5, 10]:
        features[f'return_{p}'] = close.pct_change(p) * 100

    # ── Candle Structure ──
    body = close - open_
    full_range = (high - low).replace(0, np.nan)
    features['body_ratio'] = body.abs() / full_range
    features['upper_wick'] = (high - close.clip(lower=open_)) / full_range
    features['lower_wick'] = (close.clip(upper=open_) - low) / full_range
    features['candle_direction'] = (close > open_).astype(int)
    features['gap'] = (open_ - close.shift(1)) / close.shift(1) * 100

    # ── Consecutive candles ──
    direction = (close > open_).astype(int)
    streaks = direction.groupby((direction != direction.shift(1)).cumsum()).cumcount() + 1
    features['streak'] = streaks * direction.map({1: 1, 0: -1})

    # ── EMAs & Trend ──
    for p in [5, 10, 21, 50]:
        ema_val = close.ewm(span=p, adjust=False).mean()
        features[f'ema_{p}_dist'] = (close - ema_val) / close * 100
        features[f'ema_{p}_slope'] = ema_val.pct_change(3) * 100

    features['ema_5_21_gap'] = (close.ewm(span=5, adjust=False).mean() -
                                 close.ewm(span=21, adjust=False).mean()) / close * 100

    # ── Momentum ──
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD Histogram
    fast_ema = close.ewm(span=12, adjust=False).mean()
    slow_ema = close.ewm(span=26, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features['macd_hist'] = macd_line - signal_line
    features['macd_hist_slope'] = features['macd_hist'].diff(2)

    # Rate of Change
    features['roc_5'] = close.pct_change(5) * 100
    features['roc_10'] = close.pct_change(10) * 100

    # ── Volatility ──
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr_val = tr.ewm(alpha=1/14, adjust=False).mean()
    features['atr'] = atr_val
    features['atr_pct'] = atr_val / close * 100
    features['atr_change'] = atr_val.pct_change(5) * 100

    # Bollinger Width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_width'] = (std20 * 4) / sma20 * 100
    features['bb_position'] = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # Intraday range
    features['intraday_range'] = (high - low) / close * 100

    # ── VOLUME FEATURES (the key advantage over index) ──
    vol_avg_20 = volume.rolling(20).mean()
    features['vol_ratio'] = volume / vol_avg_20.replace(0, np.nan)
    features['vol_surge'] = (volume > vol_avg_20 * 1.5).astype(int)
    features['vol_trend'] = vol_avg_20.pct_change(10) * 100

    # OBV
    obv_direction = np.sign(close.diff()).fillna(0)
    obv_val = (obv_direction * volume).cumsum()
    obv_ema = obv_val.ewm(span=10, adjust=False).mean()
    features['obv_slope'] = obv_val.diff(5) / volume.rolling(5).mean().replace(0, np.nan)
    features['obv_vs_ema'] = (obv_val - obv_ema) / obv_ema.abs().replace(0, np.nan) * 100

    # VWAP
    df_copy = df.copy()
    df_copy['date'] = df_copy['Time'].dt.date
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).groupby(df_copy['date']).cumsum()
    cum_vol = volume.groupby(df_copy['date']).cumsum()
    vwap_val = cum_tp_vol / cum_vol.replace(0, np.nan)
    features['vwap_dist'] = (close - vwap_val) / close * 100

    # Volume × Price interaction
    features['vol_price_corr'] = volume.rolling(10).corr(close)

    # ── Time Features ──
    features['hour'] = df['Time'].dt.hour
    features['minute_bucket'] = df['Time'].dt.minute // 15  # 0-3
    features['day_of_week'] = df['Time'].dt.dayofweek

    # ── Previous candle context ──
    features['prev_body_ratio'] = features['body_ratio'].shift(1)
    features['prev_vol_ratio'] = features['vol_ratio'].shift(1)
    features['prev_direction'] = features['candle_direction'].shift(1)

    return features, atr_val


def create_labels(df, lookahead=5, threshold_pct=0.15):
    """
    Label each candle:
      1 = price goes UP by threshold_pct in next N candles
      0 = price goes DOWN by threshold_pct in next N candles
      NaN = no clear movement (skip)
    """
    close = df['Close'].astype(float)
    future_high = close.rolling(lookahead).max().shift(-lookahead)
    future_low = close.rolling(lookahead).min().shift(-lookahead)

    up_move = (future_high - close) / close * 100
    down_move = (close - future_low) / close * 100

    labels = pd.Series(np.nan, index=df.index)
    labels[up_move >= threshold_pct] = 1       # Up
    labels[down_move >= threshold_pct] = 0     # Down

    # Both triggered → use which is bigger
    both = (up_move >= threshold_pct) & (down_move >= threshold_pct)
    labels[both & (up_move >= down_move)] = 1
    labels[both & (down_move > up_move)] = 0

    return labels


# ─────────── Model ───────────

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with anti-overfitting config."""
    model = xgb.XGBClassifier(
        n_estimators=200,          # ↓ from 300 (less overfitting)
        max_depth=3,               # ↓ from 5 (shallower = generalizes better)
        learning_rate=0.03,        # ↓ from 0.05 (slower learning)
        subsample=0.7,             # ↓ from 0.8 (more randomness)
        colsample_bytree=0.6,     # ↓ from 0.8 (use less features per tree)
        min_child_weight=10,       # ↑ from 5 (need more samples per leaf)
        reg_alpha=0.5,             # ↑ from 0.1 (L1 regularization)
        reg_lambda=3.0,            # ↑ from 1.0 (L2 regularization)
        gamma=0.3,                 # NEW: min loss reduction for split
        max_delta_step=1,          # NEW: limits each tree's prediction
        scale_pos_weight=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return model, train_acc, test_acc, test_pred, test_proba


# ─────────── Backtest ───────────

def in_window(t, window_config):
    w1 = window_config['w1']
    w2 = window_config['w2']
    in_w1 = w1 and w1[0] <= t <= w1[1]
    in_w2 = w2 and w2[0] <= t <= w2[1]
    return in_w1 or in_w2


def run_backtest(df, predictions, probabilities, atr_val, confidence_threshold, window_config):
    """Backtest ML predictions with ultra-selective filtering."""
    trades = []
    pos = None
    prev_date = None
    total_signals = 0
    filtered_signals = 0

    w1 = window_config['w1']
    w2 = window_config['w2']
    sqoff = window_config['sqoff']

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time()
        curr_date = row['Time'].date()
        close = float(row['Close'])
        high_v = float(row['High'])
        low_v = float(row['Low'])
        curr_atr = float(atr_val.iloc[i]) if i < len(atr_val) else 0

        # Day reset
        if prev_date and curr_date != prev_date:
            if pos:
                pc = float(df.iloc[i-1]['Close'])
                pnl = _pnl(pos, pc)
                trades.append(Trade(pos.direction, pos.entry_price, pc,
                                    pos.entry_time, df.iloc[i-1]['Time'],
                                    round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                    "DAY_END", pos.confidence))
                pos = None
        prev_date = curr_date

        # W1 close (if both windows exist, close W1 positions before gap)
        if pos and w1 and w2 and t > w1[1] and t < w2[0]:
            if pos.entry_time.time() <= w1[1]:
                pnl = _pnl(pos, close)
                trades.append(Trade(pos.direction, pos.entry_price, close,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                    "W1_CLOSE", pos.confidence))
                pos = None
            continue

        # Square off
        if pos and t >= sqoff:
            pnl = _pnl(pos, close)
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                "SQUARE_OFF", pos.confidence))
            pos = None
            continue

        # Check if in any active window
        in_active = False
        if w1 and w1[0] <= t <= w1[1]: in_active = True
        if w2 and w2[0] <= t <= sqoff: in_active = True
        if w1 and not w2 and w1[0] <= t <= sqoff: in_active = True  # 'full' mode
        if not in_active:
            continue

        # SL check
        if pos:
            if pos.direction == "LONG" and low_v <= pos.stop_loss:
                pnl = _pnl(pos, pos.stop_loss)
                trades.append(Trade(pos.direction, pos.entry_price, pos.stop_loss,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                    "STOP_LOSS", pos.confidence))
                pos = None
            elif pos and pos.direction == "SHORT" and high_v >= pos.stop_loss:
                pnl = _pnl(pos, pos.stop_loss)
                trades.append(Trade(pos.direction, pos.entry_price, pos.stop_loss,
                                    pos.entry_time, row['Time'],
                                    round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                    "STOP_LOSS", pos.confidence))
                pos = None

        # Trail SL
        if pos and curr_atr > 0:
            if pos.direction == "LONG":
                new_sl = high_v - curr_atr * TRAIL_ATR_MULT
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
            else:
                new_sl = low_v + curr_atr * TRAIL_ATR_MULT
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl

        # ML prediction
        if i >= len(predictions):
            continue

        pred = predictions[i]
        prob_up = probabilities[i][1] if len(probabilities[i]) > 1 else 0.5
        prob_down = probabilities[i][0] if len(probabilities[i]) > 1 else 0.5

        # ─── ULTRA-SELECTIVE: Only trade when confidence > threshold ───
        is_buy = pred == 1 and prob_up >= confidence_threshold
        is_sell = pred == 0 and prob_down >= confidence_threshold

        if is_buy or is_sell:
            total_signals += 1

        # Opposite signal close
        if is_buy and pos and pos.direction == "SHORT" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                "OPPOSITE", pos.confidence))
            pos = None
        elif is_sell and pos and pos.direction == "LONG" and (i - pos.entry_idx) >= MIN_HOLD:
            pnl = _pnl(pos, close)
            trades.append(Trade(pos.direction, pos.entry_price, close,
                                pos.entry_time, row['Time'],
                                round(pnl, 2), round(pnl/pos.entry_price*100, 3),
                                "OPPOSITE", pos.confidence))
            pos = None

        # New entry
        can_enter = in_window(t, window_config)
        if not pos and can_enter and curr_atr > 0:
            if is_buy:
                sl = close - curr_atr * SL_ATR_MULT
                pos = Position("LONG", close, row['Time'], sl, i, prob_up)
                filtered_signals += 1
            elif is_sell:
                sl = close + curr_atr * SL_ATR_MULT
                pos = Position("SHORT", close, row['Time'], sl, i, prob_down)
                filtered_signals += 1

    return trades, total_signals, filtered_signals


def _pnl(pos, exit_price):
    return (exit_price - pos.entry_price) if pos.direction == "LONG" else (pos.entry_price - exit_price)


# ─────────── Report ───────────

def print_report(symbol, trades, test_acc, train_acc, total_signals, filtered_signals,
                 total_candles, trading_days, date_range, feature_importance=None):
    if not trades:
        print(f"\n  {symbol}: No trades at confidence threshold {CONFIDENCE_THRESHOLD}")
        return

    pnl_pts = [t.pnl_pts for t in trades]
    pnl_pct = [t.pnl_pct for t in trades]
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

    avg_conf = np.mean([t.confidence for t in trades])

    print(f"\n  {'='*60}")
    print(f"  📈 {symbol} — ML STRATEGY RESULTS")
    print(f"  {date_range} | {trading_days} days")
    print(f"  {'='*60}")

    print(f"\n  🤖 MODEL")
    print(f"  Train accuracy:   {train_acc:.1%}")
    print(f"  Test accuracy:    {test_acc:.1%}")
    print(f"  Avg confidence:   {avg_conf:.1%}")

    print(f"\n  📊 SELECTIVITY")
    print(f"  Total candles:    {total_candles}")
    print(f"  High-conf signals: {total_signals}")
    print(f"  Trades taken:     {filtered_signals} ({filtered_signals/max(total_candles,1)*100:.2f}% of candles)")

    print(f"\n  💰 PERFORMANCE")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Win rate:         {wr:.1f}%")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Total P&L:        {sum(pnl_pts):+.2f} pts ({sum(pnl_pct):+.3f}%)")
    print(f"  Avg P&L/trade:    {np.mean(pnl_pts):+.2f} pts")
    print(f"  Best trade:       {max(pnl_pts):+.2f} pts")
    print(f"  Worst trade:      {min(pnl_pts):+.2f} pts")
    print(f"  Prof days:        {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")

    reasons = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
    print(f"\n  🔍 CLOSE REASONS")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        r_t = [t for t in trades if t.close_reason == r]
        r_wr = sum(1 for t in r_t if t.pnl_pts > 0)/len(r_t)*100
        print(f"    {r:20s}: {c:4d} | Win: {r_wr:.0f}%")

    # Top features
    if feature_importance is not None:
        print(f"\n  🧠 TOP 10 FEATURES")
        top = feature_importance.head(10)
        for _, row in top.iterrows():
            print(f"    {row['feature']:25s}: {row['importance']:.4f}")

    print(f"  {'='*60}")


# ─────────── Process One Stock ───────────

def process_stock(symbol, file_path, confidence_threshold, window_name='afternoon', detailed=False):
    """Full pipeline for one stock: features → train → backtest."""
    df = pd.read_csv(file_path)
    if len(df) < 500:
        return None

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    if 'Volume' not in df.columns:
        return None

    trading_days = df['Time'].dt.date.nunique()
    date_range = f"{df['Time'].iloc[0].strftime('%Y-%m-%d')} → {df['Time'].iloc[-1].strftime('%Y-%m-%d')}"

    # Feature engineering
    features, atr_val = engineer_features(df)
    labels = create_labels(df, LOOKAHEAD_CANDLES, MOVE_THRESHOLD_PCT)

    # Combine and drop NaN
    combined = features.copy()
    combined['label'] = labels
    combined = combined.dropna()

    if len(combined) < 200:
        return None

    feature_cols = [c for c in combined.columns if c != 'label']
    X = combined[feature_cols].values
    y = combined['label'].values.astype(int)

    # Time-based split
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) < 100 or len(X_test) < 50:
        return None

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    model, train_acc, test_acc, test_pred, test_proba = train_model(
        X_train_s, y_train, X_test_s, y_test
    )

    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Backtest on TEST data only (no lookahead bias)
    test_start_idx = combined.index[split_idx]
    df_test = df.loc[test_start_idx:].reset_index(drop=True)
    atr_test = atr_val.loc[test_start_idx:].reset_index(drop=True)

    # Generate predictions for all test candles
    test_features = features.loc[test_start_idx:]
    test_features_clean = test_features.dropna()

    if len(test_features_clean) < 50:
        return None

    X_all_test = scaler.transform(test_features_clean[feature_cols].values)
    all_pred = model.predict(X_all_test)
    all_proba = model.predict_proba(X_all_test)

    # Map predictions back to full test DataFrame
    full_pred = np.zeros(len(df_test))
    full_proba = np.full((len(df_test), 2), 0.5)

    clean_idx = test_features_clean.index - test_start_idx
    for j, ci in enumerate(clean_idx):
        if 0 <= ci < len(df_test):
            full_pred[ci] = all_pred[j]
            full_proba[ci] = all_proba[j]

    # Backtest each window config
    window_config = WINDOW_CONFIGS[window_name]
    trades, total_signals, filtered_signals = run_backtest(
        df_test, full_pred, full_proba, atr_test, confidence_threshold, window_config
    )

    # Report
    if detailed:
        print(f"\n  ⏰ Window: {window_name.upper()}")
        print_report(symbol, trades, test_acc, train_acc, total_signals, filtered_signals,
                     len(df_test), df_test['Time'].dt.date.nunique(),
                     f"{df_test['Time'].iloc[0].strftime('%Y-%m-%d')} → {df_test['Time'].iloc[-1].strftime('%Y-%m-%d')}",
                     fi)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/{symbol}_xgb.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{symbol}_scaler.pkl")

    # Return summary
    win_rate = sum(1 for t in trades if t.pnl_pts > 0) / max(len(trades), 1) * 100
    total_pnl = sum(t.pnl_pct for t in trades) if trades else 0
    gp = sum(t.pnl_pts for t in trades if t.pnl_pts > 0) if trades else 0
    gl = abs(sum(t.pnl_pts for t in trades if t.pnl_pts <= 0)) if trades else 1
    pf = gp / gl if gl > 0 else 0

    # For 'all' mode: also test other windows and return comparison
    window_results = None
    if window_name == 'all_internal':
        window_results = {}
        for wn in ['morning', 'afternoon', 'both', 'full']:
            wc = WINDOW_CONFIGS[wn]
            wt, ws, wf = run_backtest(df_test, full_pred, full_proba, atr_test, confidence_threshold, wc)
            w_wr = sum(1 for t in wt if t.pnl_pts > 0) / max(len(wt), 1) * 100
            w_pnl = sum(t.pnl_pct for t in wt) if wt else 0
            w_gp = sum(t.pnl_pts for t in wt if t.pnl_pts > 0) if wt else 0
            w_gl = abs(sum(t.pnl_pts for t in wt if t.pnl_pts <= 0)) if wt else 1
            w_pf = w_gp / w_gl if w_gl > 0 else 0
            window_results[wn] = {'trades': len(wt), 'win_rate': round(w_wr, 1),
                                   'pnl_pct': round(w_pnl, 3), 'pf': round(w_pf, 2)}

    return {
        'symbol': symbol, 'test_acc': round(test_acc * 100, 1),
        'train_acc': round(train_acc * 100, 1),
        'trades': len(trades), 'win_rate': round(win_rate, 1),
        'pnl_pct': round(total_pnl, 3), 'profit_factor': round(pf, 2),
        'signals': total_signals, 'filtered': filtered_signals,
        'all_trades': trades, 'window_results': window_results,
    }


# ─────────── Main ───────────

def main():
    global CONFIDENCE_THRESHOLD, ACTIVE_WINDOW
    if not HAS_XGB or not HAS_SKLEARN:
        print("❌ Install: pip install xgboost scikit-learn")
        return

    parser = argparse.ArgumentParser(description="ML Stock Strategy (XGBoost)")
    parser.add_argument("--stock", default=None)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--window", default=ACTIVE_WINDOW,
                        choices=['morning', 'afternoon', 'both', 'full', 'all'],
                        help="Trading window: morning (9:15-10:30), afternoon (1:00-3:24), "
                             "both (morning+afternoon), full (9:15-3:24), all (compare all)")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence
    ACTIVE_WINDOW = args.window
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find files — fix duplicate matching
    if args.stock:
        stock_upper = args.stock.upper()
        all_files = glob.glob(f"{args.data_dir}/*_*min.csv")
        files = []
        seen = set()
        for f in all_files:
            fname = os.path.basename(f)
            sym = fname.split('_')[0].upper()
            if sym == stock_upper and f not in seen:
                files.append(f)
                seen.add(f)
                break  # Only take first match
    else:
        files = sorted(set(glob.glob(f"{args.data_dir}/*_*min.csv")))

    if not files:
        print(f"❌ No data in {args.data_dir}/. Run fetch_stock_data.py first.")
        return

    # Determine window mode
    test_all_windows = ACTIVE_WINDOW == 'all'
    window_to_use = 'all_internal' if test_all_windows else ACTIVE_WINDOW

    print(f"🤖 ML STOCK STRATEGY v2 — Anti-Overfitting + Window Testing")
    print(f"Confidence: {CONFIDENCE_THRESHOLD:.0%} | Window: {ACTIVE_WINDOW}")
    print(f"Anti-overfit: depth=3, reg_alpha=0.5, reg_lambda=3.0, gamma=0.3")
    print(f"Found {len(files)} stocks")
    print(f"{'='*60}")

    results = []
    all_trades = []
    window_comparison = {}  # For --window all

    for idx, fp in enumerate(sorted(files)):
        symbol = os.path.basename(fp).split("_")[0]
        print(f"[{idx+1}/{len(files)}] {symbol}...", end=" ", flush=True)

        try:
            if test_all_windows:
                # Test all windows and show comparison
                result = process_stock(symbol, fp, CONFIDENCE_THRESHOLD,
                                       window_name='all_internal',
                                       detailed=args.detailed or bool(args.stock))
                if result and result.get('window_results'):
                    window_comparison[symbol] = result['window_results']
                    # Pick the best window for ranking
                    best_w = max(result['window_results'].items(),
                                 key=lambda x: x[1]['pnl_pct'])
                    result['best_window'] = best_w[0]
                    result['trades'] = best_w[1]['trades']
                    result['win_rate'] = best_w[1]['win_rate']
                    result['pnl_pct'] = best_w[1]['pnl_pct']
                    result['profit_factor'] = best_w[1]['pf']
                    print(f"Acc: {result['test_acc']:.1f}% | Best: {best_w[0]} ({best_w[1]['win_rate']:.1f}% win, {best_w[1]['pnl_pct']:+.2f}%)")
            else:
                result = process_stock(symbol, fp, CONFIDENCE_THRESHOLD,
                                       window_name=window_to_use,
                                       detailed=args.detailed or bool(args.stock))
                if result:
                    print(f"Acc: {result['test_acc']:.1f}% | {result['trades']} trades | Win: {result['win_rate']:.1f}% | P&L: {result['pnl_pct']:+.2f}%")

            if result:
                results.append(result)
                for t in result['all_trades']:
                    all_trades.append({
                        'symbol': symbol, 'direction': t.direction,
                        'entry_price': t.entry_price, 'exit_price': t.exit_price,
                        'pnl_pts': t.pnl_pts, 'pnl_pct': t.pnl_pct,
                        'entry_time': t.entry_time, 'exit_time': t.exit_time,
                        'close_reason': t.close_reason, 'confidence': t.confidence,
                    })
            else:
                print("⚠️ Insufficient data")
        except Exception as e:
            print(f"❌ {e}")

    if not results:
        print("❌ No results.")
        return

    # Window comparison table (--window all)
    if test_all_windows and window_comparison:
        print(f"\n{'='*100}")
        print(f"  ⏰ WINDOW COMPARISON (per stock)")
        print(f"{'='*100}")
        print(f"  {'Symbol':>12} | {'Morning':>20} | {'Afternoon':>20} | {'Both':>20} | {'Full Day':>20}")
        print(f"  {'-'*96}")
        for sym, wrs in sorted(window_comparison.items()):
            parts = []
            for w in ['morning', 'afternoon', 'both', 'full']:
                d = wrs.get(w, {})
                t = d.get('trades', 0)
                wr = d.get('win_rate', 0)
                pnl = d.get('pnl_pct', 0)
                parts.append(f"{t:>3}t {wr:>4.0f}% {pnl:>+6.2f}%")
            print(f"  {sym:>12} | {parts[0]:>20} | {parts[1]:>20} | {parts[2]:>20} | {parts[3]:>20}")

    # Ranking
    sorted_results = sorted(results, key=lambda r: r['win_rate'], reverse=True)

    print(f"\n{'='*95}")
    print(f"  ML RANKINGS — Confidence ≥ {CONFIDENCE_THRESHOLD:.0%} | Window: {ACTIVE_WINDOW}")
    print(f"{'='*95}")
    header = f"  {'Rank':>4} {'Symbol':>12} {'Train%':>7} {'Test%':>6} {'Trades':>7} {'Win%':>6} {'P&L%':>8} {'PF':>5}"
    if test_all_windows:
        header += f" {'BestWin':>8}"
    print(header)
    print(f"  {'-'*75}")

    for rank, r in enumerate(sorted_results[:args.top], 1):
        star = "⭐" if rank <= 5 else "  "
        line = f"  {rank:>3}. {r['symbol']:>12} {r['train_acc']:>6.1f}% {r['test_acc']:>5.1f}% {r['trades']:>7} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+7.2f}% {r['profit_factor']:>5.2f}"
        if test_all_windows and 'best_window' in r:
            line += f" {r['best_window']:>8}"
        print(f"{line} {star}")

    # Summary
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(sum(1 for t in r['all_trades'] if t.pnl_pts > 0) for r in results)
    overall_wr = total_wins / max(total_trades, 1) * 100
    profitable = [r for r in results if r['pnl_pct'] > 0]
    avg_test_acc = np.mean([r['test_acc'] for r in results])
    avg_train_acc = np.mean([r['train_acc'] for r in results])

    print(f"\n  📊 OVERALL")
    print(f"  Stocks:       {len(results)}")
    print(f"  Avg train:    {avg_train_acc:.1f}% → test: {avg_test_acc:.1f}% (gap: {avg_train_acc - avg_test_acc:.1f}%)")
    print(f"  Total trades: {total_trades} | Win: {overall_wr:.1f}%")
    print(f"  Profitable:   {len(profitable)}/{len(results)}")
    print(f"{'='*95}")

    # Save
    save_data = [{k: v for k, v in r.items() if k not in ('all_trades', 'window_results')} for r in sorted_results]
    pd.DataFrame(save_data).to_csv(f"{RESULTS_DIR}/ml_rankings.csv", index=False)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"{RESULTS_DIR}/ml_all_trades.csv", index=False)

    top5 = sorted_results[:5]
    print(f"\n🎯 TOP 5:")
    for r in top5:
        extra = f" | Best: {r.get('best_window', ACTIVE_WINDOW)}" if test_all_windows else ""
        print(f"  ⭐ {r['symbol']}: Test {r['test_acc']:.1f}% | Win {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f}{extra}")
    print(f"\n  💾 {MODEL_DIR}/ | {RESULTS_DIR}/ml_rankings.csv")


if __name__ == "__main__":
    main()
