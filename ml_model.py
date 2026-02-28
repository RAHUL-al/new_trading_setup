"""
ml_model.py — XGBoost + LSTM Ensemble Trading Model for NIFTY

Architecture:
  1. Feature Engineering: 30+ indicators from OHLC (no volume needed)
  2. XGBoost: Gradient boosted trees on tabular features
  3. LSTM: Sequential model on price sequences (PyTorch)
  4. Ensemble: Weighted average of both predictions
  5. Confidence Filter: Only trade when ensemble confidence > threshold

Usage:
    python ml_model.py --mode train     # Train models on historical data
    python ml_model.py --mode backtest  # Backtest with trained models
    python ml_model.py --mode both      # Train + backtest

Requirements:
    pip install xgboost torch scikit-learn pandas numpy joblib
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import warnings
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import List, Optional, Tuple
warnings.filterwarnings('ignore')

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not installed. Run: pip install torch")

# Check for XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ─────────── Config ───────────
MODEL_DIR = "ml_models"
SEQUENCE_LENGTH = 60         # LSTM looks back 60 candles
PREDICTION_HORIZON = 3       # Predict direction over next 3 candles
CONFIDENCE_THRESHOLD = 0.60  # Only trade when confidence > 60%
ENSEMBLE_WEIGHT_XGB = 0.5    # XGBoost weight in ensemble
ENSEMBLE_WEIGHT_LSTM = 0.5   # LSTM weight in ensemble
TRAIN_SPLIT = 0.8            # 80% train, 20% test

TRADING_START = dt_time(12, 30)
TRADING_END = dt_time(15, 10)
SQUARE_OFF_TIME = dt_time(15, 24)
MARKET_OPEN = dt_time(9, 16)
MARKET_CLOSE = dt_time(15, 30)


# ─────────── Feature Engineering ───────────

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD indicator."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(close, period=20, std_dev=2):
    """Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic oscillator."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 30+ features from OHLC data."""
    feat = pd.DataFrame(index=df.index)

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    open_ = df['Open'].astype(float)

    # ── Price returns ──
    feat['return_1'] = close.pct_change(1)
    feat['return_3'] = close.pct_change(3)
    feat['return_5'] = close.pct_change(5)
    feat['return_10'] = close.pct_change(10)

    # ── Volatility ──
    feat['volatility_5'] = close.pct_change().rolling(5).std()
    feat['volatility_10'] = close.pct_change().rolling(10).std()
    feat['volatility_20'] = close.pct_change().rolling(20).std()

    # ── True Range & ATR ──
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    feat['atr_7'] = tr.ewm(alpha=1/7, adjust=False).mean()
    feat['atr_ratio'] = feat['atr_7'] / feat['atr_14'].replace(0, np.nan)

    # ── Moving Averages ──
    feat['ema_9'] = close.ewm(span=9, adjust=False).mean()
    feat['ema_21'] = close.ewm(span=21, adjust=False).mean()
    feat['ema_50'] = close.ewm(span=50, adjust=False).mean()
    feat['sma_20'] = close.rolling(20).mean()

    # EMA distances (normalized)
    feat['dist_ema9'] = (close - feat['ema_9']) / feat['atr_14'].replace(0, np.nan)
    feat['dist_ema21'] = (close - feat['ema_21']) / feat['atr_14'].replace(0, np.nan)
    feat['dist_ema50'] = (close - feat['ema_50']) / feat['atr_14'].replace(0, np.nan)
    feat['ema_cross'] = (feat['ema_9'] - feat['ema_21']) / feat['atr_14'].replace(0, np.nan)

    # ── RSI ──
    feat['rsi_14'] = calculate_rsi(close, 14)
    feat['rsi_7'] = calculate_rsi(close, 7)
    feat['rsi_diff'] = feat['rsi_7'] - feat['rsi_14']

    # ── MACD ──
    macd_line, signal_line, histogram = calculate_macd(close)
    feat['macd'] = macd_line / feat['atr_14'].replace(0, np.nan)
    feat['macd_signal'] = signal_line / feat['atr_14'].replace(0, np.nan)
    feat['macd_hist'] = histogram / feat['atr_14'].replace(0, np.nan)

    # ── Bollinger Bands ──
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close)
    bb_width = bb_upper - bb_lower
    feat['bb_position'] = (close - bb_lower) / bb_width.replace(0, np.nan)
    feat['bb_width'] = bb_width / close

    # ── Stochastic ──
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    feat['stoch_k'] = stoch_k
    feat['stoch_d'] = stoch_d

    # ── Candle Patterns ──
    body = close - open_
    full_range = high - low
    feat['body_ratio'] = body / full_range.replace(0, np.nan)           # +1 = strong bullish, -1 = strong bearish
    feat['upper_wick'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range.replace(0, np.nan)
    feat['lower_wick'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range.replace(0, np.nan)
    feat['body_size'] = body.abs() / feat['atr_14'].replace(0, np.nan)  # Normalized body size
    feat['range_size'] = full_range / feat['atr_14'].replace(0, np.nan)

    # ── Momentum ──
    feat['momentum_5'] = close - close.shift(5)
    feat['momentum_10'] = close - close.shift(10)
    feat['acceleration'] = feat['momentum_5'] - feat['momentum_5'].shift(5)

    # ── High/Low position ──
    feat['high_5'] = (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min()).replace(0, np.nan)
    feat['high_10'] = (close - low.rolling(10).min()) / (high.rolling(10).max() - low.rolling(10).min()).replace(0, np.nan)

    # ── Time features ──
    if 'Time' in df.columns:
        feat['hour'] = df['Time'].dt.hour
        feat['minute'] = df['Time'].dt.minute
        feat['day_of_week'] = df['Time'].dt.dayofweek
        feat['minutes_since_open'] = (df['Time'].dt.hour - 9) * 60 + (df['Time'].dt.minute - 15)

    return feat


def create_target(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """Target: 1 if price goes UP over next N candles, 0 if DOWN."""
    future_return = df['Close'].shift(-horizon) - df['Close']
    return (future_return > 0).astype(int)


# ─────────── LSTM Model (PyTorch) ───────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last time step
        return self.fc(last_hidden).squeeze()


# ─────────── Training ───────────

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    print("\n🟢 Training XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    # Feature importance
    importance = model.feature_importances_
    return model, test_acc, importance


def train_lstm(X_train, y_train, X_test, y_test, input_size, epochs=50, batch_size=64, lr=0.001):
    """Train LSTM model."""
    print("\n🔵 Training LSTM...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len=SEQUENCE_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_test_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                test_preds.extend(output.cpu().numpy())
                test_targets.extend(y_batch.numpy())

        test_preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
        test_acc = accuracy_score(test_targets, test_preds_binary)
        scheduler.step(train_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {test_acc:.4f}")

    print(f"  Best test accuracy: {best_test_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_test_acc


# ─────────── Ensemble Prediction ───────────

def ensemble_predict(xgb_model, lstm_model, scaler, features_df, feature_names, device='cpu'):
    """Generate ensemble predictions with confidence scores."""
    X = features_df[feature_names].values
    X_scaled = scaler.transform(X)

    # XGBoost probabilities
    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]

    # LSTM probabilities
    lstm_proba = np.full(len(X_scaled), 0.5)  # default
    if HAS_TORCH and lstm_model is not None:
        lstm_model.eval()
        with torch.no_grad():
            for i in range(SEQUENCE_LENGTH, len(X_scaled)):
                seq = torch.FloatTensor(X_scaled[i-SEQUENCE_LENGTH:i]).unsqueeze(0).to(device)
                pred = lstm_model(seq).cpu().item()
                lstm_proba[i] = pred

    # Ensemble
    ensemble_proba = ENSEMBLE_WEIGHT_XGB * xgb_proba + ENSEMBLE_WEIGHT_LSTM * lstm_proba

    # Signals: buy if proba > 0.5 + threshold, sell if proba < 0.5 - threshold
    buy_threshold = 0.5 + (CONFIDENCE_THRESHOLD - 0.5)
    sell_threshold = 0.5 - (CONFIDENCE_THRESHOLD - 0.5)

    signals = pd.DataFrame(index=features_df.index)
    signals['buy'] = ensemble_proba > buy_threshold
    signals['sell'] = ensemble_proba < sell_threshold
    signals['confidence'] = np.abs(ensemble_proba - 0.5) * 2  # 0 to 1 scale
    signals['xgb_proba'] = xgb_proba
    signals['lstm_proba'] = lstm_proba
    signals['ensemble_proba'] = ensemble_proba

    return signals


# ─────────── Backtest ───────────

@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    confidence: float
    close_reason: str


def backtest_ml(df, signals, atr_series, sl_mult=2.0, min_hold=3):
    """Backtest using ML ensemble signals."""
    trades = []
    open_pos = None
    entry_idx = 0

    for i in range(len(df)):
        row = df.iloc[i]
        t = row['Time'].time() if hasattr(row['Time'], 'time') else row['Time']
        close = float(row['Close'])
        atr = float(atr_series.iloc[i]) if i < len(atr_series) else 0

        if not (MARKET_OPEN <= t <= MARKET_CLOSE):
            continue

        # Square off
        if open_pos and t >= SQUARE_OFF_TIME:
            pnl = (close - open_pos['entry']) if open_pos['dir'] == "CE" else (open_pos['entry'] - close)
            trades.append(Trade(open_pos['dir'], open_pos['entry'], close,
                                open_pos['time'], row['Time'], round(pnl, 2),
                                open_pos['conf'], "SQUARE_OFF"))
            open_pos = None
            continue

        # Stop loss check
        if open_pos:
            if open_pos['dir'] == "CE" and close <= open_pos['sl']:
                pnl = close - open_pos['entry']
                trades.append(Trade(open_pos['dir'], open_pos['entry'], close,
                                    open_pos['time'], row['Time'], round(pnl, 2),
                                    open_pos['conf'], "STOP_LOSS"))
                open_pos = None
            elif open_pos['dir'] == "PE" and close >= open_pos['sl']:
                pnl = open_pos['entry'] - close
                trades.append(Trade(open_pos['dir'], open_pos['entry'], close,
                                    open_pos['time'], row['Time'], round(pnl, 2),
                                    open_pos['conf'], "STOP_LOSS"))
                open_pos = None

        # Trailing SL
        if open_pos and atr > 0:
            if open_pos['dir'] == "CE":
                new_sl = close - (atr * sl_mult)
                if new_sl > open_pos['sl']:
                    open_pos['sl'] = new_sl
            elif open_pos['dir'] == "PE":
                new_sl = close + (atr * sl_mult)
                if new_sl < open_pos['sl']:
                    open_pos['sl'] = new_sl

        # Signal processing
        if i >= len(signals):
            continue
        sig = signals.iloc[i]
        is_buy = bool(sig.get('buy', False))
        is_sell = bool(sig.get('sell', False))
        conf = float(sig.get('confidence', 0))

        # Opposite signal close (with min hold)
        if is_buy and open_pos and open_pos['dir'] == "PE" and (i - entry_idx) >= min_hold:
            pnl = open_pos['entry'] - close
            trades.append(Trade(open_pos['dir'], open_pos['entry'], close,
                                open_pos['time'], row['Time'], round(pnl, 2),
                                open_pos['conf'], "OPPOSITE_SIGNAL"))
            open_pos = None

        elif is_sell and open_pos and open_pos['dir'] == "CE" and (i - entry_idx) >= min_hold:
            pnl = close - open_pos['entry']
            trades.append(Trade(open_pos['dir'], open_pos['entry'], close,
                                open_pos['time'], row['Time'], round(pnl, 2),
                                open_pos['conf'], "OPPOSITE_SIGNAL"))
            open_pos = None

        # New entry
        if not open_pos and TRADING_START <= t <= TRADING_END and conf >= (CONFIDENCE_THRESHOLD - 0.5) * 2:
            if is_buy and atr > 0:
                open_pos = {'dir': 'CE', 'entry': close, 'sl': close - atr * sl_mult,
                            'time': row['Time'], 'conf': conf}
                entry_idx = i
            elif is_sell and atr > 0:
                open_pos = {'dir': 'PE', 'entry': close, 'sl': close + atr * sl_mult,
                            'time': row['Time'], 'conf': conf}
                entry_idx = i

    return trades


def print_ml_report(trades, title="ML ENSEMBLE"):
    """Print backtest results."""
    if not trades:
        print(f"\n❌ {title}: No trades generated!")
        return

    pnl_list = [t.pnl for t in trades]
    total = sum(pnl_list)
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    win_rate = len(wins) / len(pnl_list) * 100

    cumulative = np.cumsum(pnl_list)
    max_dd = (cumulative - np.maximum.accumulate(cumulative)).min()

    daily = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        daily[day] = daily.get(day, 0) + t.pnl
    prof_days = sum(1 for v in daily.values() if v > 0)

    avg_conf = np.mean([t.confidence for t in trades])

    print(f"\n{'='*70}")
    print(f"  {title} RESULTS")
    print(f"{'='*70}")
    print(f"  Total P&L:       {total:+.2f} pts")
    print(f"  Total trades:    {len(trades)}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Avg win:         {np.mean(wins):+.2f} pts" if wins else "  Avg win: N/A")
    print(f"  Avg loss:        {np.mean(losses):+.2f} pts" if losses else "  Avg loss: N/A")
    print(f"  Max drawdown:    {max_dd:.2f} pts")
    print(f"  Avg confidence:  {avg_conf:.2f}")
    print(f"  Profitable days: {prof_days}/{len(daily)} ({prof_days/max(len(daily),1)*100:.0f}%)")
    print(f"  Best trade:      {max(pnl_list):+.2f} pts")
    print(f"  Worst trade:     {min(pnl_list):+.2f} pts")
    print(f"{'='*70}")

    # Save trades
    trades_df = pd.DataFrame([{
        'direction': t.direction, 'entry_price': t.entry_price,
        'exit_price': t.exit_price, 'pnl': t.pnl,
        'confidence': t.confidence, 'close_reason': t.close_reason,
        'entry_time': t.entry_time, 'exit_time': t.exit_time,
    } for t in trades])
    trades_df.to_csv("ml_backtest_trades.csv", index=False)
    print(f"  💾 Trades saved to: ml_backtest_trades.csv")


# ─────────── Main Pipeline ───────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=["train", "backtest", "both"])
    parser.add_argument("--file", default="nifty_3min_data.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.60)
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    # Try 3-min first, then 5-min, then 1-min
    data_file = args.file
    if not os.path.exists(data_file):
        for alt in ["nifty_5min_data.csv", "nifty_1min_data.csv"]:
            if os.path.exists(alt):
                data_file = alt
                break

    if not os.path.exists(data_file):
        print("❌ No data file found. Run fetch_nifty_data.py first.")
        return

    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    print(f"Loaded {len(df)} candles | {df['Time'].dt.date.nunique()} days")

    # Feature engineering
    print("\n📊 Engineering features...")
    features = engineer_features(df)
    target = create_target(df, horizon=PREDICTION_HORIZON)

    # ATR for backtest
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/14, adjust=False).mean()

    # Drop NaN rows
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features[valid_mask].reset_index(drop=True)
    target = target[valid_mask].reset_index(drop=True)
    df_valid = df[valid_mask].reset_index(drop=True)
    atr_valid = atr_series[valid_mask].reset_index(drop=True)

    feature_names = features.columns.tolist()
    print(f"  Features: {len(feature_names)}")
    print(f"  Valid samples: {len(features)}")

    # Scale features
    scaler = StandardScaler()
    X = features.values
    y = target.values

    # Train/test split (time-based, no shuffle!)
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.2f} (1=up)")
    print(f"  Test class balance:  {y_test.mean():.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ─── TRAIN ───
    if args.mode in ["train", "both"]:
        # XGBoost
        if HAS_XGB:
            xgb_model, xgb_acc, importance = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
            joblib.dump(xgb_model, f"{MODEL_DIR}/xgb_model.pkl")
            joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

            # Top features
            top_idx = np.argsort(importance)[-10:][::-1]
            print("\n  Top 10 features:")
            for idx in top_idx:
                print(f"    {feature_names[idx]:25s} importance: {importance[idx]:.4f}")

        # LSTM
        if HAS_TORCH:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lstm_model, lstm_acc = train_lstm(
                X_train_scaled, y_train, X_test_scaled, y_test,
                input_size=len(feature_names), epochs=args.epochs,
            )
            torch.save(lstm_model.state_dict(), f"{MODEL_DIR}/lstm_model.pt")

        # Save config
        config = {
            'feature_names': feature_names,
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'input_size': len(feature_names),
            'data_file': data_file,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        }
        with open(f"{MODEL_DIR}/config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Models saved to {MODEL_DIR}/")

    # ─── BACKTEST ───
    if args.mode in ["backtest", "both"]:
        print("\n" + "=" * 70)
        print("  BACKTESTING ML ENSEMBLE")
        print("=" * 70)

        # Load models
        xgb_model = None
        lstm_model = None

        if HAS_XGB and os.path.exists(f"{MODEL_DIR}/xgb_model.pkl"):
            xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
            scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

        if HAS_TORCH and os.path.exists(f"{MODEL_DIR}/lstm_model.pt"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lstm_model = LSTMModel(input_size=len(feature_names)).to(device)
            lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_model.pt", map_location=device, weights_only=True))

        if xgb_model is None:
            print("❌ No trained XGBoost model found. Run with --mode train first.")
            return

        # Generate signals on TEST data only (no cheating)
        test_features = features.iloc[split_idx:].reset_index(drop=True)
        test_df = df_valid.iloc[split_idx:].reset_index(drop=True)
        test_atr = atr_valid.iloc[split_idx:].reset_index(drop=True)

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu' if HAS_TORCH else 'cpu'
        signals = ensemble_predict(xgb_model, lstm_model, scaler, test_features, feature_names, device_str)

        total_signals = signals['buy'].sum() + signals['sell'].sum()
        print(f"  Total signals: {total_signals} ({signals['buy'].sum()} buy, {signals['sell'].sum()} sell)")
        print(f"  Avg confidence: {signals['confidence'].mean():.3f}")

        # Run backtest
        trades = backtest_ml(test_df, signals, test_atr, sl_mult=2.0, min_hold=3)
        print_ml_report(trades, "XGBoost + LSTM ENSEMBLE")


if __name__ == "__main__":
    main()
