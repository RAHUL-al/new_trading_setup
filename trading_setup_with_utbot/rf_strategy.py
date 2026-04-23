"""
rf_strategy.py — Multi-Timeframe Random Forest + LSTM Strategy for NIFTY

STRATEGY CONCEPT:
  Phase 1 (09:15-11:00): Observation — RF builds market regime confidence
  Phase 2 (11:00-15:15): Active trading
    LONG:  RF=BULL + StochRSI_1m < 10 (oversold pullback) + UT_Bot confirm
    SHORT: RF=BEAR + StochRSI_1m > 90 (overbought rally) + UT_Bot confirm

  Regime changes (BULL→BEAR or BEAR→BULL) trigger position flips.

MODELS:
  1. Random Forest — Market regime classifier (BULL/BEAR/CHOP)
     Trained on multi-timeframe features (1m + 3m + 5m)
  2. LSTM (optional) — Short-term momentum confirmation
     Trained on 1-min sequential patterns

ZERO LOOK-AHEAD:
  Regime labels are generated from FUTURE price movement (EMA slope),
  but only used for training. At inference, the model predicts from
  current + past data only.

Usage:
    python rf_strategy.py                                    # Train + backtest
    python rf_strategy.py --test-from 2026-04-01             # April test
    python rf_strategy.py --no-lstm                          # RF only
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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed. LSTM will be disabled. Run: pip install torch")


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5

# Trading windows
OBSERVATION_END = dt_time(11, 0)        # No trades before 11:00 AM
ENTRY_START = dt_time(11, 0)            # Trading starts at 11:00 AM
ENTRY_END = dt_time(15, 15)             # No new trades after 3:15 PM
SQUARE_OFF = dt_time(15, 24)            # Force close at 3:24 PM

# StochRSI entry thresholds
STOCH_BUY_THRESHOLD = 10       # StochRSI K < 10 → oversold pullback → BUY
STOCH_SELL_THRESHOLD = 90      # StochRSI K > 90 → overbought rally → SELL

# StochRSI parameters
STOCH_RSI_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3

# Regime label: forward-looking EMA slope window
REGIME_FORWARD_CANDLES_5M = 30   # 30 five-min candles = 150 minutes
REGIME_SLOPE_THRESHOLD = 0.02   # Min EMA slope for BULL/BEAR classification

# Lot sizing
LOT_SIZE = 65
BASE_LOTS = 2

# LSTM
SEQ_LEN = 20

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════

def calc_rma(series, period):
    """Wilder's smoothing (RMA)."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_atr(df, period=14):
    """Average True Range using RMA."""
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
    """RSI using RMA."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = calc_rma(gain, period)
    avg_loss = calc_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_stochastic_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """
    Stochastic RSI = Stochastic oscillator applied to RSI values.
    Returns: stoch_k, stoch_d (both as Series)
    """
    rsi = calc_rsi(close, rsi_period)

    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    denom = (rsi_high - rsi_low).replace(0, 1e-10)
    stoch_rsi = (rsi - rsi_low) / denom * 100

    stoch_k = stoch_rsi.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()

    return stoch_k, stoch_d


def calc_ut_bot_direction(close, atr, key_value=1.0):
    """
    UT Bot trailing stop + direction.
    Returns: trail_stop (array), direction (array: +1 bullish, -1 bearish)
    """
    close_arr = close.values if hasattr(close, 'values') else np.array(close)
    atr_arr = atr.values if hasattr(atr, 'values') else np.array(atr)
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
    """Standard EMA."""
    return series.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════
#  FEATURE ENGINEERING — 3 TIMEFRAMES
# ═══════════════════════════════════════════

def build_features_1min(df):
    """Build 1-minute features (entry timing + momentum)."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    stoch_k, stoch_d = calc_stochastic_rsi(close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
                                            STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    trail, dirn = calc_ut_bot_direction(close, atr, ATR_KEY_VALUE)

    features = pd.DataFrame(index=df.index)

    # Core indicators
    features['atr_1m'] = atr
    features['rsi_1m'] = rsi
    features['stoch_k_1m'] = stoch_k
    features['stoch_d_1m'] = stoch_d
    features['stoch_kd_diff_1m'] = stoch_k - stoch_d
    features['ut_dir_1m'] = dirn
    features['close_vs_trail_1m'] = close.values - trail

    # Price momentum
    features['mom_3_1m'] = close.pct_change(3) * 100
    features['mom_5_1m'] = close.pct_change(5) * 100
    features['mom_10_1m'] = close.pct_change(10) * 100

    # Candle features
    features['body_1m'] = close - opn
    features['body_pct_1m'] = (close - opn) / opn * 100
    features['upper_wick_1m'] = high - close.where(close > opn, opn)
    features['lower_wick_1m'] = close.where(close < opn, opn) - low
    features['range_1m'] = high - low

    # Volatility
    features['std_5_1m'] = close.rolling(5).std()
    features['std_10_1m'] = close.rolling(10).std()

    # Moving averages
    ema_5 = calc_ema(close, 5)
    ema_10 = calc_ema(close, 10)
    ema_20 = calc_ema(close, 20)
    features['close_vs_ema5_1m'] = close - ema_5
    features['close_vs_ema10_1m'] = close - ema_10
    features['close_vs_ema20_1m'] = close - ema_20
    features['ema5_vs_ema10_1m'] = ema_5 - ema_10

    # High/Low channels
    features['close_vs_high5_1m'] = close - high.rolling(5).max()
    features['close_vs_low5_1m'] = close - low.rolling(5).min()

    return features


def build_features_3min(df_3m, df_1m):
    """Build 3-minute features and align to 1-min index via merge_asof."""
    close = df_3m['Close'].astype(float)
    high = df_3m['High'].astype(float)
    low = df_3m['Low'].astype(float)
    opn = df_3m['Open'].astype(float)

    atr = calc_atr(df_3m, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    stoch_k, stoch_d = calc_stochastic_rsi(close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
                                            STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    trail, dirn = calc_ut_bot_direction(close, atr, ATR_KEY_VALUE)

    ema_20 = calc_ema(close, 20)
    ema_50 = calc_ema(close, 50)

    feat = pd.DataFrame(index=df_3m.index)
    feat['Time'] = df_3m['Time']
    feat['atr_3m'] = atr.values
    feat['rsi_3m'] = rsi.values
    feat['stoch_k_3m'] = stoch_k.values
    feat['stoch_d_3m'] = stoch_d.values
    feat['stoch_kd_diff_3m'] = (stoch_k - stoch_d).values
    feat['ut_dir_3m'] = dirn
    feat['close_vs_trail_3m'] = close.values - trail
    feat['mom_3_3m'] = (close.pct_change(3) * 100).values
    feat['mom_5_3m'] = (close.pct_change(5) * 100).values
    feat['range_3m'] = (high - low).values
    feat['body_3m'] = (close - opn).values
    feat['ema20_3m'] = ema_20.values
    feat['ema50_3m'] = ema_50.values
    feat['ema20_vs_ema50_3m'] = (ema_20 - ema_50).values
    feat['ema20_slope_3m'] = (ema_20.diff(3) / 3).values  # slope over 3 bars

    # Merge to 1-min
    feat['Time'] = pd.to_datetime(feat['Time'])
    df_1m_time = pd.DataFrame({'Time': pd.to_datetime(df_1m['Time'])})

    merged = pd.merge_asof(
        df_1m_time.sort_values('Time'),
        feat.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    return merged.drop('Time', axis=1).reset_index(drop=True)


def build_features_5min(df_5m, df_1m):
    """Build 5-minute features (regime backbone) and align to 1-min index."""
    close = df_5m['Close'].astype(float)
    high = df_5m['High'].astype(float)
    low = df_5m['Low'].astype(float)
    opn = df_5m['Open'].astype(float)

    atr = calc_atr(df_5m, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    stoch_k, stoch_d = calc_stochastic_rsi(close, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
                                            STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    trail, dirn = calc_ut_bot_direction(close, atr, ATR_KEY_VALUE)

    ema_20 = calc_ema(close, 20)
    ema_50 = calc_ema(close, 50)

    feat = pd.DataFrame(index=df_5m.index)
    feat['Time'] = df_5m['Time']
    feat['atr_5m'] = atr.values
    feat['rsi_5m'] = rsi.values
    feat['stoch_k_5m'] = stoch_k.values
    feat['stoch_d_5m'] = stoch_d.values
    feat['stoch_kd_diff_5m'] = (stoch_k - stoch_d).values
    feat['ut_dir_5m'] = dirn
    feat['close_vs_trail_5m'] = close.values - trail
    feat['mom_3_5m'] = (close.pct_change(3) * 100).values
    feat['mom_5_5m'] = (close.pct_change(5) * 100).values
    feat['mom_10_5m'] = (close.pct_change(10) * 100).values
    feat['range_5m'] = (high - low).values
    feat['body_5m'] = (close - opn).values
    feat['ema20_5m'] = ema_20.values
    feat['ema50_5m'] = ema_50.values
    feat['ema20_vs_ema50_5m'] = (ema_20 - ema_50).values
    feat['ema20_slope_5m'] = (ema_20.diff(3) / 3).values
    feat['close_vs_ema20_5m'] = (close - ema_20).values
    feat['close_vs_ema50_5m'] = (close - ema_50).values
    feat['high_10_5m'] = high.rolling(10).max().values
    feat['low_10_5m'] = low.rolling(10).min().values
    feat['close_vs_high10_5m'] = (close.values - high.rolling(10).max().values)
    feat['close_vs_low10_5m'] = (close.values - low.rolling(10).min().values)

    # Merge to 1-min
    feat['Time'] = pd.to_datetime(feat['Time'])
    df_1m_time = pd.DataFrame({'Time': pd.to_datetime(df_1m['Time'])})

    merged = pd.merge_asof(
        df_1m_time.sort_values('Time'),
        feat.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    return merged.drop('Time', axis=1).reset_index(drop=True)


# ═══════════════════════════════════════════
#  REGIME LABEL GENERATION (ZERO LOOK-AHEAD)
# ═══════════════════════════════════════════

def generate_regime_labels(df_5m, forward_candles=30, slope_threshold=0.02):
    """
    Generate regime labels from 5-min data based on forward EMA20 slope.

    For each candle at time t:
      - Compute EMA20 at t and EMA20 at t + forward_candles
      - slope = (EMA20[t+N] - EMA20[t]) / EMA20[t] * 100
      - BULL (+1) if slope > threshold
      - BEAR (-1) if slope < -threshold
      - CHOP (0)  otherwise

    These labels are ONLY used for training. At inference, the model
    predicts regime from current/past features only.
    """
    close = df_5m['Close'].astype(float)
    ema_20 = calc_ema(close, 20)

    n = len(df_5m)
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


def map_5m_labels_to_1m(df_5m, labels_5m, df_1m):
    """Map 5-minute regime labels to 1-minute index using merge_asof."""
    label_df = pd.DataFrame({
        'Time': pd.to_datetime(df_5m['Time']),
        'regime_label': labels_5m,
    })

    df_1m_time = pd.DataFrame({'Time': pd.to_datetime(df_1m['Time'])})

    merged = pd.merge_asof(
        df_1m_time.sort_values('Time'),
        label_df.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    return merged['regime_label'].fillna(0).astype(int).values


# ═══════════════════════════════════════════
#  LSTM MODEL (OPTIONAL)
# ═══════════════════════════════════════════

if HAS_TORCH:
    class LSTMClassifier(nn.Module):
        """2-layer LSTM for sequential regime confirmation."""
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]
            out = self.dropout(last_hidden)
            return self.fc(out)


    class SequenceDataset(Dataset):
        """Sliding window dataset for LSTM training."""
        def __init__(self, features, labels, seq_len=20):
            self.features = features
            self.labels = labels
            self.seq_len = seq_len

        def __len__(self):
            return max(0, len(self.features) - self.seq_len)

        def __getitem__(self, idx):
            x = self.features[idx:idx + self.seq_len]
            y = self.labels[idx + self.seq_len - 1]
            y_mapped = y + 1  # -1→0, 0→1, 1→2
            return torch.FloatTensor(x), torch.LongTensor([y_mapped]).squeeze()


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=500):
    """Train Random Forest regime classifier."""
    # Map labels: -1→0, 0→1, 1→2
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
        verbose=1,
    )

    print(f"\n  Training Random Forest ({n_estimators} trees)...")
    model.fit(X_train, y_train_mapped)

    # Predictions
    train_pred = model.predict(X_train) - 1
    test_pred = model.predict(X_test) - 1

    train_acc = accuracy_score(y_train, train_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    print(f"\n  RF Train accuracy: {train_acc:.1f}%")
    print(f"  RF Test accuracy:  {test_acc:.1f}%")

    # Class report
    print(f"\n  Test Classification Report:")
    target_names = ['BEAR (-1)', 'CHOP (0)', 'BULL (+1)']
    print(classification_report(y_test + 1, test_pred + 1, target_names=target_names, zero_division=0))

    return model


def train_lstm(X_train, y_train, X_test, y_test, scaler, seq_len=20,
               epochs=30, batch_size=256, lr=0.001):
    """Train LSTM sequential classifier."""
    if not HAS_TORCH:
        print("  ⚠️ PyTorch not available. Skipping LSTM.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  LSTM Device: {device}")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = SequenceDataset(X_train_scaled, y_train, seq_len)
    test_dataset = SequenceDataset(X_test_scaled, y_test, seq_len)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("  ⚠️ Not enough data for LSTM sequences. Skipping LSTM.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    model = LSTMClassifier(input_size=n_features, hidden_size=64,
                           num_layers=2, dropout=0.3).to(device)

    # Class weights
    class_counts = np.bincount(y_train + 1, minlength=3).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()

        avg_test_loss = test_loss / max(len(test_loader), 1)
        scheduler.step(avg_test_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_acc = train_correct / max(train_total, 1) * 100
            test_acc = test_correct / max(test_total, 1) * 100
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss/max(len(train_loader),1):.4f} Acc: {train_acc:.1f}% | "
                  f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.1f}%")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.cpu()

    # Final accuracy
    model.eval()
    test_dataset_full = SequenceDataset(scaler.transform(X_test), y_test, seq_len)
    test_loader_full = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader_full:
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    print(f"\n  LSTM Final Test Accuracy: {correct/max(total,1)*100:.1f}%")

    return model


def predict_lstm_full(model, features_scaled, seq_len=20):
    """Generate LSTM predictions for the full dataset."""
    n = len(features_scaled)
    predictions = np.zeros(n, dtype=int)

    if model is None or not HAS_TORCH:
        return predictions

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(seq_len, n + 1):
            seq = features_scaled[i - seq_len:i]
            x = torch.FloatTensor(seq).unsqueeze(0).to(device)
            outputs = model(x)
            pred_class = outputs.argmax(dim=1).item()
            predictions[i - 1] = pred_class - 1  # unmap

    return predictions


def ensemble_predict(rf_pred, lstm_pred):
    """
    Combine RF and LSTM predictions.
    Both agree → use that signal (HIGH confidence)
    One active, other HOLD → use active signal (MEDIUM confidence)
    Conflicting → HOLD (sit out)
    """
    n = len(rf_pred)
    final = np.zeros(n, dtype=int)

    for i in range(n):
        r = rf_pred[i]
        l = lstm_pred[i] if i < len(lstm_pred) else 0

        if r == l:
            final[i] = r
        elif r != 0 and l == 0:
            final[i] = r
        elif r == 0 and l != 0:
            final[i] = l
        else:
            final[i] = 0  # conflicting → HOLD

    return final


# ═══════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════

def backtest_strategy(df, regime_predictions, stoch_k_vals, stoch_d_vals,
                      ut_dir_vals, atr_vals,
                      stoch_buy_thresh=10, stoch_sell_thresh=90):
    """
    Backtest the multi-timeframe regime + StochRSI strategy.

    Rules:
      PHASE 1 (09:15-11:00): No trades — observation only
      PHASE 2 (11:00-15:15): Active trading
        LONG:  regime=BULL AND StochRSI_K < buy_thresh AND UT_Bot=bullish
        SHORT: regime=BEAR AND StochRSI_K > sell_thresh AND UT_Bot=bearish
      REGIME CHANGE: Close existing + open opposite if StochRSI confirms

    Exit:
      - Trailing SL (ATR-based)
      - Opposite regime signal
      - Square off at 15:24
    """
    close = df['Close'].astype(float)
    high_v = df['High'].astype(float)
    low_v = df['Low'].astype(float)

    pos = None
    all_trades = []
    daily_results = {}
    prev_date = None

    # Lot management
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

        # ── Day boundary ──
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

            # Lot adjustment
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

        # ── Square off ──
        if pos and t >= SQUARE_OFF:
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "SQUARE_OFF",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
            continue

        # ── Skip if before observation period ends ──
        in_window = ENTRY_START <= t <= ENTRY_END

        # ── SL check ──
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

        # ── Trail SL update ──
        if pos:
            if pos['dir'] == "LONG":
                new_sl = c - curr_atr * ATR_KEY_VALUE
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
            elif pos['dir'] == "SHORT":
                new_sl = c + curr_atr * ATR_KEY_VALUE
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl

        # ── Get current indicators ──
        regime = regime_predictions[i]
        sk = stoch_k_vals[i] if not np.isnan(stoch_k_vals[i]) else 50
        ut_d = ut_dir_vals[i]

        # ── Regime change — close existing position ──
        if regime == 1 and pos and pos['dir'] == "SHORT":
            # Regime flipped to BULL while SHORT → close
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "REGIME_FLIP",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        elif regime == -1 and pos and pos['dir'] == "LONG":
            # Regime flipped to BEAR while LONG → close
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "REGIME_FLIP",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None

        # ── New entry ──
        if not pos and in_window and curr_atr >= MIN_ATR:
            # BUY: regime=BULL + StochRSI oversold + UT Bot bullish
            if regime == 1 and sk < stoch_buy_thresh and ut_d == 1:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}

            # SELL: regime=BEAR + StochRSI overbought + UT Bot bearish
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


# ═══════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════

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
    print(f"  🌲 RANDOM FOREST + STOCHRSI — {label}")
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

    # Daily results
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


def print_detailed_daily_log(all_trades, daily_results, log_file="rf_detailed_log.txt"):
    """Print AND save detailed per-trade log."""
    sorted_days = sorted(daily_results.keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*110}")
    out(f"  📋 DETAILED TRADE LOG — RANDOM FOREST + STOCHRSI")
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
        out(f"  │ 📅 {day} ({day_name}) | Lots: {lots} (qty={lots*LOT_SIZE}) | Trades: {len(trades)} (W:{wins} L:{losses_count} {wr:.0f}%) | Day P&L: {day_pnl:+.2f} {icon} | Cum: {cum_pnl:+.2f}")
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
    out(f"  GRAND TOTAL: {cum_pnl:+.2f} pts")
    out(f"{'='*110}")

    log_path = os.path.join(SCRIPT_DIR, log_file)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n💾 Detailed log saved: {log_path}")


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    global ATR_KEY_VALUE, MIN_ATR, SEQ_LEN, ENTRY_START, ENTRY_END, SQUARE_OFF
    global STOCH_BUY_THRESHOLD, STOCH_SELL_THRESHOLD, OBSERVATION_END

    parser = argparse.ArgumentParser(description="Multi-TF Random Forest + StochRSI Strategy (NIFTY)")
    parser.add_argument("--file-1m", default=os.path.join(SCRIPT_DIR, "nifty_1min_data.csv"), help="1-min data")
    parser.add_argument("--file-3m", default=os.path.join(SCRIPT_DIR, "nifty_3min_data.csv"), help="3-min data")
    parser.add_argument("--file-5m", default=os.path.join(SCRIPT_DIR, "nifty_5min_data.csv"), help="5-min data")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--stoch-buy", type=float, default=STOCH_BUY_THRESHOLD,
                        help="StochRSI K threshold for BUY (default: 10)")
    parser.add_argument("--stoch-sell", type=float, default=STOCH_SELL_THRESHOLD,
                        help="StochRSI K threshold for SELL (default: 90)")
    parser.add_argument("--regime-thresh", type=float, default=REGIME_SLOPE_THRESHOLD,
                        help="EMA slope threshold for regime classification")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="LSTM lookback window")
    parser.add_argument("--lstm-epochs", type=int, default=30, help="LSTM training epochs")
    parser.add_argument("--rf-trees", type=int, default=500, help="Random Forest n_estimators")
    parser.add_argument("--obs-end", type=str, default="11:00",
                        help="End of observation period (HH:MM). No trades before this.")
    parser.add_argument("--window-end", type=str, default="15:15", help="Entry window end (HH:MM)")
    parser.add_argument("--square-off", type=str, default="15:24", help="Square off time (HH:MM)")
    parser.add_argument("--test-from", type=str, default="2026-04-01",
                        help="Test data start date (YYYY-MM-DD)")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM, use RF only")
    parser.add_argument("--lstm-only", action="store_true", help="Skip RF, use LSTM only")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    STOCH_BUY_THRESHOLD = args.stoch_buy
    STOCH_SELL_THRESHOLD = args.stoch_sell
    SEQ_LEN = args.seq_len

    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    OBSERVATION_END = parse_time(args.obs_end)
    ENTRY_START = OBSERVATION_END  # Trading starts when observation ends
    ENTRY_END = parse_time(args.window_end)
    SQUARE_OFF = parse_time(args.square_off)

    test_from_date = datetime.strptime(args.test_from, "%Y-%m-%d").date()

    # ── Load data ──
    print(f"{'='*70}")
    print(f"  🌲 MULTI-TIMEFRAME RANDOM FOREST + STOCHRSI STRATEGY")
    print(f"{'='*70}")

    print(f"\n📊 Loading data...")
    try:
        df_1m = pd.read_csv(args.file_1m)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file_1m}")
        return
    df_1m['Time'] = pd.to_datetime(df_1m['Time'])
    df_1m = df_1m.sort_values('Time').reset_index(drop=True)
    print(f"  1-min: {len(df_1m):,} candles | {df_1m['Time'].dt.date.nunique()} days")

    try:
        df_3m = pd.read_csv(args.file_3m)
        df_3m['Time'] = pd.to_datetime(df_3m['Time'])
        df_3m = df_3m.sort_values('Time').reset_index(drop=True)
        print(f"  3-min: {len(df_3m):,} candles | {df_3m['Time'].dt.date.nunique()} days")
    except FileNotFoundError:
        print(f"⚠️  3-min file not found: {args.file_3m}")
        df_3m = None

    try:
        df_5m = pd.read_csv(args.file_5m)
        df_5m['Time'] = pd.to_datetime(df_5m['Time'])
        df_5m = df_5m.sort_values('Time').reset_index(drop=True)
        print(f"  5-min: {len(df_5m):,} candles | {df_5m['Time'].dt.date.nunique()} days")
    except FileNotFoundError:
        print(f"❌ 5-min file required for regime detection. File not found: {args.file_5m}")
        return

    print(f"  Date range: {df_1m['Time'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_1m['Time'].iloc[-1].strftime('%Y-%m-%d')}")

    print(f"\n  Strategy Config:")
    print(f"    Observation:  09:15 - {args.obs_end} (no trades)")
    print(f"    Trading:      {args.obs_end} - {args.window_end}")
    print(f"    Square off:   {args.square_off}")
    print(f"    StochRSI BUY: K < {STOCH_BUY_THRESHOLD}")
    print(f"    StochRSI SELL: K > {STOCH_SELL_THRESHOLD}")
    print(f"    ATR: RMA({ATR_PERIOD}) × {ATR_KEY_VALUE} | Min: {MIN_ATR}")
    print(f"    Test from:    {args.test_from}")

    # ── Build features ──
    print(f"\n📊 Building features...")
    feat_1m = build_features_1min(df_1m)
    print(f"  1-min features: {feat_1m.shape[1]} columns")

    if df_3m is not None:
        feat_3m = build_features_3min(df_3m, df_1m)
        print(f"  3-min features: {feat_3m.shape[1]} columns")
    else:
        feat_3m = None

    feat_5m = build_features_5min(df_5m, df_1m)
    print(f"  5-min features: {feat_5m.shape[1]} columns")

    # Combine all features
    feature_parts = [feat_1m]
    if feat_3m is not None:
        feature_parts.append(feat_3m)
    feature_parts.append(feat_5m)
    features = pd.concat(feature_parts, axis=1)

    # Remove any duplicate columns
    features = features.loc[:, ~features.columns.duplicated()]
    print(f"  Total combined features: {features.shape[1]} columns")

    # ── Generate regime labels from 5-min ──
    print(f"\n🏷️  Generating regime labels (from 5-min data)...")
    regime_labels_5m = generate_regime_labels(df_5m, REGIME_FORWARD_CANDLES_5M,
                                              args.regime_thresh)
    bull_5m = (regime_labels_5m == 1).sum()
    bear_5m = (regime_labels_5m == -1).sum()
    chop_5m = (regime_labels_5m == 0).sum()
    print(f"  5-min labels: BULL={bull_5m} ({bull_5m/len(regime_labels_5m)*100:.1f}%) | "
          f"BEAR={bear_5m} ({bear_5m/len(regime_labels_5m)*100:.1f}%) | "
          f"CHOP={chop_5m} ({chop_5m/len(regime_labels_5m)*100:.1f}%)")

    # Map to 1-min
    labels = map_5m_labels_to_1m(df_5m, regime_labels_5m, df_1m)
    bull_1m = (labels == 1).sum()
    bear_1m = (labels == -1).sum()
    chop_1m = (labels == 0).sum()
    print(f"  1-min mapped:  BULL={bull_1m} ({bull_1m/len(labels)*100:.1f}%) | "
          f"BEAR={bear_1m} ({bear_1m/len(labels)*100:.1f}%) | "
          f"CHOP={chop_1m} ({chop_1m/len(labels)*100:.1f}%)")

    # ── Filter to trading window ──
    times = df_1m['Time'].dt.time
    # For training, use all market hours to capture full patterns
    window_mask = (times >= dt_time(9, 20)) & (times <= ENTRY_END)
    print(f"\n  Window candles: {window_mask.sum()} (of {len(df_1m)})")

    # ── Clean features ──
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    # ── Train/Test split ──
    dates = df_1m['Time'].dt.date

    train_mask = (dates < test_from_date) & window_mask
    test_mask = (dates >= test_from_date) & window_mask

    train_dates_set = set(dates[dates < test_from_date].unique())
    test_dates_set = set(dates[dates >= test_from_date].unique())

    X_train = features[train_mask].values
    y_train = labels[train_mask]
    X_test = features[test_mask].values
    y_test = labels[test_mask]

    print(f"\n  Train: {len(X_train)} samples ({len(train_dates_set)} days) "
          f"[up to {args.test_from}]")
    print(f"  Test:  {len(X_test)} samples ({len(test_dates_set)} days) "
          f"[{args.test_from} onwards]")

    # ── Train Random Forest ──
    rf_model = None
    if not args.lstm_only:
        print(f"\n🌳 Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train, X_test, y_test,
                                        n_estimators=args.rf_trees)

        # Feature importance
        importance = rf_model.feature_importances_
        feat_names = features.columns.tolist()
        top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:15]
        print(f"\n  📊 TOP 15 FEATURES (Random Forest):")
        for name, imp in top_features:
            print(f"    {name:>30}: {imp:.4f}")

    # ── Train LSTM ──
    lstm_model = None
    scaler = StandardScaler()
    scaler.fit(X_train)

    if not args.no_lstm and HAS_TORCH:
        print(f"\n🧠 Training LSTM model (seq_len={SEQ_LEN}, epochs={args.lstm_epochs})...")
        lstm_model = train_lstm(X_train, y_train, X_test, y_test, scaler,
                                seq_len=SEQ_LEN, epochs=args.lstm_epochs)

    # ── Generate predictions for backtest ──
    print(f"\n🔮 Generating predictions...")

    full_features = features.values

    # RF predictions
    if rf_model is not None:
        rf_full_pred = np.zeros(len(df_1m), dtype=int)
        rf_test_pred = rf_model.predict(X_test) - 1  # unmap
        rf_full_pred[test_mask] = rf_test_pred
        print(f"  RF:     BULL={sum(rf_test_pred==1)} BEAR={sum(rf_test_pred==-1)} "
              f"CHOP={sum(rf_test_pred==0)}")
    else:
        rf_full_pred = np.zeros(len(df_1m), dtype=int)

    # LSTM predictions
    if lstm_model is not None:
        full_scaled = scaler.transform(full_features)
        lstm_full_pred_raw = predict_lstm_full(lstm_model, full_scaled, SEQ_LEN)
        lstm_full_pred = np.zeros(len(df_1m), dtype=int)
        lstm_full_pred[test_mask] = lstm_full_pred_raw[test_mask]
        lstm_test_pred = lstm_full_pred_raw[test_mask]
        print(f"  LSTM:   BULL={sum(lstm_test_pred==1)} BEAR={sum(lstm_test_pred==-1)} "
              f"CHOP={sum(lstm_test_pred==0)}")
    else:
        lstm_full_pred = np.zeros(len(df_1m), dtype=int)

    # Ensemble
    if rf_model is not None and lstm_model is not None:
        final_predictions = ensemble_predict(rf_full_pred, lstm_full_pred)
        ens_test = final_predictions[test_mask]
        print(f"  Ensemble: BULL={sum(ens_test==1)} BEAR={sum(ens_test==-1)} "
              f"CHOP={sum(ens_test==0)}")
    elif rf_model is not None:
        final_predictions = rf_full_pred
        print("  Using Random Forest only")
    elif lstm_model is not None:
        final_predictions = lstm_full_pred
        print("  Using LSTM only")
    else:
        print("❌ No models trained!")
        return

    # ── Prepare backtest indicators ──
    atr_vals = calc_atr(df_1m, ATR_PERIOD).values
    stoch_k_vals, stoch_d_vals = calc_stochastic_rsi(
        df_1m['Close'].astype(float), STOCH_RSI_PERIOD, STOCH_RSI_PERIOD,
        STOCH_K_SMOOTH, STOCH_D_SMOOTH
    )
    stoch_k_vals = stoch_k_vals.values
    stoch_d_vals = stoch_d_vals.values
    _, ut_dir_vals = calc_ut_bot_direction(
        df_1m['Close'].astype(float), calc_atr(df_1m, ATR_PERIOD), ATR_KEY_VALUE
    )

    # ── Backtest on test set ──
    print(f"\n🚀 Running backtest on TEST data ({args.test_from} onwards)...")
    test_start = min(test_dates_set)
    test_end = max(test_dates_set)
    test_df_mask = dates >= test_from_date
    df_test = df_1m[test_df_mask].reset_index(drop=True)
    pred_test = final_predictions[test_df_mask]
    atr_test = atr_vals[test_df_mask]
    stoch_k_test = stoch_k_vals[test_df_mask]
    stoch_d_test = stoch_d_vals[test_df_mask]
    ut_dir_test = ut_dir_vals[test_df_mask]

    all_trades, daily_results = backtest_strategy(
        df_test, pred_test, stoch_k_test, stoch_d_test,
        ut_dir_test, atr_test,
        stoch_buy_thresh=STOCH_BUY_THRESHOLD,
        stoch_sell_thresh=STOCH_SELL_THRESHOLD,
    )

    print_results(all_trades, daily_results, f"TEST ({test_start} → {test_end})")
    print_detailed_daily_log(all_trades, daily_results)

    # ── Save trades ──
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_path = os.path.join(SCRIPT_DIR, "rf_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"\n💾 Trades saved: {trades_path}")

    # ── Save models ──
    if rf_model is not None:
        rf_path = os.path.join(SCRIPT_DIR, "rf_regime_model.pkl")
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"💾 Random Forest saved: {rf_path}")

    if lstm_model is not None and HAS_TORCH:
        lstm_path = os.path.join(SCRIPT_DIR, "rf_lstm_model.pt")
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'input_size': lstm_model.lstm.input_size,
            'hidden_size': lstm_model.lstm.hidden_size,
            'num_layers': lstm_model.lstm.num_layers,
            'seq_len': SEQ_LEN,
        }, lstm_path)
        print(f"💾 LSTM model saved: {lstm_path}")

    # Save scaler
    scaler_path = os.path.join(SCRIPT_DIR, "rf_feature_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Feature scaler saved: {scaler_path}")

    # Save feature columns
    cols_path = os.path.join(SCRIPT_DIR, "rf_feature_columns.pkl")
    with open(cols_path, 'wb') as f:
        pickle.dump(features.columns.tolist(), f)
    print(f"💾 Feature columns saved: {cols_path}")

    # ── Save Training Metadata ──
    metadata = {
        "strategy": "Random Forest + StochRSI + UT Bot (Multi-TF)",
        "train_samples": int(len(X_train)),
        "train_days": int(len(train_dates_set)),
        "train_up_to": args.test_from,
        "test_samples": int(len(X_test)),
        "test_days": int(len(test_dates_set)),
        "test_from": args.test_from,
        "features": int(features.shape[1]),
        "rf_trees": args.rf_trees,
        "stoch_buy_threshold": STOCH_BUY_THRESHOLD,
        "stoch_sell_threshold": STOCH_SELL_THRESHOLD,
        "observation_end": args.obs_end,
        "last_trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(SCRIPT_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Training metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
