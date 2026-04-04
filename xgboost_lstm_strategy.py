"""
xgboost_lstm_strategy.py — XGBoost + LSTM Ensemble Strategy for NIFTY

ZERO LOOK-AHEAD DESIGN:
  Labels are generated from the COMPLETED next candle's return.
  features[t] → label = sign(close[t+1] - close[t])
  At inference: we use features on the current candle to predict
  the next candle's direction — using only past/present data.

MODELS:
  1. XGBoost — tabular feature classifier (BUY/SELL/HOLD)
  2. LSTM   — 20-candle sequential pattern detector (short-term pickup)
  3. Ensemble — combines both for final signal

Usage:
    python xgboost_lstm_strategy.py                             # Train + backtest
    python xgboost_lstm_strategy.py --threshold 5               # 5 pts for signal
    python xgboost_lstm_strategy.py --seq-len 20                # LSTM lookback
    python xgboost_lstm_strategy.py --test-from 2025-01-01      # Test split date
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import os
from datetime import datetime, time as dt_time
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("❌ XGBoost not installed. Run: pip install xgboost")
    exit(1)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("❌ PyTorch not installed. Run: pip install torch")
    exit(1)

from sklearn.preprocessing import StandardScaler


# ─────────── Config ───────────
ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5

ENTRY_START = dt_time(9, 20)      # 9:20 AM
ENTRY_END = dt_time(15, 15)       # 3:15 PM
SQUARE_OFF = dt_time(15, 24)      # 3:24 PM

LOT_SIZE = 65
BASE_LOTS = 2

THRESHOLD = 5.0                   # Min pts for buy/sell label (next candle return)
SEQ_LEN = 20                      # LSTM lookback window

# Fixed train/test split
TRAIN_END_YEAR = 2024
TEST_START_YEAR = 2025


# ─────────── Indicators (identical to catboost_strategy.py) ───────────

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


def calc_ut_bot_direction(close, atr, key_value=1.0):
    """Returns trail_stop and direction arrays."""
    n = len(close)
    trail_stop = np.zeros(n)
    direction = np.zeros(n)
    trail_stop[0] = close[0]
    direction[0] = 1

    for i in range(1, n):
        nloss = atr[i] * key_value
        prev_ts = trail_stop[i-1]
        prev_dir = direction[i-1]

        if prev_dir == 1:
            new_ts = close[i] - nloss
            trail_stop[i] = max(new_ts, prev_ts)
            if close[i] < trail_stop[i]:
                direction[i] = -1
                trail_stop[i] = close[i] + nloss
            else:
                direction[i] = 1
        else:
            new_ts = close[i] + nloss
            trail_stop[i] = min(new_ts, prev_ts)
            if close[i] > trail_stop[i]:
                direction[i] = 1
                trail_stop[i] = close[i] - nloss
            else:
                direction[i] = -1

    return trail_stop, direction


# ─────────── Feature Engineering ───────────

def build_features_1min(df):
    """Build features from 1-minute data."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    trail, dirn = calc_ut_bot_direction(close.values, atr.values, ATR_KEY_VALUE)

    features = pd.DataFrame(index=df.index)
    features['atr_1m'] = atr
    features['rsi_1m'] = rsi
    features['ut_dir_1m'] = dirn
    features['close_vs_trail_1m'] = close.values - trail

    # Price momentum
    features['mom_3'] = close.pct_change(3) * 100
    features['mom_5'] = close.pct_change(5) * 100
    features['mom_10'] = close.pct_change(10) * 100

    # Candle features
    features['body_1m'] = close - opn
    features['body_pct_1m'] = (close - opn) / opn * 100
    features['upper_wick_1m'] = high - close.where(close > opn, opn)
    features['lower_wick_1m'] = close.where(close < opn, opn) - low
    features['range_1m'] = high - low

    # Volatility
    features['std_5'] = close.rolling(5).std()
    features['std_10'] = close.rolling(10).std()

    # Moving averages
    features['sma_5'] = close.rolling(5).mean()
    features['sma_10'] = close.rolling(10).mean()
    features['sma_20'] = close.rolling(20).mean()
    features['close_vs_sma5'] = close - features['sma_5']
    features['close_vs_sma10'] = close - features['sma_10']
    features['sma5_vs_sma10'] = features['sma_5'] - features['sma_10']

    # High/Low channels
    features['high_5'] = high.rolling(5).max()
    features['low_5'] = low.rolling(5).min()
    features['close_vs_high5'] = close - features['high_5']
    features['close_vs_low5'] = close - features['low_5']

    return features


def build_features_2min(df_2m, df_1m):
    """
    Build features from 2-min data and align to 1-min index.
    Uses forward-fill to map 2-min features to the 1-min timeframe.
    """
    close_2m = df_2m['Close'].astype(float)
    high_2m = df_2m['High'].astype(float)
    low_2m = df_2m['Low'].astype(float)

    atr_2m = calc_atr(df_2m, ATR_PERIOD)
    rsi_2m = calc_rsi(close_2m, 14)
    trail_2m, dir_2m = calc_ut_bot_direction(close_2m.values, atr_2m.values, ATR_KEY_VALUE)

    feat_2m = pd.DataFrame(index=df_2m.index)
    feat_2m['Time'] = df_2m['Time']
    feat_2m['atr_2m'] = atr_2m.values
    feat_2m['rsi_2m'] = rsi_2m.values
    feat_2m['ut_dir_2m'] = dir_2m
    feat_2m['close_vs_trail_2m'] = close_2m.values - trail_2m
    feat_2m['mom_3_2m'] = close_2m.pct_change(3).values * 100
    feat_2m['mom_5_2m'] = close_2m.pct_change(5).values * 100
    feat_2m['range_2m'] = (high_2m - low_2m).values
    feat_2m['body_2m'] = (close_2m - df_2m['Open'].astype(float)).values

    # Merge to 1-min by time (forward fill)
    feat_2m['Time'] = pd.to_datetime(feat_2m['Time'])
    df_1m_time = pd.DataFrame({'Time': pd.to_datetime(df_1m['Time'])})

    merged = pd.merge_asof(
        df_1m_time.sort_values('Time'),
        feat_2m.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    return merged.drop('Time', axis=1).reset_index(drop=True)


# ─────────── Label Generation (ZERO LOOK-AHEAD) ───────────

def generate_labels_no_lookahead(close_series, threshold=5.0):
    """
    ZERO LOOK-AHEAD label generation.

    label[t] = class based on return of candle t+1
      = close[t+1] - close[t]

    During training: features[t] paired with label[t]
    At inference:    features[t] used to PREDICT what happens at t+1

    This is standard supervised learning — NOT look-ahead bias.
    The label encodes what ALREADY HAPPENED at the next timestamp.
    The last row gets label=0 (unknown future — dropped from training).

    Returns array of labels: 1=BUY, -1=SELL, 0=HOLD
    """
    close = close_series.astype(float).values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(n - 1):
        # Return of the NEXT completed candle
        next_return = close[i + 1] - close[i]

        if next_return > threshold:
            labels[i] = 1   # BUY — next candle went up significantly
        elif next_return < -threshold:
            labels[i] = -1  # SELL — next candle went down significantly
        else:
            labels[i] = 0   # HOLD — next candle was flat

    # Last row: we don't know the future → HOLD (will be excluded from training)
    labels[-1] = 0

    return labels


# ─────────── LSTM Model Definition ───────────

class LSTMClassifier(nn.Module):
    """
    2-layer LSTM for short-term sequential pattern detection.
    Input: (batch, seq_len, n_features)
    Output: (batch, 3) — softmax over BUY/HOLD/SELL
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: SELL(-1), HOLD(0), BUY(1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state from the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logits = self.fc(out)  # (batch, 3)
        return logits


class SequenceDataset(Dataset):
    """Sliding window dataset for LSTM training."""
    def __init__(self, features, labels, seq_len=20):
        self.features = features  # numpy array (n_samples, n_features)
        self.labels = labels      # numpy array (n_samples,)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        # Window of seq_len feature vectors
        x = self.features[idx:idx + self.seq_len]
        # Label is for the LAST candle in the window
        y = self.labels[idx + self.seq_len - 1]
        # Map labels: -1→0, 0→1, 1→2 for CrossEntropyLoss
        y_mapped = y + 1  # -1→0, 0→1, 1→2
        return torch.FloatTensor(x), torch.LongTensor([y_mapped]).squeeze()


# ─────────── Training Functions ───────────

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost multi-class classifier."""
    # Map labels: -1→0, 0→1, 1→2 for XGBoost
    y_train_mapped = y_train + 1
    y_test_mapped = y_test + 1

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        random_state=42,
        use_label_encoder=False,
        tree_method='hist',
        verbosity=1,
    )

    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_test, y_test_mapped)],
        verbose=100,
    )

    # Predictions (mapped back: 0→-1, 1→0, 2→1)
    train_pred = model.predict(X_train) - 1
    test_pred = model.predict(X_test) - 1

    train_acc = (train_pred == y_train).mean() * 100
    test_acc = (test_pred == y_test).mean() * 100
    print(f"\n  XGBoost Train accuracy: {train_acc:.1f}%")
    print(f"  XGBoost Test accuracy:  {test_acc:.1f}%")

    return model


def train_lstm(X_train, y_train, X_test, y_test, scaler, seq_len=20,
               epochs=30, batch_size=256, lr=0.001):
    """Train LSTM sequential classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  LSTM Device: {device}")

    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets
    train_dataset = SequenceDataset(X_train_scaled, y_train, seq_len)
    test_dataset = SequenceDataset(X_test_scaled, y_test, seq_len)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("  ⚠️ Not enough data for LSTM sequences. Skipping LSTM.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    model = LSTMClassifier(input_size=n_features, hidden_size=64,
                           num_layers=2, dropout=0.3).to(device)

    # Class weights to handle imbalance (HOLD is usually dominant)
    class_counts = np.bincount(y_train + 1, minlength=3).astype(float)
    class_counts = np.maximum(class_counts, 1.0)  # avoid division by zero
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 3  # normalize
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
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

        # Eval
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

        # Early stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.cpu()

    # Final accuracy
    model.eval()
    test_dataset_full = SequenceDataset(X_test_scaled, y_test, seq_len)
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


# ─────────── Ensemble Prediction ───────────

def ensemble_predict(xgb_pred, lstm_pred):
    """
    Combine XGBoost and LSTM predictions.

    Rules:
      Both agree          → use that signal (HIGH confidence)
      One says BUY/SELL,
        other says HOLD   → use the active signal (MEDIUM confidence)
      Conflicting signals → HOLD (sit out)
    """
    n = len(xgb_pred)
    final = np.zeros(n, dtype=int)

    for i in range(n):
        xg = xgb_pred[i]
        ls = lstm_pred[i] if i < len(lstm_pred) else 0

        if xg == ls:
            # Both agree
            final[i] = xg
        elif xg != 0 and ls == 0:
            # XGBoost has signal, LSTM neutral
            final[i] = xg
        elif xg == 0 and ls != 0:
            # LSTM short-term pickup, XGBoost neutral
            final[i] = ls
        else:
            # Conflicting (BUY vs SELL) → sit out
            final[i] = 0

    return final


def predict_lstm_full(model, features_scaled, seq_len=20):
    """
    Generate LSTM predictions for the full dataset.
    First seq_len-1 rows get HOLD (not enough history).
    """
    n = len(features_scaled)
    predictions = np.zeros(n, dtype=int)

    if model is None:
        return predictions

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(seq_len, n + 1):
            seq = features_scaled[i - seq_len:i]
            x = torch.FloatTensor(seq).unsqueeze(0).to(device)
            outputs = model(x)
            pred_class = outputs.argmax(dim=1).item()
            # Unmap: 0→-1, 1→0, 2→1
            predictions[i - 1] = pred_class - 1

    return predictions


# ─────────── Backtest ───────────

def backtest_predictions(df, predictions, atr_vals):
    """Backtest using ensemble predictions as signals."""
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

        # Signal from ensemble
        pred = predictions[i]

        # Opposite signal close
        if pred == 1 and pos and pos['dir'] == "SHORT":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None
        elif pred == -1 and pos and pos['dir'] == "LONG":
            trade = _make_trade(pos, c, df.iloc[i]['Time'], "OPPOSITE",
                                current_lots, pos.get('initial_sl'))
            all_trades.append(trade)
            _add_daily(daily_results, curr_date, trade)
            pos = None

        # New entry
        if not pos and in_window and curr_atr >= MIN_ATR:
            if pred == 1:
                sl = c - curr_atr * ATR_KEY_VALUE
                pos = {'dir': 'LONG', 'entry': c, 'sl': sl, 'initial_sl': sl,
                       'entry_time': df.iloc[i]['Time'], 'entry_idx': i}
            elif pred == -1:
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


# ─────────── Reports ───────────

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

    # Drawdown
    cumulative = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (peak - cumulative).max()

    trading_days = sum(1 for d in daily_results.values() if len(d['trades']) > 0)
    win_days = sum(1 for d in daily_results.values() if d['pnl'] > 0)

    print(f"\n{'='*60}")
    print(f"  🤖 XGBOOST + LSTM ENSEMBLE — {label}")
    print(f"{'='*60}")
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

    print(f"\n{'='*60}")


def print_detailed_daily_log(all_trades, daily_results, log_file="xgboost_lstm_detailed_log.txt"):
    """Print AND save detailed per-trade explanation for each day."""
    sorted_days = sorted(daily_results.keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*110}")
    out(f"  📋 DETAILED TRADE LOG — XGBOOST + LSTM ENSEMBLE")
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
        out(f"  Day Summary: {day_pnl:+.2f} pts ({day_name}) | Raw trades: {len(trades)} | Multiplier: {lots // BASE_LOTS}x")

    out(f"\n{'='*110}")
    out(f"  GRAND TOTAL: {cum_pnl:+.2f} pts")
    out(f"{'='*110}")

    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n💾 Detailed log saved: {log_file}")


# ─────────── Main ───────────

def main():
    global ATR_KEY_VALUE, MIN_ATR, THRESHOLD, SEQ_LEN, ENTRY_START, ENTRY_END, SQUARE_OFF

    parser = argparse.ArgumentParser(description="XGBoost + LSTM Ensemble Strategy (NIFTY)")
    parser.add_argument("--file-1m", default="nifty_1min_data.csv", help="1-min data")
    parser.add_argument("--file-2m", default="nifty_2min_data.csv", help="2-min data")
    parser.add_argument("--atr-key", type=float, default=ATR_KEY_VALUE)
    parser.add_argument("--min-atr", type=float, default=MIN_ATR)
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Min pts for buy/sell label (next candle return)")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                        help="LSTM lookback window (candles)")
    parser.add_argument("--lstm-epochs", type=int, default=30,
                        help="LSTM training epochs")
    parser.add_argument("--window-start", type=str, default="09:20",
                        help="Entry window start (HH:MM)")
    parser.add_argument("--window-end", type=str, default="15:15",
                        help="Entry window end (HH:MM)")
    parser.add_argument("--square-off", type=str, default="15:24",
                        help="Square off time (HH:MM)")
    parser.add_argument("--test-from", type=str, default="2025-01-01",
                        help="Test data start date (YYYY-MM-DD)")
    parser.add_argument("--xgboost-only", action="store_true",
                        help="Use only XGBoost (skip LSTM)")
    parser.add_argument("--lstm-only", action="store_true",
                        help="Use only LSTM (skip XGBoost)")
    args = parser.parse_args()

    ATR_KEY_VALUE = args.atr_key
    MIN_ATR = args.min_atr
    THRESHOLD = args.threshold
    SEQ_LEN = args.seq_len

    # Parse time windows
    def parse_time(s):
        h, m = map(int, s.split(':'))
        return dt_time(h, m)

    ENTRY_START = parse_time(args.window_start)
    ENTRY_END = parse_time(args.window_end)
    SQUARE_OFF = parse_time(args.square_off)

    test_from_date = datetime.strptime(args.test_from, "%Y-%m-%d").date()

    # ── Load data ──
    print(f"Loading 1-min data: {args.file_1m}")
    try:
        df_1m = pd.read_csv(args.file_1m)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file_1m}")
        return

    print(f"Loading 2-min data: {args.file_2m}")
    try:
        df_2m = pd.read_csv(args.file_2m)
    except FileNotFoundError:
        print(f"⚠️  2-min file not found, using 1-min only")
        df_2m = None

    df_1m['Time'] = pd.to_datetime(df_1m['Time'])
    df_1m = df_1m.sort_values('Time').reset_index(drop=True)

    if df_2m is not None and len(df_2m) > 0:
        df_2m['Time'] = pd.to_datetime(df_2m['Time'])
        df_2m = df_2m.sort_values('Time').reset_index(drop=True)
    else:
        df_2m = None

    total_candles = len(df_1m)
    total_days = df_1m['Time'].dt.date.nunique()
    print(f"1-min: {total_candles:,} candles | {total_days} days")
    if df_2m is not None:
        print(f"2-min: {len(df_2m):,} candles")
    print(f"Date range: {df_1m['Time'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_1m['Time'].iloc[-1].strftime('%Y-%m-%d')}")

    print(f"\n🤖 XGBOOST + LSTM ENSEMBLE STRATEGY (ZERO LOOK-AHEAD)")
    print(f"ATR: RMA({ATR_PERIOD}) × {ATR_KEY_VALUE} | Min ATR: {MIN_ATR}")
    print(f"Window: {ENTRY_START.strftime('%H:%M')} - {ENTRY_END.strftime('%H:%M')} | "
          f"Square off: {SQUARE_OFF.strftime('%H:%M')}")
    print(f"Label threshold: {THRESHOLD} pts (next candle return)")
    print(f"LSTM seq len: {SEQ_LEN} candles")
    print(f"{'='*60}")

    # ── Build features ──
    print(f"\n📊 Building features...")
    feat_1m = build_features_1min(df_1m)
    print(f"  1-min features: {feat_1m.shape[1]} columns")

    if df_2m is not None:
        feat_2m = build_features_2min(df_2m, df_1m)
        features = pd.concat([feat_1m, feat_2m], axis=1)
        print(f"  2-min features: {feat_2m.shape[1]} columns")
    else:
        features = feat_1m

    print(f"  Total features: {features.shape[1]} columns")

    # ── Generate ZERO LOOK-AHEAD labels ──
    print(f"\n🏷️  Generating labels (ZERO LOOK-AHEAD, threshold={THRESHOLD})...")
    labels = generate_labels_no_lookahead(df_1m['Close'], THRESHOLD)
    buy_count = (labels == 1).sum()
    sell_count = (labels == -1).sum()
    hold_count = (labels == 0).sum()
    print(f"  BUY:  {buy_count} ({buy_count/len(labels)*100:.1f}%)")
    print(f"  SELL: {sell_count} ({sell_count/len(labels)*100:.1f}%)")
    print(f"  HOLD: {hold_count} ({hold_count/len(labels)*100:.1f}%)")

    # ── Filter to trading window only ──
    times = df_1m['Time'].dt.time
    window_mask = (times >= ENTRY_START) & (times <= ENTRY_END)
    print(f"\n  Window candles: {window_mask.sum()} (of {len(df_1m)})")

    # ── Clean features ──
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    # ── Train/Test split (by date, no leakage) ──
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

    # ── Train XGBoost ──
    xgb_model = None
    if not args.lstm_only:
        print(f"\n🌳 Training XGBoost model...")
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

        # Feature importance
        importance = xgb_model.feature_importances_
        feat_names = features.columns.tolist()
        top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
        print(f"\n  📊 TOP 10 FEATURES (XGBoost):")
        for name, imp in top_features:
            print(f"    {name:>25}: {imp:.4f}")

    # ── Train LSTM ──
    lstm_model = None
    scaler = StandardScaler()
    scaler.fit(X_train)

    if not args.xgboost_only:
        print(f"\n🧠 Training LSTM model (seq_len={SEQ_LEN}, epochs={args.lstm_epochs})...")
        lstm_model = train_lstm(X_train, y_train, X_test, y_test, scaler,
                                seq_len=SEQ_LEN, epochs=args.lstm_epochs)

    # ── Generate predictions for backtest ──
    print(f"\n🔮 Generating ensemble predictions...")

    # Full feature array for test period
    full_features = features.values

    # XGBoost predictions
    if xgb_model is not None:
        xgb_full_pred = np.zeros(len(df_1m), dtype=int)
        xgb_test_pred = xgb_model.predict(X_test) - 1  # unmap
        xgb_full_pred[test_mask] = xgb_test_pred
        print(f"  XGBoost: BUY={sum(xgb_test_pred==1)} SELL={sum(xgb_test_pred==-1)} "
              f"HOLD={sum(xgb_test_pred==0)}")
    else:
        xgb_full_pred = np.zeros(len(df_1m), dtype=int)

    # LSTM predictions
    if lstm_model is not None:
        # Scale full features and predict
        full_scaled = scaler.transform(full_features)
        lstm_full_pred_raw = predict_lstm_full(lstm_model, full_scaled, SEQ_LEN)
        # Only keep test period
        lstm_full_pred = np.zeros(len(df_1m), dtype=int)
        lstm_full_pred[test_mask] = lstm_full_pred_raw[test_mask]
        lstm_test_pred = lstm_full_pred_raw[test_mask]
        print(f"  LSTM:    BUY={sum(lstm_test_pred==1)} SELL={sum(lstm_test_pred==-1)} "
              f"HOLD={sum(lstm_test_pred==0)}")
    else:
        lstm_full_pred = np.zeros(len(df_1m), dtype=int)

    # Ensemble
    if xgb_model is not None and lstm_model is not None:
        ensemble_pred = ensemble_predict(xgb_full_pred, lstm_full_pred)
        ensemble_test_pred = ensemble_pred[test_mask]
        print(f"  Ensemble: BUY={sum(ensemble_test_pred==1)} SELL={sum(ensemble_test_pred==-1)} "
              f"HOLD={sum(ensemble_test_pred==0)}")
        final_predictions = ensemble_pred
    elif xgb_model is not None:
        final_predictions = xgb_full_pred
        print("  Using XGBoost only (no LSTM)")
    elif lstm_model is not None:
        final_predictions = lstm_full_pred
        print("  Using LSTM only (no XGBoost)")
    else:
        print("❌ No models trained!")
        return

    # ── ATR for backtest ──
    atr_vals = calc_atr(df_1m, ATR_PERIOD).values

    # ── Backtest on test set ──
    print(f"\n🚀 Running backtest on TEST data ({args.test_from} onwards)...")
    test_start = min(test_dates_set)
    test_end = max(test_dates_set)
    test_df_mask = dates >= test_from_date
    df_test = df_1m[test_df_mask].reset_index(drop=True)
    pred_test_full = final_predictions[test_df_mask]
    atr_test = atr_vals[test_df_mask]

    all_trades, daily_results = backtest_predictions(df_test, pred_test_full, atr_test)

    print_results(all_trades, daily_results, f"TEST ({test_start} → {test_end})")

    # ── Detailed per-trade daily log ──
    print_detailed_daily_log(all_trades, daily_results)

    # ── Save trades ──
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("xgboost_lstm_trades.csv", index=False)
        print(f"\n💾 Trades saved: xgboost_lstm_trades.csv")

    # ── Save models ──
    if xgb_model is not None:
        xgb_model.save_model("xgboost_nifty_model.json")
        print(f"💾 XGBoost model saved: xgboost_nifty_model.json")

    if lstm_model is not None:
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'input_size': lstm_model.lstm.input_size,
            'hidden_size': lstm_model.lstm.hidden_size,
            'num_layers': lstm_model.lstm.num_layers,
            'seq_len': SEQ_LEN,
        }, "lstm_nifty_model.pt")
        print(f"💾 LSTM model saved: lstm_nifty_model.pt")

    # Save scaler
    with open("feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Feature scaler saved: feature_scaler.pkl")

    # Save feature column names for live engine
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(features.columns.tolist(), f)
    print(f"💾 Feature columns saved: feature_columns.pkl")


if __name__ == "__main__":
    main()
