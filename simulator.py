"""
simulator.py — Market Replay Simulator

Replays historical CSV candle data through the XGBoost+LSTM model one-by-one,
simulating live trading. Designed to work with the dashboard via callbacks.

Usage (standalone test):
    python simulator.py --date 2026-03-20 --speed 60

Usage (via FastAPI):
    Imported by backend/routes/simulator_routes.py
"""

import pandas as pd
import numpy as np
import os
import time
import json
import pickle
import asyncio
from collections import deque
from datetime import datetime, time as dt_time
from typing import Optional, Callable, Dict, Any, List

# ─── XGBoost + LSTM imports ───
try:
    from xgboost_lstm_strategy import (
        build_features_1min as xgl_build_features_1min,
        build_features_2min as xgl_build_features_2min,
        calc_atr as xgl_calc_atr,
        LSTMClassifier,
    )
    HAS_XGBOOST_STRATEGY = True
except ImportError:
    HAS_XGBOOST_STRATEGY = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────── Config ───────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_1M = os.environ.get("CSV_1M", os.path.join(PROJECT_ROOT, "nifty_1min_data.csv"))
CSV_2M = os.environ.get("CSV_2M", os.path.join(PROJECT_ROOT, "nifty_2min_data.csv"))

# XGBoost + LSTM model paths
XGB_MODEL_PATH = os.environ.get("XGB_MODEL", os.path.join(PROJECT_ROOT, "xgboost_nifty_model.json"))
LSTM_MODEL_PATH = os.environ.get("LSTM_MODEL", os.path.join(PROJECT_ROOT, "lstm_nifty_model.pt"))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join(PROJECT_ROOT, "feature_scaler.pkl"))
FEATURE_COLS_PATH = os.environ.get("FEATURE_COLS", os.path.join(PROJECT_ROOT, "feature_columns.pkl"))

ATR_PERIOD = 14
ATR_KEY_VALUE = 1.0
MIN_ATR = 6.5

WINDOW_START = dt_time(9, 20)
WINDOW_END = dt_time(15, 15)
SQUARE_OFF_TIME = dt_time(15, 24)

FEATURE_COLS_1M = [
    'atr_1m', 'rsi_1m', 'ut_dir_1m', 'close_vs_trail_1m',
    'mom_3', 'mom_5', 'mom_10',
    'body_1m', 'body_pct_1m', 'upper_wick_1m', 'lower_wick_1m', 'range_1m',
    'std_5', 'std_10',
    'sma_5', 'sma_10', 'sma_20',
    'close_vs_sma5', 'close_vs_sma10', 'sma5_vs_sma10',
    'high_5', 'low_5', 'close_vs_high5', 'close_vs_low5',
]

FEATURE_COLS_2M = [
    'atr_2m', 'rsi_2m', 'ut_dir_2m', 'close_vs_trail_2m',
    'mom_3_2m', 'mom_5_2m', 'range_2m', 'body_2m',
]

ALL_FEATURE_COLS = FEATURE_COLS_1M + FEATURE_COLS_2M


class SimulationState:
    """Holds the current state of a running simulation."""

    def __init__(self):
        self.running = False
        self.paused = False
        self.speed = 10  # candles per second
        self.current_index = 0
        self.total_candles = 0
        self.position = None
        self.trades: List[Dict] = []
        self.daily_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.current_date = ""
        self.end_date = ""
        self.candle_times: List[float] = []  # timing per candle

        # Lot management
        self.current_lots = 2
        self.accumulated_loss = 0.0
        self.recovering = False


def _ensemble_predict_single(xgb_pred, lstm_pred):
    """Combine XGBoost and LSTM predictions for a single candle."""
    if xgb_pred == lstm_pred:
        return xgb_pred
    elif xgb_pred != 0 and lstm_pred == 0:
        return xgb_pred
    elif xgb_pred == 0 and lstm_pred != 0:
        return lstm_pred
    else:
        return 0  # Conflicting → HOLD


class MarketSimulator:
    """
    Replays historical candles through XGBoost+LSTM models,
    streaming results via an async callback.
    """

    def __init__(self, on_event: Optional[Callable] = None):
        self.state = SimulationState()
        self.on_event = on_event  # async callback(event_type, data)

        # XGBoost + LSTM models
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.feature_columns = ALL_FEATURE_COLS
        self.lstm_seq_len = 20
        self.lstm_feature_buffer = deque(maxlen=20)

        # Data
        self.df_1m_full = None
        self.df_2m_full = None
        self._stop_requested = False

        # Aliases for the active engine's feature builders
        self._build_features_1min = None
        self._build_features_2min = None

    def load_data(self):
        """Load CSV data and model(s) based on selected engine."""
        if self.df_1m_full is not None:
            return True  # Already loaded

        if not os.path.exists(CSV_1M):
            return False

        self.df_1m_full = pd.read_csv(CSV_1M)
        self.df_1m_full['Time'] = pd.to_datetime(self.df_1m_full['Time']).dt.tz_localize(None)
        self.df_1m_full = self.df_1m_full.sort_values('Time').reset_index(drop=True)

        if os.path.exists(CSV_2M):
            self.df_2m_full = pd.read_csv(CSV_2M)
            self.df_2m_full['Time'] = pd.to_datetime(self.df_2m_full['Time']).dt.tz_localize(None)
            self.df_2m_full = self.df_2m_full.sort_values('Time').reset_index(drop=True)

        return self._load_xgboost_lstm()

    def _load_xgboost_lstm(self):
        """Load XGBoost + LSTM models, scaler, and feature builders."""
        if not HAS_XGBOOST_STRATEGY:
            print("xgboost_lstm_strategy.py not importable")
            return False

        loaded_any = False

        # XGBoost
        if HAS_XGBOOST and os.path.exists(XGB_MODEL_PATH):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(XGB_MODEL_PATH)
            print(f"XGBoost model loaded: {XGB_MODEL_PATH}")
            loaded_any = True
        else:
            print(f"XGBoost model not found: {XGB_MODEL_PATH}")

        # LSTM
        if HAS_TORCH and os.path.exists(LSTM_MODEL_PATH):
            try:
                checkpoint = torch.load(LSTM_MODEL_PATH, map_location='cpu', weights_only=False)
                input_size = checkpoint['input_size']
                hidden_size = checkpoint['hidden_size']
                num_layers = checkpoint['num_layers']
                self.lstm_seq_len = checkpoint.get('seq_len', 20)
                self.lstm_feature_buffer = deque(maxlen=self.lstm_seq_len)

                self.lstm_model = LSTMClassifier(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=0.0,
                )
                self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                self.lstm_model.eval()
                print(f"LSTM model loaded: {LSTM_MODEL_PATH} (seq_len={self.lstm_seq_len})")
                loaded_any = True
            except Exception as e:
                print(f"LSTM model load failed: {e}")
        else:
            print(f"LSTM model not found: {LSTM_MODEL_PATH}")

        # Scaler
        if HAS_SKLEARN and os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"Feature scaler loaded: {SCALER_PATH}")

        # Feature columns
        if os.path.exists(FEATURE_COLS_PATH):
            with open(FEATURE_COLS_PATH, "rb") as f:
                self.feature_columns = pickle.load(f)
            print(f"Feature columns loaded: {len(self.feature_columns)} features")

        self._build_features_1min = xgl_build_features_1min
        self._build_features_2min = xgl_build_features_2min

        if not loaded_any:
            print("No XGBoost or LSTM model loaded!")
            return False

        print(f"Engine: XGBoost+LSTM loaded (XGB={'yes' if self.xgb_model else 'no'}, "
              f"LSTM={'yes' if self.lstm_model else 'no'})")
        return True

    def get_available_dates(self) -> List[str]:
        """Return list of unique trading dates in the CSV."""
        if self.df_1m_full is None:
            self.load_data()
        if self.df_1m_full is None:
            return []
        dates = self.df_1m_full['Time'].dt.date.unique()
        return [str(d) for d in sorted(dates)]

    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Send event to the dashboard."""
        if self.on_event:
            await self.on_event(event_type, data)

    def _close_position(self, exit_price: float, reason: str, exit_time: str) -> Dict:
        """Close current position and return trade record."""
        pos = self.state.position
        if not pos:
            return {}

        if pos['dir'] == 'LONG':
            raw_pnl = exit_price - pos['entry']
        else:
            raw_pnl = pos['entry'] - exit_price
        
        raw_pnl = round(raw_pnl, 2)
        lot_multiplier = self.state.current_lots // 2
        pnl = raw_pnl * lot_multiplier
        pnl = round(pnl, 2)

        self.state.daily_pnl += pnl
        if pnl > 0:
            self.state.wins += 1
        elif pnl < 0:
            self.state.losses += 1

        trade = {
            'dir': pos['dir'],
            'entry': pos['entry'],
            'exit': round(exit_price, 2),
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'sl': round(pos['sl'], 2),
            'pnl': pnl,
            'raw_pnl': raw_pnl,
            'lots': self.state.current_lots,
            'reason': reason,
        }
        self.state.trades.append(trade)
        self.state.position = None
        return trade

    async def run_simulation(self, start_date: str, end_date: str, speed: int = 10, warmup_candles: int = 500):
        """
        Main simulation loop. Replays candles between start_date and end_date.
        Speed = candles per second (1 = real-time, 60 = fast).
        """
        if not self.load_data():
            await self.emit("error", {"message": "CSV data or models not found"})
            return

        if self.xgb_model is None and self.lstm_model is None:
            await self.emit("error", {"message": "No XGBoost or LSTM model found"})
            return

        if self._build_features_1min is None:
            await self.emit("error", {"message": f"Feature builders not available"})
            return

        # Reset state
        self.state = SimulationState()
        self.state.running = True
        self.state.speed = speed
        self.state.start_date = start_date
        self.state.end_date = end_date
        self._stop_requested = False

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        # Get simulation candles
        sim_mask = (self.df_1m_full['Time'] >= start_dt) & (self.df_1m_full['Time'] < end_dt)
        df_sim = self.df_1m_full[sim_mask].reset_index(drop=True)

        if len(df_sim) == 0:
            await self.emit("error", {"message": f"No data for {start_date} to {end_date}"})
            self.state.running = False
            return

        # Get warm-up candles
        warmup_mask = self.df_1m_full['Time'] < start_dt
        df_warmup = self.df_1m_full[warmup_mask].tail(warmup_candles).reset_index(drop=True)

        # Get 2-min warm-up
        df_2m_warmup = pd.DataFrame()
        if self.df_2m_full is not None:
            warmup_2m_mask = self.df_2m_full['Time'] < start_dt
            df_2m_warmup = self.df_2m_full[warmup_2m_mask].tail(warmup_candles).reset_index(drop=True)

        self.state.total_candles = len(df_sim)

        # Broadcast warm-up candles to the dashboard
        warmup_payload = [
            {
                "time": row["Time"].isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
            for row in df_warmup.to_dict('records')
        ]
        await self.emit("warmup_data", warmup_payload)

        # Reset LSTM buffer for fresh simulation
        self.lstm_feature_buffer = deque(maxlen=self.lstm_seq_len)

        await self.emit("sim_start", {
            "start_date": start_date,
            "end_date": end_date,
            "total_candles": len(df_sim),
            "warmup_candles": len(df_warmup),
            "speed": speed,
            "engine": "xgboost_lstm",
        })

        prev_date = None

        # ── Main candle-by-candle loop ──
        for i in range(len(df_sim)):
            if self._stop_requested:
                break

            # Handle pause
            while self.state.paused and not self._stop_requested:
                await asyncio.sleep(0.1)

            t_candle_start = time.perf_counter()
            self.state.current_index = i

            row = df_sim.iloc[i]
            candle_time = row['Time']
            t = candle_time.time()
            curr_date = candle_time.date()
            close_price = float(row['Close'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            time_str = candle_time.strftime('%H:%M')

            # ── Day boundary ──
            if prev_date and curr_date != prev_date:
                if self.state.position:
                    trade = self._close_position(
                        float(df_sim.iloc[i - 1]['Close']),
                        "DAY_END",
                        df_sim.iloc[i - 1]['Time'].strftime('%H:%M')
                    )
                    await self.emit("trade_close", trade)

                # ── Lot adjustment for Martingale recovery ──
                if prev_date:
                    day_pnl = self.state.daily_pnl
                    if day_pnl < 0:
                        self.state.accumulated_loss += day_pnl
                        self.state.recovering = True
                        self.state.current_lots += 2
                    elif day_pnl > 0 and self.state.recovering:
                        self.state.accumulated_loss += day_pnl
                        if self.state.accumulated_loss >= 0:
                            self.state.current_lots = 2
                            self.state.accumulated_loss = 0.0
                            self.state.recovering = False

                await self.emit("day_change", {
                    "date": str(curr_date),
                    "daily_pnl": round(self.state.daily_pnl, 2),
                    "trades": len(self.state.trades),
                    "wins": self.state.wins,
                    "losses": self.state.losses,
                })
                self.state.daily_pnl = 0.0

            self.state.current_date = str(curr_date)
            prev_date = curr_date

            # ── Build features using warm-up + candles seen so far ──
            t_feat_start = time.perf_counter()

            df_today_so_far = df_sim.iloc[:i + 1].copy()
            df_combined = pd.concat([df_warmup, df_today_so_far], ignore_index=True)

            feat_1m = self._build_features_1min(df_combined)

            # Build 2-min features
            if len(df_2m_warmup) > 0:
                today_start = df_today_so_far['Time'].iloc[0]
                df_2m_today = df_today_so_far.set_index('Time').resample(
                    '2min', label='left', closed='left'
                ).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                }).dropna().reset_index()

                warmup_2m_filtered = df_2m_warmup[df_2m_warmup['Time'] < today_start]
                df_2m_combined = pd.concat([warmup_2m_filtered, df_2m_today], ignore_index=True)
            else:
                df_2m_combined = df_combined.set_index('Time').resample(
                    '2min', label='left', closed='left'
                ).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                }).dropna().reset_index()

            feat_2m = self._build_features_2min(df_2m_combined, df_combined)

            all_features = pd.concat([feat_1m, feat_2m], axis=1)
            last_feat = all_features.iloc[-1]

            current_atr = float(last_feat.get('atr_1m', 0))
            current_rsi = float(last_feat.get('rsi_1m', 50))
            ut_dir = float(last_feat.get('ut_dir_1m', 0))

            # Build feature vector
            feature_vector = [float(last_feat.get(f, 0)) for f in self.feature_columns]
            feature_vector = [0 if (v != v or abs(v) == float('inf')) else v for v in feature_vector]

            t_feat_elapsed = (time.perf_counter() - t_feat_start) * 1000

            # ── Predict ──
            t_pred_start = time.perf_counter()
            xgb_pred = 0
            lstm_pred = 0

            # ── XGBoost prediction ──
            if self.xgb_model is not None:
                xgb_raw = self.xgb_model.predict(np.array([feature_vector]))[0]
                xgb_pred = int(xgb_raw) - 1  # unmap: 0->-1, 1->0, 2->1

            # ── LSTM prediction ──
            if self.lstm_model is not None and self.scaler is not None:
                full_feature_row = np.array(feature_vector, dtype=np.float32)
                scaled_row = self.scaler.transform(full_feature_row.reshape(1, -1))[0]
                self.lstm_feature_buffer.append(scaled_row)

                if len(self.lstm_feature_buffer) >= self.lstm_seq_len:
                    seq_array = np.array(list(self.lstm_feature_buffer), dtype=np.float32)
                    x_tensor = torch.FloatTensor(seq_array).unsqueeze(0)
                    with torch.no_grad():
                        output = self.lstm_model(x_tensor)
                        pred_class = output.argmax(dim=1).item()
                        lstm_pred = pred_class - 1  # unmap: 0->-1, 1->0, 2->1

            # Ensemble
            pred = _ensemble_predict_single(xgb_pred, lstm_pred)

            t_pred_elapsed = (time.perf_counter() - t_pred_start) * 1000

            signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            signal_name = signal_map.get(pred, "HOLD")
            xgb_signal = signal_map.get(xgb_pred, "HOLD")
            lstm_signal = signal_map.get(lstm_pred, "HOLD")

            in_window = WINDOW_START <= t <= WINDOW_END
            atr_ok = current_atr >= MIN_ATR

            # ── Square off at 15:24 ──
            if self.state.position and t >= SQUARE_OFF_TIME:
                trade = self._close_position(close_price, "SQUARE_OFF", time_str)
                await self.emit("trade_close", trade)

            # ── SL check ──
            if self.state.position:
                pos = self.state.position
                if pos['dir'] == 'LONG' and low_price <= pos['sl']:
                    trade = self._close_position(pos['sl'], "TRAIL_SL", time_str)
                    await self.emit("trade_close", trade)
                elif pos['dir'] == 'SHORT' and high_price >= pos['sl']:
                    trade = self._close_position(pos['sl'], "TRAIL_SL", time_str)
                    await self.emit("trade_close", trade)

            # ── Trail SL update ──
            if self.state.position:
                pos = self.state.position
                if pos['dir'] == 'LONG':
                    new_sl = close_price - current_atr * ATR_KEY_VALUE
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
                elif pos['dir'] == 'SHORT':
                    new_sl = close_price + current_atr * ATR_KEY_VALUE
                    if new_sl < pos['sl']:
                        pos['sl'] = new_sl

            # ── Opposite signal close ──
            if pred == 1 and self.state.position and self.state.position['dir'] == 'SHORT':
                trade = self._close_position(close_price, "OPPOSITE", time_str)
                await self.emit("trade_close", trade)
            elif pred == -1 and self.state.position and self.state.position['dir'] == 'LONG':
                trade = self._close_position(close_price, "OPPOSITE", time_str)
                await self.emit("trade_close", trade)

            # ── New entry ──
            if not self.state.position and pred != 0 and in_window and atr_ok and t < SQUARE_OFF_TIME:
                if pred == 1:
                    sl = close_price - current_atr * ATR_KEY_VALUE
                    self.state.position = {
                        'dir': 'LONG', 'entry': close_price,
                        'sl': sl, 'entry_time': time_str,
                    }
                elif pred == -1:
                    sl = close_price + current_atr * ATR_KEY_VALUE
                    self.state.position = {
                        'dir': 'SHORT', 'entry': close_price,
                        'sl': sl, 'entry_time': time_str,
                    }

                await self.emit("trade_open", {
                    "dir": self.state.position['dir'],
                    "entry": close_price,
                    "sl": round(self.state.position['sl'], 2),
                    "time": time_str,
                    "atr": round(current_atr, 2),
                })

            # ── Calculate unrealized P&L ──
            unrealized = 0
            if self.state.position:
                if self.state.position['dir'] == 'LONG':
                    raw_unrealized = close_price - self.state.position['entry']
                else:
                    raw_unrealized = self.state.position['entry'] - close_price
                unrealized = raw_unrealized * (self.state.current_lots // 2)

            t_candle_elapsed = (time.perf_counter() - t_candle_start) * 1000
            self.state.candle_times.append(t_candle_elapsed)

            # ── Feature dict for dashboard ──
            feature_data = {}
            for col in ALL_FEATURE_COLS:
                val = float(last_feat.get(col, 0))
                if val != val or abs(val) == float('inf'):
                    val = 0
                feature_data[col] = round(val, 4)

            # ── Emit candle event ──
            candle_event = {
                "index": i,
                "total": len(df_sim),
                "time": candle_time.isoformat(),
                "time_str": time_str,
                "date": str(curr_date),
                "open": float(row['Open']),
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "signal": signal_name,
                "prediction": int(pred),
                "engine": "xgboost_lstm",
                "atr": round(current_atr, 2),
                "rsi": round(current_rsi, 2),
                "ut_dir": int(ut_dir),
                "in_window": in_window,
                "atr_ok": atr_ok,
                "position": {
                    "dir": self.state.position['dir'],
                    "entry": self.state.position['entry'],
                    "sl": round(self.state.position['sl'], 2),
                    "unrealized": round(unrealized, 2),
                } if self.state.position else None,
                "daily_pnl": round(self.state.daily_pnl + unrealized, 2),
                "total_trades": self.state.wins + self.state.losses,
                "wins": self.state.wins,
                "losses": self.state.losses,
                "timing": {
                    "features_ms": round(t_feat_elapsed, 1),
                    "predict_ms": round(t_pred_elapsed, 1),
                    "total_ms": round(t_candle_elapsed, 1),
                },
                "features": feature_data,
            }

            # Add XGBoost/LSTM sub-predictions for the ensemble engine
            candle_event["xgb_signal"] = xgb_signal
            candle_event["lstm_signal"] = lstm_signal

            await self.emit("candle", candle_event)

            # ── Sleep to control replay speed ──
            if speed > 0:
                delay = 1.0 / speed
                await asyncio.sleep(delay)

        # ── Simulation complete ──
        if self.state.position:
            last_close = float(df_sim.iloc[-1]['Close'])
            trade = self._close_position(last_close, "SIM_END", df_sim.iloc[-1]['Time'].strftime('%H:%M'))
            await self.emit("trade_close", trade)

        total_trades = self.state.wins + self.state.losses
        wr = (self.state.wins / total_trades * 100) if total_trades > 0 else 0
        avg_timing = np.mean(self.state.candle_times) if self.state.candle_times else 0

        await self.emit("sim_end", {
            "total_trades": total_trades,
            "wins": self.state.wins,
            "losses": self.state.losses,
            "win_rate": round(wr, 1),
            "total_pnl": round(sum(t['pnl'] for t in self.state.trades), 2),
            "trades": self.state.trades,
            "avg_candle_ms": round(avg_timing, 1),
            "stopped": self._stop_requested,
        })

        self.state.running = False

    def stop(self):
        """Request simulation stop."""
        self._stop_requested = True

    def pause(self):
        """Toggle pause state."""
        self.state.paused = not self.state.paused
        return self.state.paused

    def set_speed(self, speed: int):
        """Change replay speed mid-simulation."""
        self.state.speed = speed


# ─────────── Standalone test ───────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Replay Simulator")
    parser.add_argument("--date", default="2026-03-20", help="Simulation date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (default: same as --date)")
    parser.add_argument("--speed", type=int, default=0, help="Candles/sec (0=instant)")
    args = parser.parse_args()

    end = args.end_date or args.date

    async def print_event(event_type, data):
        if event_type == "candle":
            pos_str = ""
            if data.get("position"):
                p = data["position"]
                lots = self.state.current_lots if hasattr(self, 'state') else 2
                pos_str = f" | {p['dir']} ({lots} lots) @ {p['entry']:.2f} SL={p['sl']:.2f} Unreal={p['unrealized']:+.2f}"

            # Show sub-predictions for XGBoost+LSTM engine
            engine_str = f" [XGB={data.get('xgb_signal','?')} LSTM={data.get('lstm_signal','?')}]"

            print(
                f"[{data['time_str']}] NIFTY={data['close']:.2f} | "
                f"{data['signal']}{engine_str} | ATR={data['atr']:.2f} | "
                f"RSI={data['rsi']:.1f}{pos_str} | "
                f"P&L={data['daily_pnl']:+.2f} | "
                f"{data['timing']['total_ms']:.0f}ms"
            )
        elif event_type == "trade_open":
            print(f"  OPENED {data['dir']} @ {data['entry']:.2f} | SL={data['sl']:.2f}")
        elif event_type == "trade_close":
            icon = "W" if data['pnl'] > 0 else "L"
            print(f"  {icon} CLOSED {data['dir']} | Entry={data['entry']:.2f} Exit={data['exit']:.2f} | P&L={data['pnl']:+.2f} | {data['reason']}")
        elif event_type == "sim_start":
            print(f"\nSimulation started: {data['start_date']} to {data['end_date']} | Engine: {data.get('engine', '?')}")
        elif event_type == "sim_end":
            print(f"\n{'='*60}")
            print(f"  SIMULATION COMPLETE")
            print(f"  Trades: {data['total_trades']} | Win rate: {data['win_rate']:.1f}%")
            print(f"  Total P&L: {data['total_pnl']:+.2f} pts")
            print(f"  Avg candle: {data['avg_candle_ms']:.1f}ms")
            print(f"{'='*60}")
        elif event_type == "day_change":
            print(f"\nDay: {data['date']} | P&L: {data['daily_pnl']:+.2f}")
        elif event_type == "error":
            print(f"ERROR: {data['message']}")

    sim = MarketSimulator(on_event=print_event)
    asyncio.run(sim.run_simulation(args.date, end, speed=args.speed))
