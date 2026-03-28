"""
simulator.py — Market Replay Simulator

Replays historical CSV candle data through the CatBoost model one-by-one,
simulating live trading. Designed to work with the dashboard via callbacks.

CRITICAL: Features and predictions are pre-computed before the replay loop
to guarantee exact parity with catboost_strategy.py backtest results.

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
import asyncio
from datetime import datetime, time as dt_time
from typing import Optional, Callable, Dict, Any, List

# Import backtest-identical feature builders
try:
    from catboost_strategy import (
        build_features_1min,
        build_features_2min,
        calc_atr,
        calc_rma,
        calc_rsi,
        calc_ut_bot_direction,
    )
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False
    print("⚠️  catboost_strategy.py not importable — features will be limited")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not installed")


# ─────────── Config ───────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_1M = os.environ.get("CSV_1M", os.path.join(PROJECT_ROOT, "nifty_1min_data.csv"))
CSV_2M = os.environ.get("CSV_2M", os.path.join(PROJECT_ROOT, "nifty_2min_data.csv"))
MODEL_PATH = os.environ.get("CATBOOST_MODEL", os.path.join(PROJECT_ROOT, "catboost_nifty_model.cbm"))

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
        self.start_date = ""
        self.end_date = ""
        self.candle_times: List[float] = []  # timing per candle


class MarketSimulator:
    """
    Replays historical candles through the CatBoost model,
    streaming results via an async callback.

    IMPORTANT: Features & predictions are pre-computed BEFORE the
    replay loop to guarantee identical results to catboost_strategy.py.
    """

    def __init__(self, on_event: Optional[Callable] = None):
        self.state = SimulationState()
        self.on_event = on_event  # async callback(event_type, data)
        self.model = None
        self.model_features = None
        self.df_1m_full = None
        self.df_2m_full = None
        self._stop_requested = False

    def load_data(self):
        """Load CSV data and model."""
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

        # Load model
        if HAS_CATBOOST and os.path.exists(MODEL_PATH):
            self.model = CatBoostClassifier()
            self.model.load_model(MODEL_PATH)
            try:
                self.model_features = self.model.feature_names_
            except Exception:
                self.model_features = None

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
            pnl = exit_price - pos['entry']
        else:
            pnl = pos['entry'] - exit_price
        pnl = round(pnl, 2)

        self.state.daily_pnl += pnl
        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        trade = {
            'dir': pos['dir'],
            'entry': pos['entry'],
            'exit': round(exit_price, 2),
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'sl': round(pos['sl'], 2),
            'pnl': pnl,
            'reason': reason,
        }
        self.state.trades.append(trade)
        self.state.position = None
        return trade

    async def run_simulation(self, start_date: str, end_date: str, speed: int = 10):
        """
        Main simulation loop. Replays candles between start_date and end_date.
        Speed = candles per second (1 = real-time, 60 = fast, 0 = instant).

        PARITY GUARANTEE: Features and predictions are pre-computed over the
        entire dataset in a single pass — identical to catboost_strategy.py.
        """
        if not self.load_data():
            await self.emit("error", {"message": "CSV data not found"})
            return

        if self.model is None:
            await self.emit("error", {"message": "CatBoost model not found"})
            return

        if not HAS_BACKTEST:
            await self.emit("error", {"message": "catboost_strategy.py not importable"})
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

        # ══════════════════════════════════════════════════════
        #  PRE-COMPUTE features & predictions  (backtest-identical)
        # ══════════════════════════════════════════════════════
        t_precompute = time.perf_counter()

        # 1) Build 1-min features over the FULL dataset
        feat_1m = build_features_1min(self.df_1m_full)

        # 2) Build 2-min features using the ACTUAL 2-min CSV + merge_asof
        if self.df_2m_full is not None and len(self.df_2m_full) > 0:
            feat_2m = build_features_2min(self.df_2m_full, self.df_1m_full)
            all_features = pd.concat([feat_1m, feat_2m], axis=1)
        else:
            all_features = feat_1m

        # 3) Clean features
        all_features = all_features.fillna(0)
        all_features = all_features.replace([np.inf, -np.inf], 0)

        # 4) Pre-compute ATR for the full dataset
        atr_all = calc_atr(self.df_1m_full, ATR_PERIOD).values

        # 5) Make predictions ONLY for window candles (like backtest)
        times_all = self.df_1m_full['Time'].dt.time
        window_mask = (times_all >= WINDOW_START) & (times_all <= WINDOW_END)

        full_predictions = np.zeros(len(self.df_1m_full), dtype=int)

        if self.model_features and len(self.model_features) > 0:
            feature_cols = self.model_features
        else:
            feature_cols = ALL_FEATURE_COLS

        X_window = all_features.loc[window_mask, [c for c in feature_cols if c in all_features.columns]]
        # Fill any missing columns with 0
        for col in feature_cols:
            if col not in X_window.columns:
                X_window[col] = 0
        X_window = X_window[feature_cols]

        if len(X_window) > 0:
            window_preds = self.model.predict(X_window).flatten().astype(int)
            full_predictions[window_mask] = window_preds

        precompute_ms = (time.perf_counter() - t_precompute) * 1000

        # ══════════════════════════════════════════════════════
        #  Slice to simulation date range
        # ══════════════════════════════════════════════════════
        sim_mask = (self.df_1m_full['Time'] >= start_dt) & (self.df_1m_full['Time'] < end_dt)
        sim_indices = self.df_1m_full.index[sim_mask].tolist()

        if len(sim_indices) == 0:
            await self.emit("error", {"message": f"No data for {start_date} to {end_date}"})
            self.state.running = False
            return

        self.state.total_candles = len(sim_indices)

        await self.emit("sim_start", {
            "start_date": start_date,
            "end_date": end_date,
            "total_candles": len(sim_indices),
            "warmup_candles": 0,
            "speed": speed,
            "precompute_ms": round(precompute_ms, 0),
        })

        prev_date = None

        # ══════════════════════════════════════════════════════
        #  CANDLE-BY-CANDLE REPLAY (using pre-computed values)
        # ══════════════════════════════════════════════════════
        for loop_i, df_idx in enumerate(sim_indices):
            if self._stop_requested:
                break

            # Handle pause
            while self.state.paused and not self._stop_requested:
                await asyncio.sleep(0.1)

            t_candle_start = time.perf_counter()
            self.state.current_index = loop_i

            row = self.df_1m_full.iloc[df_idx]
            candle_time = row['Time']
            t = candle_time.time()
            curr_date = candle_time.date()
            close_price = float(row['Close'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            time_str = candle_time.strftime('%H:%M')

            # Read pre-computed values
            pred = int(full_predictions[df_idx])
            curr_atr = float(atr_all[df_idx])
            feat_row = all_features.iloc[df_idx]
            current_rsi = float(feat_row.get('rsi_1m', 50))
            ut_dir = float(feat_row.get('ut_dir_1m', 0))

            # ── Day boundary ──
            if prev_date and curr_date != prev_date:
                if self.state.position:
                    prev_idx = sim_indices[loop_i - 1] if loop_i > 0 else df_idx
                    trade = self._close_position(
                        float(self.df_1m_full.iloc[prev_idx]['Close']),
                        "DAY_END",
                        self.df_1m_full.iloc[prev_idx]['Time'].strftime('%H:%M')
                    )
                    await self.emit("trade_close", trade)

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

            in_window = WINDOW_START <= t <= WINDOW_END
            atr_ok = curr_atr >= MIN_ATR

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
                    new_sl = close_price - curr_atr * ATR_KEY_VALUE
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
                elif pos['dir'] == 'SHORT':
                    new_sl = close_price + curr_atr * ATR_KEY_VALUE
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
            if not self.state.position and in_window and atr_ok and t < SQUARE_OFF_TIME:
                if pred == 1:
                    sl = close_price - curr_atr * ATR_KEY_VALUE
                    self.state.position = {
                        'dir': 'LONG', 'entry': close_price,
                        'sl': sl, 'entry_time': time_str,
                    }
                elif pred == -1:
                    sl = close_price + curr_atr * ATR_KEY_VALUE
                    self.state.position = {
                        'dir': 'SHORT', 'entry': close_price,
                        'sl': sl, 'entry_time': time_str,
                    }

                if self.state.position:
                    await self.emit("trade_open", {
                        "dir": self.state.position['dir'],
                        "entry": close_price,
                        "sl": round(self.state.position['sl'], 2),
                        "time": time_str,
                        "atr": round(curr_atr, 2),
                    })

            signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            signal_name = signal_map.get(pred, "HOLD")

            # ── Calculate unrealized P&L ──
            unrealized = 0
            if self.state.position:
                if self.state.position['dir'] == 'LONG':
                    unrealized = close_price - self.state.position['entry']
                else:
                    unrealized = self.state.position['entry'] - close_price

            t_candle_elapsed = (time.perf_counter() - t_candle_start) * 1000
            self.state.candle_times.append(t_candle_elapsed)

            # ── Feature dict for dashboard ──
            feature_data = {}
            for col in ALL_FEATURE_COLS:
                val = float(feat_row.get(col, 0))
                if val != val or abs(val) == float('inf'):
                    val = 0
                feature_data[col] = round(val, 4)

            # ── Emit candle event ──
            await self.emit("candle", {
                "index": loop_i,
                "total": len(sim_indices),
                "time": candle_time.isoformat(),
                "time_str": time_str,
                "date": str(curr_date),
                "open": float(row['Open']),
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "signal": signal_name,
                "prediction": int(pred),
                "atr": round(curr_atr, 2),
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
                    "features_ms": round(t_candle_elapsed, 1),
                    "predict_ms": 0,
                    "total_ms": round(t_candle_elapsed, 1),
                },
                "features": feature_data,
            })

            # ── Sleep to control replay speed ──
            if speed > 0:
                delay = 1.0 / speed
                await asyncio.sleep(delay)

        # ── Simulation complete ──
        if self.state.position:
            last_idx = sim_indices[-1]
            last_close = float(self.df_1m_full.iloc[last_idx]['Close'])
            trade = self._close_position(last_close, "SIM_END", self.df_1m_full.iloc[last_idx]['Time'].strftime('%H:%M'))
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
            "precompute_ms": round(precompute_ms, 0),
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
                pos_str = f" | {p['dir']} @ {p['entry']:.2f} SL={p['sl']:.2f} Unreal={p['unrealized']:+.2f}"
            print(
                f"[{data['time_str']}] NIFTY={data['close']:.2f} | "
                f"{data['signal']} | ATR={data['atr']:.2f} | "
                f"RSI={data['rsi']:.1f}{pos_str} | "
                f"P&L={data['daily_pnl']:+.2f} | "
                f"{data['timing']['total_ms']:.0f}ms"
            )
        elif event_type == "trade_open":
            print(f"  🟢 OPENED {data['dir']} @ {data['entry']:.2f} | SL={data['sl']:.2f}")
        elif event_type == "trade_close":
            icon = "✅" if data['pnl'] > 0 else "❌"
            print(f"  {icon} CLOSED {data['dir']} | Entry={data['entry']:.2f} Exit={data['exit']:.2f} | P&L={data['pnl']:+.2f} | {data['reason']}")
        elif event_type == "sim_start":
            precomp = data.get('precompute_ms', 0)
            print(f"\n🚀 Simulation started: {data['start_date']} → {data['end_date']} | {data['total_candles']} candles | Speed: {data['speed']}x | Pre-compute: {precomp:.0f}ms")
        elif event_type == "sim_end":
            print(f"\n{'='*60}")
            print(f"  SIMULATION COMPLETE")
            print(f"  Trades: {data['total_trades']} | Win rate: {data['win_rate']:.1f}%")
            print(f"  Total P&L: {data['total_pnl']:+.2f} pts")
            print(f"  Avg candle: {data['avg_candle_ms']:.1f}ms")
            print(f"{'='*60}")
        elif event_type == "day_change":
            print(f"\n📅 Day: {data['date']} | P&L: {data['daily_pnl']:+.2f}")
        elif event_type == "error":
            print(f"❌ {data['message']}")

    sim = MarketSimulator(on_event=print_event)
    asyncio.run(sim.run_simulation(args.date, end, speed=args.speed))
