import re
import sys

with open("simulator.py", "r", encoding="utf-8") as f:
    text = f.read()

new_imports = """
try:
    from catboost_strategy import (
        build_features_1min,
        build_features_2min,
        calc_atr,
        calc_rma,
        calc_rsi,
        calc_ut_bot_direction,
        generate_labels,
        ALL_FEATURE_COLS,
    )
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False
    print("⚠️  catboost_strategy.py not importable — features will be limited")
"""

# Replace imports
text = re.sub(r"try:\n    from catboost_strategy import \([\s\S]*?print\(\"⚠️  catboost_strategy.py not importable — features will be limited\"\)", new_imports.strip(), text)


# Replace run_simulation
new_run_sim = '''
    async def run_simulation(self, start_date: str, end_date: str, speed: int = 10):
        """
        True Walk-Forward Simulation loop.
        1. Trains a fresh CatBoost model on the 1-year trailing data up to start_date.
        2. Simulates candle-by-candle evaluation mirroring catboost_live_engine.py.
        """
        if not self.load_data():
            await self.emit("error", {"message": "CSV data not found"})
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
        one_year_ago = start_dt - pd.Timedelta(days=365)

        # ══════════════════════════════════════════════════════
        #  WALK-FORWARD TRAINING (Last 365 Days)
        # ══════════════════════════════════════════════════════
        t_precompute = time.perf_counter()

        train_mask_1m = (self.df_1m_full['Time'] >= one_year_ago) & (self.df_1m_full['Time'] < start_dt)
        df_train_1m = self.df_1m_full[train_mask_1m].copy()

        if len(df_train_1m) == 0:
            await self.emit("error", {"message": f"No 1-year warmup data found before {start_date} for training."})
            self.state.running = False
            return

        sim_mask_1m = (self.df_1m_full['Time'] >= start_dt) & (self.df_1m_full['Time'] < end_dt)
        sim_indices = self.df_1m_full.index[sim_mask_1m].tolist()

        if len(sim_indices) == 0:
            await self.emit("error", {"message": f"No data for {start_date} to {end_date}"})
            self.state.running = False
            return

        self.state.total_candles = len(sim_indices)

        await self.emit("sim_start", {
            "start_date": start_date,
            "end_date": end_date,
            "total_candles": len(sim_indices),
            "warmup_candles": len(df_train_1m),
            "speed": speed,
            "precompute_ms": 0,
            "message": "Training Walk-Forward Model..."
        })

        # --- Build Features for Training ---
        feat_1m_tr = build_features_1min(df_train_1m)
        
        df_train_2m = None
        if self.df_2m_full is not None:
            train_mask_2m = (self.df_2m_full['Time'] >= one_year_ago) & (self.df_2m_full['Time'] < start_dt)
            df_train_2m = self.df_2m_full[train_mask_2m].copy()
            feat_2m_tr = build_features_2min(df_train_2m, df_train_1m)
            X_train = pd.concat([feat_1m_tr, feat_2m_tr], axis=1)
        else:
            X_train = feat_1m_tr

        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        labels = generate_labels(df_train_1m, lookahead=5, threshold=8.0)

        times_tr = df_train_1m['Time'].dt.time
        window_mask_tr = (times_tr >= WINDOW_START) & (times_tr <= WINDOW_END)
        
        X_train_win = X_train.loc[window_mask_tr, [c for c in ALL_FEATURE_COLS if c in X_train.columns]]
        for col in ALL_FEATURE_COLS:
            if col not in X_train_win.columns:
                X_train_win[col] = 0
        X_train_win = X_train_win[ALL_FEATURE_COLS]
        y_train_win = labels[window_mask_tr]

        self.model = CatBoostClassifier(iterations=250, depth=6, random_state=42, verbose=0)
        
        # Train model asynchronously to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.model.fit, X_train_win, y_train_win)

        precompute_ms = (time.perf_counter() - t_precompute) * 1000
        prev_date = None

        # ══════════════════════════════════════════════════════
        #  CANDLE-BY-CANDLE REPLAY (Mimicking Live Engine)
        # ══════════════════════════════════════════════════════
        for loop_i, df_idx in enumerate(sim_indices):
            if self._stop_requested:
                break

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

            # Build incremental DataFrame (warmup + today up to this candle)
            df_sim_today = self.df_1m_full.iloc[sim_indices[0]:df_idx+1]
            df_combined = pd.concat([df_train_1m, df_sim_today], ignore_index=True)
            
            feat_1m_curr = build_features_1min(df_combined)

            if self.df_2m_full is not None:
                sim_2m_mask = (self.df_2m_full['Time'] >= start_dt) & (self.df_2m_full['Time'] <= candle_time)
                df_sim_today_2m = self.df_2m_full[sim_2m_mask]
                df_combined_2m = pd.concat([df_train_2m, df_sim_today_2m], ignore_index=True)
                feat_2m_curr = build_features_2min(df_combined_2m, df_combined)
                all_features = pd.concat([feat_1m_curr, feat_2m_curr], axis=1)
            else:
                all_features = feat_1m_curr
                
            all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
            feat_row = all_features.iloc[-1]

            X_curr = pd.DataFrame([feat_row], columns=all_features.columns)
            for col in ALL_FEATURE_COLS:
                if col not in X_curr.columns:
                    X_curr[col] = 0
            X_curr = X_curr[ALL_FEATURE_COLS]

            pred = 0
            in_window = WINDOW_START <= t <= WINDOW_END
            if in_window:
                pred = int(self.model.predict(X_curr)[0])

            curr_atr = float(feat_row.get('atr_1m', 0))
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

                if self.state.daily_pnl < 0:
                    self.state.accumulated_loss += self.state.daily_pnl
                    self.state.recovering = True
                    self.state.current_lots += 2
                elif self.state.daily_pnl > 0 and self.state.recovering:
                    self.state.accumulated_loss += self.state.daily_pnl
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
                    "lots": self.state.current_lots,
                })
                self.state.daily_pnl = 0.0

            self.state.current_date = str(curr_date)
            prev_date = curr_date
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
                    raw_unrealized = close_price - self.state.position['entry']
                else:
                    raw_unrealized = self.state.position['entry'] - close_price
                unrealized = raw_unrealized * self.state.current_lots

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
'''
start_idx = text.find("    async def run_simulation")
end_idx = text.find("    def stop(self):")
text = text[:start_idx] + new_run_sim + text[end_idx:]

with open("simulator.py", "w", encoding="utf-8") as f:
    f.write(text)
print("SUCCESS!")
