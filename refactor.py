import os
import re

print("Starting 2-min refactoring...")

# 1. Refactor catboost_strategy.py
with open("catboost_strategy.py", "r", encoding="utf-8") as f:
    strategy = f.read()

# Replace config
strategy = strategy.replace("LOOKAHEAD = 5", "LOOKAHEAD = 3")

# Replace build_features section
features_start = strategy.find("def build_features_1min(df):")
features_end = strategy.find("# ─────────── Label Generation ───────────")
if features_start != -1 and features_end != -1:
    new_features = '''def build_features_2m(df):
    """Build all features exclusively from 2-minute data."""
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    opn = df['Open'].astype(float)

    atr = calc_atr(df, ATR_PERIOD)
    rsi = calc_rsi(close, 14)
    trail, dirn = calc_ut_bot_direction(close.values, atr.values, ATR_KEY_VALUE)

    features = pd.DataFrame(index=df.index)
    features['atr_2m'] = atr
    features['rsi_2m'] = rsi
    features['ut_dir_2m'] = dirn
    features['close_vs_trail_2m'] = close.values - trail

    features['mom_3_2m'] = close.pct_change(3) * 100
    features['mom_5_2m'] = close.pct_change(5) * 100
    features['mom_10_2m'] = close.pct_change(10) * 100

    features['body_2m'] = close - opn
    features['body_pct_2m'] = (close - opn) / opn.replace(0, 1e-10) * 100
    features['upper_wick_2m'] = high - close.where(close > opn, opn)
    features['lower_wick_2m'] = close.where(close < opn, opn) - low
    features['range_2m'] = high - low

    features['std_5_2m'] = close.rolling(5).std()
    features['std_10_2m'] = close.rolling(10).std()

    features['sma_5_2m'] = close.rolling(5).mean()
    features['sma_10_2m'] = close.rolling(10).mean()
    features['sma_20_2m'] = close.rolling(20).mean()
    features['close_vs_sma5_2m'] = close - features['sma_5_2m']
    features['close_vs_sma10_2m'] = close - features['sma_10_2m']
    features['sma5_vs_sma10_2m'] = features['sma_5_2m'] - features['sma_10_2m']

    features['high_5_2m'] = high.rolling(5).max()
    features['low_5_2m'] = low.rolling(5).min()
    features['close_vs_high5_2m'] = close - features['high_5_2m']
    features['close_vs_low5_2m'] = close - features['low_5_2m']

    return features

'''
    strategy = strategy[:features_start] + new_features + strategy[features_end:]

# Replace main method parsing to remove --file-1m
strategy = re.sub(r'parser\.add_argument\("--file-1m".*\n\s+parser', 'parser', strategy)
strategy = re.sub(r'print\(f"Loading 1-min data:.*\n.*?except.*?\n.*?\n.*?return\n', '', strategy, flags=re.DOTALL)

# Make sure df_2m is not checked using args.file_1m
strategy = strategy.replace("df_1m =", "df_2m =")
strategy = strategy.replace("df_1m['Time']", "df_2m['Time']")
strategy = strategy.replace("len(df_1m)", "len(df_2m)")
strategy = strategy.replace("1-min data", "2-min data")
strategy = strategy.replace("1-min: ", "2-min: ")

# Replace main execution features
strategy = re.sub(r'feat_1m = build_features_1min\(.*?\)\n.*?if df_2m is not None:.*?else:\n.*?features = feat_1m', 
                  'features = build_features_2m(df_2m)', strategy, flags=re.DOTALL)

with open("catboost_strategy_2m.py", "w", encoding="utf-8") as f:
    f.write(strategy)
print("catboost_strategy_2m.py created.")

# 2. Refactor simulator.py
with open("simulator.py", "r", encoding="utf-8") as f:
    sim = f.read()

sim = sim.replace('FEATURE_COLS_1M = [', 'FEATURE_COLS_2M = [')
sim = sim.replace("'atr_1m', 'rsi_1m'", "'atr_2m', 'rsi_2m'") # Very raw replacement - we will simplify
with open("simulator_2m.py", "w", encoding="utf-8") as f:
    f.write(sim)
print("Done refs")

