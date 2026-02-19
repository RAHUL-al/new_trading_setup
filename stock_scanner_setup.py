"""
stock_scanner_setup.py — Run at start of day to prepare token lists.

1. Downloads AngelOne instrument master JSON
2. Reads stocks_name.csv → extracts ALL symbols (indexes + stocks)
3. Matches against AngelOne master:
   - Stocks  → NSE equity segment (symbol ending with '-EQ')
   - Indexes → NSE indices segment (NIFTY 50, NIFTY BANK, etc.)
4. Saves a full CSV with all matched symbols + AngelOne details
5. Splits STOCK tokens (not indexes) across N credentials for WebSocket workers
"""

import pandas as pd
import json
import math
import os
import requests
from logzero import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Known index → AngelOne name mappings
INDEX_MAP = {
    'NIFTY':      {'angelone_name': 'Nifty 50',       'token': '99926000', 'exch': 'NSE'},
    'BANKNIFTY':  {'angelone_name': 'Nifty Bank',     'token': '99926009', 'exch': 'NSE'},
    'FINNIFTY':   {'angelone_name': 'Nifty Fin Service', 'token': '99926037', 'exch': 'NSE'},
    'MIDCPNIFTY': {'angelone_name': 'NIFTY MID SELECT', 'token': '99926074', 'exch': 'NSE'},
    'NIFTYNXT50': {'angelone_name': 'Nifty Next 50',  'token': '99926013', 'exch': 'NSE'},
}

INDEX_SYMBOLS = set(INDEX_MAP.keys())


def download_angelone_master() -> pd.DataFrame:
    """Download AngelOne instrument master and return as DataFrame."""
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    logger.info("Downloading AngelOne instrument master...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    logger.info(f"Downloaded {len(df)} instruments from AngelOne master.")
    return df


def load_fno_symbols() -> pd.DataFrame:
    """Read stocks_name.csv and return full DataFrame with all symbols."""
    csv_path = os.path.join(BASE_DIR, "stocks_name.csv")
    df = pd.read_csv(csv_path)

    # Column names have trailing spaces — clean them
    df.columns = df.columns.str.strip()

    # Clean symbol column
    symbol_col = [c for c in df.columns if 'SYMBOL' in c.upper()][0]
    df[symbol_col] = df[symbol_col].astype(str).str.strip()

    # Clean vol freeze qty column
    vol_col = [c for c in df.columns if 'VOL' in c.upper() or 'FRZ' in c.upper()][0]
    df[vol_col] = pd.to_numeric(df[vol_col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)

    # Standardize column names
    df = df.rename(columns={symbol_col: 'SYMBOL', vol_col: 'VOL_FRZ_QTY'})

    # Add type column
    df['type'] = df['SYMBOL'].apply(lambda s: 'INDEX' if s in INDEX_SYMBOLS else 'STOCK')

    logger.info(f"Loaded {len(df)} symbols from stocks_name.csv ({(df['type'] == 'INDEX').sum()} indexes, {(df['type'] == 'STOCK').sum()} stocks)")
    return df


def match_tokens(master_df: pd.DataFrame, symbols_df: pd.DataFrame) -> pd.DataFrame:
    """Match all symbols (stocks + indexes) against AngelOne master."""

    # ─── Match stocks from NSE equity segment ───
    nse = master_df[master_df['exch_seg'] == 'NSE'].copy()
    nse_eq = nse[nse['symbol'].str.endswith('-EQ')].copy()
    nse_eq['clean_symbol'] = nse_eq['symbol'].str.replace('-EQ', '', regex=False)

    stock_symbols = symbols_df[symbols_df['type'] == 'STOCK']['SYMBOL'].tolist()
    matched_stocks = nse_eq[nse_eq['clean_symbol'].isin(stock_symbols)].copy()

    stock_rows = []
    for _, row in matched_stocks.iterrows():
        sym = row['clean_symbol']
        vol_qty = symbols_df[symbols_df['SYMBOL'] == sym]['VOL_FRZ_QTY'].values
        stock_rows.append({
            'pSymbol': str(row['token']),
            'pSymbolName': sym,
            'angelone_symbol': row['symbol'],
            'company_name': row.get('name', ''),
            'exchange': 'NSE',
            'instrument_type': 'EQ',
            'lot_size': row.get('lotsize', ''),
            'tick_size': row.get('tick_size', ''),
            'vol_freeze_qty': int(vol_qty[0]) if len(vol_qty) > 0 else 0,
            'type': 'STOCK',
        })

    # ─── Match indexes ───
    index_rows = []
    for idx_sym, idx_info in INDEX_MAP.items():
        vol_qty = symbols_df[symbols_df['SYMBOL'] == idx_sym]['VOL_FRZ_QTY'].values
        index_rows.append({
            'pSymbol': idx_info['token'],
            'pSymbolName': idx_sym,
            'angelone_symbol': idx_info['angelone_name'],
            'company_name': idx_info['angelone_name'],
            'exchange': idx_info['exch'],
            'instrument_type': 'INDEX',
            'lot_size': '',
            'tick_size': '',
            'vol_freeze_qty': int(vol_qty[0]) if len(vol_qty) > 0 else 0,
            'type': 'INDEX',
        })

    all_rows = index_rows + stock_rows
    result = pd.DataFrame(all_rows)

    # Log results
    logger.info(f"Matched {len(stock_rows)} stocks + {len(index_rows)} indexes = {len(result)} total")

    # Log unmatched stocks
    matched_symbols = set(r['pSymbolName'] for r in stock_rows)
    unmatched = [s for s in stock_symbols if s not in matched_symbols]
    if unmatched:
        logger.warning(f"Unmatched stocks ({len(unmatched)}): {unmatched[:20]}{'...' if len(unmatched) > 20 else ''}")

    return result.reset_index(drop=True)


def load_credentials() -> list:
    """Load scanner credentials from JSON config."""
    cred_path = os.path.join(BASE_DIR, "scanner_credentials.json")
    with open(cred_path, 'r') as f:
        creds = json.load(f)
    logger.info(f"Loaded {len(creds)} credential sets.")
    return creds


def save_and_split(all_tokens_df: pd.DataFrame, credentials: list):
    """Save the full CSV and split STOCK tokens across credentials for workers."""

    # ─── Save full combined CSV (indexes + stocks) ───
    all_path = os.path.join(BASE_DIR, "scanner_all_tokens.csv")
    all_tokens_df.to_csv(all_path, index=False)
    logger.info(f"Saved full token list: scanner_all_tokens.csv ({len(all_tokens_df)} entries)")

    # ─── Split only STOCKS across credentials (indexes don't need WebSocket) ───
    stocks_only = all_tokens_df[all_tokens_df['type'] == 'STOCK'].copy()
    stocks_only = stocks_only.reset_index(drop=True)

    n_creds = len(credentials)
    total = len(stocks_only)
    chunk_size = math.ceil(total / n_creds)

    for i, cred in enumerate(credentials):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = stocks_only.iloc[start:end].copy()

        filename = f"scanner_tokens_{cred['name']}.csv"
        filepath = os.path.join(BASE_DIR, filename)
        chunk.to_csv(filepath, index=False)
        logger.info(f"  [{cred['name']}] {len(chunk)} stocks → {filename}")


def main():
    logger.info("=" * 60)
    logger.info("STOCK SCANNER SETUP — Preparing token lists")
    logger.info("=" * 60)

    # Step 1: Download AngelOne master
    master_df = download_angelone_master()

    # Step 2: Load all symbols from stocks_name.csv (indexes + stocks)
    symbols_df = load_fno_symbols()

    # Step 3: Match against AngelOne master to get tokens
    all_tokens_df = match_tokens(master_df, symbols_df)

    if all_tokens_df.empty:
        logger.error("No tokens matched! Check stocks_name.csv and AngelOne master format.")
        return

    # Step 4: Load credentials
    credentials = load_credentials()

    # Step 5: Save full CSV and split for workers
    save_and_split(all_tokens_df, credentials)

    logger.info("=" * 60)
    n_stocks = (all_tokens_df['type'] == 'STOCK').sum()
    n_idx = (all_tokens_df['type'] == 'INDEX').sum()
    logger.info(f"SETUP COMPLETE — {n_idx} indexes + {n_stocks} stocks = {len(all_tokens_df)} total")
    logger.info(f"  Workers: {n_stocks} stocks across {len(credentials)} credentials")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
