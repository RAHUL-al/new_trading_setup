import pandas as pd
from datetime import datetime
import datetime as dt
import math
from logzero import logger


def stocks_filter():
    today_date_str = datetime.now().strftime("%Y%m%d")

    # Load and clean symbol list (from NSE F&O quantity freeze CSV)
    symbols_df = pd.read_csv("stocks_name.csv")
    symbols_df.columns = symbols_df.columns.str.strip()
    symbols_list = symbols_df["SYMBOL"].astype(str).str.strip().unique().tolist()

    # Load the AngelOne scrip master data (single file has all exchanges)
    data_file = f"data_{today_date_str}.csv"
    df = pd.read_csv(data_file, low_memory=False)
    df.columns = df.columns.str.strip()

    logger.info(f"Loaded {len(df)} instruments from {data_file}")

    # Filter by symbol names from stocks_name.csv
    filtered_df = df[df["pSymbolName"].astype(str).str.strip().isin(symbols_list)]

    logger.info(f"Filtered to {len(filtered_df)} instruments matching stocks_name.csv")

    # Remove Futures on Stocks (keep options and equity)
    filtered_df = filtered_df[filtered_df['pInstType'] != 'FUTSTK']

    # Remove BSE instruments (AngelOne uses "BSE" not "bse_cm")
    filtered_df = filtered_df[filtered_df['pExchSeg'] != 'BSE']

    # Clean strike price — AngelOne gives it as float like "17500.000000"
    if 'dStrikePrice;' in filtered_df.columns:
        filtered_df['dStrikePrice;'] = pd.to_numeric(
            filtered_df['dStrikePrice;'], errors='coerce'
        ).fillna(0).astype(int)

    # Clean expiry date — AngelOne gives it as string like "02MAR2026" or ""
    def parse_expiry(expiry_str):
        try:
            expiry_str = str(expiry_str).strip()
            if not expiry_str or expiry_str in ("-1", "", "nan"):
                return None
            return datetime.strptime(expiry_str, "%d%b%Y").date()
        except (ValueError, TypeError):
            return None

    filtered_df["lExpiryDate"] = filtered_df["lExpiryDate"].apply(parse_expiry)
    filtered_df = filtered_df.dropna(subset=["lExpiryDate"])

    # Select only required columns
    final_df = filtered_df[[
        "pSymbol", "pExchSeg", "pSymbolName", "pTrdSymbol",
        "lLotSize", "dStrikePrice;", "pOptionType", "pInstType", "lExpiryDate"
    ]].copy()

    # --- NSE Cash Market stocks (for equity websocket) ---
    # AngelOne uses "NSE" for cash market
    df_nse_cm = final_df[final_df["pExchSeg"] == "NSE"]
    df_nse_cm.to_csv(f"stocks_csv_{today_date_str}.csv", index=False)

    # Split into 3 parts for parallel websocket subscriptions
    num_parts = 3
    rows_per_file = math.ceil(len(df_nse_cm) / num_parts) if len(df_nse_cm) > 0 else 1

    for i in range(num_parts):
        start_idx = i * rows_per_file
        end_idx = start_idx + rows_per_file
        df_part = df_nse_cm.iloc[start_idx:end_idx]
        df_part.to_csv(f"stocks_csv_{i+1}.csv", index=False)

    logger.info(f"Created stocks_csv_1/2/3.csv with {len(df_nse_cm)} NSE stocks")

    # --- F&O instruments expiring this month (for options trading) ---
    today = dt.datetime.today()
    current_month = today.month
    current_year = today.year

    df_fo_current_month = final_df[
        (final_df["lExpiryDate"].apply(lambda x: x.month) == current_month) &
        (final_df["lExpiryDate"].apply(lambda x: x.year) == current_year)
    ].copy()

    df_fo_current_month["Position"] = 0
    df_fo_current_month.to_csv(f"final_stock_csv_{today_date_str}.csv", index=False)

    logger.info(
        f"Created final_stock_csv_{today_date_str}.csv with "
        f"{len(df_fo_current_month)} current-month F&O instruments"
    )

    print(f"Filtered CSVs created successfully for {today_date_str}.")


if __name__ == "__main__":
    stocks_filter()
