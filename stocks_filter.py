import pandas as pd
from datetime import datetime
import datetime as dt
from dateutil.relativedelta import relativedelta
import math


def stocks_filter():
    today_date_str = datetime.now().strftime("%Y%m%d")

    # Load and clean symbol list
    symbols_df = pd.read_csv("stocks_name.csv")
    symbols_df.columns = symbols_df.columns.str.strip()
    symbols_list = symbols_df["SYMBOL"].astype(str).str.strip().unique().tolist()

    # Load data
    df1 = pd.read_csv(f"data_{today_date_str}.csv")
    df2 = pd.read_csv(f"stocks_data_{today_date_str}.csv", low_memory=False)

    # Clean columns
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Filter symbol match
    filtered_df1 = df1[df1["pSymbolName"].astype(str).str.strip().isin(symbols_list)]
    filtered_df2 = df2[df2["pSymbolName"].astype(str).str.strip().isin(symbols_list)]

    # Merge
    final_df = pd.concat([filtered_df1, filtered_df2], ignore_index=True)

    # Filters
    final_df = final_df[final_df['pGroup'] != 'BL']
    final_df = final_df[final_df['pInstType'] != 'FUTSTK']
    final_df = final_df[final_df['pExchSeg'] != 'bse_cm']

    # Clean strike price
    if 'dStrikePrice;' in final_df.columns:
        final_df['dStrikePrice;'] = (final_df['dStrikePrice;'] / 100).astype(int)

    # # Clean expiry date
    final_df = final_df[pd.to_numeric(final_df["lExpiryDate"], errors="coerce").notnull()]
    final_df["lExpiryDate"] = final_df["lExpiryDate"].astype(float)

    def safe_convert_expiry(ts):
        try:
            return dt.datetime.fromtimestamp(ts).date() + relativedelta(years=10)
        except (OSError, OverflowError, ValueError):
            return None

    final_df["lExpiryDate"] = final_df["lExpiryDate"].apply(safe_convert_expiry)
    final_df = final_df.dropna(subset=["lExpiryDate"])
    
    final_df = final_df[[
        "pSymbol", "pExchSeg", "pSymbolName", "pTrdSymbol",
        "lLotSize", "dStrikePrice;", "pOptionType", "pInstType", "lExpiryDate"
    ]]
    
    df = final_df[final_df["pExchSeg"]=="nse_cm"]
    df.to_csv(f"stocks_csv_{today_date_str}.csv",index=False)
    num_parts = 3

    rows_per_file = math.ceil(len(df) / num_parts)

    for i in range(num_parts):
        start_idx = i * rows_per_file
        end_idx = start_idx + rows_per_file
        df_part = df.iloc[start_idx:end_idx]
        df_part.to_csv(f"stocks_csv_{i+1}.csv", index=False)
    
    today = dt.datetime.today()
    current_month = today.month
    current_year = today.year

    final_df = final_df[
        (final_df["lExpiryDate"].apply(lambda x: x.month) == current_month) &
        (final_df["lExpiryDate"].apply(lambda x: x.year) == current_year)
    ]    
    
    final_df["Position"] = 0
    final_df.to_csv(f"final_stock_csv_{today_date_str}.csv", index=False)

    print(f"Filtered final_stock_csv_{today_date_str}.csv created successfully.")

stocks_filter()
