import pandas as pd
import os
import requests
from datetime import datetime
from logzero import logger


# AngelOne SmartAPI Scrip Master URL (no login needed)
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"


def download_scrip_master():
    """Download the full instrument list from AngelOne SmartAPI."""
    logger.info("Downloading AngelOne scrip master...")
    response = requests.get(SCRIP_MASTER_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    logger.info(f"Downloaded {len(df)} instruments from AngelOne.")
    return df


def create_angleone_dataframe(file_path):
    """Download scrip master and save as CSV."""
    df = download_scrip_master()

    # Rename columns to match existing pipeline (pSymbol, pSymbolName, etc.)
    column_mapping = {
        "token": "pSymbol",
        "symbol": "pTrdSymbol",
        "name": "pSymbolName",
        "expiry": "lExpiryDate",
        "strike": "dStrikePrice;",
        "lotsize": "lLotSize",
        "instrumenttype": "pInstType",
        "exch_seg": "pExchSeg",
        "tick_size": "tick_size",
    }
    df = df.rename(columns=column_mapping)

    # Add pOptionType from the symbol name (CE/PE)
    df["pOptionType"] = df["pTrdSymbol"].apply(
        lambda x: "CE" if str(x).endswith("CE") else ("PE" if str(x).endswith("PE") else "")
    )

    df.to_csv(file_path, index=False)
    logger.info(f"Saved scrip master to {file_path}")
    return df


def filter_nse_fo(df):
    """Filter for NSE F&O instruments only."""
    df_fo = df[df["pExchSeg"] == "NFO"]
    return df_fo


def filter_nse_cm(df):
    """Filter for NSE Cash Market instruments only."""
    df_cm = df[df["pExchSeg"] == "NSE"]
    return df_cm


def delete_old_files(today_file_name):
    """Delete old CSV files from previous days."""
    today_date_str = datetime.now().strftime("%Y%m%d")
    folder_path = "."
    preserve_files = {
        today_file_name,
        "stocks_name.csv",
        f"stocks_data_{today_date_str}.csv",
        f"final_stock_csv_{today_date_str}.csv",
        f"stocks_csv_{today_date_str}.csv",
        "stocks_csv_1.csv",
        "stocks_csv_2.csv",
        "stocks_csv_3.csv",
        "future_and_options_token.csv",
    }

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and file_name not in preserve_files:
            # Only delete dated data files
            if file_name.startswith("data_"):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                    logger.info(f"Deleted old file: {file_name}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_name}: {e}")


def create_angleone_df():
    """Main function: download scrip master, clean old files, return DataFrame."""
    today_date_str = datetime.now().strftime("%Y%m%d")
    file_path = f"data_{today_date_str}.csv"

    delete_old_files(file_path)

    if os.path.exists(file_path):
        logger.info(f"Using existing file: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
    else:
        df = create_angleone_dataframe(file_path)

    logger.info(f"Loaded {len(df)} instruments.")
    return df


if __name__ == "__main__":
    create_angleone_df()
