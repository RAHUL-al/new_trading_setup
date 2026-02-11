"""
symbol_found.py — Auto-select nearest expiry NIFTY CE/PE options (₹110-150)
Uses AngelOne scrip master + SmartAPI for LTP checks.
Stores selections in Redis for fast access by TradingBot.
"""

from SmartApi import SmartConnect
from logzero import logger
import pyotp
import pandas as pd
import redis
import ujson as json
import datetime
import pytz
import time
from multiprocessing import Process

# ─────────── Config ───────────
TOTP_TOKEN = "33OUTDUE57WS3TUPHPLFUCGHFM"
API_KEY = "Ytt1NkKD"
CLIENT_ID = "R865920"
PWD = "7355"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "Rahul@7355"

INDEX_TOKEN = "99926000"         # NIFTY 50 index
TRADING_SYMBOLS_KEY = "Trading_symbol"
POSITIONS_KEY = "active_positions"
OPTION_TOKENS_CHANNEL = "option_tokens_updated"

PRICE_MIN = 110
PRICE_MAX = 150
REFRESH_INTERVAL = 30  # seconds

INDIA_TZ = pytz.timezone("Asia/Kolkata")

r = redis.StrictRedis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
    db=0, decode_responses=True
)


# ─────────── AngelOne API ───────────
def connect_angleone():
    """Login to AngelOne SmartAPI and return the SmartConnect object."""
    totp = pyotp.TOTP(TOTP_TOKEN).now()
    smart_api = SmartConnect(API_KEY)
    smart_api.generateSession(CLIENT_ID, PWD, totp)
    logger.info("AngelOne API connected successfully.")
    return smart_api


def get_ltp(smart_api, exchange, trading_symbol, token):
    """Get Last Traded Price for a given instrument via AngelOne REST API."""
    try:
        data = smart_api.ltpData(exchange, trading_symbol, token)
        if data and data.get("status") and data.get("data"):
            ltp = data["data"].get("ltp")
            if ltp is not None:
                return float(ltp)
    except Exception as e:
        logger.error(f"LTP fetch failed for {trading_symbol}: {e}")
    return None


# ─────────── Scrip Master ───────────
def load_nifty_options():
    """Load NIFTY OPTIDX from AngelOne scrip master CSV."""
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    scrip_file = f"data_{today_str}.csv"

    try:
        df = pd.read_csv(scrip_file, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Scrip master not found: {scrip_file}. Run create_angleone_csv.py first.")
        return None

    # Filter for NIFTY options on NFO
    nifty_opts = df[
        (df["pSymbolName"].astype(str).str.strip() == "NIFTY") &
        (df["pInstType"].astype(str).str.strip() == "OPTIDX") &
        (df["pExchSeg"].astype(str).str.strip() == "NFO")
    ].copy()

    if nifty_opts.empty:
        logger.error("No NIFTY OPTIDX found in scrip master.")
        return None

    # Parse expiry dates
    nifty_opts["expiry_parsed"] = pd.to_datetime(
        nifty_opts["lExpiryDate"].astype(str).str.strip(),
        format="%d%b%Y", errors="coerce"
    )
    nifty_opts = nifty_opts.dropna(subset=["expiry_parsed"])

    # Parse strike price — handle "dStrikePrice;" column
    strike_col = "dStrikePrice;" if "dStrikePrice;" in nifty_opts.columns else "dStrikePrice"
    nifty_opts["strike_raw"] = pd.to_numeric(nifty_opts[strike_col], errors="coerce")
    nifty_opts = nifty_opts.dropna(subset=["strike_raw"])
    # AngelOne scrip master stores strikes as 100x actual (e.g., 2515000 = ₹25150)
    nifty_opts["strike"] = nifty_opts["strike_raw"] / 100.0

    logger.info(f"Loaded {len(nifty_opts)} NIFTY options from scrip master.")
    return nifty_opts


def get_nearest_expiry(options_df):
    """Get nearest expiry date (including today for expiry-day trading)."""
    today = pd.Timestamp.now().normalize()
    future_expiries = options_df[options_df["expiry_parsed"] >= today]["expiry_parsed"].unique()

    if len(future_expiries) == 0:
        logger.error("No future expiries found.")
        return None

    nearest = min(future_expiries)
    logger.info(f"Nearest expiry: {nearest.strftime('%d%b%Y')}")
    return nearest


def find_option_in_price_range(smart_api, options_df, expiry, nifty_ltp, option_type):
    """
    Find a CE or PE option with LTP between PRICE_MIN and PRICE_MAX.

    Smart bidirectional strategy:
    1. Start at ATM strike
    2. If LTP > PRICE_MAX → go OTM (CE: higher strikes, PE: lower strikes)
    3. If LTP < PRICE_MIN → go ITM (CE: lower strikes, PE: higher strikes)
    4. Stop when price overshoots in the other direction
    """
    atm_strike = round(nifty_ltp / 50) * 50
    logger.info(f"NIFTY LTP={nifty_ltp}, ATM strike={atm_strike}, searching {option_type}...")

    # Filter for this expiry and option type
    filtered = options_df[
        (options_df["expiry_parsed"] == expiry) &
        (options_df["pOptionType"].astype(str).str.strip() == option_type)
    ].copy()

    if filtered.empty:
        logger.error(f"No {option_type} options found for expiry {expiry}")
        return None, None

    available_strikes = sorted(filtered["strike"].unique())

    # Check ATM first to determine search direction
    atm_ltp = _check_strike(smart_api, filtered, atm_strike, option_type)

    if atm_ltp is not None and PRICE_MIN <= atm_ltp <= PRICE_MAX:
        # ATM is in range — return it
        row = filtered[filtered["strike"] == atm_strike].iloc[0]
        token = str(row["pSymbol"]).strip()
        trading_symbol = str(row["pTrdSymbol"]).strip()
        logger.info(f"✅ Selected {option_type}: {trading_symbol} (token={token}) LTP=₹{atm_ltp}")
        return trading_symbol, token

    if atm_ltp is not None and atm_ltp > PRICE_MAX:
        # Too expensive at ATM → go OTM (price decreases)
        if option_type == "CE":
            candidates = [s for s in available_strikes if s > atm_strike]
        else:
            candidates = [s for s in available_strikes if s < atm_strike]
            candidates = list(reversed(candidates))
        search_direction = "OTM"
    else:
        # Too cheap at ATM (or ATM not found) → go ITM (price increases)
        if option_type == "CE":
            candidates = [s for s in available_strikes if s < atm_strike]
            candidates = list(reversed(candidates))  # Start closest to ATM
        else:
            candidates = [s for s in available_strikes if s > atm_strike]
        search_direction = "ITM"

    logger.info(f"  ATM LTP={atm_ltp}, searching {search_direction}...")

    for strike in candidates:
        row = filtered[filtered["strike"] == strike].iloc[0]
        token = str(row["pSymbol"]).strip()
        trading_symbol = str(row["pTrdSymbol"]).strip()

        ltp = get_ltp(smart_api, "NFO", trading_symbol, token)
        if ltp is None:
            continue

        logger.info(f"  {option_type} Strike {strike:.0f}: {trading_symbol} LTP={ltp}")

        if PRICE_MIN <= ltp <= PRICE_MAX:
            logger.info(f"✅ Selected {option_type}: {trading_symbol} (token={token}) LTP=₹{ltp}")
            return trading_symbol, token

        # If going OTM (prices decreasing) and price drops below range → stop
        if search_direction == "OTM" and ltp < PRICE_MIN:
            logger.info(f"  {option_type} LTP {ltp} below ₹{PRICE_MIN}, stopping OTM search.")
            break

        # If going ITM (prices increasing) and price goes above range → stop
        if search_direction == "ITM" and ltp > PRICE_MAX:
            logger.info(f"  {option_type} LTP {ltp} above ₹{PRICE_MAX}, stopping ITM search.")
            break

    logger.warning(f"No {option_type} option found in ₹{PRICE_MIN}-{PRICE_MAX} range")
    return None, None


def _check_strike(smart_api, filtered_df, strike, option_type):
    """Check LTP for a specific strike. Returns LTP or None."""
    rows = filtered_df[filtered_df["strike"] == strike]
    if rows.empty:
        return None
    row = rows.iloc[0]
    token = str(row["pSymbol"]).strip()
    trading_symbol = str(row["pTrdSymbol"]).strip()
    ltp = get_ltp(smart_api, "NFO", trading_symbol, token)
    if ltp is not None:
        logger.info(f"  {option_type} Strike {strike:.0f}: {trading_symbol} LTP={ltp}")
    return ltp


# ─────────── Main Loop ───────────
def has_open_position():
    """Check if there's an active position in Redis."""
    try:
        if r.exists(POSITIONS_KEY):
            positions = json.loads(r.get(POSITIONS_KEY))
            return bool(positions)
    except Exception as e:
        logger.error(f"Error checking positions: {e}")
    return False


def select_and_store_symbols(smart_api, options_df, expiry):
    """Select CE/PE options in price range and store in Redis."""
    # Get NIFTY LTP from Redis
    nifty_ltp_str = r.get(INDEX_TOKEN)
    if not nifty_ltp_str:
        logger.warning("NIFTY LTP not available in Redis yet. Waiting...")
        return False

    nifty_ltp = float(nifty_ltp_str)
    logger.info(f"NIFTY LTP: {nifty_ltp}")

    ce_symbol, ce_token = find_option_in_price_range(
        smart_api, options_df, expiry, nifty_ltp, "CE"
    )
    pe_symbol, pe_token = find_option_in_price_range(
        smart_api, options_df, expiry, nifty_ltp, "PE"
    )

    if not ce_token or not pe_token:
        logger.error("Failed to find both CE and PE options in range.")
        return False

    # Check if tokens changed from previous selection
    old_data = r.get(TRADING_SYMBOLS_KEY)
    new_data = {
        "CE": [ce_symbol, ce_token],
        "PE": [pe_symbol, pe_token],
    }

    r.set(TRADING_SYMBOLS_KEY, json.dumps(new_data))
    logger.info(f"Stored Trading_symbol: CE={ce_symbol}/{ce_token}, PE={pe_symbol}/{pe_token}")

    # Notify websocket if tokens changed
    if old_data:
        old = json.loads(old_data)
        old_ce = old.get("CE", [None, None])[1]
        old_pe = old.get("PE", [None, None])[1]
        if old_ce != ce_token or old_pe != pe_token:
            r.publish(OPTION_TOKENS_CHANNEL, json.dumps(new_data))
            logger.info("Published option_tokens_updated (tokens changed)")
    else:
        r.publish(OPTION_TOKENS_CHANNEL, json.dumps(new_data))
        logger.info("Published option_tokens_updated (first selection)")

    return True


def trading_symbol_loop():
    """Main loop: continuously select option contracts."""
    india_tz = pytz.timezone("Asia/Kolkata")
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)

    # Connect to AngelOne
    smart_api = connect_angleone()

    # Load scrip master
    options_df = load_nifty_options()
    if options_df is None:
        logger.error("Cannot start without scrip master data.")
        return

    # Get nearest expiry
    expiry = get_nearest_expiry(options_df)
    if expiry is None:
        logger.error("Cannot start without valid expiry.")
        return

    logger.info("Starting option contract selection loop...")

    while True:
        now = datetime.datetime.now(india_tz).time()

        if now < market_open or now >= market_close:
            logger.info("Outside market hours. Waiting...")
            time.sleep(30)
            continue

        try:
            # Only refresh if no active position
            if not has_open_position():
                select_and_store_symbols(smart_api, options_df, expiry)
            else:
                logger.info("Position open — skipping symbol refresh")

        except Exception as e:
            logger.error(f"Error in symbol selection: {e}")
            # Reconnect API on failure
            try:
                smart_api = connect_angleone()
            except Exception as re_err:
                logger.error(f"Reconnection failed: {re_err}")

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    P0 = Process(target=trading_symbol_loop)
    P0.start()
    P0.join()