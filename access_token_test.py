from neo_api_client import NeoAPI
import os

access_token = os.getenv("access_token")

access_token = access_token
consumer_key="cR7ZW_66Z5zmvEj_35GDBKpuxYga", 
consumer_secret="sbtJCz6vbHrSURnjXBGSIcSIMmka",

client = NeoAPI(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    environment="prod",
    access_token=access_token
)

order = client.place_order(
    exchange_segment="nse_fo",
    product="MIS",
    price="0",
    order_type="MKT",
    quantity="50",
    validity="DAY",
    trading_symbol="NIFTY24AUG24300CE",
    transaction_type="B"
)
print(order)
