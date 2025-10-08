# from SmartApi.smartWebSocketV2 import SmartWebSocketV2
# from SmartApi import SmartConnect
# from logzero import logger
# import pyotp
# import pandas as pd
# import redis
# import threading
# import time
# import ujson as json
# import pytz
# from multiprocessing import Process, Manager
# import ta
# import numpy as np  
# from neo_api_client import NeoAPI
# import pandas as pd
# import json
# import threading
# import time
# import datetime
# import pandas as pd
# import threading
# import time


# token = "33OUTDUE57WS3TUPHPLFUCGHFM"
# api_key = "Ytt1NkKD"
# clientId = "R865920"
# pwd = '7355'
# correlation_id = "Rahul_7355"

# ltp_data = {}

# symbol_map = {}
# shared_tokens = []
# r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)


# def load_tokens_from_csv():
#     now = datetime.datetime.now()
#     market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
#     market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

#     while market_open < now < market_close:
#         try:
#             data = r.get("Trading_symbol")
#             if data:
#                 data_dict = json.loads(data)
#                 CE_token = str(data_dict["CE"][1]).strip()
#                 PE_token = str(data_dict["PE"][1]).strip()

#                 shared_tokens[:] = [CE_token, PE_token]
#                 r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
#                 logger.info(f"Updated shared tokens: {list(shared_tokens)}")
#                 time.sleep(5)
#             else:
#                 logger.error("No token data found in Redis")
#         except Exception as e:
#             logger.exception("Error loading tokens from Redis: %s", e)
#     else:
#         logger.info("Market closed or not open yet. Sleeping...")


# def connect_api():
#     totp = pyotp.TOTP(token).now()
#     smartApi = SmartConnect(api_key)
#     smartApi.generateSession(clientId, pwd, totp)
#     FEED_TOKEN = smartApi.getfeedToken()
#     return FEED_TOKEN

# last_candle_time_map = {}
# last_candle_map = {}


# def run_websocket():
#     def on_data(wsapp, message):
#         if message != b'\x00':
#             try:
#                 tick = message
#                 token = str(tick.get('token'))
#                 ltp = tick.get('last_traded_price')
#                 if token and ltp:
#                     price = ltp / 100
#                     r.set(token,price)
#                     logger.info(f"[Price] {token} @  -> {price}")

#             except Exception as e:
#                 logger.error(f"Error processing tick: {e}")


#     def on_open(wsapp):
#         logger.info("WebSocket Opened")
#         tokens = json.loads(r.get("selective_trading_symbol"))
#         print(tokens)
#         token_list = [{"exchangeType": 2, "tokens": tokens}]
#         sws.subscribe(correlation_id, 1, token_list)


#     def on_error(wsapp, error):
#         logger.error(f"WebSocket Error: {error}")
#         logger.info("Reconnecting in 3 seconds...")
#         time.sleep(3)
#         threading.Thread(target=load_tokens_from_csv).start().join()
#         threading.Thread(target=run_websocket).start().join()

#     def on_close(wsapp):
#         logger.info("WebSocket Closed")
#         logger.info("Reconnecting in 3 seconds...")
#         time.sleep(3)
#         threading.Thread(target=load_tokens_from_csv).start().join()
#         threading.Thread(target=run_websocket).start().join

#     FEED_TOKEN = connect_api()
#     sws = SmartWebSocketV2(token, api_key, clientId, FEED_TOKEN)
#     sws.on_open = on_open
#     sws.on_data = on_data
#     sws.on_error = on_error
#     sws.on_close = on_close

#     logger.info("Connecting WebSocket...")
#     sws.connect()



# if __name__ == "__main__":
#     loader = Process(target=load_tokens_from_csv)
#     wsproc = Process(target=run_websocket)

#     loader.start()
#     wsproc.start()

#     wsproc.join()



# from SmartApi.smartWebSocketV2 import SmartWebSocketV2
# from SmartApi import SmartConnect
# from logzero import logger
# import pyotp
# import pandas as pd
# import redis
# import threading
# import time
# import ujson as json
# import pytz
# from multiprocessing import Process, Manager
# import ta
# import numpy as np  
# from neo_api_client import NeoAPI
# import pandas as pd
# import json
# import threading
# import time
# import datetime
# import asyncio

# token = "33OUTDUE57WS3TUPHPLFUCGHFM"
# api_key = "Ytt1NkKD"
# clientId = "R865920"
# pwd = '7355'
# correlation_id = "Rahul_7355"

# ltp_data = {}
# symbol_map = {}
# shared_tokens = []
# r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# # Global WebSocket instance
# sws = None
# ws_thread = None
# ws_running = False
# ws_lock = threading.Lock()  # Add a lock for thread-safe WebSocket operations

# def has_active_positions():
#     try:
#         if r.exists("active_positions"):
#             positions = json.loads(r.get("active_positions"))
#             return len(positions) > 0
#     except Exception as e:
#         logger.exception("Error checking active positions: %s", e)
#     return False

# def load_tokens_from_csv():
#     global shared_tokens
#     now = datetime.datetime.now()
#     market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
#     market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

#     previous_tokens = set(shared_tokens)
    
#     while market_open < now < market_close:
#         try:
#             data = r.get("Trading_symbol")
#             if data:
#                 data_dict = json.loads(data)
#                 CE_token = str(data_dict["CE"][1]).strip()
#                 PE_token = str(data_dict["PE"][1]).strip()

#                 new_tokens = [CE_token, PE_token]
                
#                 if set(new_tokens) != set(previous_tokens):
#                     logger.info(f"Tokens changed from {previous_tokens} to {new_tokens}")
                    
#                     if not has_active_positions():
#                         with ws_lock:
#                             shared_tokens[:] = new_tokens
#                             r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
#                             logger.info(f"Updated shared tokens: {list(shared_tokens)}")
                            
#                             # Restart WebSocket if it's running
#                             if ws_running:
#                                 logger.info("Restarting WebSocket with new tokens")
#                                 stop_websocket()
#                                 time.sleep(2)
#                                 start_websocket()
#                     else:
#                         logger.info("Active positions detected, skipping token update")
                            
#                     previous_tokens = set(new_tokens)
#                 else:
#                     with ws_lock:
#                         shared_tokens[:] = new_tokens
                    
#                 time.sleep(5)
#             else:
#                 logger.error("No token data found in Redis")
#                 time.sleep(5)
#         except Exception as e:
#             logger.exception("Error loading tokens from Redis: %s", e)
#             time.sleep(5)
#     else:
#         logger.info("Market closed or not open yet. Sleeping...")

# def connect_api():
#     totp = pyotp.TOTP(token).now()
#     smartApi = SmartConnect(api_key)
#     smartApi.generateSession(clientId, pwd, totp)
#     FEED_TOKEN = smartApi.getfeedToken()
#     return FEED_TOKEN

# def stop_websocket():
#     global sws, ws_running
#     with ws_lock:
#         if sws:
#             try:
#                 ws_running = False
#                 sws.close_connection()
#             except Exception as e:
#                 logger.error(f"Error closing WebSocket: {e}")
#             finally:
#                 sws = None
#         # Wait a moment for the connection to properly close
#         time.sleep(1)

# def start_websocket():
#     global ws_thread, ws_running
    
#     with ws_lock:
#         # Ensure any existing thread is terminated
#         if ws_thread and ws_thread.is_alive():
#             return
            
#         ws_thread = threading.Thread(target=run_websocket)
#         ws_thread.daemon = True
#         ws_thread.start()

# def run_websocket():
#     global sws, ws_running
    
#     def on_data(wsapp, message):
#         if message != b'\x00':
#             try:
#                 tick = message
#                 token = str(tick.get('token'))
#                 ltp = tick.get('last_traded_price')
#                 if token and ltp:
#                     price = ltp / 100
#                     r.set(token, price)
#                     logger.info(f"[Price] {token} -> {price}")

#             except Exception as e:
#                 logger.error(f"Error processing tick: {e}")

#     def on_open(wsapp):
#         global sws
#         logger.info("WebSocket Opened")
#         try:
#             if sws is not None:
#                 tokens_data = r.get("selective_trading_symbol")
#                 if tokens_data:
#                     tokens = json.loads(tokens_data)
#                     logger.info(f"Subscribing to tokens: {tokens}")
#                     token_list = [{"exchangeType": 2, "tokens": tokens}]
#                     sws.subscribe(correlation_id, 1, token_list)
#                 else:
#                     logger.error("No selective_trading_symbol found in Redis")
#             else:
#                 logger.warning("WebSocket instance is None in on_open callback")
#         except Exception as e:
#             logger.error(f"Error subscribing to tokens: {e}")

#     def on_error(wsapp, error):
#         logger.error(f"WebSocket Error: {error}")
#         logger.info("Reconnecting in 3 seconds...")
#         time.sleep(3)
#         if ws_running:
#             stop_websocket()
#             start_websocket()

#     def on_close(wsapp):
#         logger.info("WebSocket Closed")
#         if ws_running:
#             logger.info("Reconnecting in 3 seconds...")
#             time.sleep(3)
#             start_websocket()

#     try:
#         FEED_TOKEN = connect_api()
#         sws = SmartWebSocketV2(token, api_key, clientId, FEED_TOKEN)
#         sws.on_open = on_open
#         sws.on_data = on_data
#         sws.on_error = on_error
#         sws.on_close = on_close

#         logger.info("Connecting WebSocket...")
#         ws_running = True
#         sws.connect()
#     except Exception as e:
#         logger.error(f"WebSocket connection failed: {e}")
#         ws_running = False
#         sws = None

# def monitor_positions():
#     previous_has_positions = False
    
#     while True:
#         try:
#             current_has_positions = has_active_positions()
            
#             if previous_has_positions and not current_has_positions:
#                 logger.info("All positions closed, checking for token updates")
                
#                 data = r.get("Trading_symbol")
#                 if data:
#                     data_dict = json.loads(data)
#                     CE_token = str(data_dict["CE"][1]).strip()
#                     PE_token = str(data_dict["PE"][1]).strip()
#                     new_tokens = [CE_token, PE_token]
                    
#                     # Check if tokens have actually changed
#                     with ws_lock:
#                         if set(new_tokens) != set(shared_tokens):
#                             logger.info(f"Updating tokens after position close: {new_tokens}")
#                             shared_tokens[:] = new_tokens
#                             r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
                            
#                             if ws_running:
#                                 logger.info("Restarting WebSocket with new tokens")
#                                 stop_websocket()
#                                 time.sleep(3)  # Add longer delay for clean shutdown
#                                 start_websocket()
#                         else:
#                             logger.info("Tokens unchanged, no need to restart WebSocket")
            
#             previous_has_positions = current_has_positions
#             time.sleep(5)
#         except Exception as e:
#             logger.error(f"Error in position monitor: {e}")
#             time.sleep(5)

# if __name__ == "__main__":
#     loader = Process(target=load_tokens_from_csv)
#     loader.start()
    
#     time.sleep(2)

#     start_websocket()
    
#     monitor_thread = threading.Thread(target=monitor_positions)
#     monitor_thread.daemon = True
#     monitor_thread.start()
    
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logger.info("Shutting down...")
#         ws_running = False
#         if sws:
#             sws.close_connection()
#         loader.terminate()


# from SmartApi.smartWebSocketV2 import SmartWebSocketV2
# from SmartApi import SmartConnect
# from logzero import logger
# import pyotp
# import pandas as pd
# import redis
# import threading
# import time
# import ujson as json
# import pytz
# from multiprocessing import Process, Manager
# import ta
# import numpy as np  
# from neo_api_client import NeoAPI
# import pandas as pd
# import json
# import threading
# import time
# import datetime
# import asyncio

# token = "33OUTDUE57WS3TUPHPLFUCGHFM"
# api_key = "Ytt1NkKD"
# clientId = "R865920"
# pwd = '7355'
# correlation_id = "Rahul_7355"

# ltp_data = {}
# symbol_map = {}
# shared_tokens = []
# r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# # Global WebSocket instance
# sws = None
# ws_thread = None
# ws_running = False

# def has_active_positions():
#     try:
#         if r.exists("active_positions"):
#             positions = json.loads(r.get("active_positions"))
#             return len(positions) > 0
#     except Exception as e:
#         logger.exception("Error checking active positions: %s", e)
#     return False

# def load_tokens_from_csv():
#     global shared_tokens
#     now = datetime.datetime.now()
#     market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
#     market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

#     previous_tokens = set(shared_tokens)
    
#     while market_open < now < market_close:
#         try:
#             data = r.get("Trading_symbol")
#             if data:
#                 data_dict = json.loads(data)
#                 CE_token = str(data_dict["CE"][1]).strip()
#                 PE_token = str(data_dict["PE"][1]).strip()

#                 new_tokens = [CE_token, PE_token]
                
#                 if set(new_tokens) != set(previous_tokens):
#                     logger.info(f"Tokens changed from {previous_tokens} to {new_tokens}")
                    
#                     if not has_active_positions():
#                         shared_tokens[:] = new_tokens
#                         r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
#                         logger.info(f"Updated shared tokens: {list(shared_tokens)}")
                        
#                         # Restart WebSocket if it's running
#                         if ws_running:
#                             logger.info("Restarting WebSocket with new tokens")
#                             stop_websocket()
#                             time.sleep(2)
#                             start_websocket()
#                     else:
#                         logger.info("Active positions detected, skipping token update")
                        
#                     previous_tokens = set(new_tokens)
#                 else:
#                     shared_tokens[:] = new_tokens
                    
#                 time.sleep(5)
#             else:
#                 logger.error("No token data found in Redis")
#                 time.sleep(5)
#         except Exception as e:
#             logger.exception("Error loading tokens from Redis: %s", e)
#             time.sleep(5)
#     else:
#         logger.info("Market closed or not open yet. Sleeping...")

# def connect_api():
#     totp = pyotp.TOTP(token).now()
#     smartApi = SmartConnect(api_key)
#     smartApi.generateSession(clientId, pwd, totp)
#     FEED_TOKEN = smartApi.getfeedToken()
#     return FEED_TOKEN

# def stop_websocket():
#     global sws, ws_running
#     if sws:
#         try:
#             ws_running = False
#             sws.close_connection()
#         except Exception as e:
#             logger.error(f"Error closing WebSocket: {e}")
#         finally:
#             sws = None
#     # Wait a moment for the connection to properly close
#     time.sleep(1)

# def start_websocket():
#     global ws_thread, ws_running
    
#     # Ensure any existing thread is terminated
#     if ws_thread and ws_thread.is_alive():
#         return
        
#     ws_thread = threading.Thread(target=run_websocket)
#     ws_thread.daemon = True
#     ws_thread.start()
#     # Don't set ws_running here - let run_websocket() set it when connected

# def run_websocket():
#     global sws, ws_running
    
#     def on_data(wsapp, message):
#         if message != b'\x00':
#             try:
#                 tick = message
#                 token = str(tick.get('token'))
#                 ltp = tick.get('last_traded_price')
#                 if token and ltp:
#                     price = ltp / 100
#                     r.set(token, price)
#                     logger.info(f"[Price] {token} -> {price}")

#             except Exception as e:
#                 logger.error(f"Error processing tick: {e}")

#     def on_open(wsapp):
#         global sws
#         logger.info("WebSocket Opened")
#         try:
#             if sws is not None:
#                 tokens_data = r.get("selective_trading_symbol")
#                 if tokens_data:
#                     tokens = json.loads(tokens_data)
#                     logger.info(f"Subscribing to tokens: {tokens}")
#                     token_list = [{"exchangeType": 2, "tokens": tokens}]
#                     sws.subscribe(correlation_id, 1, token_list)
#                 else:
#                     logger.error("No selective_trading_symbol found in Redis")
#             else:
#                 logger.warning("WebSocket instance is None in on_open callback")
#         except Exception as e:
#             logger.error(f"Error subscribing to tokens: {e}")

#     def on_error(wsapp, error):
#         logger.error(f"WebSocket Error: {error}")
#         logger.info("Reconnecting in 3 seconds...")
#         time.sleep(3)
#         if ws_running:
#             stop_websocket()
#             start_websocket()

#     def on_close(wsapp):
#         logger.info("WebSocket Closed")
#         if ws_running:
#             logger.info("Reconnecting in 3 seconds...")
#             time.sleep(3)
#             start_websocket()

#     try:
#         FEED_TOKEN = connect_api()
#         sws = SmartWebSocketV2(token, api_key, clientId, FEED_TOKEN)
#         sws.on_open = on_open
#         sws.on_data = on_data
#         sws.on_error = on_error
#         sws.on_close = on_close

#         logger.info("Connecting WebSocket...")
#         sws.connect()
#         ws_running = True
#     except Exception as e:
#         logger.error(f"WebSocket connection failed: {e}")
#         ws_running = False
#         sws = None

# def monitor_positions():
#     previous_has_positions = False
    
#     while True:
#         try:
#             current_has_positions = has_active_positions()
            
#             if previous_has_positions and not current_has_positions:
#                 logger.info("All positions closed, checking for token updates")
                
#                 data = r.get("Trading_symbol")
#                 if data:
#                     data_dict = json.loads(data)
#                     CE_token = str(data_dict["CE"][1]).strip()
#                     PE_token = str(data_dict["PE"][1]).strip()
#                     new_tokens = [CE_token, PE_token]
                    
#                     if set(new_tokens) != set(shared_tokens):
#                         logger.info(f"Updating tokens after position close: {new_tokens}")
#                         shared_tokens[:] = new_tokens
#                         r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
                        
#                         if ws_running:
#                             stop_websocket()
#                             time.sleep(3)  # Add longer delay for clean shutdown
#                             start_websocket()
            
#             previous_has_positions = current_has_positions
#             time.sleep(5)
#         except Exception as e:
#             logger.error(f"Error in position monitor: {e}")
#             time.sleep(5)

# if __name__ == "__main__":
#     loader = Process(target=load_tokens_from_csv)
#     loader.start()
    
#     time.sleep(2)

#     start_websocket()
    
#     monitor_thread = threading.Thread(target=monitor_positions)
#     monitor_thread.daemon = True
#     monitor_thread.start()
    
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logger.info("Shutting down...")
#         ws_running = False
#         if sws:
#             sws.close_connection()
#         loader.terminate()







from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from SmartApi import SmartConnect
from logzero import logger
import pyotp
import pandas as pd
import redis
import threading
import time
import ujson as json
import pytz
from multiprocessing import Process, Manager
import ta
import numpy as np  
from neo_api_client import NeoAPI
import pandas as pd
import json
import threading
import time
import datetime
import asyncio

token = "33OUTDUE57WS3TUPHPLFUCGHFM"
api_key = "Ytt1NkKD"
clientId = "R865920"
pwd = '7355'
correlation_id = "Rahul_7355"

ltp_data = {}
symbol_map = {}
shared_tokens = []
r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# Global WebSocket instance
sws = None
ws_thread = None
ws_running = False
ws_lock = threading.Lock()

def has_active_positions():
    try:
        if r.exists("active_positions"):
            positions = json.loads(r.get("active_positions"))
            return len(positions) > 0
    except Exception as e:
        logger.exception("Error checking active positions: %s", e)
    return False

def load_tokens_from_csv():
    global shared_tokens
    now = datetime.datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    previous_tokens = set(shared_tokens)
    
    while market_open < now < market_close:
        try:
            data = r.get("Trading_symbol")
            if data:
                data_dict = json.loads(data)
                CE_token = str(data_dict["CE"][1]).strip()
                PE_token = str(data_dict["PE"][1]).strip()

                new_tokens = [CE_token, PE_token]
                
                if set(new_tokens) != set(previous_tokens):
                    logger.info(f"Tokens changed from {previous_tokens} to {new_tokens}")
                    
                    if not has_active_positions():
                        with ws_lock:
                            shared_tokens[:] = new_tokens
                            r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
                            logger.info(f"Updated shared tokens: {list(shared_tokens)}")
                            
                            # Unsubscribe from old tokens and subscribe to new ones
                            if ws_running and sws:
                                try:
                                    # Unsubscribe from old tokens
                                    if previous_tokens:
                                        old_token_list = [{"exchangeType": 2, "tokens": list(previous_tokens)}]
                                        sws.unsubscribe(correlation_id, 1, old_token_list)
                                        logger.info(f"Unsubscribed from old tokens: {list(previous_tokens)}")
                                    
                                    # Subscribe to new tokens
                                    new_token_list = [{"exchangeType": 2, "tokens": new_tokens}]
                                    sws.subscribe(correlation_id, 1, new_token_list)
                                    logger.info(f"Subscribed to new tokens: {new_tokens}")
                                    
                                except Exception as e:
                                    logger.error(f"Error updating subscription: {e}")
                                    # If subscription update fails, restart WebSocket
                                    stop_websocket()
                                    time.sleep(2)
                                    start_websocket()
                    else:
                        logger.info("Active positions detected, skipping token update")
                        
                    previous_tokens = set(new_tokens)
                else:
                    shared_tokens[:] = new_tokens
                    
                time.sleep(5)
            else:
                logger.error("No token data found in Redis")
                time.sleep(5)
        except Exception as e:
            logger.exception("Error loading tokens from Redis: %s", e)
            time.sleep(5)
    else:
        logger.info("Market closed or not open yet. Sleeping...")

def connect_api():
    totp = pyotp.TOTP(token).now()
    smartApi = SmartConnect(api_key)
    smartApi.generateSession(clientId, pwd, totp)
    FEED_TOKEN = smartApi.getfeedToken()
    return FEED_TOKEN

def stop_websocket():
    global sws, ws_running
    if sws:
        try:
            ws_running = False
            sws.close_connection()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        finally:
            sws = None
    # Wait a moment for the connection to properly close
    time.sleep(1)

def start_websocket():
    global ws_thread, ws_running
    
    # Ensure any existing thread is terminated
    if ws_thread and ws_thread.is_alive():
        return
        
    ws_thread = threading.Thread(target=run_websocket)
    ws_thread.daemon = True
    ws_thread.start()

def run_websocket():
    global sws, ws_running
    
    def on_data(wsapp, message):
        if message != b'\x00':
            try:
                tick = message
                token = str(tick.get('token'))
                ltp = tick.get('last_traded_price')
                if token and ltp:
                    price = ltp / 100
                    r.set(token, price)
                    logger.info(f"[Price] {token} -> {price}")

            except Exception as e:
                logger.error(f"Error processing tick: {e}")

    def on_open(wsapp):
        global sws, ws_running
        logger.info("WebSocket Opened")
        ws_running = True
        try:
            if sws is not None:
                tokens_data = r.get("selective_trading_symbol")
                if tokens_data:
                    tokens = json.loads(tokens_data)
                    logger.info(f"Subscribing to tokens: {tokens}")
                    token_list = [{"exchangeType": 2, "tokens": tokens}]
                    sws.subscribe(correlation_id, 1, token_list)
                else:
                    logger.error("No selective_trading_symbol found in Redis")
            else:
                logger.warning("WebSocket instance is None in on_open callback")
        except Exception as e:
            logger.error(f"Error subscribing to tokens: {e}")

    def on_error(wsapp, error):
        logger.error(f"WebSocket Error: {error}")
        if ws_running:
            logger.info("Reconnecting in 3 seconds...")
            time.sleep(3)
            stop_websocket()
            start_websocket()

    def on_close(wsapp):
        logger.info("WebSocket Closed")
        if ws_running:
            logger.info("Reconnecting in 3 seconds...")
            time.sleep(3)
            start_websocket()

    try:
        FEED_TOKEN = connect_api()
        sws = SmartWebSocketV2(token, api_key, clientId, FEED_TOKEN)
        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close

        logger.info("Connecting WebSocket...")
        sws.connect()
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        ws_running = False
        sws = None
        # Try to reconnect after delay
        if ws_running:
            time.sleep(3)
            start_websocket()

def monitor_positions():
    previous_has_positions = False
    
    while True:
        try:
            current_has_positions = has_active_positions()
            
            if previous_has_positions and not current_has_positions:
                logger.info("All positions closed, checking for token updates")
                
                data = r.get("Trading_symbol")
                if data:
                    data_dict = json.loads(data)
                    CE_token = str(data_dict["CE"][1]).strip()
                    PE_token = str(data_dict["PE"][1]).strip()
                    new_tokens = [CE_token, PE_token]
                    
                    if set(new_tokens) != set(shared_tokens):
                        logger.info(f"Updating tokens after position close: {new_tokens}")
                        
                        with ws_lock:
                            shared_tokens[:] = new_tokens
                            r.set("selective_trading_symbol", json.dumps(list(shared_tokens)))
                            
                            if ws_running and sws:
                                try:
                                    # Unsubscribe from old tokens and subscribe to new ones
                                    old_tokens = list(set(shared_tokens) - set(new_tokens))
                                    if old_tokens:
                                        old_token_list = [{"exchangeType": 2, "tokens": old_tokens}]
                                        sws.unsubscribe(correlation_id, 1, old_token_list)
                                        logger.info(f"Unsubscribed from old tokens: {old_tokens}")
                                    
                                    new_token_list = [{"exchangeType": 2, "tokens": new_tokens}]
                                    sws.subscribe(correlation_id, 1, new_token_list)
                                    logger.info(f"Subscribed to new tokens: {new_tokens}")
                                    
                                except Exception as e:
                                    logger.error(f"Error updating subscription: {e}")
                                    # If subscription update fails, restart WebSocket
                                    stop_websocket()
                                    time.sleep(3)
                                    start_websocket()
            
            previous_has_positions = current_has_positions
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in position monitor: {e}")
            time.sleep(5)

if __name__ == "__main__":
    loader = Process(target=load_tokens_from_csv)
    loader.start()
    
    time.sleep(2)

    start_websocket()
    
    monitor_thread = threading.Thread(target=monitor_positions)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        ws_running = False
        if sws:
            sws.close_connection()
        loader.terminate()