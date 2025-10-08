# from datetime import datetime, timedelta
# import time
# import redis

# r = redis.StrictRedis(host='localhost', port=6379, password='Rahul@7355', db=0, decode_responses=True)

# market_open_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
# datetime_now = datetime.now()
# market_close_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
# stoploss = 0

# while market_open_time < datetime_now < market_close_time:
#     signal = None
#     if r.exists("buy_signal"):
#         signal = r.get("buy_signal")
#         if signal == "true":
#             print("Buy signal detected!")
#     elif r.exists("sell_signal"):
#         signal = r.get("sell_signal")
#         if signal == "true":
#             print("Sell signal detected!")

# def start_process():
#     longPoint = 0
#     sellpoint = 0
#     ShortPoint = 0
#     GainPoint = 0
#     TotalCount = 0
#     increaseShort = 0
#     increaseGain = 0
#     short_flag = 0
#     flag = 0
#     total_number_of_trade_taken = 0
#     short_trade_taken = 0
#     long_trade_taken = 0
#     Stoploss_hit_in_long_trade = 0
#     StopLoss_hit_in_short_trade = 0
#     stoploss_candle_for_short = 0
#     stoploss_candle_for_long = 0
#     short_trade_continue = 0
#     long_trade_continue = 0
#     short_trade_continue_handle = 0
#     long_trade_continue_handle = 0
#     StopLossForShort = 0
#     StopLossForLong = 0
#     Stoploss_trail_for_short = 0
#     Stoploss_trail_for_long = 0
#     order_id_for_short = 0
#     order_id_for_long = 0
#     price = 0
#     CandleSize = 0
#     StopLoss_point = 0
#     Quantity = 0
#     last_traded_value_for_short = 0
   
#     last_traded_value_for_long = 0
    
#     TotalPointGain = 0
#     current_traded_value_for_short = 0
#     current_traded_value_for_long = 0
#     pSymbol = str(0)
    
    
    
    
    
    
#     shorttable = f"banknifty_short_{datetime.now().strftime('%d%m%Y')}"
#     longtable = f"banknifty_long_{datetime.now().strftime('%d%m%Y')}"
    
#     shorttable_exists = sa.inspect(engine).has_table("%s"%(f"{shorttable}"))
#     longtable_exists = sa.inspect(engine).has_table("%s"%(f"{longtable}"))

#     if shorttable_exists:
#         query = f"SELECT * FROM `{shorttable}` LIMIT 1"
#         result = connection.execute(text(query)).fetchone()
#         if result:
#             order_id_for_short = result[0]
#             stoploss_candle_for_short = result[1]
#             StopLossForShort = result[2]
#             sellpoint = result[3]
#             low = result[4]
#             short_flag = result[5]
#             long_trade_continue_handle = result[6]
#             Stoploss_trail_for_short = result[7]
#             short_trade_continue = result[8]
#             price = result[9]
#             CandleSize = result[10]
#             print("Short variables reinitialized from database.")
            
        
#     if longtable_exists:
#         query = f"SELECT * FROM `{longtable}` LIMIT 1"
#         result = connection.execute(text(query)).fetchone()

#         if result:
#             order_id_for_long = result[0] 
#             stoploss_candle_for_long = result[1] 
#             StopLossForLong = result[2]  
#             longPoint = result[3] 
#             high = result[4] 
#             flag = result[5] 
#             short_trade_continue_handle = result[6] 
#             long_trade_continue_handle=result[7]
#             Stoploss_trail_for_long = result[8] 
#             long_trade_continue = result[9]
#             price = result[10] 
#             CandleSize = result[11] 
#             print("Long variables reinitialized from database.")
            
#     trading_symbol = str(0)
#     i = -1
#     j = -2


#     while True:
#         current_time = datetime.now()

#         if market_open_time <= current_time <= market_close_time:
#             exchange = "NSE"
#             symboltoken = "99926009"
            
#             maindata = history(exchange,symboltoken)
            
#             if maindata.empty:
#                 print("No data available. Waiting for the next trading session.")
#                 time.sleep(2)
#                 continue
            
#             signals = bot_alerts(maindata)
#             data = pd.concat([maindata,signals],axis=1)
#             strategy1(data)
#             os.system('cls' if os.name == 'nt' else 'clear')
#             print(data.tail(15))
            
#             if len(data) > abs(j) and len(data) > abs(i):
                

#                 if data["sell_signal"].iloc[j] == -1 and short_flag == 0:  # Enter short position
#                     if StopLoss_point == 1:
#                         Quantity = 15
                        
#                     else:
#                         Quantity = 15
                    
#                     ltp = data["Close"].iloc[j] - 300
#                     trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"PE")
#                     order_response = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
                    
#                     last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
#                     while last_traded_value_for_short is None:
#                         last_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
                        
#                     last_traded_value_for_short = int(float(last_traded_value_for_short))
                    
#                     if sa.inspect(engine).has_table(position_handle):
#                         update_query = text(
#                             f"""UPDATE `{position_handle}` SET 
#                             Quantity = :Quantity_value, 
#                             last_traded_value_for_short = :last_traded_value_for_short_value,
#                             pSymbol = :pSymbol_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'Quantity_value':Quantity,
#                             'last_traded_value_for_short_value': last_traded_value_for_short,
#                             'pSymbol_value':pSymbol,
#                         })
#                         connection.commit()
                    
        
#                     print(order_response)
#                     if order_id_for_short is not None:
#                         stoploss_candle_for_short = data["Open"].iloc[j] - data["Close"].iloc[j]
#                         StopLossForShort = data["High"].iloc[j]
                        
#                         sellpoint = data["Close"].iloc[j]
#                         low = data["Low"].iloc[j]
#                         short_flag = 1
#                         long_trade_continue_handle = 0
#                         total_number_of_trade_taken += 1
#                         short_trade_taken += 1
                        
#                         stockdata = {
#                                 "order_id_for_short":order_id_for_short,
#                                 "stoploss_candle_for_short":stoploss_candle_for_short,
#                                 "StopLossForShort":StopLossForShort,
#                                 "sellpoint":sellpoint,
#                                 "low":low,
#                                 "short_flag":short_flag,
#                                 "long_trade_continue_handle":long_trade_continue_handle,
#                                 "Stoploss_trail_for_short":Stoploss_trail_for_short,
#                                 "short_trade_continue":short_trade_continue,
#                                 "price":price,
#                                 "CandleSize":CandleSize,
#                             }
#                         df = pd.DataFrame([stockdata])
#                         dbload.createtable(shorttable,df)
                        
#                         print(f"taking position at the point {sellpoint} at index {j}")
                    
#                     else:
#                         print("Order id is None for Short.")
#                         break
                    
                    
#                 elif (data["Close"].iloc[i] > StopLossForShort) and short_flag == 1:  # Stoploss condition in short position
#                     square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
#                     current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
#                     while current_traded_value_for_short is None:
#                         current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
#                     current_traded_value_for_short = int(float(current_traded_value_for_short))
#                     TotalPointGain += current_traded_value_for_short - last_traded_value_for_short
#                     StopLoss_point = 1
                    
#                     if sa.inspect(engine).has_table(position_handle):
#                         update_query = text(
#                             f"""
#                             UPDATE `{position_handle}`
#                             SET
#                                 StopLoss_point = :StopLoss_point_value,
#                                 TotalPointGain = :TotalPointGain_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'TotalPointGain_value': TotalPointGain,
#                             'StopLoss_point_value': StopLoss_point,
#                         })
                        
#                         connection.commit()
                    
#                     increaseShort = sellpoint - StopLossForShort
#                     ShortPoint += increaseShort
#                     TotalCount += increaseShort
#                     Stoploss_trail_for_short = 0
#                     short_flag = 0
#                     point2 = data["Close"].iloc[i]
#                     short_trade_continue_handle = 1
#                     short_trade_continue = 1
#                     StopLoss_hit_in_short_trade += 1
#                     connection.execute(text(f"DROP TABLE `{shorttable}`"))
#                     print("Enter in stoploss condition.")
#                     time.sleep(2)
#                     print(f"StopLoss hit at point : {point2} and index is {i} and difference is {increaseShort}")
                    
                    
                    
#                 elif (data["buy_signal"].iloc[j] == 1) and (short_flag == 1):  # Exit short position
#                     square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
#                     current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
#                     while current_traded_value_for_short is None:
#                         current_traded_value_for_short = kotak_function_file.fetch_ltp(pSymbol)
#                     current_traded_value_for_short = int(float(current_traded_value_for_short))
#                     TotalPointGain += current_traded_value_for_short - last_traded_value_for_short
#                     StopLoss_point = 1
                    
#                     if sa.inspect(engine).has_table(position_handle):
                        
#                         update_query = text(
#                             f"""UPDATE `{position_handle}` SET 
#                             TotalPointGain = :TotalPointGain_value, 
#                             StopLoss_point = :StopLoss_point_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'TotalPointGain_value': TotalPointGain,
#                             'StopLoss_point_value': StopLoss_point,
#                         })
#                         connection.commit()
                    
#                     increaseShort = sellpoint - data["Close"].iloc[j]
#                     ShortPoint += increaseShort
#                     TotalCount += increaseShort
#                     short_flag = 0
#                     point = data["Close"].iloc[j]
#                     connection.execute(text(f"DROP TABLE `{shorttable}`"))
#                     print("Enter in exit short position condition")
#                     time.sleep(2)
                    
#                     print(f"Coming outside from the trade at the point {point} at index {j} and difference is {increaseShort}")
                    

    
                    
#                 if data["buy_signal"].iloc[j] == 1 and flag == 0: # Enter Long position
#                     if StopLoss_point == 1:
#                         Quantity = 15
#                     else:
#                         Quantity = 15
       
#                     ltp = data["Close"].iloc[j] + 300
#                     trading_symbol,pSymbol = kotak_function_file.select_trading_symbol("BANKNIFTY",ltp,"CE")
#                     order_response1 = kotak_function_file.place_order_for_buy_index(str(trading_symbol),Quantity)
#                     last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     while last_traded_value_for_long is None:
#                         last_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     last_traded_value_for_long = int(float(last_traded_value_for_long))
                    
#                     if sa.inspect(engine).has_table(position_handle):
                        
#                         update_query = text(
#                             f"""UPDATE `{position_handle}` SET 
#                             Quantity = :Quantity_value, 
#                             last_traded_value_for_long = :last_traded_value_for_long_value,
#                             pSymbol = :pSymbol_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'Quantity_value':Quantity,
#                             'last_traded_value_for_long_value': last_traded_value_for_long,
#                             'pSymbol_value':pSymbol,
#                         })
#                         connection.commit()
                    
                    
#                     print(order_response1)
#                     time.sleep(3)
#                     order_id_for_long = order_response1.get('nOrdNo')
#                     if order_id_for_long is not None:
#                         stoploss_candle_for_long = data["Close"].iloc[j] - data["Open"].iloc[j]
                        
#                         StopLossForLong = data["Low"].iloc[j]
#                         longPoint = data["Close"].iloc[j]
#                         high = data["High"].iloc[j]
#                         flag = 1
#                         total_number_of_trade_taken += 1
#                         long_trade_taken += 1
#                         short_trade_continue_handle = 0
                        
                        
#                         stockdata = {
#                             "order_id_for_long":order_id_for_long,
#                             "stoploss_candle_for_long":stoploss_candle_for_long,
#                             "StopLossForLong":StopLossForLong,
#                             "longPoint":longPoint,
#                             "high":high,
#                             "flag":flag,
#                             "short_trade_continue_handle":short_trade_continue_handle,
#                             "long_trade_continue_handle":long_trade_continue_handle,
#                             "Stoploss_trail_for_long":Stoploss_trail_for_long,
#                             "long_trade_continue":long_trade_continue,
#                             "price":price,
#                             "CandleSize":CandleSize,
#                         }
#                         df = pd.DataFrame([stockdata])
#                         dbload.createtable(longtable,df)
                        
#                         print(f"Taking position at point {longPoint} at index {j}")
                        
#                     else:
#                         print("Order id is None in Long position.")
#                         break
                    
                    
                    
#                 elif data["Close"].iloc[i] < StopLossForLong and flag == 1: # stoploss condition for long position
#                     square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
#                     current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     while current_traded_value_for_long is None:
#                         current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     current_traded_value_for_long = int(float(current_traded_value_for_long))
#                     TotalPointGain += current_traded_value_for_long - last_traded_value_for_long
#                     StopLoss_point = 1
#                     if sa.inspect(engine).has_table(position_handle):
                        
#                         update_query = text(
#                             f"""UPDATE `{position_handle}` SET 
#                             TotalPointGain = :TotalPointGain_value, 
#                             StopLoss_point = :StopLoss_point_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'TotalPointGain_value': TotalPointGain,
#                             'StopLoss_point_value': StopLoss_point,
#                         })
                        
#                         connection.commit()
#                     increaseGain = StopLossForLong - longPoint
#                     GainPoint += increaseGain
#                     TotalCount += increaseGain
#                     Stoploss_trail_for_long = 0
#                     long_trade_continue = 1
#                     long_trade_continue_handle = 1
#                     flag = 0
#                     point4 = data["Close"].iloc[i]
#                     connection.execute(text(f"DROP TABLE `{longtable}`"))
#                     print("drop the longtable because enter in stoploss condition")
#                     time.sleep(3)
#                     Stoploss_hit_in_long_trade += 1
#                     print(f"Coming outside from the trade at the point {point4} at the index {i} and difference is {increaseGain}")  
                    
                    
                    
                    
#                 elif data["sell_signal"].iloc[j] == -1 and flag == 1: # Exit long position
#                     square_off_position = kotak_function_file.square_off_position(str(trading_symbol),"S",Quantity)
#                     current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     while current_traded_value_for_long is None:
#                         current_traded_value_for_long = kotak_function_file.fetch_ltp(pSymbol)
#                     current_traded_value_for_long = int(float(current_traded_value_for_long))
#                     TotalPointGain += current_traded_value_for_long - last_traded_value_for_long
#                     StopLoss_point = 1
#                     if sa.inspect(engine).has_table(position_handle):
                        
#                         update_query = text(
#                             f"""UPDATE `{position_handle}` SET 
#                             TotalPointGain = :TotalPointGain_value, 
#                             StopLoss_point = :StopLoss_point_value
#                             """
#                         )
#                         connection.execute(update_query, {
#                             'TotalPointGain_value': TotalPointGain,
#                             'StopLoss_point_value': StopLoss_point,
#                         })
#                         connection.commit()
#                     increaseGain = data["Close"].iloc[j] - longPoint
#                     GainPoint += increaseGain
#                     TotalCount += increaseGain
#                     flag = 0
#                     point3 = data["Close"].iloc[j]
#                     connection.execute(text(f"DROP TABLE `{longtable}`"))
#                     print("Drop the longtable becasue i am in exit long positon.")
#                     time.sleep(2)
                        
#                     print(f"Coming outside from the trade at the point {point3} at the index {j} and difference is {increaseGain}")