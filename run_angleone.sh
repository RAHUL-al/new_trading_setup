#!/bin/bash
cd /home/ubuntu
source /home/ubuntu/stocks_env/bin/activate
cd /home/ubuntu/backup_three_twenty_strategy

nohup python3 angleone_websocket1.py >> angleone1.log 2>&1 &
nohup python3 angleone_websocket2.py >> angleone2.log 2>&1 &
nohup python3 angleone_websocket3.py >> angleone3.log 2>&1 &
