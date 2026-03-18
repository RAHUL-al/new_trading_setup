#!/bin/bash
cd /root/Trading_setup_code
source /root/Trading_setup_code/venv/bin/activate

nohup python3 angleone_websocket1.py >> angleone1.log 2>&1 &

