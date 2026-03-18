#! /bin/bash
cd /root/Trading_setup_code
source /root/Trading_setup_code/venv/bin/activate

nohup python3 create_angleone_csv.py >> create_angleone_csv.log 2>&1 &
nohup python3 stocks_filter.py >> stocks_filter.log 2>&1 &

