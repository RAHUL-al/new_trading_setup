#! /bin/bash
cd /home/ubuntu/
source /home/ubuntu/stocks_env/bin/activate
cd /home/ubuntu/backup_three_twenty_strategy

nohup python create_angleone_csv.py >> create_angleone_csv.log 2>&1 &
nohup python stocks_filter.py >> stocks_filter.log 2>&1 &
