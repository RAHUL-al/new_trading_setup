#!/bin/bash
cd /home/ubuntu
source /home/ubuntu/stocks_env/bin/activate

cd /home/ubuntu/backup_three_twenty_strategy

nohup python test.py >> test.log 2>&1 &
