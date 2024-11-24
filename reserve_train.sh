#!/bin/bash

target_pid=288519  # 대기할 프로세스의 PID
interval=300        # 확인 주기 (초)

echo "Start waiting."
while ps -p $target_pid > /dev/null 2>&1; do
  sleep $interval
done

# 프로세스 종료 후 실행할 명령어
nohup bash train.sh > helicopter.log 2>&1  &
echo "Process completed. 'train.sh' started in background."