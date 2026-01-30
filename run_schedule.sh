#!/bin/bash

# Hàm kiểm tra giờ
check_time() {
  HOUR=$(date +%H)
  MIN=$(date +%M)
  if [ $HOUR -eq 0 ] && [ $MIN -lt 5 ]; then
    return 1
  else
    return 0
  fi
}

APP_PID=0

while true; do
  if check_time; then
    if [ $APP_PID -eq 0 ] || ! kill -0 $APP_PID 2>/dev/null; then
      echo "[$(date '+%H:%M:%S')] Daytime -> starting app..."
      python3 /app/main_metric3.py --send_api --store_id vn300  &
      python3 /app/main_metric3.py --send_api --store_id vn316  &
      APP_PID=$!
    fi
  else
    if [ $APP_PID -ne 0 ] && kill -0 $APP_PID 2>/dev/null; then
      echo "[$(date '+%H:%M:%S')] Nighttime -> stopping app..."
      kill $APP_PID
      wait $APP_PID 2>/dev/null
      APP_PID=0
    fi
    echo "[$(date '+%H:%M:%S')] Sleeping 5 min before recheck..."
    sleep 300
  fi
  sleep 60
done
