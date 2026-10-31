#!/usr/bin/env bash
# 精度看门狗：预测文件超过 N 分钟不增长就杀掉卡住的评测，并记录原因。
#
# 为什么用"产物是否增长"而不是"GPU 是否忙"：
# 2026-10-31 那次卡死时 GPU 一直显示 100%，最容易骗过人的就是这个指标。
#
# 用法（与评测命令并行启动）：
#   bash scripts/watchdog_accuracy.sh <predictions.jsonl> [STALL_SECONDS] [PID...]
set -u

TARGET="${1:?需要传入 predictions.jsonl 路径}"
STALL="${2:-900}"                 # 默认 15 分钟无增长判定卡死
shift 2 2>/dev/null || true
WATCH_PIDS=("$@")

LOG=/root/logs/watchdog_accuracy.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "看门狗启动：目标=$TARGET 阈值=${STALL}s PID=${WATCH_PIDS[*]:-未指定}"

last_count=-1
last_change=$(date +%s)

while true; do
  if [ ! -f "$TARGET" ]; then
    sleep 30
    continue
  fi
  count=$(wc -l < "$TARGET" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$count" != "$last_count" ]; then
    log "进度 $count（较上次 +$((count - last_count))）"
    last_count=$count
    last_change=$now
  else
    idle=$((now - last_change))
    if [ "$idle" -ge "$STALL" ]; then
      log "超过 ${idle}s 无增长（停在 $count 条），判定卡死，开始清理"
      for pid in "${WATCH_PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        kill "$pid" 2>/dev/null && log "  killed $pid"
      done
      pkill -f 'runner.py _accuracy' 2>/dev/null && log "  killed runner.py _accuracy"
      pkill -9 -f 'evalscope' 2>/dev/null && log "  killed evalscope"
      log "清理完成，停在 $count 条；该结果按规则不作为成绩"
      exit 0
    fi
  fi

  # 评测进程都不在了就退出
  if [ "${#WATCH_PIDS[@]}" -gt 0 ]; then
    alive=0
    for pid in "${WATCH_PIDS[@]}"; do
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    if [ "$alive" -eq 0 ]; then
      log "被监控进程已退出（最终 $count 条），看门狗结束"
      exit 0
    fi
  fi
  sleep 60
done
