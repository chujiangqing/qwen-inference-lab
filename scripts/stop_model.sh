#!/usr/bin/env bash
# 停止 Qwen3.8-27B-FP8 推理服务：优先用 PID 文件，超时后逐级升级信号。
#
# 用法：bash scripts/stop_model.sh [--force]
# 退出码：0 已停止；1 未找到进程或停止失败。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

PID_FILE="$(eval "$("$PY" "$SCRIPT_DIR/render_runtime.py" --profile baseline)" && printf '%s' "$PID_FILE")"
PORT="$(eval "$("$PY" "$SCRIPT_DIR/render_runtime.py" --profile baseline)" && printf '%s' "$PORT")"

pids=()
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null && pids+=("$pid")
fi
if [[ ${#pids[@]} -eq 0 ]]; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(pgrep -f "vllm serve" 2>/dev/null || true)
fi

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "未发现正在运行的 vLLM 服务"
  rm -f -- "$PID_FILE"
  exit 1
fi

echo "准备停止 pid: ${pids[*]}"
for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

for _ in $(seq 1 60); do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [[ $alive -eq 0 ]] && break
  sleep 2
done

if [[ $alive -ne 0 ]]; then
  if [[ $FORCE -eq 1 ]]; then
    echo "SIGTERM 超时，强制 SIGKILL"
    for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    sleep 2
  else
    echo "进程未在 120s 内退出；确认后可加 --force 强制结束" >&2
    exit 1
  fi
fi

rm -f -- "$PID_FILE"
sleep 1
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
fi
echo "服务已停止（端口 ${PORT}）"
