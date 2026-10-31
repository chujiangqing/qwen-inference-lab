#!/usr/bin/env bash
# 一次实验的评测入口：检查环境与服务 → 串行执行性能/精度 → 输出结构化结果。
#
# 用法：
#   bash scripts/run_benchmark.sh --label E00_BASELINE --mode all
#   bash scripts/run_benchmark.sh --label E00_PRETEND --pretend      # 只做前置检查
#
# 所有评测口径由 eval/runner.py 决定，本脚本不复制任何 ISL/OSL/并发常量。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$SCRIPT_DIR")"
PY="${PYTHON:-python3}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8088/v1}"
MODEL="${MODEL:-qwen3.8-27b-fp8}"
LABEL="E00_BASELINE"
MODE="all"
TOKENIZER="${TOKENIZER:-builtin}"
PERF_REQUESTS="${PERF_REQUESTS:-100}"
TIMEOUT="${REQUEST_TIMEOUT:-600}"
BATCH="${EVAL_BATCH_SIZE:-2}"
PRETEND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --perf-requests) PERF_REQUESTS="$2"; shift 2 ;;
    --request-timeout) TIMEOUT="$2"; shift 2 ;;
    --eval-batch-size) BATCH="$2"; shift 2 ;;
    --pretend) PRETEND="--pretend"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

exec "$PY" "$ROOT/backend/benchmark/benchmark_manager.py" \
  --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" --mode "$MODE" \
  --tokenizer "$TOKENIZER" --perf-requests "$PERF_REQUESTS" \
  --request-timeout "$TIMEOUT" --eval-batch-size "$BATCH" $PRETEND
