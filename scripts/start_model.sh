#!/usr/bin/env bash
# 启动 Qwen3.8-27B-FP8 推理服务（vLLM），带启动前检查、日志、PID 与启动记录。
#
# 用法：
#   bash scripts/start_model.sh                        # 默认 profile=baseline
#   bash scripts/start_model.sh --profile accuracy     # 40960 上下文，供精度评测
#   bash scripts/start_model.sh --profile mtp          # MTP 对照
#   bash scripts/start_model.sh --dry-run              # 只打印将要执行的命令
#   bash scripts/start_model.sh --skip-preflight       # 跳过启动前检查（不推荐）
#
# 退出码：0 启动成功且 /v1/models 可用；1 启动失败；2 参数/配置错误。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$SCRIPT_DIR")"
PY="${PYTHON:-python3}"

PROFILE="baseline"
DRY_RUN=0
SKIP_PREFLIGHT=0
OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --set) OVERRIDES+=(--set "$2"); shift 2 ;;
    --set=*) OVERRIDES+=(--set "${1#*=}"); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

eval "$("$PY" "$SCRIPT_DIR/render_runtime.py" --profile "$PROFILE" "${OVERRIDES[@]}")"

# vLLM 的运行时 JIT（flashinfer / torch.compile）会用 subprocess 调 ninja、nvcc，
# 必须把 venv/bin 与 CUDA 工具链放进 PATH，否则会以 FileNotFoundError: 'ninja' 失败。
export PATH="$VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "profile      : $PROFILE（$PROFILE_DESC）"
echo "模型         : $MODEL_PATH"
echo "服务名/端口  : $SERVED_NAME / $HOST:$PORT"
echo "上下文/批上限: $MAX_MODEL_LEN / max_num_seqs=$MAX_NUM_SEQS"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '命令         : %s serve %q' "$VENV/bin/vllm" "$MODEL_PATH"
  printf ' %q' "${VLLM_ARGS[@]}"
  printf '\n'
  exit 0
fi

# ---------- 启动前检查 ----------
if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  command -v nvidia-smi >/dev/null || { echo "缺少 nvidia-smi" >&2; exit 1; }
  [[ -d "$MODEL_PATH" ]] || { echo "模型目录不存在：$MODEL_PATH" >&2; exit 1; }
  [[ -f "$MODEL_PATH/config.json" ]] || { echo "缺少 config.json" >&2; exit 1; }
  [[ -x "$VENV/bin/vllm" ]] || { echo "vLLM 未安装：$VENV/bin/vllm" >&2; exit 1; }

  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [[ "${busy:-0}" -gt 0 ]]; then
    echo "GPU 上已有 $busy 个计算进程，拒绝启动（评测必须独占）：" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 1
  fi

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ -n "$free_mib" && "$free_mib" -lt 30000 ]]; then
    echo "可用显存仅 ${free_mib} MiB，低于 30000 MiB 安全线；请先释放显存" >&2
    exit 1
  fi
  echo "显存可用 : ${free_mib} MiB"

  if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "端口 $PORT 已被占用，拒绝重复启动：" >&2
    ss -ltnp 2>/dev/null | grep ":$PORT " >&2 || true
    exit 1
  fi
  # PID 文件可能是上次强杀后残留的：只有进程确实还是 vLLM 才拒绝启动
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null \
       && tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null | grep -q 'vllm'; then
      echo "已有 vLLM 服务在运行（pid=$old_pid），先执行 scripts/stop_model.sh" >&2
      exit 1
    fi
    echo "清理过期 PID 文件（pid=${old_pid:-空} 不是 vLLM 进程）"
    rm -f -- "$PID_FILE"
  fi
fi

mkdir -p -- "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/serve_${PROFILE}_${STAMP}.log"
META="$LOG_DIR/launch_${PROFILE}_${STAMP}.json"

EXTRA=()
if [[ -n "${!API_KEY_ENV:-}" ]]; then
  EXTRA+=(--api-key "${!API_KEY_ENV}")
  echo "鉴权         : 启用（来自环境变量 $API_KEY_ENV）"
else
  echo "鉴权         : 未设置 $API_KEY_ENV，接口为无鉴权模式"
fi

echo "日志         : $LOG"
nohup "$VENV/bin/vllm" serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "${EXTRA[@]}" >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "已拉起进程   : pid=$PID，等待 /v1/models 就绪（最多 ${STARTUP_TIMEOUT}s）"

start_epoch=$(date +%s)
ready=0
while [[ $(( $(date +%s) - start_epoch )) -lt $STARTUP_TIMEOUT ]]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "服务进程已退出，启动失败。日志尾部：" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  if curl -fsS -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 3
done

elapsed=$(( $(date +%s) - start_epoch ))
if [[ $ready -ne 1 ]]; then
  echo "启动超时（${elapsed}s）：/v1/models 未就绪。日志尾部：" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

echo "服务已就绪   : ${elapsed}s"
"$PY" - "$META" "$PROFILE" "$MODEL_PATH" "$LOG" "$PID" "$elapsed" "${VLLM_ARGS[@]}" <<'PY'
import json, subprocess, sys, datetime
meta, profile, model, log, pid, elapsed, *args = sys.argv[1:]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
data = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "profile": profile, "model_path": model, "log": log, "pid": int(pid),
        "startup_seconds": int(elapsed), "vllm_arguments": args,
        "gpu_after_start": gpu}
with open(meta, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("启动记录     :", meta)
PY

echo "接口地址     : http://127.0.0.1:${PORT}/v1"
echo "停止服务     : bash scripts/stop_model.sh"
