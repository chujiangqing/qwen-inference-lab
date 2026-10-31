#!/usr/bin/env bash
# 阶段 0：环境与评测工具审计。只读，不安装、不启动模型、不修改服务器环境。
#
# 用法：
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3.8-27B-FP8 bash scripts/task0_audit.sh
#   bash scripts/task0_audit.sh [输出目录]
#
# 可选环境变量：PYTHON（解释器）、MODEL_PATH（模型目录）、VRAM_GIB（实测显存）
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$SCRIPT_DIR")"
NOW="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$ROOT/reports/task0_$NOW}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "未找到 Python 3；阶段 0 的检测脚本需要 Python >= 3.10。" >&2
  exit 2
fi

mkdir -p -- "$OUT" || exit 2

PY_VERSION="$("$PYTHON" --version 2>&1)"
echo "输出目录：$OUT"
echo "解释器：$PYTHON ($PY_VERSION)"
echo "开始时间：$(date +%Y-%m-%dT%H:%M:%S%z)"
echo

run_step() {
  local name="$1"
  shift
  echo "---- $name ----"
  "$@" 2>&1 | tee -- "$OUT/$name.log"
  local code="${PIPESTATUS[0]}"
  echo "$name 退出码：$code"
  echo
  return 0
}

run_step gpu "$PYTHON" "$SCRIPT_DIR/check_gpu.py" --out-dir "$OUT"

if [[ -n "${MODEL_PATH:-}" ]]; then
  run_step environment "$PYTHON" "$SCRIPT_DIR/check_environment.py" \
    --out-dir "$OUT" --model-path "$MODEL_PATH"
else
  run_step environment "$PYTHON" "$SCRIPT_DIR/check_environment.py" --out-dir "$OUT"
fi

run_step eval_tool "$PYTHON" "$SCRIPT_DIR/check_eval_tool.py" --out-dir "$OUT"

if [[ -n "${MODEL_PATH:-}" ]]; then
  if [[ -n "${VRAM_GIB:-}" ]]; then
    run_step model "$PYTHON" "$SCRIPT_DIR/check_model.py" \
      --model-path "$MODEL_PATH" --out-dir "$OUT" --vram-gib "$VRAM_GIB"
  else
    run_step model "$PYTHON" "$SCRIPT_DIR/check_model.py" \
      --model-path "$MODEL_PATH" --out-dir "$OUT"
  fi
else
  echo "跳过模型目录检查：未设置 MODEL_PATH"
  echo "服务器上应设置为 Qwen3.8-27B-FP8 的实际路径后重跑。"
  echo
fi

{
  echo "# 阶段 0 审计结果"
  echo
  echo "- 时间：$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "- 主机：$(hostname 2>/dev/null || echo unknown)"
  echo "- 解释器：$PYTHON"
  echo "- 输出目录：\`$OUT\`"
  echo
  echo "| 检测项 | 退出码 | 主要告警 |"
  echo "|---|---:|---|"
  for step in gpu environment eval_tool model; do
    log="$OUT/$step.log"
    if [[ -f "$log" ]]; then
      code="$(grep -o '退出码：[0-9]*' "$log" | tail -1 | grep -o '[0-9]*')"
      note="$(grep '^\[告警\]' "$log" | head -3 | tr '\n' ' ' | tr -d '|')"
      echo "| $step | ${code:-?} | ${note:-无} |"
    fi
  done
  echo
  echo "退出码约定：0 通过，1 阻断（缺文件/无 GPU），2 告警（可用但需人工确认）。"
  echo
  echo "下一步：在交互终端执行 \`bash eval/run.sh --mode check\`，确认评测依赖。"
} > "$OUT/summary.md"

echo "汇总：$OUT/summary.md"
echo "请将 $OUT 目录整体保留（含四份 JSON 与四份日志），作为阶段 0 的原始证据。"
