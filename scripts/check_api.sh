#!/usr/bin/env bash
# 接口健康检查：/v1/models 是否存在目标模型 + completions/chat 各一次真实请求。
#
# 用法：bash scripts/check_api.sh [--port 8088] [--model SERVICE_NAME] [--timeout 120]
# 退出码：0 全部通过；1 有失败项。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
PORT=""
MODEL=""
TIMEOUT=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PORT" || -z "$MODEL" ]]; then
  eval "$("$PY" "$SCRIPT_DIR/render_runtime.py" --profile baseline)"
  PORT="${PORT:-8088}"
  MODEL="${MODEL:-$SERVED_NAME}"
fi

BASE="http://127.0.0.1:${PORT}/v1"
echo "检查目标：$BASE  模型：$MODEL"

"$PY" - "$BASE" "$MODEL" "$TIMEOUT" <<'PY'
import json, os, sys, time, urllib.error, urllib.request

base, model, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
key = os.environ.get("OPENAI_API_KEY", "EMPTY")
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
failures = []


def call(path, payload=None, method="POST"):
    url = base + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8", "replace"))
    return body, time.time() - started


print("1) GET /v1/models")
try:
    body, elapsed = call("/models", method="GET")
    ids = [item.get("id") for item in body.get("data", [])]
    print(f"   耗时 {elapsed:.2f}s  返回模型：{ids}")
    if model not in ids:
        failures.append(f"/v1/models 未返回 {model!r}")
    else:
        card = next(item for item in body["data"] if item.get("id") == model)
        mml = card.get("max_model_len")
        print(f"   max_model_len={mml}  context={card.get('context_length')}")
        if mml is not None and int(mml) <= 32768:
            print("   注意：max_model_len<=32768，eval.zip 会跳过精度阶段（见 docs/eval_工具评测审计）")
except Exception as error:
    failures.append(f"/v1/models 失败：{type(error).__name__}: {error}")

print("2) POST /v1/completions")
try:
    body, elapsed = call("/completions", {
        "model": model, "prompt": "The capital of France is", "max_tokens": 16,
        "temperature": 0, "ignore_eos": False, "stream": False})
    text = (body.get("choices") or [{}])[0].get("text", "")
    usage = body.get("usage", {})
    print(f"   耗时 {elapsed:.2f}s  输出={text!r}")
    print(f"   usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    if not text.strip():
        failures.append("completions 返回空文本")
    if usage.get("completion_tokens") != 16:
        failures.append(f"completion_tokens={usage.get('completion_tokens')}，期望 16")
except Exception as error:
    failures.append(f"/v1/completions 失败：{type(error).__name__}: {error}")

print("3) POST /v1/chat/completions")
try:
    body, elapsed = call("/chat/completions", {
        "model": model, "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": "用一个词回答：水的化学式是？"}]})
    message = (body.get("choices") or [{}])[0].get("message", {})
    content = message.get("content")
    print(f"   耗时 {elapsed:.2f}s  content={content!r}")
    if not content:
        print(f"   reasoning_content 非空：{bool(message.get('reasoning_content'))}")
except Exception as error:
    failures.append(f"/v1/chat/completions 失败：{type(error).__name__}: {error}")

print()
if failures:
    for item in failures:
        print("[失败]", item)
    sys.exit(1)
print("全部检查通过")
PY
