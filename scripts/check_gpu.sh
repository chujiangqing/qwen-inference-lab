#!/usr/bin/env bash
# 只读 GPU 检测的 POSIX 包装；实际逻辑在 check_gpu.py。
# 用法：bash scripts/check_gpu.sh [--out-dir DIR]
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/check_gpu.py" "$@"
