#!/usr/bin/env bash
# 只读系统环境检测的 POSIX 包装；实际逻辑在 check_environment.py。
# 用法：bash scripts/check_environment.sh [--model-path PATH] [--out-dir DIR]
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/check_environment.py" "$@"
