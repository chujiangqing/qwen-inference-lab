"""全局配置：路径与常量集中在这里，其它模块不硬编码路径。"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "inference_lab.db"
RUNTIME_CONFIG = CONFIG_DIR / "runtime.json"
EVAL_ENTRY = ROOT / "eval" / "run.sh"

API_HOST = os.environ.get("LAB_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("LAB_API_PORT", "8090"))
BASE_URL = os.environ.get("LAB_BASE_URL", "http://127.0.0.1:8088/v1")
SERVED_MODEL = os.environ.get("LAB_MODEL", "qwen3.8-27b-fp8")


def runtime_config() -> dict:
    return json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8-sig"))


def venv_python() -> Path:
    return Path(runtime_config()["server"]["venv"]) / "bin" / "python"

def model_path() -> str | None:
    """模型目录（用于把 tokenizer 固定成模型自身的，保证 ISL 口径一致）。"""
    return (runtime_config().get("model") or {}).get("path")


def default_tokenizer() -> str:
    """默认用模型自身 tokenizer：服务端 ISL 才能精确等于 8192。

    官方 runner.py 的默认是 builtin(tiktoken)，那样服务端实际长度会漂移；
    我们的所有基线数据都是用模型 tokenizer 采的，所以内部默认必须与之一致。
    """
    path = model_path()
    return path if path and Path(path).is_dir() else "builtin"