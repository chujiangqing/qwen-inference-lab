#!/usr/bin/env python3
"""把 configs/runtime.json 的某个 profile 渲染成 shell 变量，供 start_model.sh eval。

用法：
    python3 scripts/render_runtime.py --profile baseline            # 输出 shell 代码
    python3 scripts/render_runtime.py --profile baseline --command  # 只输出 vllm 参数（一行）

只依赖标准库；不读取环境变量以外的任何外部状态。
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "runtime.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def vllm_arguments(profile: dict, config: dict, api_key: str | None = None) -> list[str]:
    """构造 vllm serve 的参数列表（不含模型路径本身）。"""
    server = config["server"]
    args = [
        "--host", str(server["host"]),
        "--port", str(server["port"]),
        "--served-model-name", str(config["model"]["served_name"]),
        "--tensor-parallel-size", str(profile["tensor_parallel_size"]),
        "--dtype", str(profile["dtype"]),
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--max-model-len", str(profile["max_model_len"]),
        "--max-num-seqs", str(profile["max_num_seqs"]),
        "--kv-cache-dtype", str(profile["kv_cache_dtype"]),
    ]
    # 批处理维度：只有显式给了才下发（不写就用 vLLM 默认值，避免改变基线口径）
    if profile.get("max_num_batched_tokens"):
        args += ["--max-num-batched-tokens", str(profile["max_num_batched_tokens"])]
    if profile.get("reasoning_parser"):
        args += ["--reasoning-parser", str(profile["reasoning_parser"])]
    # 只在显式要求时开启前缀缓存；关闭时不发 --no-* 形式，避免不同 vLLM 版本参数名不一致
    if profile.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if profile.get("speculative_config"):
        args += ["--speculative-config", json.dumps(profile["speculative_config"], separators=(",", ":"))]
    if api_key:
        args += ["--api-key", api_key]
    args += [str(item) for item in profile.get("extra_args", [])]
    return args


def coerce(value: str):
    """把 --set 的字符串值转成 JSON 标量/字面量，便于覆盖数值、布尔和对象。"""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(profile: dict, overrides: list[str]) -> dict:
    merged = dict(profile)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式：{item}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"--set 缺少键名：{item}")
        merged[key] = coerce(raw.strip())
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="渲染 vLLM 运行时配置")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--command", action="store_true", help="只打印完整命令行")
    parser.add_argument("--json", action="store_true", help="只打印解析后的配置")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="覆盖 profile 中的字段，可重复")
    args = parser.parse_args(argv)

    config = load(args.config)
    profiles = config.get("profiles", {})
    if args.profile not in profiles:
        print(f"未知 profile：{args.profile}；可选：{', '.join(profiles)}", file=sys.stderr)
        return 2
    profile = apply_overrides(profiles[args.profile], args.overrides)
    model_path = config["model"]["path"]
    tokenizer = config["model"].get("tokenizer") or model_path
    cli = vllm_arguments(profile, config)
    if config["model"].get("tokenizer"):
        cli += ["--tokenizer", str(tokenizer)]

    if args.json:
        print(json.dumps({"profile": args.profile, "model_path": model_path,
                          "tokenizer": tokenizer, "arguments": cli}, ensure_ascii=False, indent=2))
        return 0
    if args.command:
        print(shlex.join([str(Path(config["server"]["venv"]) / "bin" / "vllm"), "serve", model_path, *cli]))
        return 0

    server = config["server"]
    out = [
        f"PROFILE={shlex.quote(args.profile)}",
        f"MODEL_PATH={shlex.quote(model_path)}",
        f"TOKENIZER={shlex.quote(tokenizer)}",
        f"SERVED_NAME={shlex.quote(config['model']['served_name'])}",
        f"VENV={shlex.quote(server['venv'])}",
        f"HOST={shlex.quote(str(server['host']))}",
        f"PORT={int(server['port'])}",
        f"LOG_DIR={shlex.quote(server['log_dir'])}",
        f"PID_FILE={shlex.quote(server['pid_file'])}",
        f"API_KEY_ENV={shlex.quote(server['api_key_env'])}",
        f"STARTUP_TIMEOUT={int(server['startup_timeout_seconds'])}",
        "VLLM_ARGS=(" + " ".join(shlex.quote(item) for item in cli) + ")",
        f"PROFILE_DESC={shlex.quote(profile.get('description', ''))}",
        f"MAX_MODEL_LEN={int(profile['max_model_len'])}",
        f"MAX_NUM_SEQS={int(profile['max_num_seqs'])}",
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
