#!/usr/bin/env python3
"""系统与软件环境检测（只读）：OS/CPU/内存/磁盘/Python/CUDA/驱动/依赖版本。

用法：
    python3 scripts/check_environment.py [--out-dir reports/task0] [--json-only]
                                         [--model-path PATH] [--skip-torch]

不做任何安装、不修改系统、不启动模型。torch 探测会短暂初始化 CUDA 上下文（不分配显存）；
完全不希望触碰 GPU 时加 --skip-torch。退出码：0 正常；2 有告警（如缺 vLLM）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from _report import (ROOT, configure_stdio, disk_report, executable_version, find_executable,
                     gib, header, memory_bytes, platform_summary, python_package_versions,
                     read_os_release, row, run, save, working_python)

PACKAGES = ("vllm", "torch", "torchvision", "torchaudio", "transformers", "tokenizers",
            "triton", "accelerate", "fastapi", "uvicorn", "pydantic", "numpy", "pandas",
            "streamlit", "plotly", "openai", "uv")

PROBE_TORCH = """
import json, sys
info = {}
try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["torch_cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gib"] = round(props.total_memory / 1024 ** 3, 2)
except Exception as error:
    info["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(info, ensure_ascii=False))
"""


def cpu_info() -> dict:
    info = {"logical_cores": os.cpu_count()}
    if sys.platform.startswith("linux"):
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
        try:
            text = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            info["cgroup_cpu_max"] = text
        except OSError:
            pass
    elif sys.platform == "win32":
        info["model"] = platform.processor()
    return info


def check_paths(args: argparse.Namespace) -> dict:
    candidates = {
        "项目根目录": ROOT,
        "评测工具目录": ROOT / "eval",
        "报告目录": ROOT / "reports",
        "模型目录": Path(args.model_path).expanduser() if args.model_path else None,
        "HF 缓存": Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser(),
        "ModelScope 缓存": Path(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope")).expanduser(),
        "评测工具环境": Path(os.environ.get("AI_INFRA_ENV_DIR",
                                            "~/.cache/ai-infra-eval/envs")).expanduser(),
    }
    return {name: disk_report(path) for name, path in candidates.items() if path}


def tool_versions() -> dict:
    tools = {}
    for name, candidates, flag in (
        ("nvcc", ("nvcc",), "--version"),
        ("docker", ("docker",), "--version"),
        ("git", ("git",), "--version"),
        ("uv", ("uv",), "--version"),
        ("bash", ("bash",), "--version"),
        ("curl", ("curl",), "--version"),
        ("wget", ("wget",), "--version"),
        ("nvidia-smi", ("nvidia-smi",), "--version"),
    ):
        path = find_executable(candidates)
        tools[name] = {"path": path, "version": executable_version(path, flag)}
    return tools


def torch_probe() -> dict:
    result = run([sys.executable, "-c", PROBE_TORCH], timeout=180)
    if not result["ok"]:
        return {"error": result["stderr"] or "torch 探测失败"}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"error": result["stdout"][:400]}


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="只读系统环境检测")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "manual")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--model-path", default=os.environ.get("QWEN_MODEL_PATH", ""),
                        help="Qwen3.8-27B-FP8 目录，用于检查磁盘与路径是否存在")
    parser.add_argument("--skip-torch", action="store_true", help="跳过 torch/CUDA 运行时探测")
    args = parser.parse_args()

    warnings: list[str] = []
    packages = python_package_versions(PACKAGES)
    payload = {
        "platform": platform_summary(),
        "os_release": read_os_release(),
        "kernel": {"release": platform.release(), "version": platform.version(), "machine": platform.machine()},
        "cpu": cpu_info(),
        "memory": {key: gib(value) for key, value in memory_bytes().items()},
        "disks": check_paths(args),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "candidates": {name: working_python(find_executable((name,)))
                                for name in ("python3.10", "python3.11", "python3.12", "python3")},
        },
        "tools": tool_versions(),
        "packages": packages,
        "env": {name: os.environ.get(name) for name in
                ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_ENDPOINT", "MODELSCOPE_CACHE",
                 "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "http_proxy", "https_proxy", "no_proxy")},
    }
    if not args.skip_torch:
        payload["torch"] = torch_probe()

    if not packages.get("vllm"):
        warnings.append("当前解释器未安装 vLLM；推理服务需要单独的 venv（见阶段 1）")
    if not packages.get("torch"):
        warnings.append("当前解释器未安装 PyTorch；torch/CUDA 探测会失败")
    for name, disk in payload["disks"].items():
        if disk.get("free_gib") is not None and name == "模型目录" and disk["free_gib"] < 5:
            warnings.append(f"{name} 剩余空间过小：{disk['free_gib']} GiB")
    if payload.get("torch", {}).get("cuda_available") is False:
        warnings.append("torch 报告 CUDA 不可用：需要检查驱动/CUDA 与 torch 版本匹配")

    if not args.json_only:
        header("系统")
        row("OS", payload["os_release"].get("PRETTY_NAME") or platform.platform())
        row("内核/架构", f"{payload['kernel']['release']} / {payload['kernel']['machine']}")
        row("CPU 逻辑核", payload["cpu"]["logical_cores"])
        row("CPU 型号", payload["cpu"].get("model"))
        row("内存 总量/可用", f"{payload['memory'].get('total')} / {payload['memory'].get('available')} GiB")
        header("Python")
        row("版本", payload["python"]["version"])
        row("解释器", payload["python"]["executable"])
        for name, path in payload["python"]["candidates"].items():
            row(name, path or "未找到")
        header("工具")
        for name, info in payload["tools"].items():
            row(name, info["version"] or "未安装")
        header("关键依赖（当前解释器）")
        for name, version in packages.items():
            row(name, version or "未安装")
        if "torch" in payload:
            header("torch / CUDA")
            for key, value in payload["torch"].items():
                row(key, value)
        header("磁盘")
        for name, disk in payload["disks"].items():
            row(name, f"{disk.get('free_gib')} GiB 可用 / {disk.get('total_gib')} GiB" if disk.get("exists")
                else "路径不存在", disk.get("path", ""))

    path = save(args.out_dir, "environment", payload)
    for item in warnings:
        print(f"[告警] {item}")
    print(f"已记录：{path}")
    return 2 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
