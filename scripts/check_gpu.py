#!/usr/bin/env python3
"""GPU 检测（只读）：型号、显存、驱动、温度、功耗、PCIe、占用进程。

用法：
    python3 scripts/check_gpu.py [--out-dir reports/task0] [--json-only]

退出码：0 检测成功；1 未检测到 nvidia-smi/GPU；2 检测成功但有告警
（显存与基准 48GB 不符、GPU 被其它进程占用等）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _report import (ROOT, configure_stdio, find_executable, header, platform_summary,
                     row, run, save)

# 基准文件记录的是 RTX 4090 48GB；用于判断是否需要向主办方复核。
DOCUMENTED_VRAM_GIB = 48.0
VRAM_MATCH_TOLERANCE_GIB = 3.0

CORE_FIELDS = [
    ("name", "型号"),
    ("memory.total", "显存总量(MiB)"),
    ("memory.used", "已用显存(MiB)"),
    ("memory.free", "剩余显存(MiB)"),
    ("driver_version", "驱动版本"),
    ("temperature.gpu", "温度(℃)"),
    ("power.draw", "功耗(W)"),
    ("power.limit", "功耗上限(W)"),
    ("utilization.gpu", "GPU 利用率(%)"),
]

OPTIONAL_FIELDS = [
    ("clocks.sm", "SM 频率(MHz)"),
    ("clocks.mem", "显存频率(MHz)"),
    ("pcie.link.gen.current", "PCIe 代数"),
    ("pcie.link.width.current", "PCIe 带宽等级"),
    ("ecc.mode.current", "ECC 模式"),
    ("persistence_mode", "持久化模式"),
    ("vbios_version", "VBIOS"),
    ("compute_cap", "计算能力"),
]

NVIDIA_SMI_FALLBACKS = (
    r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    r"C:\Windows\System32\nvidia-smi.exe",
)


def locate_smi() -> str | None:
    return find_executable(("nvidia-smi",), NVIDIA_SMI_FALLBACKS)


def query(smi: str, fields: list[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    """Return one dict per GPU; unsupported fields are dropped one by one."""
    names = [name for name, _ in fields]
    result = run([smi, f"--query-gpu={','.join(names)}", "--format=csv,noheader,nounits"], timeout=30)
    if not result["ok"]:
        return [], [f"整体查询失败：{result['stderr'] or result['code']}"]
    rows = []
    for line in result["stdout"].splitlines():
        if not line.strip():
            continue
        rows.append(dict(zip(names, [cell.strip() for cell in line.split(",")])))
    if rows:
        return rows, []
    return [], ["查询无输出"]


def query_optional(smi: str, gpu_index: str, fields: list[tuple[str, str]]) -> dict:
    values, unsupported = {}, []
    for name, _label in fields:
        result = run([smi, "-i", gpu_index, f"--query-gpu={name}", "--format=csv,noheader,nounits"], timeout=20)
        text = result["stdout"].strip()
        if result["ok"] and text and not text.startswith(("[N/A]", "[Not Supported]", "N/A")):
            values[name] = text
        else:
            unsupported.append(name)
    return {"values": values, "unsupported": unsupported}


def processes(smi: str) -> list[dict]:
    """Return compute processes; unreadable rows are kept and marked."""
    result = run([smi, "--query-compute-apps=pid,process_name,used_memory",
                  "--format=csv,noheader,nounits"], timeout=30)
    entries = []
    if result["ok"]:
        for line in result["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3 or not parts[0]:
                continue
            readable = parts[0].isdigit() and not parts[1].startswith("[")
            entries.append({"pid": parts[0], "process": parts[1], "memory_mib": parts[2],
                            "readable": readable})
    return entries


def to_number(value: str | None) -> float | None:
    """nvidia-smi returns numbers or '[N/A]'/'[Not Supported]'."""
    if value is None:
        return None
    text = str(value).strip().replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="只读 GPU 检测")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "manual")
    parser.add_argument("--json-only", action="store_true", help="只写 JSON，不打印明细")
    args = parser.parse_args()

    warnings: list[str] = []
    smi = locate_smi()
    payload: dict = {"nvidia_smi": smi, "platform": platform_summary(), "gpus": [],
                     "warnings": warnings}

    if not smi:
        payload["error"] = "未找到 nvidia-smi：本机不是 NVIDIA 环境，或未安装驱动"
        path = save(args.out_dir, "gpu", payload)
        print(f"未找到 nvidia-smi。已记录：{path}")
        return 1

    rows, errors = query(smi, CORE_FIELDS)
    payload["query_errors"] = errors
    listing = run([smi, "-L"], timeout=30)
    payload["nvidia_smi_L"] = listing["stdout"]

    if not rows:
        payload["error"] = "nvidia-smi 存在但没有返回 GPU 行"
        path = save(args.out_dir, "gpu", payload)
        print(f"未检测到 GPU。已记录：{path}")
        return 1

    for index, entry in enumerate(rows):
        gpu_index = str(entry.get("index", index))
        total_mib = to_number(entry.get("memory.total"))
        total_gib = round(total_mib / 1024, 2) if total_mib else None
        extra = query_optional(smi, gpu_index, OPTIONAL_FIELDS)
        gpu = {
            "index": gpu_index,
            "name": entry.get("name"),
            "memory_total_mib": total_mib,
            "memory_total_gib": total_gib,
            "memory_used_mib": to_number(entry.get("memory.used")),
            "memory_free_mib": to_number(entry.get("memory.free")),
            "driver_version": entry.get("driver_version"),
            "temperature_c": to_number(entry.get("temperature.gpu")),
            "power_draw_w": to_number(entry.get("power.draw")),
            "power_limit_w": to_number(entry.get("power.limit")),
            "utilization_percent": to_number(entry.get("utilization.gpu")),
            "optional": extra["values"],
            "unsupported_fields": extra["unsupported"],
        }
        if total_gib is not None:
            gpu["vram_matches_documented_48gb"] = abs(total_gib - DOCUMENTED_VRAM_GIB) <= VRAM_MATCH_TOLERANCE_GIB
            if not gpu["vram_matches_documented_48gb"]:
                warnings.append(f"GPU{gpu_index} 显存 {total_gib} GiB 与基准记录的 48GB 不一致，"
                                f"需向主办方确认是否有 48GB 变体；所有显存参数必须按实测值重新计算")
        payload["gpus"].append(gpu)

    usage = processes(smi)
    payload["compute_processes"] = usage
    known = [item for item in usage if item["readable"]]
    unknown = [item for item in usage if not item["readable"]]
    if known:
        warnings.append("检测到已有进程占用 GPU，正式评测前必须确认没有其它服务在跑："
                        + "; ".join(f"pid={i['pid']} {i['process']} {i['memory_mib']}MiB" for i in known))
    if unknown:
        warnings.append(f"有 {len(unknown)} 个进程正在使用 GPU 但当前权限读不到详情；"
                        f"请用 nvidia-smi 人工确认（评测期间不应有其它负载）")

    if not args.json_only:
        header("GPU")
        for gpu in payload["gpus"]:
            row("GPU 编号", gpu["index"])
            row("型号", gpu["name"])
            row("显存总量", f"{gpu['memory_total_gib']} GiB")
            row("显存已用/剩余", f"{gpu['memory_used_mib']} / {gpu['memory_free_mib']} MiB")
            row("驱动版本", gpu["driver_version"])
            row("温度", f"{gpu['temperature_c']} ℃")
            row("功耗（当前/上限）", f"{gpu['power_draw_w']} / {gpu['power_limit_w']} W")
            row("GPU 利用率", f"{gpu['utilization_percent']} %")
            for key, value in gpu["optional"].items():
                row(key, value)
            if gpu["unsupported_fields"]:
                row("不支持的字段", ", ".join(gpu["unsupported_fields"]))
            row("与基准 48GB 一致", gpu.get("vram_matches_documented_48gb"))
        if usage:
            header("占用 GPU 的进程")
            for item in usage:
                row(f"pid {item['pid']}", f"{item['process']}  {item['memory_mib']} MiB")

    path = save(args.out_dir, "gpu", payload)
    for item in warnings:
        print(f"[告警] {item}")
    print(f"已记录：{path}")
    return 2 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
