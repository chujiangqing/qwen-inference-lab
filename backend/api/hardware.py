"""硬件资源接口：直接复用 monitor/ 下的两个监控模块，本层只做字段映射。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core.config import ROOT  # noqa: E402
from monitor import gpu_monitor, system_monitor  # noqa: E402

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("")
def hardware() -> dict:
    system = system_monitor.snapshot(ROOT)
    memory = system.get("memory_gib") or {}
    # 保持前端已使用的字段名（MemTotal / MemAvailable / cpu_count），不动前端
    return {"gpu": gpu_monitor.snapshot(),
            "memory_gib": {"MemTotal": memory.get("total"), "MemAvailable": memory.get("available")},
            "cpu_count": (system.get("cpu") or {}).get("count"),
            "cpu": system.get("cpu"),
            "disk": system.get("disk")}
