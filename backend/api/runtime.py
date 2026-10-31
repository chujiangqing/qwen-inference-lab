"""模型运行管理接口：服务状态、启停、日志、启动前检查。

业务逻辑在 core/runtime_manager.py 与 core/resource_manager.py，本层只做 HTTP 转换。
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core import resource_manager, runtime_manager  # noqa: E402
from core.config import runtime_config  # noqa: E402

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


class StartRequest(BaseModel):
    profile: str = "baseline"
    overrides: dict = {}


@router.get("/status")
def status() -> dict:
    config = runtime_config()
    return {"service": runtime_manager.status(),
            "profiles": runtime_manager.profiles(),
            "log_dir": config["server"]["log_dir"],
            "pid_file": config["server"]["pid_file"]}


@router.get("/preflight")
def preflight(profile: str = "baseline") -> dict:
    """起服务前检查：GPU 空闲？显存够？端口没占？KV 预算够？"""
    return resource_manager.can_start(profile)


@router.post("/start")
def start(request: StartRequest) -> dict:
    result = runtime_manager.start(request.profile, request.overrides)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("output", "")[-1500:])
    return result


@router.post("/stop")
def stop() -> dict:
    return runtime_manager.stop()


@router.get("/logs")
def logs(lines: int = 60) -> dict:
    return runtime_manager.logs(lines)
