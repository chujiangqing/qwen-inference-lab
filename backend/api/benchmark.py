"""Benchmark 接口：前置检查、异步启动评测、查询任务状态。

评测一律走 core/experiment_manager.py（串行独占 + 官方工具 + 入库），
API 层不复制任何评测口径。

**为什么是异步**：一次性能评测是小时级（4 档 × 100 请求）。
如果 POST /run 同步阻塞，浏览器/反向代理必然超时，前端就只能把 timeout 设成 86400 秒，
那实际上是"假装异步"。这里的做法是：POST 立刻返回 job_id，真正的执行在后台线程里，
前端用 GET /jobs/{id} 轮询。

注意：后台线程只跑评测逻辑本身，串行独占仍由 task_scheduler 保证，
所以即使有人连点两次，也只会有一个真正在跑。
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core import experiment_manager  # noqa: E402
from core.config import ROOT  # noqa: E402

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

# job_id -> {status, label, mode, started_at, finished_at, result, error}
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


class RunRequest(BaseModel):
    label: str = "E00_BASELINE"
    mode: str = "all"
    tokenizer: str = ""
    perf_requests: int = 100
    request_timeout: float = 600
    eval_batch_size: int = 2

    def config(self):
        return experiment_manager.make_config(mode=self.mode, tokenizer=self.tokenizer or None,
                                              perf_requests=self.perf_requests,
                                              request_timeout=self.request_timeout,
                                              eval_batch_size=self.eval_batch_size)


@router.post("/pretend")
def pretend(request: RunRequest) -> dict:
    """只做前置检查：依赖、服务、上下文、是否有别的任务在跑。"""
    return experiment_manager.preflight(request.mode, request.config())


def _run_job(job_id: str, request: RunRequest) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(status="running", started_at=time.time())
    try:
        result = experiment_manager.run(request.label, request.mode, request.config())
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="完成", result=result, finished_at=time.time())
    except (ValueError, RuntimeError, OSError) as error:
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="失败", error=f"{type(error).__name__}: {error}",
                                 finished_at=time.time())


@router.post("/run")
def run(request: RunRequest) -> dict:
    """异步启动一次评测：立刻返回 job_id，前端用 /jobs/{job_id} 轮询。"""
    if request.mode not in experiment_manager.MODES:
        raise HTTPException(status_code=400, detail=f"未知 mode：{request.mode}")
    preflight = experiment_manager.preflight(request.mode, request.config())
    if not preflight.get("ok"):
        raise HTTPException(status_code=409, detail={"message": "前置检查未通过",
                                                    "problems": preflight.get("problems")})

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"job_id": job_id, "label": request.label, "mode": request.mode,
                         "status": "排队", "started_at": None, "finished_at": None,
                         "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True,
                     name=f"benchmark-{job_id}").start()
    return {"job_id": job_id, "status": "排队",
            "message": "已在后台执行；用 /api/benchmark/jobs/{job_id} 查询，完成后实验会出现在 /api/experiments"}


@router.get("/jobs")
def list_jobs() -> dict:
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda item: item.get("started_at") or 0, reverse=True)
    return {"jobs": jobs[:20]}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    elapsed = None
    if job.get("started_at"):
        elapsed = round((job.get("finished_at") or time.time()) - job["started_at"], 1)
    return {**job, "elapsed_seconds": elapsed}


@router.get("/jobs/{job_id}/progress")
def job_progress(job_id: str, lines: int = 30) -> dict:
    """进度：已完成几个并发档 + 最新日志尾巴。

    进度用"产物计数"而不是解析进度条：每个并发档跑完官方工具会写一个
    performance/<cN>/summary.json，数文件个数最可靠。
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")

    label = str(job.get("label") or "")
    folder = None
    candidates = sorted((ROOT / "experiments").glob(f"{label}*"), key=lambda p: p.stat().st_mtime,
                        reverse=True) if label else []
    if candidates and candidates[0].is_dir():
        folder = candidates[0]

    cells_done = cells_total = None
    log_tail = ""
    if folder:
        summaries = list(folder.glob("performance/*/summary.json"))
        cells_done = len(summaries)
        cells_total = 4                     # 官方协议固定 4 档
        logs = sorted(folder.glob("*/console.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            try:
                text = logs[0].read_text(encoding="utf-8", errors="replace")
                log_tail = "\n".join(text.splitlines()[-max(1, min(lines, 200)):])
            except OSError:
                log_tail = "(日志读取失败)"

    elapsed = None
    if job.get("started_at"):
        elapsed = round((job.get("finished_at") or time.time()) - job["started_at"], 1)

    percent = None
    if cells_done is not None and cells_total:
        percent = min(99, round(cells_done / cells_total * 100)) if job.get("status") != "完成" else 100
    return {"job_id": job_id, "status": job.get("status"), "label": label,
            "elapsed_seconds": elapsed, "output_dir": str(folder) if folder else None,
            "cells_done": cells_done, "cells_total": cells_total, "percent": percent,
            "log_tail": log_tail, "error": job.get("error")}


@router.post("/auto-tuner/dry-run")
def tuner_dry_run() -> dict:
    done = subprocess.run([sys.executable, str(ROOT / "backend" / "optimizer" / "search_space.py"),
                           "--json"], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise HTTPException(status_code=500, detail=(done.stderr or done.stdout)[-1200:])
    return {"output": done.stdout}
