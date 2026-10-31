"""实验管理接口：列表、详情。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "storage"))

import database  # noqa: E402
import experiment_repository as repo  # noqa: E402
from core.config import DB_PATH, ROOT  # noqa: E402

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("")
def list_experiments(limit: int = 50) -> dict:
    database.init(ROOT)
    return {"experiments": repo.list_experiments(DB_PATH, limit=limit)}


@router.get("/compare")
def compare(baseline: int, candidate: int) -> dict:
    """两个实验的并排对比：逐并发档的吞吐/延迟变化百分比。

    baseline/candidate 传实验 id；两者都必须有性能数据，否则明确报错而不是返回空表。
    """
    database.init(ROOT)
    base = repo.experiment_detail(DB_PATH, baseline)
    cand = repo.experiment_detail(DB_PATH, candidate)
    if not base or not cand:
        raise HTTPException(status_code=404, detail="实验不存在")
    if not (base.get("performance") and cand.get("performance")):
        raise HTTPException(status_code=409, detail="所选实验缺少性能数据，无法对比")

    sys.path.insert(0, str(BACKEND / "analyzer"))
    import comparison  # noqa: E402

    summary = comparison.summarize(base["performance"], cand["performance"])
    base_meta = base.get("experiment") or {}
    cand_meta = cand.get("experiment") or {}
    return {"baseline": {"id": baseline, "name": base_meta.get("experiment_name")},
            "candidate": {"id": candidate, "name": cand_meta.get("experiment_name")},
            **summary}


@router.get("/{experiment_id}")
def detail(experiment_id: int) -> dict:
    database.init(ROOT)
    data = repo.experiment_detail(DB_PATH, experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="实验不存在")
    return data
