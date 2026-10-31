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


def chinese_label(name: str) -> str:
    """把实验名翻译成中文说明，供界面直接展示。"""
    import re

    text = str(name or "")
    upper = text.upper()

    if upper.startswith("E00_BASELINE"):
        return "基线（未优化）"
    if upper.startswith("BASE20"):
        return "基线复测（20 请求/档）"
    if upper.startswith("FINAL"):
        mtp = "MTP " if "mtp" in text.lower() else ""
        count = re.search(r"(\d+)REQ", upper)
        req = f"{count.group(1)} 请求/档" if count else ""
        return f"正式复测：基线 + {mtp}（{req}）" if mtp else f"正式复测（{req}）"
    if upper.startswith("COMBO"):
        body = text[len("COMBO_"):]
        parts = []
        if "mtp" in body.lower():
            parts.append("MTP")
        seq = re.search(r"seq(\d+)", body)
        if seq:
            parts.append(f"{seq.group(1)} 路")
        ctx = re.search(r"ctx(\d+)", body)
        if ctx:
            parts.append(f"上下文 {int(ctx.group(1)) // 1024}k")
        return "组合候选：" + " + ".join(parts) if parts else "组合候选"
    if upper.startswith("QUALITY"):
        return "三类任务抽检"
    if upper.startswith("ACC_"):
        mtp = "MTP" if "MTP" in upper else "基线"
        return f"精度评测（{mtp}）"
    if upper.startswith("AUTOTUNER"):
        body = text[len("AUTOTUNER_"):]
        body = re.sub(r"_\d{8}_\d{6}.*$", "", body)
        parts = []
        if "mtp" in body.lower():
            parts.append("MTP 投机解码")
        if "kv_auto" in body.lower():
            parts.append("KV 用 bf16")
        seq = re.search(r"seq(\d+)", body)
        if seq:
            parts.append(f"并发 {seq.group(1)}")
        tok = re.search(r"tok(\d+)", body)
        if tok:
            parts.append(f"prefill 批 {tok.group(1)}")
        ctx = re.search(r"ctx(\d+)", body)
        if ctx:
            parts.append(f"上下文 {int(ctx.group(1)) // 1024}k")
        return "候选：" + " + ".join(parts) if parts else "候选"
    return text


@router.get("")
def list_experiments(limit: int = 50) -> dict:
    database.init(ROOT)
    rows = repo.list_experiments(DB_PATH, limit=limit)
    for row in rows:
        row["display_name"] = chinese_label(row.get("experiment_name") or "")
    return {"experiments": rows}


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


@router.get("/request-profile")
def request_profile() -> dict:
    """每个实验的每档请求数（用于判断两次实验是否同口径可对比）。"""
    database.init(ROOT)
    with database.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT experiment_id, MAX(request_count) AS requests, COUNT(*) AS levels "
            "FROM performance_metrics GROUP BY experiment_id").fetchall()
    return {"profile": {str(row[0]): {"requests": row[1], "levels": row[2]} for row in rows}}


@router.get("/{experiment_id}")
def detail(experiment_id: int) -> dict:
    database.init(ROOT)
    data = repo.experiment_detail(DB_PATH, experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="实验不存在")
    return data
