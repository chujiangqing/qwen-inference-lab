"""报告与产物接口：Markdown 报告、CSV 导出、原始文件哈希校验。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "storage"))

import artifact_repository as artifacts  # noqa: E402
import database  # noqa: E402
from core.config import DB_PATH, EXPERIMENTS_DIR, ROOT  # noqa: E402

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{experiment_name}", response_class=PlainTextResponse)
def report(experiment_name: str) -> str:
    candidates = [
        EXPERIMENTS_DIR / experiment_name / "benchmark_result.json",
        EXPERIMENTS_DIR / experiment_name / "experiment_report.md",
    ]
    for folder in EXPERIMENTS_DIR.glob(f"{experiment_name}*"):
        candidates += [folder / "experiment_report.md", folder / "report.md",
                       folder / "performance" / "report.md"]
    for path in candidates:
        if path.is_file() and path.suffix == ".md":
            return path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail=f"未找到 {experiment_name} 的报告")


@router.get("/export/csv", response_class=PlainTextResponse)
def export_csv() -> str:
    database.init(ROOT)
    path = artifacts.export_csv(DB_PATH, ROOT / "data" / "experiments.csv")
    return path.read_text(encoding="utf-8-sig")


@router.get("/{experiment_id}/verify")
def verify(experiment_id: int) -> dict:
    database.init(ROOT)
    return artifacts.verify(DB_PATH, experiment_id)
