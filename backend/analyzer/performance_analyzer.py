"""性能分析：把逐档指标变成"结论 + 推荐"，供报告与前端共用。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.concurrency import analyze as analyze_curve  # noqa: E402


def rows(cells: list[dict]) -> list[dict]:
    """统一成报告用的行（毫秒 → 秒统一在这里做，避免各处口径不一）。"""
    result = []
    for cell in sorted(cells, key=lambda item: item.get("concurrency") or 0):
        result.append({
            "concurrency": cell.get("concurrency"),
            "status": cell.get("status"),
            "requests": cell.get("request_count"),
            "ttft_p50_s": _scale(cell.get("ttft_p50"), 0.001, 2),
            "ttft_p95_s": _scale(cell.get("ttft_p95"), 0.001, 2),
            "tpot_p50_ms": _scale(cell.get("tpot_p50"), 1, 2),
            "tpot_p95_ms": _scale(cell.get("tpot_p95"), 1, 2),
            "e2e_p50_s": _scale(cell.get("e2e_p50"), 0.001, 2),
            "e2e_p95_s": _scale(cell.get("e2e_p95"), 0.001, 2),
            "request_throughput": cell.get("request_throughput"),
            "output_throughput": cell.get("output_throughput"),
        })
    return result


def analyze(cells: list[dict]) -> dict:
    curve = analyze_curve(cells)
    return {"rows": rows(cells), "conclusion": curve.get("conclusion"),
            "recommendation": curve.get("recommendation"),
            "best_concurrency": curve.get("best_concurrency"),
            "best_output_throughput": curve.get("best_output_throughput"),
            "saturation": curve.get("saturation")}


def _scale(value, factor: float, digits: int):
    if value is None:
        return None
    try:
        return round(float(value) * factor, digits)
    except (TypeError, ValueError):
        return None
