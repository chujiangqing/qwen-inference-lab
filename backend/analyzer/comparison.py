"""多实验对比：给"优化前后"提供统一的差值与显著性标注。"""
from __future__ import annotations

METRICS = ("output_throughput", "request_throughput", "ttft_p95", "tpot_p95", "e2e_p95")


def _pct(before: float | None, after: float | None) -> float | None:
    if not before or after is None:
        return None
    return round((after / before - 1) * 100, 2)


def compare_cells(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    """按并发对齐两组的指标，给出百分比变化（正数=变大）。"""
    base_map = {item.get("concurrency"): item for item in baseline}
    rows = []
    for cell in sorted(candidate, key=lambda item: item.get("concurrency") or 0):
        reference = base_map.get(cell.get("concurrency"))
        if not reference:
            continue
        row = {"concurrency": cell.get("concurrency")}
        for metric in METRICS:
            row[f"{metric}_base"] = reference.get(metric)
            row[f"{metric}_new"] = cell.get(metric)
            row[f"{metric}_delta_pct"] = _pct(reference.get(metric), cell.get(metric))
        rows.append(row)
    return rows


def summarize(baseline: list[dict], candidate: list[dict]) -> dict:
    rows = compare_cells(baseline, candidate)
    if not rows:
        return {"conclusion": "两组数据没有共同并发档，无法对比"}
    best = max(rows, key=lambda item: item.get("output_throughput_new") or 0)
    delta = best.get("output_throughput_delta_pct")
    direction = "提升" if (delta or 0) > 0 else "下降"
    return {"rows": rows, "best_concurrency": best["concurrency"],
            "throughput_delta_pct": delta,
            "conclusion": f"并发 {best['concurrency']} 下输出吞吐{direction} {abs(delta or 0):.1f}%"}
