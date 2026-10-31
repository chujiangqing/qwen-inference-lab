"""精度分析：给出分数、置信区间与"能不能下结论"的判断。

100 样本的二项标准误约 ±4 个百分点，小于该幅度的差异不能当成结论（写报告时的硬约束）。
"""
from __future__ import annotations

import json
import math


def binomial_stderr(score: float, samples: int) -> float | None:
    if not samples or not 0 <= score <= 1:
        return None
    return math.sqrt(score * (1 - score) / samples)


def analyze(accuracy_rows: list[dict]) -> dict:
    if not accuracy_rows:
        return {"conclusion": "尚无精度结果"}
    row = sorted(accuracy_rows, key=lambda item: item.get("id") or 0)[-1]
    score = row.get("prompt_level_strict")
    samples = row.get("sample_count") or 0
    stderr = binomial_stderr(score or 0.0, samples)
    raw = {}
    if row.get("raw_scores"):
        try:
            raw = json.loads(row["raw_scores"])
        except (TypeError, ValueError):
            raw = {}
    return {
        "dataset": row.get("dataset"),
        "samples": samples,
        "prompt_level_strict": score,
        "binomial_stderr": round(stderr, 4) if stderr else None,
        "confidence_interval_95": [round(max(0.0, (score or 0) - 1.96 * stderr), 4),
                                   round(min(1.0, (score or 0) + 1.96 * stderr), 4)] if stderr else None,
        "other_metrics": {key: raw.get(key) for key in
                          ("inst_level_strict", "prompt_level_loose", "inst_level_loose")},
        "conclusion": (f"{score:.1%}（{samples} 样本，±{1.96 * stderr:.1%} @95%）"
                       if score is not None and stderr else "精度数据不完整"),
    }


def is_meaningful_difference(a: float | None, b: float | None, samples: int = 100) -> bool:
    """两个分数之差是否超出 1.96 个标准误（近似：用两比例的最大标准误）。"""
    if a is None or b is None:
        return False
    stderr = max(binomial_stderr(a, samples) or 0, binomial_stderr(b, samples) or 0)
    return abs(a - b) > 1.96 * stderr
