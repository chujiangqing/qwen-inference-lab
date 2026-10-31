"""推荐：从 AutoTuner 的候选里给出"用哪个配置、为什么、有什么风险"。

硬规则（需求书 8.3）：未测质量的配置不得评为唯一最优；质量低于阈值直接出局。
"""
from __future__ import annotations


def recommend(trials: list[dict], baseline: dict | None = None,
              quality_threshold: float = 0.70) -> dict:
    usable = [item for item in trials
              if item.get("valid") and item.get("output_throughput") and item.get("score") is not None]
    if not usable:
        return {"conclusion": "没有有效候选，无法推荐", "recommended": None}

    unqualified = [item for item in usable
                   if item.get("quality") is not None and item["quality"] < quality_threshold]
    usable = [item for item in usable if item not in unqualified]
    if not usable:
        return {"conclusion": f"所有候选质量都低于门槛 {quality_threshold}", "recommended": None}

    ranked = sorted(usable, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    quality_unknown = [item for item in ranked if item.get("quality") is None]

    risks: list[str] = []
    if best.get("quality") is None:
        risks.append("该配置尚未跑 IFBench，质量未验证，不能作为最终交付配置")
    if best.get("status") != "完成":
        risks.append(f"该配置的评测状态是 {best.get('status')}，不是完整")
    if best.get("vram_peak_gib") and baseline and baseline.get("vram_peak_gib"):
        if best["vram_peak_gib"] > baseline["vram_peak_gib"] + 2:
            risks.append(f"峰值显存比基线高 {best['vram_peak_gib'] - baseline['vram_peak_gib']:.1f} GiB，长期运行需观察")

    return {
        "recommended": best.get("candidate"),
        "overrides": best.get("overrides"),
        "output_throughput": best.get("output_throughput"),
        "e2e_p95_ms": best.get("e2e_p95_ms"),
        "vram_peak_gib": best.get("vram_peak_gib"),
        "quality": best.get("quality"),
        "score": best.get("score"),
        "runner_up": ranked[1].get("candidate") if len(ranked) > 1 else None,
        "quality_unverified": [item.get("candidate") for item in quality_unknown],
        "risks": risks,
        "conclusion": (f"推荐 {best.get('candidate')}：评分 {best.get('score'):.4f}，"
                       f"输出 {best.get('output_throughput'):.2f} token/s"
                       + ("；但质量未验证，需补 IFBench" if best.get("quality") is None else "")),
    }
