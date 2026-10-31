"""MTP 维度：性能增益、质量变化、以及"无限思考循环"这类失败的判定。"""
from __future__ import annotations


def analyze(baseline: dict | None, mtp: dict | None, baseline_accuracy: dict | None = None,
            mtp_accuracy: dict | None = None) -> dict:
    """
    baseline / mtp: 各含 output_throughput、ttft_p95_ms、tpot_p95_ms、e2e_p95_ms
    *_accuracy:    各含 status、prompt_level_strict、samples、issues
    """
    result: dict = {"baseline": baseline, "mtp": mtp, "risks": [], "conclusion": ""}
    if baseline and mtp and baseline.get("output_throughput") and mtp.get("output_throughput"):
        result["throughput_gain"] = round(mtp["output_throughput"] / baseline["output_throughput"] - 1, 4)
        if baseline.get("tpot_p95_ms") and mtp.get("tpot_p95_ms"):
            result["tpot_improvement"] = round(1 - mtp["tpot_p95_ms"] / baseline["tpot_p95_ms"], 4)

    if mtp_accuracy:
        status = mtp_accuracy.get("status")
        if status != "完整":
            result["risks"].append(f"MTP 精度评测状态为 {status}：{'; '.join(mtp_accuracy.get('issues') or [])}")
        score = mtp_accuracy.get("prompt_level_strict")
        base_score = (baseline_accuracy or {}).get("prompt_level_strict")
        if score is not None and base_score is not None:
            result["quality_delta"] = round(score - base_score, 4)
            if score < base_score - 0.05:
                result["risks"].append(f"质量下降 {base_score - score:.1%}，超过 5 个百分点阈值")
    if mtp and mtp.get("status") and mtp["status"] != "完整":
        result["risks"].append(f"MTP 性能评测未完整：{mtp['status']}")

    gain = result.get("throughput_gain")
    if result["risks"]:
        result["conclusion"] = ("MTP 存在风险，默认关闭；" + "；".join(result["risks"]))
    elif gain and gain > 0.10:
        result["conclusion"] = f"MTP 吞吐提升 {gain:.1%} 且未发现质量下降，可考虑按场景开启"
    elif gain:
        result["conclusion"] = f"MTP 吞吐仅提升 {gain:.1%}，收益不足以抵消复杂度，建议保持关闭"
    else:
        result["conclusion"] = "缺少 MTP 对照数据"
    return result
