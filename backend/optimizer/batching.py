"""批处理维度：从 AutoTuner 的候选结果里挑出 max_num_seqs / max_num_batched_tokens 的最优组合。"""
from __future__ import annotations


def analyze(trials: list[dict]) -> dict:
    batching = [item for item in trials
                if item.get("dimension") == "batching" and item.get("valid")
                and item.get("output_throughput")]
    if not batching:
        return {"conclusion": "尚无批处理维度的有效结果"}
    ranked = sorted(batching, key=lambda item: item["output_throughput"], reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    by_seq: dict[int, list[float]] = {}
    for item in batching:
        seq = (item.get("overrides") or {}).get("max_num_seqs")
        if seq is not None:
            by_seq.setdefault(int(seq), []).append(item["output_throughput"])
    return {
        "best": {"candidate": best["candidate"], "overrides": best.get("overrides"),
                 "output_throughput": best["output_throughput"],
                 "e2e_p95": best.get("e2e_p95_ms"), "score": best.get("score")},
        "worst": {"candidate": worst["candidate"], "output_throughput": worst["output_throughput"]},
        "gain_vs_worst": round(best["output_throughput"] / worst["output_throughput"], 3),
        "by_max_num_seqs": {seq: round(max(values), 2) for seq, values in sorted(by_seq.items())},
        "conclusion": (f"{best['candidate']} 吞吐最高（{best['output_throughput']:.2f} token/s），"
                       f"比最差配置快 {best['output_throughput'] / worst['output_throughput']:.2f}×"),
    }
