"""并发维度分析：找吞吐饱和点、量化"加并发不涨吞吐但延迟暴涨"的现象。"""
from __future__ import annotations


def analyze(cells: list[dict]) -> dict:
    """cells 为 result_parser.PerfCell 的字典形式（按并发升序）。"""
    usable = [cell for cell in cells if cell.get("status") == "完整" and cell.get("output_throughput")]
    if len(usable) < 2:
        return {"conclusion": "有效档位不足，无法分析并发曲线", "points": usable}
    usable.sort(key=lambda cell: cell["concurrency"])
    base = usable[0]
    best = max(usable, key=lambda cell: cell["output_throughput"])
    saturation = None
    for previous, current in zip(usable, usable[1:]):
        gain = current["output_throughput"] / previous["output_throughput"] - 1
        latency = (current.get("ttft_p95") or 0) / (previous.get("ttft_p95") or 1)
        if gain < 0.05 and latency > 1.3:
            saturation = {"from": previous["concurrency"], "to": current["concurrency"],
                          "throughput_gain": round(gain, 4), "ttft_p95_ratio": round(latency, 2)}
            break
    # "最优"按建议口径：有饱和点就取饱和前那一档，否则取吞吐最高档
    recommended = saturation["from"] if saturation else best["concurrency"]
    return {
        "base_concurrency": base["concurrency"],
        "base_output_throughput": base["output_throughput"],
        "best_concurrency": recommended,
        "peak_concurrency": best["concurrency"],
        "best_output_throughput": best["output_throughput"],
        "speedup_vs_base": round(best["output_throughput"] / base["output_throughput"], 3),
        "saturation": saturation,
        "conclusion": ("已定位饱和点：" + f"并发 {saturation['from']}→{saturation['to']} 时吞吐仅增 "
                       f"{saturation['throughput_gain'] * 100:.1f}%，而 TTFT P95 变为 "
                       f"{saturation['ttft_p95_ratio']}×" if saturation else
                       "未观察到明显饱和（每档吞吐仍有 >5% 提升）"),
        "recommendation": (f"推荐并发 {best['concurrency']}：吞吐最高 {best['output_throughput']:.2f} token/s"
                           if not saturation else
                           f"推荐并发 {saturation['from']}：再往上吞吐不涨、延迟显著变差"),
    }
