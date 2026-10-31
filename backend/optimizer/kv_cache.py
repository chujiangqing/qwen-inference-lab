"""KV Cache 维度：把实测的 KV/token 与并发需求换算成"能撑多少并发"，用于解释饱和点。"""
from __future__ import annotations


def analyze(facts: dict, configs: list[dict]) -> dict:
    """
    facts: 来自 reports/*/model.json（weights_gib / kv_per_token_kib / vram_gib）
    configs: 待评估的配置列表，每项含 max_num_seqs / max_model_len / gpu_memory_utilization
    """
    weights = facts.get("weights_gib")
    per_token_kib = facts.get("kv_per_token_kib")
    vram = facts.get("vram_gib")
    if not (weights and per_token_kib and vram):
        return {"conclusion": "缺少实测的权重/KV/显存数据，无法计算 KV 容量"}
    rows = []
    for config in configs:
        utilization = float(config.get("gpu_memory_utilization", 0.9))
        max_len = int(config.get("max_model_len", 32768))
        seqs = int(config.get("max_num_seqs", 4))
        usable = vram * utilization - weights - 1.5
        capacity = max(0.0, usable) * 1024 ** 2 / per_token_kib
        needed = max_len * seqs
        rows.append({"max_num_seqs": seqs, "max_model_len": max_len,
                     "usable_for_kv_gib": round(usable, 2),
                     "capacity_tokens": int(capacity),
                     "needed_tokens": needed,
                     "max_parallel_full_length": int(capacity // max_len) if max_len else None,
                     "fits": capacity >= needed})
    feasible = [row for row in rows if row["fits"]]
    return {
        "kv_per_token_kib": per_token_kib,
        "rows": rows,
        "max_feasible_max_num_seqs": max((row["max_num_seqs"] for row in feasible), default=None),
        "conclusion": (f"单 token KV {per_token_kib} KiB；在 "
                       f"{rows[0]['max_model_len']} 上下文下最多可同时容纳 "
                       f"{feasible and max(row['max_parallel_full_length'] for row in feasible)} 路满长请求"
                       if rows else "无配置"),
    }
