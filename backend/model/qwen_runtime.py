"""模型侧的运行时知识与校验：这个模型支持什么、某个 profile 合不合法。

与 scripts/render_runtime.py 的分工：
  * render_runtime.py 负责把 profile 变成 vLLM 命令行参数（给 shell 用）；
  * qwen_runtime.py 负责回答"这些参数对这个模型合不合法"，给 preflight 用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.memory_estimator import budget  # noqa: E402
from model.qwen_config import ModelFacts, facts as read_facts  # noqa: E402


def supports_mtp(model: ModelFacts) -> bool:
    """config.json 里有 mtp.* 层才谈得上投机解码。"""
    return bool(model.mtp_num_hidden_layers)


def max_context_by_model(model: ModelFacts) -> int | None:
    return model.max_position_embeddings


def recommended_kv_dtype(model: ModelFacts) -> str:
    """FP8 KV 在 Ada（sm89）上可用，能把 KV 体积压到一半；这是 48GB 卡能开大上下文的前提。"""
    return "fp8"


def validate_profile(profile: dict, model: ModelFacts, vram_gib: float) -> dict:
    """返回 {"ok": bool, "problems": [...], "notes": [...], "budget": {...}}。"""
    problems: list[str] = []
    notes: list[str] = []

    max_len = int(profile.get("max_model_len") or 32768)
    seqs = int(profile.get("max_num_seqs") or 4)
    util = float(profile.get("gpu_memory_utilization") or 0.9)
    kv_dtype = str(profile.get("kv_cache_dtype") or "auto")

    limit = max_context_by_model(model)
    if limit and max_len > limit:
        problems.append(f"max_model_len={max_len} 超过模型上限 max_position_embeddings={limit}")

    if profile.get("speculative_config") and not supports_mtp(model):
        problems.append("profile 开了投机解码，但模型没有 MTP 层（mtp_num_hidden_layers 为空）")

    if model.is_multimodal:
        notes.append("多模态模型：视觉塔会额外占用显存，实际可用 KV 略小于估算")

    result_budget = None
    if model.full_attention_layers and model.num_key_value_heads and model.head_dim:
        per_token_bytes = 2 * model.full_attention_layers * model.num_key_value_heads * model.head_dim
        if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            per_token_bytes //= 2          # fp8 每元素 1 字节，fp16/bf16 是 2 字节
        result_budget = budget(vram_gib=vram_gib, weights_gib=model.weights_gib or 0.0,
                               kv_per_token_kib=per_token_bytes / 1024,
                               gpu_memory_utilization=util,
                               max_model_len=max_len, max_num_seqs=seqs)
        if not result_budget["fits"]:
            problems.append(f"KV 预算不足：需要 {result_budget['tokens_needed']} token，"
                            f"只能容纳 {result_budget['kv_capacity_tokens']}")
    else:
        notes.append("混合注意力结构缺少层数/头数信息，无法估算 KV 预算")

    if model.is_hybrid:
        notes.append(f"混合注意力：{model.num_hidden_layers} 层里仅 {model.full_attention_layers} 层保存 KV")

    return {"ok": not problems, "problems": problems, "notes": notes, "budget": result_budget,
            "model": model.to_dict()}


def validate(model_path: Path, profile: dict, vram_gib: float) -> dict:
    try:
        model = read_facts(Path(model_path))
    except (OSError, ValueError) as error:
        return {"ok": False, "problems": [f"读取模型配置失败：{type(error).__name__}: {error}"],
                "notes": [], "budget": None, "model": None}
    return validate_profile(profile, model, vram_gib)
