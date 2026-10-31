"""显存预算估算：权重 + KV Cache + 激活预留，判断某个参数组合会不会 OOM。

被两处复用：
  * `optimizer/search_space.py` —— 网格搜索时先剔掉一定 OOM 的候选；
  * `scripts/check_model.py`    —— 阶段 0 的显存预算表。
公式只对 GQA 全注意力层成立；混合线性注意力（本模型 64 层里只有 16 层全注意力）
必须用 `full_attention_layers` 计数，否则会高估 4 倍。
"""
from __future__ import annotations

ACTIVATION_RESERVE_GIB = 1.5      # 激活、CUDA graph、通信缓冲的经验预留
KV_DTYPE_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "auto": 2.0, "fp16": 2.0, "bf16": 2.0}


def kv_per_token_bytes(layers: int, kv_heads: int, head_dim: int, dtype: str = "fp8") -> int:
    """每 token 的 KV 体积（K 和 V 各一份）。"""
    return int(2 * layers * kv_heads * head_dim * KV_DTYPE_BYTES.get(dtype, 2.0))


def capacity_tokens(*, vram_gib: float, weights_gib: float, per_token_bytes: float,
                    gpu_memory_utilization: float = 0.9) -> int:
    usable = vram_gib * gpu_memory_utilization - weights_gib - ACTIVATION_RESERVE_GIB
    if usable <= 0 or per_token_bytes <= 0:
        return 0
    return int(usable * 1024 ** 3 / per_token_bytes)


def capacity_tokens_from_kib(*, vram_gib: float, weights_gib: float, kv_per_token_kib: float,
                             gpu_memory_utilization: float = 0.9) -> int:
    usable = vram_gib * gpu_memory_utilization - weights_gib - ACTIVATION_RESERVE_GIB
    if usable <= 0 or kv_per_token_kib <= 0:
        return 0
    return int(usable * 1024 ** 2 / kv_per_token_kib)


def budget(*, vram_gib: float, weights_gib: float, kv_per_token_kib: float,
           gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int) -> dict:
    """返回一份可直接写进报告的预算表。"""
    usable = vram_gib * gpu_memory_utilization - weights_gib - ACTIVATION_RESERVE_GIB
    capacity = max(0.0, usable) * 1024 ** 2 / kv_per_token_kib if kv_per_token_kib else 0
    needed = max_model_len * max_num_seqs
    return {
        "usable_for_kv_gib": round(usable, 2),
        "kv_per_token_kib": kv_per_token_kib,
        "kv_capacity_tokens": int(capacity),
        "tokens_needed": needed,
        "max_parallel_full_length": int(capacity // max_model_len) if max_model_len else None,
        "fits": capacity >= needed,
    }


if __name__ == "__main__":
    # 本模型实测：16 个全注意力层、4 个 KV 头、head_dim 256、fp8
    per_token = kv_per_token_bytes(16, 4, 256, "fp8")
    print(f"KV/token = {per_token} B = {per_token / 1024:.1f} KiB")
    print(budget(vram_gib=47.99, weights_gib=28.75, kv_per_token_kib=per_token / 1024,
                 gpu_memory_utilization=0.9, max_model_len=32768, max_num_seqs=4))
