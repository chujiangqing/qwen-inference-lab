#!/usr/bin/env python3
"""模型目录只读审计：文件完整性、架构、权重体积、KV Cache 与显存预算。

用法：
    python3 scripts/check_model.py --model-path /model/ModelScope/Qwen/Qwen3.8-27B-FP8 \
        [--vram-gib 48] [--max-model-len 32768] [--max-num-seqs 4] [--gpu-memory-utilization 0.90]

只读取 config.json 与文件大小，不加载权重、不 import torch。
退出码：0 通过；1 目录/关键文件缺失；2 有告警（显存预算不足、上下文不够等）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _report import ROOT, configure_stdio, gib, header, row, save

REQUIRED_HINTS = ("config.json",)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")
# 精度评测固定 max_tokens=32768；eval.zip 在 max_model_len <= 32768 时会直接跳过精度阶段。
EVAL_MAX_TOKENS = 32768
# 运行时激活、CUDA graph、通信缓冲等预留，按经验值给出并在报告中标注为估算。
ACTIVATION_RESERVE_GIB = 1.5
KV_DTYPE_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "auto": 2.0, "fp16": 2.0, "bf16": 2.0}


def load_config(path: Path) -> dict:
    # utf-8-sig 容忍 BOM：个别导出工具会给 config.json 加 BOM。
    return json.loads((path / "config.json").read_text(encoding="utf-8-sig"))


def weight_files(path: Path) -> list[Path]:
    return [item for item in path.rglob("*") if item.is_file() and item.suffix in WEIGHT_SUFFIXES]


def architecture(config: dict) -> dict:
    text = config.get("text_config", config)  # 多模态/嵌套配置的兜底
    heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads", heads)
    hidden = text.get("hidden_size")
    head_dim = text.get("head_dim") or (hidden // heads if hidden and heads else None)
    layers = text.get("num_hidden_layers")
    # 混合注意力（Qwen3-Next / Qwen3.5 一类）：线性注意力层不保存 KV Cache，
    # 只有每个 full_attention_interval 层中的全注意力层才计入 KV 预算。
    layer_types = text.get("layer_types") or []
    interval = text.get("full_attention_interval")
    full_layers = None
    if layer_types:
        full_layers = sum(1 for item in layer_types
                          if str(item) in {"full_attention", "full", "full_attn", "attention"})
    elif interval and layers:
        full_layers = max(1, (int(layers) + int(interval) - 1) // int(interval))
    elif layers:
        full_layers = int(layers)
    return {
        "model_type": text.get("model_type") or config.get("model_type"),
        "architectures": text.get("architectures") or config.get("architectures"),
        "num_hidden_layers": layers,
        "full_attention_layers": full_layers,
        "full_attention_interval": interval,
        "is_hybrid": bool(full_layers and layers and full_layers < layers),
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": text.get("max_position_embeddings"),
        "torch_dtype": text.get("torch_dtype") or config.get("torch_dtype"),
        "quantization_config": config.get("quantization_config") or text.get("quantization_config"),
        "tie_word_embeddings": text.get("tie_word_embeddings"),
        "vocab_size": text.get("vocab_size"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "is_mla": any(key in text for key in ("kv_lora_rank", "qk_nope_head_dim")),
        "is_moe": "num_experts" in text or "n_routed_experts" in text,
        "num_experts_per_tok": text.get("num_experts_per_tok") or text.get("moe_top_k"),
    }


def tokenizer_files(path: Path) -> dict:
    expected = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                "special_tokens_map.json", "generation_config.json")
    found = {name: (path / name).is_file() for name in expected}
    return {"expected": found,
            "usable": bool(found["tokenizer_config.json"] and (found["tokenizer.json"] or found["vocab.json"]))}


def kv_budget(arch: dict, args: argparse.Namespace, weights_gib: float) -> dict:
    details: dict = {"kv_dtype": args.kv_cache_dtype,
                     "kv_dtype_bytes": KV_DTYPE_BYTES.get(args.kv_cache_dtype, 2.0),
                     "activation_reserve_gib": ACTIVATION_RESERVE_GIB}
    if arch["is_mla"]:
        details["note"] = "检测到 MLA 结构，逐 token KV 体积与 GQA 公式不同，此处不做数值估算"
        return details
    layers, kv_heads, head_dim = arch["num_hidden_layers"], arch["num_key_value_heads"], arch["head_dim"]
    kv_layers = arch.get("full_attention_layers") or layers
    if not all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)):
        details["note"] = "config.json 缺少层数/头数，无法估算 KV Cache"
        return details
    per_token = 2 * kv_layers * kv_heads * head_dim * details["kv_dtype_bytes"]
    details["kv_layers"] = kv_layers
    if arch.get("is_hybrid"):
        details["note"] = (f"混合注意力：{layers} 层中仅 {kv_layers} 层全注意力保存 KV Cache；"
                           f"线性注意力层使用固定大小状态，未计入")
    usable = args.vram_gib * args.gpu_memory_utilization - weights_gib - ACTIVATION_RESERVE_GIB
    capacity_tokens = max(0.0, usable) * 1024 ** 3 / per_token
    needed_tokens = args.max_num_seqs * args.max_model_len
    details.update({
        "kv_per_token_bytes": int(per_token),
        "kv_per_token_kib": round(per_token / 1024, 2),
        "usable_for_kv_gib": round(usable, 2),
        "kv_capacity_tokens": int(capacity_tokens),
        "kv_capacity_full_length_requests": int(capacity_tokens // args.max_model_len) if args.max_model_len else None,
        "tokens_needed_for_max_num_seqs": needed_tokens,
        "fits": capacity_tokens >= needed_tokens,
    })
    return details


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="模型目录只读审计")
    parser.add_argument("--model-path", required=True, help="模型目录，例如 /model/ModelScope/Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "manual")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--vram-gib", type=float, default=48.0, help="实测显存（GiB），来自 check_gpu.py")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    args = parser.parse_args()

    path = Path(args.model_path).expanduser()
    warnings: list[str] = []
    payload: dict = {"model_path": str(path), "vram_gib": args.vram_gib,
                     "runtime": {"max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
                                 "gpu_memory_utilization": args.gpu_memory_utilization,
                                 "kv_cache_dtype": args.kv_cache_dtype},
                     "warnings": warnings}

    if not path.is_dir():
        payload["error"] = "模型目录不存在"
        saved = save(args.out_dir, "model", payload)
        print(f"模型目录不存在：{path}\n已记录：{saved}")
        return 1

    missing = [name for name in REQUIRED_HINTS if not (path / name).is_file()]
    if missing:
        payload["error"] = f"缺少关键文件：{missing}"
        saved = save(args.out_dir, "model", payload)
        print(f"缺少关键文件：{missing}\n已记录：{saved}")
        return 1

    try:
        config = load_config(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        payload["error"] = f"config.json 无法解析：{error}"
        saved = save(args.out_dir, "model", payload)
        print(f"config.json 无法解析：{error}\n已记录：{saved}")
        return 1
    arch = architecture(config)
    payload["config_summary"] = arch
    weights = weight_files(path)
    weights_bytes = sum(item.stat().st_size for item in weights)
    weights_gib = gib(weights_bytes)
    index_file = path / "model.safetensors.index.json"
    payload["weights"] = {
        "files": len(weights),
        "extensions": sorted({item.suffix for item in weights}),
        "total_gib": weights_gib,
        "shards_declared": (len(json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {}))
                            if index_file.is_file() else None),
        "has_index": index_file.is_file(),
        "quantization": (arch["quantization_config"] or {}).get("quant_method") if arch["quantization_config"] else None,
    }
    payload["tokenizer"] = tokenizer_files(path)
    payload["kv_budget"] = kv_budget(arch, args, weights_gib or 0.0)

    if not weights:
        warnings.append("未找到 safetensors/bin/gguf 权重文件")
    if not payload["tokenizer"]["usable"]:
        warnings.append("tokenizer 文件不完整：AIPerf 以模型 tokenizer 生成输入时会失败")
    if weights_gib and weights_gib > args.vram_gib * args.gpu_memory_utilization:
        warnings.append(f"权重 {weights_gib} GiB 已超过可用显存 "
                        f"{round(args.vram_gib * args.gpu_memory_utilization, 2)} GiB")
    if isinstance(arch["max_position_embeddings"], int) and arch["max_position_embeddings"] < args.max_model_len:
        warnings.append(f"max_position_embeddings={arch['max_position_embeddings']} < max_model_len={args.max_model_len}")
    if args.max_model_len <= EVAL_MAX_TOKENS:
        warnings.append(f"max_model_len={args.max_model_len} <= {EVAL_MAX_TOKENS}：eval.zip 会把精度阶段判为"
                        f"配置异常并跳过（见 docs/eval_工具审计.md），精度基线需要更大的上下文或调整评测口径")
    budget = payload["kv_budget"]
    if budget.get("fits") is False:
        warnings.append(f"KV Cache 容量 {budget['kv_capacity_tokens']} tokens 不足以支撑 "
                        f"max_num_seqs={args.max_num_seqs} × max_model_len={args.max_model_len}")

    if not args.json_only:
        header("模型目录")
        row("路径", path)
        row("config.json", "存在")
        row("模型类型", arch["model_type"])
        row("架构", ",".join(arch["architectures"] or []) if arch["architectures"] else None)
        row("层数", arch["num_hidden_layers"])
        row("全注意力层 / 混合", f"{arch['full_attention_layers']} / {arch['is_hybrid']}")
        row("hidden_size / heads / kv_heads / head_dim",
            f"{arch['hidden_size']} / {arch['num_attention_heads']} / {arch['num_key_value_heads']} / {arch['head_dim']}")
        row("max_position_embeddings", arch["max_position_embeddings"])
        row("dtype / 量化", f"{arch['torch_dtype']} / {payload['weights']['quantization']}")
        row("MoE / MLA / MTP 层", f"{arch['is_moe']} / {arch['is_mla']} / {arch['mtp_num_hidden_layers']}")
        header("权重")
        row("文件数", payload["weights"]["files"])
        row("总大小", f"{weights_gib} GiB")
        row("分片索引", payload["weights"]["has_index"])
        header("Tokenizer")
        for name, present in payload["tokenizer"]["expected"].items():
            row(name, "存在" if present else "缺失")
        header("显存预算（估算）")
        row("显存 / 利用率", f"{args.vram_gib} GiB / {args.gpu_memory_utilization}")
        row("权重", f"{weights_gib} GiB")
        for key in ("kv_per_token_kib", "usable_for_kv_gib", "kv_capacity_tokens",
                    "kv_capacity_full_length_requests", "tokens_needed_for_max_num_seqs", "fits"):
            if key in budget:
                row(key, budget[key])
        if budget.get("note"):
            row("说明", budget["note"])

    saved = save(args.out_dir, "model", payload)
    for item in warnings:
        print(f"[告警] {item}")
    if not warnings:
        print("模型目录检查通过")
    print(f"已记录：{saved}")
    return 2 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
