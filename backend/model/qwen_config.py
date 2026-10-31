"""读取模型自身配置，产出"这个模型到底长什么样"的事实。

用途：显存预算、KV 计算、以及回答"为什么是 FP8/为什么上下文能开多大"。
只读 config.json，不加载权重。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelFacts:
    path: str
    model_type: str | None
    architectures: list[str]
    num_hidden_layers: int | None
    full_attention_layers: int | None
    full_attention_interval: int | None
    is_hybrid: bool
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    max_position_embeddings: int | None
    vocab_size: int | None
    quantization: str | None
    mtp_num_hidden_layers: int | None
    is_multimodal: bool
    weight_files: int
    weights_gib: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _text_config(config: dict) -> dict:
    return config.get("text_config", config)


def read_config(model_path: Path) -> dict:
    return json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8-sig"))


def facts(model_path: Path) -> ModelFacts:
    model_path = Path(model_path)
    config = read_config(model_path)
    text = _text_config(config)
    layers = text.get("num_hidden_layers")
    layer_types = text.get("layer_types") or []
    interval = text.get("full_attention_interval")
    if layer_types:
        full = sum(1 for item in layer_types
                   if str(item) in {"full_attention", "full", "full_attn", "attention"})
    elif interval and layers:
        full = max(1, (int(layers) + int(interval) - 1) // int(interval))
    else:
        full = layers

    weights = [item for item in model_path.rglob("*")
               if item.is_file() and item.suffix in {".safetensors", ".bin"}]
    total_bytes = sum(item.stat().st_size for item in weights)
    heads = text.get("num_attention_heads")
    hidden = text.get("hidden_size")
    quant = (config.get("quantization_config") or text.get("quantization_config") or {})

    return ModelFacts(
        path=str(model_path),
        model_type=text.get("model_type") or config.get("model_type"),
        architectures=text.get("architectures") or config.get("architectures") or [],
        num_hidden_layers=layers,
        full_attention_layers=full,
        full_attention_interval=interval,
        is_hybrid=bool(full and layers and full < int(layers)),
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=text.get("num_key_value_heads", heads),
        head_dim=text.get("head_dim") or (int(hidden) // int(heads) if hidden and heads else None),
        max_position_embeddings=text.get("max_position_embeddings"),
        vocab_size=text.get("vocab_size"),
        quantization=quant.get("quant_method"),
        mtp_num_hidden_layers=text.get("mtp_num_hidden_layers"),
        is_multimodal=bool(config.get("vision_config")),
        weight_files=len(weights),
        weights_gib=round(total_bytes / 1024 ** 3, 2) if total_bytes else None,
    )


def tokenizer_ready(model_path: Path) -> dict:
    model_path = Path(model_path)
    expected = ("tokenizer_config.json", "tokenizer.json", "vocab.json",
                "merges.txt", "special_tokens_map.json")
    present = {name: (model_path / name).is_file() for name in expected}
    return {"files": present,
            "usable": bool(present["tokenizer_config.json"] and
                           (present["tokenizer.json"] or present["vocab.json"]))}


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="读取模型配置事实")
    parser.add_argument("--model-path",
                        default=os.environ.get("QWEN_MODEL_PATH",
                                               "/model/ModelScope/Qwen/Qwen3.8-27B-FP8"))
    args = parser.parse_args()
    print(json.dumps({"facts": facts(Path(args.model_path)).to_dict(),
                      "tokenizer": tokenizer_ready(Path(args.model_path))},
                     ensure_ascii=False, indent=2))
