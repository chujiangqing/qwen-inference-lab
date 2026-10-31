"""资源管理：GPU 是否空闲、显存够不够、端口占没占。

启动服务前必须过这三关（需求书 原则 5：避免并发跑、避免 OOM 后无脑重试）。
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import runtime_config  # noqa: E402
from model.memory_estimator import budget  # noqa: E402
from model.qwen_config import facts as model_facts  # noqa: E402
from monitor.gpu_monitor import snapshot  # noqa: E402

MIN_FREE_GIB = 30.0      # 启动前至少剩余多少显存（权重 28.75 GiB + 激活）


def gpu_state() -> dict:
    data = snapshot()
    if not data.get("available"):
        return {"available": False, "error": data.get("error")}
    gpu = data["gpus"][0]
    free_gib = (gpu.get("memory_free_mib") or 0) / 1024
    used_gib = (gpu.get("memory_used_mib") or 0) / 1024
    return {"available": True, "name": gpu.get("name"),
            "total_gib": round((gpu.get("memory_total_mib") or 0) / 1024, 2),
            "used_gib": round(used_gib, 2), "free_gib": round(free_gib, 2),
            "utilization": gpu.get("utilization"),
            "has_processes": used_gib > 1.0}


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def can_start(profile: str = "baseline") -> dict:
    """开跑前的三道检查：GPU 空闲、显存足够、端口没被占。"""
    config = runtime_config()
    port = int(config["server"]["port"])
    model_path = Path(config["model"]["path"])
    state = gpu_state()
    problems: list[str] = []

    if not state.get("available"):
        problems.append(f"GPU 不可用：{state.get('error')}")
    elif state["used_gib"] > 1.0:
        problems.append(f"GPU 已被占用 {state['used_gib']} GiB，评测必须独占")
    elif state["free_gib"] < MIN_FREE_GIB:
        problems.append(f"可用显存 {state['free_gib']} GiB 低于安全线 {MIN_FREE_GIB} GiB")

    if not port_free(port):
        problems.append(f"端口 {port} 已被占用")

    if not (model_path / "config.json").is_file():
        problems.append(f"模型目录缺少 config.json：{model_path}")

    estimate = None
    if (model_path / "config.json").is_file():
        profile_conf = (config.get("profiles") or {}).get(profile, {})
        model = model_facts(model_path)
        if model.full_attention_layers and model.num_key_value_heads and model.head_dim:
            per_token = 2 * model.full_attention_layers * model.num_key_value_heads * model.head_dim
            estimate = budget(vram_gib=state.get("total_gib") or 48.0,
                              weights_gib=model.weights_gib or 0.0,
                              kv_per_token_kib=per_token / 1024,
                              gpu_memory_utilization=float(profile_conf.get("gpu_memory_utilization", 0.9)),
                              max_model_len=int(profile_conf.get("max_model_len", 32768)),
                              max_num_seqs=int(profile_conf.get("max_num_seqs", 4)))
            if not estimate["fits"]:
                problems.append(f"该 profile 的 KV 预算不足：需要 {estimate['tokens_needed']} token，"
                                f"只能容纳 {estimate['kv_capacity_tokens']}")

    return {"ok": not problems, "problems": problems, "gpu": state,
            "port": port, "port_free": port_free(port), "kv_budget": estimate}


if __name__ == "__main__":
    import json
    print(json.dumps(can_start(), ensure_ascii=False, indent=2))
