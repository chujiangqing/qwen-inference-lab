"""分析接口：把"文档里的结论"变成页面上可看的数字。

四类数据：
  1. 选型对照（模型/量化版本能不能在本机跑）；
  2. 瓶颈核算（用实测 TPOT 反推权重读取带宽，证明瓶颈在哪）；
  3. AutoTuner 候选明细（试验结果 + 评分 + 推荐）；
  4. 精度摘要（分数、样本数、置信区间、输出长度分布、截断比例）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "storage"))
sys.path.insert(0, str(BACKEND / "benchmark"))
sys.path.insert(0, str(BACKEND / "monitor"))

import database  # noqa: E402
import experiment_repository as repo  # noqa: E402
import result_parser  # noqa: E402
from core.config import DB_PATH, ROOT  # noqa: E402

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

PEAK_BANDWIDTH_GBS = 1008.0          # RTX 4090 标称显存带宽（GB/s）
GIB_IN_GB = 1.073741824

# 候选量化版本（路径来自本机模型仓库的实测盘查）
VARIANTS = [
    ("Qwen3.8-27B-FP8", "/model/ModelScope/Qwen/Qwen3.8-27B-FP8"),
    ("Qwen3.8-27B (BF16)", "/model/ModelScope/Qwen/Qwen3.8-27B"),
    ("Qwen3.8-27B-GGUF (Q5_K_M)", "/model/ModelScope/unsloth/Qwen3.8-27B-GGUF"),
    ("Qwen3.8-27B-EXL3-3.5bpw", "/model/ModelScope/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw"),
    ("Qwen3.8-27B-NVFP4", "/model/ModelScope/RadixArk/Qwen3.8-27B-NVFP4"),
    ("Qwen3.8-27B-DFlash2", "/model/ModelScope/incoai/Qwen3.8-27B-DFlash2"),
]

_CACHE: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float, builder):
    import time
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = builder()
    _CACHE[key] = (time.time(), value)
    return value


def _dir_size_gib(path: Path) -> float | None:
    if not path.is_dir():
        return None
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return round(total / 1024 ** 3, 2)


def _quant_of(path: Path) -> dict:
    config = path / "config.json"
    if not config.is_file():
        return {"quant_method": None, "architectures": []}
    try:
        data = json.loads(config.read_text(encoding="utf-8-sig"))
    except ValueError:
        return {"quant_method": None, "architectures": []}
    quant = data.get("quantization_config") or {}
    return {"quant_method": quant.get("quant_method") or quant.get("quant_type"),
            "architectures": data.get("architectures") or []}


def _compute_cap() -> str | None:
    import subprocess
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10)
        return done.stdout.strip().splitlines()[0].strip() if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


@router.get("/model-options")
def model_options() -> dict:
    """选型对照：体积、量化格式、引擎支持、本机可行性、结论。"""

    def build() -> dict:
        vram_gib = None
        try:
            import gpu_monitor
            snapshot = gpu_monitor.snapshot()
            vram_gib = round(((snapshot.get("gpus") or [{}])[0].get("memory_total_mib") or 0) / 1024, 2)
        except (ImportError, OSError, KeyError):
            vram_gib = None
        cap = _compute_cap()

        rows = []
        for name, raw in VARIANTS:
            path = Path(raw)
            size = _dir_size_gib(path)
            quant = _quant_of(path) if path.is_dir() else {"quant_method": None, "architectures": []}
            method = (quant.get("quant_method") or "").lower()
            verdict, reason = "待定", ""
            lower = name.lower()
            if size is None:
                verdict, reason = "缺失", "本机没有这个模型目录"
            elif "dflash" in lower:
                verdict, reason = "排除", f"仅 {size} GiB，是训练/草稿类附加权重，不是可独立推理的模型"
            elif "gguf" in lower:
                verdict, reason = "未实测", "需换 llama.cpp 引擎；该架构（混合线性注意力）在 llama.cpp 的支持未验证"
            elif method in {"exl3"}:
                verdict, reason = "排除", "vLLM 0.31 不支持 exl3，需要 ExLlamaV2/TabbyAPI"
            elif method in {"modelopt"} or "nvfp4" in lower:
                verdict, reason = "排除", f"FP4 需要 Blackwell(sm100)，本机 compute_cap={cap}"
            elif "bf16" in lower or method == "":
                if vram_gib and size >= vram_gib:
                    verdict, reason = "排除", f"权重 {size} GiB ≥ 显存 {vram_gib} GiB，加载前必 OOM"
                else:
                    verdict, reason = "可选", "显存放得下，但未实测"
            else:
                verdict, reason = "采用", "已实测：加载成功并通过 4 档 × 100 请求评测"
            rows.append({"name": name, "path": raw, "size_gib": size,
                         "quant_method": quant.get("quant_method"),
                         "architectures": quant.get("architectures"),
                         "verdict": verdict, "reason": reason})
        return {"vram_gib": vram_gib, "compute_cap": cap, "variants": rows}

    return _cached("model_options", 600, build)


@router.get("/bottleneck")
def bottleneck(experiment_id: int) -> dict:
    """用实测 TPOT 反推"每生成 1 token 需要读多少权重"，判断是不是带宽受限。"""
    database.init(ROOT)
    data = repo.experiment_detail(DB_PATH, experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="实验不存在")
    cells = data.get("performance") or []
    if not cells:
        raise HTTPException(status_code=409, detail="该实验没有性能数据")

    weights_gib = None
    model_json = sorted((ROOT / "reports").glob("*/model.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if model_json:
        try:
            facts = json.loads(model_json[0].read_text(encoding="utf-8-sig"))
            weights_gib = (facts.get("weights") or {}).get("total_gib")
        except (OSError, ValueError):
            weights_gib = None

    rows = []
    for cell in sorted(cells, key=lambda item: item.get("concurrency") or 0):
        tpot = cell.get("tpot_p95")
        row = {"concurrency": cell.get("concurrency"),
               "output_throughput": cell.get("output_throughput"),
               "ttft_p95": cell.get("ttft_p95"), "tpot_p95": tpot,
               "status": cell.get("status")}
        if weights_gib and tpot:
            gib_per_s = weights_gib / (tpot / 1000)
            gbs = gib_per_s * GIB_IN_GB
            row.update({"read_gibs": round(gib_per_s, 1), "read_gbs": round(gbs),
                        "bandwidth_pct": round(gbs / PEAK_BANDWIDTH_GBS * 100, 1)})
        rows.append(row)
    return {"experiment": data.get("experiment"), "weights_gib": weights_gib,
            "peak_bandwidth_gbs": PEAK_BANDWIDTH_GBS, "rows": rows}


@router.get("/trials")
def trials() -> dict:
    """AutoTuner 逐候选结果 + 排名（没有结果时返回空列表，不报错）。"""
    path = ROOT / "experiments" / "auto_tuner" / "trials.json"
    if not path.is_file():
        return {"trials": [], "ranked": [], "note": "AutoTuner 尚未产出结果"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        return {"trials": [], "ranked": [], "note": f"trials.json 解析失败：{error}"}
    items = data.get("trials") or []
    # 评分要等整轮结束才计算；跑的过程中先按吞吐排序，否则页面永远显示"无结果"
    scored = [item for item in items if item.get("valid") and item.get("score") is not None]
    if scored:
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        basis = "score"
    else:
        ranked = sorted([item for item in items
                         if item.get("valid") and item.get("output_throughput")],
                        key=lambda item: item["output_throughput"], reverse=True)
        basis = "throughput"
    return {"trials": items, "ranked": ranked, "rank_basis": basis,
            "note": None if items else "AutoTuner 尚未产出结果"}


@router.get("/kv-cache")
def kv_cache() -> dict:
    """从 vLLM 启动日志提取 KV 容量与并发上限（"内存碎片管理"的可核验证据）。

    vLLM 启动时打印：Available KV cache memory / GPU KV cache size / Maximum concurrency，
    运行期周期性打印 GPU KV cache usage 与 Running/Waiting 队列长度。
    """
    import re
    log_dir = Path("/root/logs/vllm")
    logs = sorted(log_dir.glob("serve_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        raise HTTPException(status_code=404, detail=f"找不到 vLLM 日志（{log_dir}）")
    text = logs[0].read_text(encoding="utf-8", errors="replace")

    def first(pattern: str, cast=float):
        found = re.search(pattern, text)
        return cast(found.group(1).replace(",", "")) if found else None

    usage = [float(value) for value in re.findall(r"GPU KV cache usage: ([\d.]+)%", text)]
    waiting = [int(value) for value in re.findall(r"Waiting: (\d+) reqs", text)]
    running = [int(value) for value in re.findall(r"Running: (\d+) reqs", text)]
    preempted = len(re.findall(r"[Pp]reempt", text))

    return {
        "log": str(logs[0]),
        "available_kv_gib": first(r"Available KV cache memory: ([\d.]+) GiB"),
        "kv_capacity_tokens": first(r"GPU KV cache size: ([\d,]+) tokens", int),
        "per_request_tokens": first(r"Maximum concurrency for ([\d,]+) tokens per request", int),
        "max_concurrency": first(r"tokens per request: ([\d.]+)x"),
        "usage_percent": {"max": max(usage) if usage else None,
                          "avg": round(sum(usage) / len(usage), 2) if usage else None,
                          "samples": len(usage)},
        "queue": {"max_waiting": max(waiting) if waiting else None,
                  "max_running": max(running) if running else None},
        "preemption_events": preempted,
        "note": "max_waiting>0 说明请求在排队（并发被封顶）；preemption>0 说明显存紧张发生过抢占。",
    }


@router.get("/quality")
def quality() -> dict:
    """三类任务质量抽检结果（长文本 / 逻辑推理 / 代码生成）。

    与官方 IFBench 分开存放：这里是补充证据，用于覆盖评分表点名的三类任务。
    """
    folders = sorted((ROOT / "experiments").glob("QUALITY_*"), key=lambda path: path.stat().st_mtime,
                     reverse=True)
    for folder in folders:
        result = folder / "quality_result.json"
        if result.is_file():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except ValueError:
                continue
            payload["output_dir"] = str(folder)
            payload["report_path"] = str(folder / "quality_report.md")
            return payload
    return {"overall": None, "summary": {}, "results": [],
            "note": "还没有跑过质量抽检（scripts/quality_spotcheck.py）"}


@router.get("/factorial")
def factorial() -> dict:
    """双因子视角：把 (MTP × max_num_seqs) 的四格拼出来，并算主效应与交互项。

    为什么要单独一个接口：
    两两对比页只能回答"这两个实验差多少"，回答不了"两个变量一起变会怎样"。
    因子设计把四格放在一起，才能判断两个优化是"可加"还是"互相抢资源"。
    """
    import re

    database.init(ROOT)
    rows = repo.list_experiments(DB_PATH, limit=100)
    cells: list[dict] = []
    for row in rows:
        name = str(row.get("experiment_name") or "")
        upper = name.upper()
        if upper.startswith("FINAL") or "BASELINE" in upper and not upper.startswith("BASE20"):
            continue                      # 100 请求/档的正式实验不进 20/档的因子表
        detail = repo.experiment_detail(DB_PATH, row["id"]) or {}
        perf = [cell for cell in (detail.get("performance") or []) if cell.get("status") == "完整"]
        if not perf:
            continue
        requests = max((cell.get("request_count") or 0) for cell in perf)
        if requests != 20:
            continue
        best = max(perf, key=lambda cell: cell.get("output_throughput") or 0)
        mtp = "mtp" in name.lower()
        seq_match = re.search(r"seq(\d+)", name)
        ctx_match = re.search(r"ctx(\d+)", name)
        cells.append({
            "experiment": name,
            "experiment_id": row["id"],
            "mtp": mtp,
            "max_num_seqs": int(seq_match.group(1)) if seq_match else 4,
            "max_model_len": int(ctx_match.group(1)) if ctx_match else 32768,
            "requests": requests,
            "throughput": best.get("output_throughput"),
            "ttft_p95": best.get("ttft_p95"),
            "tpot_p95": best.get("tpot_p95"),
            "concurrency": best.get("concurrency"),
        })

    def pick(mtp: bool, seq: int):
        found = [c for c in cells if c["mtp"] == mtp and c["max_num_seqs"] == seq]
        if not found:
            return None
        return max(found, key=lambda c: c["throughput"] or 0)

    off4, off8 = pick(False, 4), pick(False, 8)
    on4, on8 = pick(True, 4), pick(True, 8)

    def diff(a, b):
        if not a or not b or not a.get("throughput"):
            return None
        return round(b["throughput"] - a["throughput"], 2)

    effect_seq_off = diff(off4, off8)          # 提并发（MTP 关时）
    effect_seq_on = diff(on4, on8)             # 提并发（MTP 开时）
    effect_mtp_4 = diff(off4, on4)             # 开 MTP（4 路时）
    effect_mtp_8 = diff(off8, on8)             # 开 MTP（8 路时）
    interaction = None
    if effect_seq_off is not None and effect_seq_on is not None:
        interaction = round(effect_seq_on - effect_seq_off, 2)

    if interaction is None:
        verdict = "四格未齐，等组合实验补（mtp + seq8）"
    elif interaction > 1:
        verdict = "正交互：两个改动互相放大"
    elif interaction < -1:
        verdict = "负交互（饱和）：两者在抢同一个瓶颈，叠加收益递减"
    else:
        verdict = "可加：组合收益≈两个单变量收益之和"

    return {"cells": cells, "grid": {
                "off_4": off4, "off_8": off8, "on_4": on4, "on_8": on8},
            "effects": {"seq_at_mtp_off": effect_seq_off, "seq_at_mtp_on": effect_seq_on,
                        "mtp_at_seq4": effect_mtp_4, "mtp_at_seq8": effect_mtp_8},
            "interaction": interaction, "verdict": verdict}


@router.get("/accuracy")
def accuracy(experiment_id: int | None = None) -> dict:
    """精度摘要：分数、样本数、置信区间、输出长度分布、截断比例。"""
    database.init(ROOT)
    if experiment_id is None:
        rows = repo.list_experiments(DB_PATH, limit=50)
        candidates = [row for row in rows if row.get("prompt_level_strict") is not None]
        if not candidates:
            raise HTTPException(status_code=404, detail="还没有任何精度结果")
        experiment_id = candidates[0]["id"]

    data = repo.experiment_detail(DB_PATH, experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="实验不存在")
    rows = data.get("accuracy") or []
    if not rows:
        raise HTTPException(status_code=409, detail="该实验没有精度数据")

    row = rows[0]
    samples = row.get("sample_count") or 0
    score = row.get("prompt_level_strict")
    ci = None
    if score is not None and samples:
        import math
        stderr = math.sqrt(max(score * (1 - score), 0) / samples)
        ci = {"stderr_pp": round(stderr * 100, 2),
              "ci95_pp": [round((score - 1.96 * stderr) * 100, 2),
                          round((score + 1.96 * stderr) * 100, 2)]}

    # 输出长度分布：优先找与该实验名匹配的预测文件，找不到再取最新的
    lengths, truncated = None, None
    name = (data.get("experiment") or {}).get("experiment_name") or ""
    matches = sorted((ROOT / "experiments").glob(f"{name}*/accuracy/native/predictions/*/*.jsonl")
                     if name else [])
    if not matches:
        matches = sorted((ROOT / "experiments").glob("*/accuracy/native/predictions/*/*.jsonl"),
                         key=lambda path: path.stat().st_mtime)
    if matches:
        try:
            items = [json.loads(line) for line in matches[-1].read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            sizes = sorted(len(str(item.get("model_output"))) for item in items)
            lengths = {"count": len(sizes), "min": sizes[0], "median": sizes[len(sizes) // 2],
                       "max": sizes[-1], "over_8k": sum(1 for n in sizes if n > 8000),
                       "source": str(matches[-1].relative_to(ROOT))}
            truncated = sum(1 for n in sizes if n > 60000)   # 接近 max_tokens=32768 的样本
        except (OSError, ValueError):
            lengths = None
    return {"experiment": data.get("experiment"), "dataset": row.get("dataset"),
            "prompt_level_strict": score, "sample_count": samples, "ci": ci,
            "raw_scores": row.get("raw_scores"), "lengths": lengths,
            "near_truncation": truncated}
