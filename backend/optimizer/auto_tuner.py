"""AutoTuner：网格搜索 → 逐候选起服务 → 跑官方评测 → 综合评分 → 推荐方案。

对应需求书 8.1（1–12 步）、8.3（评分公式与硬门槛）、8.4（输出字段）。
硬性约束：
  * 同一时刻只有一个评测（由 benchmark_manager 的文件锁保证）；
  * 每个候选都先停旧服务再起新服务，参数只通过 `--set` 覆盖，不手改配置文件；
  * OOM 或完整性异常直接判无效，不重试（需求书 原则 5）；
  * 质量未测的配置不得评为最优（需求书 8.3）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import search_space  # noqa: E402
from core.config import default_tokenizer  # noqa: E402
from monitor.gpu_monitor import PeakSampler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrialRecord:
    candidate: str
    dimension: str
    overrides: dict
    status: str = "pending"
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    output_dir: str | None = None
    # 指标
    output_throughput: float | None = None      # 四档中最高的输出 token/s
    request_throughput: float | None = None
    ttft_p95_ms: float | None = None            # 取吞吐最高那一档
    tpot_p95_ms: float | None = None
    e2e_p95_ms: float | None = None
    best_concurrency: int | None = None
    vram_peak_gib: float | None = None
    quality: float | None = None                # prompt_level_strict
    score: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    print("执行：" + " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def restart_service(overrides: dict, profile: str) -> tuple[bool, str]:
    run(["bash", str(ROOT / "scripts" / "stop_model.sh"), "--force"])
    time.sleep(8)                      # 等显存真正释放，否则启动前检查会判 GPU 忙
    cmd = ["bash", str(ROOT / "scripts" / "start_model.sh"), "--profile", profile]
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={json.dumps(value, separators=(',', ':'))}"]
    done = run(cmd, timeout=1800)
    if done.returncode != 0:           # 显存释放慢时重试一次，不掩盖真实错误
        time.sleep(10)
        run(["bash", str(ROOT / "scripts" / "stop_model.sh"), "--force"])
        time.sleep(5)
        done = run(cmd, timeout=1800)
    tail = (done.stdout or done.stderr or "").strip().splitlines()[-3:]
    if done.returncode != 0:
        return False, " | ".join(tail)
    # 启动成功还不够：必须确认参数真的进了 vLLM 命令行，否则实验等于没做
    ok, detail = verify_startup_arguments(overrides)
    return ok, detail or " | ".join(tail)


FLAG_OF = {
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "kv_cache_dtype": "--kv-cache-dtype",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "speculative_config": "--speculative-config",
}


def verify_startup_arguments(overrides: dict, log_dir: Path | None = None) -> tuple[bool, str]:
    """读最近一次启动记录，逐条确认覆盖参数出现在 vLLM 命令行里。"""
    if not overrides:
        return True, ""
    log_dir = log_dir or Path("/root/logs/vllm")
    records = sorted(log_dir.glob("launch_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not records:
        return False, f"找不到启动记录（{log_dir}/launch_*.json），无法确认参数生效"
    try:
        meta = json.loads(records[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return False, f"启动记录无法解析：{type(error).__name__}"
    argv = [str(item) for item in (meta.get("vllm_arguments") or [])]
    problems = []
    for key, value in overrides.items():
        flag = FLAG_OF.get(key)
        if not flag:
            problems.append(f"{key} 没有对应的命令行参数映射，无法核验")
            continue
        if flag not in argv:
            problems.append(f"{flag} 未出现在启动命令中")
            continue
        actual = argv[argv.index(flag) + 1] if argv.index(flag) + 1 < len(argv) else ""
        if isinstance(value, dict):
            try:
                same = json.loads(actual) == value
            except ValueError:
                same = False
        else:
            same = str(actual) == str(value)
        if not same:
            problems.append(f"{flag} 实际是 {actual!r}，期望 {value!r}")
    if problems:
        return False, "参数未生效：" + "；".join(problems)
    return True, f"参数已生效：{', '.join(FLAG_OF.get(k, k) for k in overrides)}"


def benchmark(space: dict, label: str, requests: int) -> tuple[dict | None, str]:
    """跑一次候选，返回 (benchmark_result, 输出的实验目录)。

    注意两点：
    1. benchmark_manager 会在 label 后面**再加一次时间戳**，所以不能按原名字直接找，
       必须 glob `experiments/<label>*` 再取最新的那个 benchmark_result.json；
    2. **必须显式传 --tokenizer 模型目录**：run_benchmark.sh 默认 builtin(tiktoken)，
       而基线是用模型 tokenizer 采的，不传就会拿两种输入口径做对比。
    """
    perf = space.get("perf") or {}
    cmd = ["bash", str(ROOT / "scripts" / "run_benchmark.sh"), "--label", label, "--mode", "perf",
           "--perf-requests", str(requests),
           "--request-timeout", str(perf.get("request_timeout", 600)),
           "--tokenizer", default_tokenizer()]
    done = run(cmd, timeout=None)
    candidates = sorted((ROOT / "experiments").glob(f"{label}*/benchmark_result.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None, (done.stderr or done.stdout or "")[-800:]
    newest = candidates[0]
    return json.loads(newest.read_text(encoding="utf-8")), str(newest.parent)


def extract_metrics(result: dict) -> dict:
    cells = (result.get("perf") or {}).get("cells") or []
    best = None
    for cell in cells:
        if cell.get("status") != "完整":
            continue
        tps = cell.get("output_throughput") or 0
        if best is None or tps > (best.get("output_throughput") or 0):
            best = cell
    if best is None:
        return {"status": "异常", "reasons": ["没有任何一档是完整状态"]}
    return {"status": "完成",
            "output_throughput": best.get("output_throughput"),
            "request_throughput": best.get("request_throughput"),
            "ttft_p95_ms": best.get("ttft_p95"),
            "tpot_p95_ms": best.get("tpot_p95"),
            "e2e_p95_ms": best.get("e2e_p95"),
            "best_concurrency": best.get("concurrency")}


def normalise(records: list[TrialRecord], space: dict) -> None:
    """按需求书 8.3 打分：指标先归一化，质量未测不得评优。"""
    scoring = space.get("scoring") or {}
    valid = [item for item in records if item.valid and item.status == "完成"]
    if not valid:
        return
    tps = [item.output_throughput or 0 for item in valid]
    lat = [item.e2e_p95_ms or 0 for item in valid]
    vram = [item.vram_peak_gib or 0 for item in valid]
    tps_max, lat_max, vram_max = max(tps) or 1, max(lat) or 1, max(vram) or 1
    alpha = float(scoring.get("alpha_throughput", 0.45))
    beta = float(scoring.get("beta_latency", 0.35))
    gamma = float(scoring.get("gamma_quality", 0.15))
    delta = float(scoring.get("delta_memory", 0.05))
    threshold = float(scoring.get("quality_threshold", 0.70))
    for item in valid:
        throughput_score = (item.output_throughput or 0) / tps_max
        latency_score = 1 - (item.e2e_p95_ms or 0) / lat_max
        memory_score = 1 - (item.vram_peak_gib or 0) / vram_max
        quality = item.quality if item.quality is not None else None
        if quality is None:
            # 未测质量：只用于排序参考，且显式降权，不允许被评为最优
            item.score = round(alpha * throughput_score + beta * latency_score + delta * memory_score, 6)
            item.reasons.append("未测 IFBench 质量，分数不含 γ 项，不得作为最终最优配置")
        elif quality < threshold:
            item.valid = False
            item.score = None
            item.reasons.append(f"质量 {quality:.4f} 低于门槛 {threshold}")
        else:
            item.score = round(alpha * throughput_score + beta * latency_score
                               + gamma * quality + delta * memory_score, 6)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="AutoTuner 网格搜索")
    parser.add_argument("--space", type=Path, default=ROOT / "configs" / "search_space.yaml")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "task0_verify")
    parser.add_argument("--requests", type=int, default=None, help="覆盖 requests_per_candidate")
    parser.add_argument("--dimension", action="append", default=[], help="只跑指定维度，可重复")
    parser.add_argument("--include", action="append", default=[], help="只跑名字包含该子串的候选")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几个候选（0=不限）")
    parser.add_argument("--budget-minutes", type=float, default=0, help="超过预算不再启动新候选")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果的候选")
    args = parser.parse_args()

    space = search_space.load_space(args.space)
    facts = search_space.load_model_facts(args.reports)
    profile = space.get("baseline_profile", "baseline")
    requests = args.requests or int((space.get("perf") or {}).get("requests_per_candidate", 20))

    candidates = [item for item in search_space.build_candidates(space, facts) if item.valid]
    if args.dimension:
        candidates = [item for item in candidates if item.dimension in args.dimension]
    if args.include:
        candidates = [item for item in candidates
                      if any(token in item.name for token in args.include)]
    if args.limit:
        candidates = candidates[:args.limit]

    state_path = ROOT / "experiments" / "auto_tuner" / "trials.json"
    history: list[TrialRecord] = []
    if args.resume and state_path.is_file():
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        history = [TrialRecord(**item) for item in raw.get("trials", [])]
    done_names = {item.candidate for item in history}

    print(f"实测事实：{json.dumps(facts, ensure_ascii=False)}")
    print(f"候选 {len(candidates)} 个，每候选 {requests} 请求/档")
    if args.dry_run:
        for item in candidates:
            print(f"  {item.name:<34} {json.dumps(item.overrides, ensure_ascii=False)}")
        return 0

    started = time.time()
    for candidate in candidates:
        if candidate.name in done_names:
            print(f"跳过（已有结果）：{candidate.name}")
            continue
        if args.budget_minutes and (time.time() - started) / 60 > args.budget_minutes:
            print(f"已达预算 {args.budget_minutes} 分钟，停止启动新候选")
            break

        record = TrialRecord(candidate=candidate.name, dimension=candidate.dimension,
                             overrides=candidate.overrides, started_at=_now())
        print(f"\n=== {candidate.name} ===")
        ok, note = restart_service(candidate.overrides, profile)
        if not ok:
            record.status = "启动/校验失败"
            record.valid = False
            record.reasons.append(note)
            record.ended_at = _now()
            history.append(record)
            _save(state_path, history, space)
            continue

        label = f"AUTOTUNER_{candidate.name}_{time.strftime('%H%M%S')}"
        with PeakSampler() as sampler:
            t0 = time.time()
            result, output_dir = benchmark(space, label, requests)
            record.duration_seconds = round(time.time() - t0, 1)
        record.vram_peak_gib = round(sampler.vram_peak_gib, 2)
        if result is None:
            record.status = "评测失败"
            record.valid = False
            record.reasons.append(output_dir or "没有生成 benchmark_result.json")
        else:
            metrics = extract_metrics(result)
            record.output_dir = output_dir          # 入库已由 benchmark_manager._persist 负责
            for key, value in metrics.items():
                if key not in ("status", "reasons", "output_dir"):
                    setattr(record, key, value)
            record.status = metrics.get("status", "未知")
            record.reasons += metrics.get("reasons", [])
            if record.status != "完成":
                record.valid = False
        record.ended_at = _now()
        history.append(record)
        _save(state_path, history, space)
        print(f"结果：状态={record.status} 输出吞吐={record.output_throughput} tok/s "
              f"峰值显存={record.vram_peak_gib} GiB 用时={record.duration_seconds}s")

    normalise(history, space)
    _save(state_path, history, space)
    _write_markdown(state_path.parent / "report.md", history, space, facts)
    ranked = sorted([item for item in history if item.valid and item.score is not None],
                    key=lambda item: item.score, reverse=True)
    print("\n=== 排名 ===")
    for rank, item in enumerate(ranked[:10], 1):
        print(f"{rank:>2}. {item.candidate:<34} score={item.score:.4f} "
              f"tps={item.output_throughput} e2e_p95={item.e2e_p95_ms} vram={item.vram_peak_gib}")
    return 0


def _now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _save(path: Path, records: list[TrialRecord], space: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _now(), "scoring": space.get("scoring"),
               "trials": [item.to_dict() for item in records]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, records: list[TrialRecord], space: dict, facts: dict) -> None:
    ranked = sorted([item for item in records if item.score is not None],
                    key=lambda item: item.score, reverse=True)
    lines = ["# AutoTuner 结果", "",
             f"- 生成时间：{_now()}",
             f"- 实测事实：`{json.dumps(facts, ensure_ascii=False)}`",
             f"- 评分权重：`{json.dumps(space.get('scoring'), ensure_ascii=False)}`", "",
             "| 排名 | 候选 | 维度 | 输出 tok/s | TTFT P95 | TPOT P95 | E2E P95 | 峰值显存 | 质量 | 评分 | 状态 |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for rank, item in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {item.candidate} | {item.dimension} | {_num(item.output_throughput, 2)} | "
            f"{_num(item.ttft_p95_ms, 0, 1)} | {_num(item.tpot_p95_ms, 2)} | {_num(item.e2e_p95_ms, 0, 1)} | "
            f"{_num(item.vram_peak_gib, 2)} | {_num(item.quality, 4)} | {_num(item.score, 4)} | {item.status} |")
    invalid = [item for item in records if not item.valid]
    if invalid:
        lines += ["", "## 无效配置（保留记录，不删除）", ""]
        for item in invalid:
            lines.append(f"- `{item.candidate}`（{item.status}）：" + "；".join(item.reasons or ["未说明"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _num(value, digits: int, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        return format(float(value), f".{digits}f")
    except (TypeError, ValueError):
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())
