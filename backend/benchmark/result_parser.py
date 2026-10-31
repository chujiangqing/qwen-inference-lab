"""读取 eval.zip 原生产物并归一化。

只读解析，不改动评测工具：所有数字都来自
`<output>/summary.json`、`performance/*/profile_export_aiperf.json|jsonl`、
`accuracy/native/**`，并保留原始路径与哈希，供数据库索引与答辩追溯。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerfCell:
    concurrency: int
    status: str
    issues: list[str] = field(default_factory=list)
    requests: int | None = None
    success: int | None = None
    errors: int | None = None
    ttft_p50: float | None = None
    ttft_p95: float | None = None
    tpot_p50: float | None = None
    tpot_p95: float | None = None
    e2e_p50: float | None = None
    e2e_p95: float | None = None
    request_throughput: float | None = None
    output_throughput: float | None = None
    benchmark_duration: float | None = None
    actual_input_tokens: dict | None = None
    artifacts: str | None = None


@dataclass
class AccuracyResult:
    status: str
    issues: list[str] = field(default_factory=list)
    prompt_level_strict: float | None = None
    inst_level_strict: float | None = None
    prompt_level_loose: float | None = None
    inst_level_loose: float | None = None
    samples: int | None = None
    execution_summary: dict | None = None
    native_report: str | None = None
    artifacts: str | None = None


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _avg(value) -> float | None:
    """AIPerf 导出既可能是 {"avg": x}，也可能是标量；两种都接受，其它一律当缺失。"""
    if isinstance(value, dict):
        candidate = value.get("avg")
        return candidate if isinstance(candidate, (int, float)) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(summary_cell: dict, tag: str) -> tuple[float | None, float | None]:
    metrics = _dict(summary_cell.get("metrics"))
    entry = _dict(metrics.get(tag))
    return entry.get("p50"), entry.get("p95")


# --------------------------------------------------------------------------- parsers
def parse_summary(output_dir: Path) -> dict:
    """读取官方 runner 生成的 summary.json / configuration.json / environment.json。"""
    output_dir = Path(output_dir)
    summary = {"output_dir": str(output_dir), "files": {}, "status": None, "issues": [],
               "started_at": None, "ended_at": None, "configuration": {}, "performance": [],
               "accuracy": None, "environment": {}}
    for name in ("summary.json", "configuration.json", "environment.json", "model_info.json", "report.md"):
        path = output_dir / name
        if path.is_file():
            summary["files"][name] = {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
    if "summary.json" in summary["files"]:
        data = load_json(output_dir / "summary.json")
        summary["status"] = data.get("status")
        summary["issues"] = data.get("issues") or []
        summary["started_at"] = data.get("started_at")
        summary["ended_at"] = data.get("ended_at")
        summary["configuration"] = data.get("configuration") or {}
    if "environment.json" in summary["files"]:
        summary["environment"] = load_json(output_dir / "environment.json")
    return summary


def parse_performance(output_dir: Path) -> list[PerfCell]:
    """性能：优先用官方逐档 summary.json（含完整性判定），缺失时回退读原生 AIPerf 导出。"""
    folder = Path(output_dir) / "performance"
    cells: list[PerfCell] = []
    if not folder.is_dir():
        return cells
    for child in sorted(folder.glob("c*"), key=lambda p: int(p.name.lstrip("c") or 0)):
        try:
            concurrency = int(child.name.lstrip("c"))
        except ValueError:
            continue
        summary_path = child / "summary.json"
        if summary_path.is_file():
            data = load_json(summary_path)
            cells.append(PerfCell(
                concurrency=concurrency,
                status=data.get("status", "未知"),
                issues=list(data.get("issues") or []),
                requests=data.get("records"),
                success=data.get("successful_requests"),
                errors=data.get("error_requests"),
                ttft_p50=_metrics(data, "time_to_first_token")[0],
                ttft_p95=_metrics(data, "time_to_first_token")[1],
                tpot_p50=_metrics(data, "inter_token_latency")[0],
                tpot_p95=_metrics(data, "inter_token_latency")[1],
                e2e_p50=_metrics(data, "request_latency")[0],
                e2e_p95=_metrics(data, "request_latency")[1],
                request_throughput=data.get("request_throughput"),
                output_throughput=data.get("output_token_throughput"),
                benchmark_duration=data.get("benchmark_duration"),
                actual_input_tokens=data.get("actual_input_tokens"),
                artifacts=str(child)))
            continue
        # 回退：只有原生导出时自己算（官方没有给出完整性判定，标记为“未核验”）
        native_path = child / "profile_export_aiperf.json"
        if not native_path.is_file():
            cells.append(PerfCell(concurrency=concurrency, status="缺失",
                                  issues=["未找到 summary.json 与 profile_export_aiperf.json"],
                                  artifacts=str(child)))
            continue
        native = load_json(native_path)
        cell = PerfCell(concurrency=concurrency, status="未核验",
                        issues=["缺少官方 summary.json，指标由原生导出直接读取"],
                        requests=_avg(native.get("request_count")),
                        success=_avg(native.get("request_count")),
                        errors=_avg(native.get("error_request_count")),
                        request_throughput=_avg(native.get("request_throughput")),
                        output_throughput=_avg(native.get("output_token_throughput")),
                        benchmark_duration=_avg(native.get("benchmark_duration")),
                        artifacts=str(child))
        for tag, prefix in (("time_to_first_token", "ttft"), ("inter_token_latency", "tpot"),
                            ("request_latency", "e2e")):
            entry = native.get(tag) or {}
            setattr(cell, f"{prefix}_p50", entry.get("p50"))
            setattr(cell, f"{prefix}_p95", entry.get("p95"))
        cells.append(cell)
    return cells


def parse_accuracy(output_dir: Path) -> AccuracyResult | None:
    folder = Path(output_dir) / "accuracy"
    if not folder.is_dir():
        return None
    summary_path = folder / "summary.json"
    if not summary_path.is_file():
        return AccuracyResult(status="缺失", issues=["accuracy/summary.json 不存在"], artifacts=str(folder))
    data = load_json(summary_path)
    metrics = data.get("metrics") or {}
    result = AccuracyResult(
        status=data.get("status", "未知"),
        issues=list(data.get("issues") or []),
        samples=data.get("generated_samples"),
        execution_summary=data.get("execution_summary"),
        native_report=data.get("native_report"),
        artifacts=str(folder))
    for name in ("prompt_level_strict", "inst_level_strict", "prompt_level_loose", "inst_level_loose"):
        entry = metrics.get(name) or {}
        setattr(result, name, entry.get("score"))
    return result


def collect_artifacts(output_dir: Path) -> list[dict]:
    """登记全部原始文件（供 experiment_artifacts 表索引）。"""
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append({"type": path.suffix.lstrip(".") or "file",
                         "path": str(path),
                         "relative": str(path.relative_to(output_dir)),
                         "size": path.stat().st_size})
    return rows


def parse_experiment_dir(folder: Path) -> dict:
    """把"官方 eval/run.sh 产物目录"归一化成与 benchmark_result.json 同构的结构。

    这样报告生成、入库、前端三条链路只有一种输入格式。
    """
    folder = Path(folder)
    summary = parse_summary(folder)
    summary["output_dir"] = str(folder)
    cells = parse_performance(folder)
    accuracy = parse_accuracy(folder)
    return {
        "experiment_id": folder.name,
        "status": summary.get("status"),
        "started_at": summary.get("started_at"),
        "ended_at": summary.get("ended_at"),
        "issues": summary.get("issues") or [],
        "perf": {"cells": [cell.__dict__ for cell in cells]},
        "accuracy": None if accuracy is None else {
            "status": accuracy.status, "samples": accuracy.samples,
            "prompt_level_strict": accuracy.prompt_level_strict,
            "inst_level_strict": accuracy.inst_level_strict,
            "prompt_level_loose": accuracy.prompt_level_loose,
            "inst_level_loose": accuracy.inst_level_loose,
            "issues": accuracy.issues,
        },
        "summary": summary,
    }