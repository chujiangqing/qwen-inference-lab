"""Benchmark 编排：一次实验 = 一段性能 + 一段精度，串行、独占、可追溯。

对应需求书 3.6 / 6.1 / 第十二章：
  - 同一时刻只允许一个评测进程（GPU 独占），用文件锁实现；
  - 性能与精度不重叠；
  - 失败也保留原始产物与状态；
  - 每段结束把结构化结果写进 SQLite，原始文件另存并按 hash 索引。
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiperf_adapter import AIPerfAdapter  # noqa: E402
from eval_adapter import EvalConfig  # noqa: E402
from evalscope_adapter import EvalScopeAdapter  # noqa: E402
from core import task_scheduler  # noqa: E402


@dataclass
class ExperimentResult:
    experiment_id: str
    status: str = "运行中"
    perf: dict | None = None
    accuracy: dict | None = None
    issues: list[str] = field(default_factory=list)
    output_dirs: dict = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "status": self.status,
                "perf": self.perf, "accuracy": self.accuracy, "issues": self.issues,
                "output_dirs": self.output_dirs, "started_at": self.started_at,
                "ended_at": self.ended_at}


class BenchmarkManager:
    LOCK_NAME = ".benchmark.lock"

    def __init__(self, project_root: Path, config: EvalConfig, label: str = "experiment") -> None:
        self.root = Path(project_root).resolve()
        self.config = config
        self.label = label
        self.lock_path = self.root / "experiments" / self.LOCK_NAME

    # ---------------------------------------------------------------- 锁
    # 统一走 core/task_scheduler.py，避免两套锁各自的文件格式互相误判。
    def _acquire(self, timeout: int = 60) -> None:
        self._lock_cm = task_scheduler.exclusive(label="benchmark", wait_seconds=timeout)
        self._lock_cm.__enter__()

    def _release(self) -> None:
        manager = getattr(self, "_lock_cm", None)
        if manager is not None:
            manager.__exit__(None, None, None)
            self._lock_cm = None

    # ---------------------------------------------------------------- 主流程
    def run(self, modes: tuple[str, ...] = ("perf", "accuracy"), lock: bool = True) -> ExperimentResult:
        experiment_id = f"{self.label}_{time.strftime('%Y%m%d_%H%M%S')}"
        result = ExperimentResult(experiment_id=experiment_id)
        result.started_at = _now()
        if lock:
            self._acquire()
        try:
            if "perf" in modes:
                adapter = AIPerfAdapter(self.root, self.config)
                preflight = adapter.preflight()
                if not preflight["ok"]:
                    result.issues += [f"性能前置检查失败：{item}" for item in preflight.get("reasons", [])]
                else:
                    output = self.root / "experiments" / experiment_id / "performance"
                    run = adapter.run(self._as_eval_output_dir(output, "perf"))
                    parsed = adapter.parse(Path(run.output_dir))
                    adapter.write_report(Path(run.output_dir), parsed)
                    result.perf = {
                        "run": run.__dict__,
                        "cells": [cell.__dict__ for cell in parsed["performance"]],
                        "summary_status": parsed["summary"].get("status"),
                    }
                    result.output_dirs["performance"] = run.output_dir
                    if run.exit_code != 0:
                        result.issues.append(f"性能阶段退出码 {run.exit_code}")

            if "accuracy" in modes:
                adapter = EvalScopeAdapter(self.root, self.config)
                preflight = adapter.preflight()
                if not preflight["ok"]:
                    result.issues += [f"精度前置检查失败：{item}" for item in preflight.get("reasons", [])]
                else:
                    output = self.root / "experiments" / experiment_id / "accuracy"
                    run = adapter.run(self._as_eval_output_dir(output, "accuracy"))
                    parsed = adapter.parse(Path(run.output_dir))
                    adapter.write_report(Path(run.output_dir), parsed)
                    accuracy = parsed["accuracy"]
                    result.accuracy = {
                        "run": run.__dict__,
                        "status": accuracy.status if accuracy else None,
                        "prompt_level_strict": accuracy.prompt_level_strict if accuracy else None,
                        "samples": accuracy.samples if accuracy else None,
                        "issues": accuracy.issues if accuracy else [],
                    }
                    result.output_dirs["accuracy"] = run.output_dir
                    if run.exit_code != 0:
                        result.issues.append(f"精度阶段退出码 {run.exit_code}")
        finally:
            if lock:
                self._release()
            result.ended_at = _now()
            if result.status == "运行中":
                bad = result.issues or any(
                    (part or {}).get("summary_status") not in (None, "完整")
                    for part in (result.perf,) if part)
                result.status = "异常" if bad else "完成"
            self._persist(result)
        return result

    def _as_eval_output_dir(self, wanted: Path, stage: str) -> Path:
        """eval/run.sh 要求 --output-dir 不存在；阶段产物目录用时间戳命名后交给它自己创建。"""
        wanted.parent.mkdir(parents=True, exist_ok=True)
        return wanted.parent / f"{wanted.name}_{time.strftime('%H%M%S')}"

    # ---------------------------------------------------------------- 留痕
    def _persist(self, result: ExperimentResult) -> None:
        """落盘 + 入库。

        入库放在这里而不是调用方：不管走 API、CLI 还是 AutoTuner，
        只要产生了一次评测结果，就一定进数据库，避免"跑了但页面空着"。
        """
        folder = self.root / "experiments" / result.experiment_id
        folder.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        result_path = folder / "benchmark_result.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        self._ingest(result_path)

    def _ingest(self, result_path: Path) -> None:
        """入库失败不能影响评测结论，所以只记录原因。"""
        try:
            sys.path.insert(0, str(self.root / "backend" / "storage"))
            import database
            import experiment_repository as repo

            db_path = database.init(self.root)
            experiment_id = repo.ingest_benchmark_result(db_path, result_path, kind="benchmark")
            self._record_hardware(db_path, experiment_id)
        except (OSError, ValueError, KeyError, ImportError) as error:
            print(f"[警告] 入库失败（不影响评测结果）：{type(error).__name__}: {error}", flush=True)

    def _record_hardware(self, db_path: Path, experiment_id: int) -> None:
        """记录评测结束时的 GPU 状态。

        hardware_metrics 表此前只建了、从没写过；这里补上唯一的写入点，
        让"这次实验在什么硬件状态下跑的"可追溯。
        """
        try:
            sys.path.insert(0, str(self.root / "backend" / "monitor"))
            import experiment_repository as repo
            import gpu_monitor

            snapshot = gpu_monitor.snapshot()
            gpus = (snapshot.get("gpus") or [{}])[0] if isinstance(snapshot, dict) else {}
            used_mib = gpus.get("memory_used_mib")
            repo.insert_hardware(
                db_path, experiment_id,
                vram_peak_gib=round(used_mib / 1024, 2) if used_mib else None,
                gpu_utilization=gpus.get("utilization"),
                temperature=gpus.get("temperature_c"),
                power=gpus.get("power_w"))
        except (OSError, ValueError, KeyError, ImportError, AttributeError) as error:
            print(f"[警告] 硬件指标入库失败：{type(error).__name__}: {error}", flush=True)


def _now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="一次完整评测（性能 + 精度）")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="E00_BASELINE")
    parser.add_argument("--mode", default="all", choices=["all", "perf", "accuracy"])
    parser.add_argument("--tokenizer", default="builtin")
    parser.add_argument("--perf-requests", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--pretend", action="store_true", help="只做前置检查，不启动评测")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config = EvalConfig(base_url=args.base_url, model=args.model, tokenizer=args.tokenizer,
                        perf_requests=args.perf_requests, request_timeout=args.request_timeout,
                        eval_batch_size=args.eval_batch_size)
    modes = {"all": ("perf", "accuracy"), "perf": ("perf",), "accuracy": ("accuracy",)}[args.mode]

    if args.pretend:
        report = {}
        if "perf" in modes:
            report["perf"] = AIPerfAdapter(root, config).preflight()
        if "accuracy" in modes:
            report["accuracy"] = EvalScopeAdapter(root, config).preflight()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(item["ok"] for item in report.values()) else 2

    manager = BenchmarkManager(root, config, label=args.label)
    result = manager.run(modes)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "完成" else 1


if __name__ == "__main__":
    raise SystemExit(main())
