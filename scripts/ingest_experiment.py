#!/usr/bin/env python3
"""把官方 eval/run.sh 的产物目录登记进 SQLite。

用法：
    python3 scripts/ingest_experiment.py experiments/E00_BASELINE_PERF_20261031_055409
    python3 scripts/ingest_experiment.py --scan            # 扫 experiments/ 下所有官方产物目录
    python3 scripts/ingest_experiment.py <dir> --kind baseline --name E00_BASELINE

只读产物、只写数据库与哈希登记；不修改任何评测文件。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "benchmark"))
sys.path.insert(0, str(ROOT / "backend" / "storage"))

import database  # noqa: E402
import experiment_repository as repo  # noqa: E402
import result_parser  # noqa: E402


def looks_like_output(folder: Path) -> bool:
    return (folder / "summary.json").is_file() or (folder / "performance").is_dir()


def ingest_one(folder: Path, kind: str, name: str | None) -> int:
    parsed = {"summary": result_parser.parse_summary(folder),
              "performance": result_parser.parse_performance(folder),
              "accuracy": result_parser.parse_accuracy(folder),
              "artifacts": result_parser.collect_artifacts(folder)}
    summary = parsed["summary"]
    summary["output_dir"] = str(folder)
    experiment_name = name or folder.name
    db_path = database.init(ROOT)
    experiment_id = repo.ingest_parsed(db_path, experiment_name=experiment_name, kind=kind,
                                       parsed=parsed)
    cells = parsed["performance"]
    accuracy = parsed["accuracy"]
    print(f"[入库] {experiment_name} → id={experiment_id} "
          f"状态={summary.get('status')} 档位={len(cells)} "
          f"精度={'—' if accuracy is None else accuracy.status}")
    for cell in cells:
        print(f"        c{cell.concurrency:<2} {cell.status:<4} "
              f"TTFT P95={_fmt(cell.ttft_p95, 0)}ms TPOT P95={_fmt(cell.tpot_p95, 2)}ms "
              f"输出={_fmt(cell.output_throughput, 2)} tok/s")
    if accuracy is not None:
        print(f"        IFBench prompt_level_strict={_fmt(accuracy.prompt_level_strict, 4)} "
              f"样本={accuracy.samples} 状态={accuracy.status}")
    return experiment_id


def _fmt(value, digits: int) -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), f".{digits}f")
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="官方评测产物入库")
    parser.add_argument("folder", nargs="?", help="官方产物目录")
    parser.add_argument("--scan", action="store_true", help="扫描 experiments/ 下全部产物目录")
    parser.add_argument("--kind", default="benchmark", help="实验类型标签，如 baseline/accuracy/auto_tuner")
    parser.add_argument("--name", help="实验名（默认用目录名）")
    args = parser.parse_args()

    if args.scan:
        roots = sorted((ROOT / "experiments").iterdir()) if (ROOT / "experiments").is_dir() else []
        targets = [item for item in roots if item.is_dir() and looks_like_output(item)]
        if not targets:
            print("没有找到官方产物目录（需要 summary.json 或 performance/）")
            return 1
        for folder in targets:
            kind = "baseline" if "PERF" in folder.name.upper() else (
                "accuracy" if "ACC" in folder.name.upper() else args.kind)
            ingest_one(folder, kind, None)
        return 0

    if not args.folder:
        parser.error("需要产物目录，或使用 --scan")
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"目录不存在：{folder}")
        return 1
    if not looks_like_output(folder):
        print(f"{folder} 不像官方产物目录（缺 summary.json 与 performance/）")
        return 2
    ingest_one(folder, args.kind, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
