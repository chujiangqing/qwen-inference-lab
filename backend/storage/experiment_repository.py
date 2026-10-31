"""实验记录：把一次评测的原始产物落成数据库行。

数据来源只认官方 runner 的产物（benchmark_result.json / summary.json / configuration.json /
逐档 summary.json / accuracy/summary.json），不重新计算任何评测口径。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from database import connect


def _now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upsert_experiment(db, *, name: str, kind: str, status: str, model: str,
                      revision: str | None = None, started_at: str | None = None,
                      finished_at: str | None = None) -> int:
    row = db.execute("SELECT id FROM experiments WHERE experiment_name = ?", (name,)).fetchone()
    if row:
        db.execute("UPDATE experiments SET status=?, started_at=COALESCE(?, started_at),"
                   " finished_at=COALESCE(?, finished_at) WHERE id=?",
                   (status, started_at, finished_at, row["id"]))
        return int(row["id"])
    cursor = db.execute(
        "INSERT INTO experiments(experiment_name, experiment_type, status, model_name,"
        " model_revision, created_at, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?)",
        (name, kind, status, model, revision, _now(), started_at, finished_at))
    return int(cursor.lastrowid)


def insert_config(db, experiment_id: int, *, startup_parameters: dict, benchmark_parameters: dict,
                  vllm_version: str | None, gpu_info: str | None, mtp_enabled: bool | None,
                  kv_cache_dtype: str | None, max_num_seqs: int | None,
                  max_num_batched_tokens: int | None) -> None:
    db.execute(
        "INSERT INTO experiment_configs(experiment_id, vllm_version, gpu_info, startup_parameters,"
        " benchmark_parameters, mtp_enabled, kv_cache_dtype, max_num_seqs, max_num_batched_tokens)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (experiment_id, vllm_version, gpu_info,
         json.dumps(startup_parameters, ensure_ascii=False),
         json.dumps(benchmark_parameters, ensure_ascii=False),
         None if mtp_enabled is None else int(bool(mtp_enabled)),
         kv_cache_dtype, max_num_seqs, max_num_batched_tokens))


def insert_performance(db, experiment_id: int, cells: list[dict]) -> None:
    for cell in cells:
        db.execute(
            "INSERT INTO performance_metrics(experiment_id, concurrency, request_count, ttft_p50,"
            " ttft_p95, tpot_p50, tpot_p95, e2e_p50, e2e_p95, request_throughput,"
            " output_throughput, status, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (experiment_id, cell.get("concurrency"), cell.get("requests"),
             cell.get("ttft_p50"), cell.get("ttft_p95"), cell.get("tpot_p50"), cell.get("tpot_p95"),
             cell.get("e2e_p50"), cell.get("e2e_p95"), cell.get("request_throughput"),
             cell.get("output_throughput"), cell.get("status"),
             json.dumps(cell, ensure_ascii=False)))


def insert_accuracy(db, experiment_id: int, accuracy: dict | None) -> None:
    if not accuracy:
        return
    db.execute(
        "INSERT INTO accuracy_metrics(experiment_id, dataset, sample_count, prompt_level_strict,"
        " raw_scores) VALUES (?,?,?,?,?)",
        (experiment_id, "ifbench", accuracy.get("samples"),
         accuracy.get("prompt_level_strict"), json.dumps(accuracy, ensure_ascii=False)))


def insert_hardware(db, experiment_id: int, *, vram_peak_gib: float | None,
                    gpu_utilization: float | None = None, temperature: float | None = None,
                    power: float | None = None) -> None:
    db.execute(
        "INSERT INTO hardware_metrics(experiment_id, timestamp, gpu_utilization, vram_used,"
        " vram_peak, temperature, power) VALUES (?,?,?,?,?,?,?)",
        (experiment_id, _now(), gpu_utilization, None, vram_peak_gib, temperature, power))


def ingest_benchmark_result(db_path: Path, result_path: Path, *, kind: str = "benchmark") -> int:
    """把一个 benchmark_result.json（benchmark_manager 产出）登记入库，返回 experiment_id。"""
    db_path = Path(db_path)
    result = json.loads(Path(result_path).read_text(encoding="utf-8-sig"))
    name = result.get("experiment_id") or Path(result_path).parent.name
    perf = result.get("perf") or {}
    run = perf.get("run") or {}
    cells = perf.get("cells") or []
    accuracy = result.get("accuracy") or {}
    with connect(db_path) as db:
        experiment_id = upsert_experiment(
            db, name=name, kind=kind, status=result.get("status", "未知"),
            model=run.get("model") or "", started_at=result.get("started_at"),
            finished_at=result.get("ended_at"))
        args = run.get("command") or []
        startup = parse_startup(args)
        insert_config(db, experiment_id,
                      startup_parameters=startup,
                      benchmark_parameters={"command": args},
                      vllm_version=None, gpu_info=None,
                      mtp_enabled=bool(startup.get("speculative_config")),
                      kv_cache_dtype=startup.get("kv_cache_dtype"),
                      max_num_seqs=startup.get("max_num_seqs"),
                      max_num_batched_tokens=startup.get("max_num_batched_tokens"))
        # 幂等：重跑同一实验时先清掉旧的指标行，避免重复累计
        db.execute("DELETE FROM performance_metrics WHERE experiment_id=?", (experiment_id,))
        db.execute("DELETE FROM accuracy_metrics WHERE experiment_id=?", (experiment_id,))
        insert_performance(db, experiment_id, cells)
        insert_accuracy(db, experiment_id, accuracy if accuracy.get("status") else None)
        _register_artifacts(db, experiment_id, Path(result_path).parent)
    return experiment_id


def parse_startup(args: list[str]) -> dict:
    """从 vllm serve 命令行里抽取可入库的启动参数（简单、可核对，不做模糊猜测）。"""
    flags = {"--max-num-seqs": "max_num_seqs", "--max-num-batched-tokens": "max_num_batched_tokens",
             "--kv-cache-dtype": "kv_cache_dtype", "--max-model-len": "max_model_len",
             "--gpu-memory-utilization": "gpu_memory_utilization",
             "--speculative-config": "speculative_config"}
    parsed: dict = {}
    for index, token in enumerate(args):
        if token in flags and index + 1 < len(args):
            value = args[index + 1]
            key = flags[token]
            if key in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
                try:
                    parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
            elif key == "speculative_config":
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed[key] = value
            else:
                parsed[key] = value
    return parsed


def _register_artifacts(db, experiment_id: int, folder: Path) -> None:
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.name.endswith(".db"):
            continue
        db.execute(
            "INSERT INTO experiment_artifacts(experiment_id, artifact_type, file_path, file_hash,"
            " created_at) VALUES (?,?,?,?,?)",
            (experiment_id, path.suffix.lstrip(".") or "file", str(path), sha256_file(path), _now()))


def list_experiments(db_path: Path, limit: int = 50) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute(
            "SELECT e.*, (SELECT MAX(output_throughput) FROM performance_metrics p"
            " WHERE p.experiment_id = e.id) AS best_output_throughput,"
            " (SELECT prompt_level_strict FROM accuracy_metrics a WHERE a.experiment_id = e.id"
            " ORDER BY a.id DESC LIMIT 1) AS prompt_level_strict"
            " FROM experiments e ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def experiment_detail(db_path: Path, experiment_id: int) -> dict:
    with connect(db_path) as db:
        experiment = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if experiment is None:
            return {}
        return {
            "experiment": dict(experiment),
            "config": [dict(row) for row in db.execute(
                "SELECT * FROM experiment_configs WHERE experiment_id=?", (experiment_id,))],
            "performance": [dict(row) for row in db.execute(
                "SELECT * FROM performance_metrics WHERE experiment_id=? ORDER BY concurrency",
                (experiment_id,))],
            "accuracy": [dict(row) for row in db.execute(
                "SELECT * FROM accuracy_metrics WHERE experiment_id=?", (experiment_id,))],
            "hardware": [dict(row) for row in db.execute(
                "SELECT * FROM hardware_metrics WHERE experiment_id=?", (experiment_id,))],
            "artifacts": [dict(row) for row in db.execute(
                "SELECT * FROM experiment_artifacts WHERE experiment_id=? ORDER BY file_path",
                (experiment_id,))],
        }

def ingest_parsed(db_path: Path, *, experiment_name: str, kind: str, parsed: dict) -> int:
    """把 result_parser 解析出来的官方产物直接入库（不要求 benchmark_result.json）。

    官方 eval/run.sh 只产出 summary.json / performance/c*/ / accuracy/，
    这个函数是"官方产物 → 数据库"的桥，不重算任何指标。
    """
    summary = parsed.get("summary") or {}
    cells = [cell.__dict__ if hasattr(cell, "__dict__") else cell for cell in (parsed.get("performance") or [])]
    accuracy = parsed.get("accuracy")
    accuracy_dict = accuracy.__dict__ if hasattr(accuracy, "__dict__") else accuracy
    config = summary.get("configuration") or {}
    with connect(Path(db_path)) as db:
        experiment_id = upsert_experiment(
            db, name=experiment_name, kind=kind, status=summary.get("status") or "未知",
            model=config.get("model") or "", revision=config.get("model_revision"),
            started_at=summary.get("started_at"), finished_at=summary.get("ended_at"))
        insert_config(db, experiment_id,
                      startup_parameters={"base_url": config.get("base_url"),
                                          "tokenizer": config.get("tokenizer")},
                      benchmark_parameters={key: config.get(key) for key in
                                            ("perf_requests", "request_timeout", "eval_batch_size",
                                             "concurrency", "warmup_requests")},
                      vllm_version=(summary.get("environment") or {}).get("vllm_version"),
                      gpu_info=str((summary.get("environment") or {}).get("platform")),
                      mtp_enabled=bool(config.get("speculative_config")),
                      kv_cache_dtype=config.get("kv_cache_dtype"),
                      max_num_seqs=config.get("max_num_seqs"),
                      max_num_batched_tokens=config.get("max_num_batched_tokens"))
        # 幂等：重跑同一实验时先清掉旧的指标行
        db.execute("DELETE FROM performance_metrics WHERE experiment_id=?", (experiment_id,))
        db.execute("DELETE FROM accuracy_metrics WHERE experiment_id=?", (experiment_id,))
        insert_performance(db, experiment_id, cells)
        insert_accuracy(db, experiment_id, accuracy_dict)
        folder = Path(summary.get("output_dir") or "")
        if folder.is_dir():
            _register_artifacts(db, experiment_id, folder)
    return experiment_id