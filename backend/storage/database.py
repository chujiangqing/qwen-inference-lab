"""SQLite 数据层：建库、建表、连接管理。

表结构严格对应需求书第九章（experiments / experiment_configs / performance_metrics /
hardware_metrics / accuracy_metrics / experiment_artifacts）。
原始评测文件不存进库，库里只放索引与结构化指标（artifact_repository 负责登记）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,              -- baseline / concurrency / batching / kv_cache / mtp / auto_tuner
    status          TEXT NOT NULL,              -- 运行中 / 完成 / 异常 / 已中断
    model_name      TEXT NOT NULL,
    model_revision  TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    UNIQUE(experiment_name)
);

CREATE TABLE IF NOT EXISTS experiment_configs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id          INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    vllm_version           TEXT,
    gpu_info               TEXT,
    startup_parameters     TEXT,               -- 完整启动命令 / JSON
    benchmark_parameters   TEXT,               -- 评测口径（来自 configuration.json）
    mtp_enabled            INTEGER,
    kv_cache_dtype         TEXT,
    max_num_seqs           INTEGER,
    max_num_batched_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id        INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    concurrency          INTEGER NOT NULL,
    request_count        INTEGER,
    ttft_p50             REAL,
    ttft_p95             REAL,
    tpot_p50             REAL,
    tpot_p95             REAL,
    e2e_p50              REAL,
    e2e_p95              REAL,
    request_throughput   REAL,
    output_throughput    REAL,
    status               TEXT,
    raw                  TEXT
);

CREATE TABLE IF NOT EXISTS hardware_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    timestamp       TEXT NOT NULL,
    gpu_utilization REAL,
    vram_used       REAL,
    vram_peak       REAL,
    temperature     REAL,
    power           REAL,
    cpu_utilization REAL,
    system_memory   REAL
);

CREATE TABLE IF NOT EXISTS accuracy_metrics (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id        INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    dataset              TEXT NOT NULL,
    sample_count         INTEGER,
    prompt_level_strict  REAL,
    raw_scores           TEXT            -- 四项原生分数完整 JSON
);

CREATE TABLE IF NOT EXISTS experiment_artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    artifact_type TEXT,
    file_path     TEXT NOT NULL,
    file_hash     TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_experiment ON performance_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_accuracy_experiment ON accuracy_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON experiment_artifacts(experiment_id);
"""


def default_path(root: Path) -> Path:
    return Path(root) / "data" / "inference_lab.db"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def init(root: Path) -> Path:
    path = default_path(root)
    with connect(path):
        pass
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="初始化实验数据库")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    path = init(args.root)
    with connect(path) as db:
        tables = [row["name"] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    print(f"数据库：{path}")
    print("表：", ", ".join(tables))
