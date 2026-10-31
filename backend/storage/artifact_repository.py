"""原始产物登记：路径、类型、SHA-256；用于"结果可追溯"与答辩时核对文件未被篡改。"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from database import connect


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def register(db_path: Path, experiment_id: int, folder: Path, *, created_at: str) -> int:
    count = 0
    with connect(db_path) as db:
        for path in sorted(Path(folder).rglob("*")):
            if not path.is_file():
                continue
            db.execute(
                "INSERT INTO experiment_artifacts(experiment_id, artifact_type, file_path,"
                " file_hash, created_at) VALUES (?,?,?,?,?)",
                (experiment_id, path.suffix.lstrip(".") or "file", str(path),
                 sha256_file(path), created_at))
            count += 1
    return count


def verify(db_path: Path, experiment_id: int) -> dict:
    """逐个文件重算哈希，报告是否与登记值一致（防止事后改动原始数据）。"""
    with connect(db_path) as db:
        rows = db.execute("SELECT file_path, file_hash FROM experiment_artifacts WHERE experiment_id=?",
                          (experiment_id,)).fetchall()
    missing, mismatched, ok = [], [], 0
    for row in rows:
        path = Path(row["file_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        if sha256_file(path) != row["file_hash"]:
            mismatched.append(str(path))
        else:
            ok += 1
    return {"checked": len(rows), "ok": ok, "missing": missing, "mismatched": mismatched,
            "intact": not missing and not mismatched}


def export_csv(db_path: Path, out_path: Path) -> Path:
    """把实验与性能指标导出成 CSV（需求书 11 章 scripts/generate_report.py 的数据源）。"""
    with connect(db_path) as db:
        rows = db.execute(
            "SELECT e.experiment_name, e.experiment_type, e.status, e.created_at,"
            " p.concurrency, p.request_count, p.ttft_p50, p.ttft_p95, p.tpot_p50, p.tpot_p95,"
            " p.e2e_p50, p.e2e_p95, p.request_throughput, p.output_throughput,"
            " a.prompt_level_strict"
            " FROM experiments e"
            " LEFT JOIN performance_metrics p ON p.experiment_id = e.id"
            " LEFT JOIN accuracy_metrics a ON a.experiment_id = e.id"
            " ORDER BY e.id, p.concurrency").fetchall()
        columns = [description[0] for description in db.execute(
            "SELECT e.experiment_name, e.experiment_type, e.status, e.created_at,"
            " p.concurrency, p.request_count, p.ttft_p50, p.ttft_p95, p.tpot_p50, p.tpot_p95,"
            " p.e2e_p50, p.e2e_p95, p.request_throughput, p.output_throughput,"
            " a.prompt_level_strict FROM experiments e"
            " LEFT JOIN performance_metrics p ON p.experiment_id = e.id"
            " LEFT JOIN accuracy_metrics a ON a.experiment_id = e.id LIMIT 1").description]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])
    return out_path
