"""把实测产物导出成可入库的小文件（供 GitHub / 交付包展示）。

只挑"能说明结论"的小文件，原始日志与逐条预测（几十 MB）不导出：
  - 每个实验的 benchmark_result.json（总体状态与四档摘要）
  - 每个并发档的 summary.json（官方工具产物，含 status/issues）
  - AutoTuner 的 trials.json 与排名报告
  - 精度 summary.json（若有）
  - 质量抽检 quality_result.json（若有）

输出：results/
  manifest.json   每份文件的来源、大小、sha256
  README.md       结果目录说明（自动生成）
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PATTERNS = [
    ("experiments/*/benchmark_result.json", "experiments/{e}/benchmark_result.json"),
    ("experiments/*/performance/c*/summary.json", "experiments/{e}/performance/{c}/summary.json"),
    ("experiments/*/accuracy/summary.json", "experiments/{e}/accuracy/summary.json"),
    ("experiments/auto_tuner/trials.json", "auto_tuner/trials.json"),
    ("experiments/auto_tuner/report.md", "auto_tuner/report.md"),
    ("experiments/QUALITY_*/quality_result.json", "quality/{e}/quality_result.json"),
    ("experiments/QUALITY_*/quality_report.md", "quality/{e}/quality_report.md"),
    ("docs/实验报告.md", "reports/实验报告.md"),
    ("docs/优化结论.md", "reports/优化结论.md"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True)

    manifest: list[dict] = []
    for pattern, template in PATTERNS:
        for source in sorted(ROOT.glob(pattern)):
            if not source.is_file():
                continue
            experiment = source.parent.name
            if template.startswith("experiments/{e}/performance"):
                experiment = source.parent.parent.parent.name
            target_rel = template.format(e=experiment, c=source.parent.name)
            target = RESULTS / target_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest.append({"path": target_rel, "bytes": target.stat().st_size,
                             "sha256": sha256(target), "source": str(source.relative_to(ROOT))})

    (RESULTS / "manifest.json").write_text(
        json.dumps({"files": manifest, "count": len(manifest)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = ["# 实测结果快照", "",
             "本目录由 `scripts/export_results.py` 自动导出，只包含能说明结论的小文件；",
             "原始日志与逐条预测体积过大，未纳入仓库。", "",
             "| 文件 | 大小 | 来源 |", "|---|---:|---|"]
    for item in manifest:
        lines.append(f"| `{item['path']}` | {item['bytes'] / 1024:.1f} KB | `{item['source']}` |")
    lines += ["", "校验：`python3 -c \"import json;print(json.load(open('results/manifest.json'))['count'])\"`",
              "", "每个 `summary.json` 的 `status` 为「完整」才计入正式成绩。"]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"导出 {len(manifest)} 个文件到 results/")
    for item in manifest[:10]:
        print(f"  {item['path']}  {item['bytes'] / 1024:.1f} KB")
    if len(manifest) > 10:
        print(f"  ... 其余 {len(manifest) - 10} 个见 results/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
