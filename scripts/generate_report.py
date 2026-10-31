#!/usr/bin/env python3
"""汇总报告：把 E00 基线、AutoTuner 结果、MTP 对照汇成一份可交付的 Markdown + CSV。

用法：
    python3 scripts/generate_report.py                # 写到 docs/实验报告.md 与 data/experiments.csv
    python3 scripts/generate_report.py --out 路径.md

数据来源（只读，不重算评测口径）：
    experiments/**/benchmark_result.json   官方评测的结构化结果
    experiments/auto_tuner/trials.json      AutoTuner 逐候选结果
    reports/**/model.json                  阶段 0 的实测模型/显存事实
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "benchmark"))
sys.path.insert(0, str(ROOT / "backend" / "optimizer"))

import concurrency  # noqa: E402
import result_parser  # noqa: E402
import mtp as mtp_module  # noqa: E402
import search_space  # noqa: E402

# 分析层：性能曲线/质量置信区间/候选推荐。导入失败不阻断报告，但要在文中标明。
try:
    from analyzer import comparison as comparison_module  # noqa: E402
    from analyzer import performance_analyzer, quality_analyzer, recommendation  # noqa: E402
    ANALYZERS_OK = True
except ImportError as error:            # pragma: no cover - 只在残缺部署时发生
    ANALYZERS_OK = False
    ANALYZER_ERROR = str(error)


def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None


def find_results() -> list[dict]:
    """两种产物都要认：run_benchmark.sh 的 benchmark_result.json，以及官方 eval/run.sh 的产物目录。"""
    experiments = ROOT / "experiments"
    results = []
    seen = set()
    for path in sorted(experiments.glob("*/benchmark_result.json")):
        data = load_json(path)
        if data:
            data["_path"] = str(path)
            results.append(data)
            seen.add(path.parent.name)
    for folder in sorted(experiments.iterdir()) if experiments.is_dir() else []:
        if not folder.is_dir() or folder.name in seen:
            continue
        if not ((folder / "summary.json").is_file() or (folder / "performance").is_dir()):
            continue
        try:
            data = result_parser.parse_experiment_dir(folder)
        except (OSError, ValueError, KeyError):
            continue
        data["_path"] = str(folder)
        results.append(data)
    return results


def facts() -> dict:
    for candidate in sorted((ROOT / "reports").glob("*/model.json"), reverse=True):
        data = load_json(candidate)
        if data:
            return search_space.load_model_facts(candidate.parent)
    return {}


def fmt(value, digits: int = 2, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    try:
        return format(float(value) * scale, f".{digits}f")
    except (TypeError, ValueError):
        return "—"


def baseline_table(results: list[dict], only_complete: bool = True) -> list[str]:
    rows = []
    for result in results:
        if only_complete and result.get("status") != "完整":
            continue
        # 参数搜索的候选不能混进"性能基线"表，否则读者会以为它们也是基线
        if str(result.get("experiment_id") or "").startswith("AUTOTUNER_"):
            continue
        cells = (result.get("perf") or {}).get("cells") or []
        for cell in cells:
            rows.append({
                "实验": result.get("experiment_id"),
                "并发": cell.get("concurrency"),
                "状态": cell.get("status"),
                "请求数": cell.get("requests"),
                "TTFT P95(s)": fmt(cell.get("ttft_p95"), 3, 0.001),
                "TPOT P95(ms)": fmt(cell.get("tpot_p95"), 3),
                "E2E P95(s)": fmt(cell.get("e2e_p95"), 3, 0.001),
                "请求/s": fmt(cell.get("request_throughput"), 5),
                "输出 tok/s": fmt(cell.get("output_throughput"), 2),
            })
    if not rows:
        return ["（暂无性能数据）"]
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in rows]
    return lines


def accuracy_table(results: list[dict]) -> list[str]:
    rows = [(result.get("experiment_id"), result.get("accuracy")) for result in results
            if result.get("accuracy")]
    if not rows:
        return ["（暂无精度数据）"]
    lines = ["| 实验 | 状态 | 样本数 | prompt_level_strict | 其余三项 |",
             "|---|---|---:|---:|---|"]
    for name, accuracy in rows:
        rest = ", ".join(f"{key}={fmt(accuracy.get(key), 4)}" for key in
                         ("inst_level_strict", "prompt_level_loose", "inst_level_loose"))
        lines.append(f"| {name} | {accuracy.get('status')} | {accuracy.get('samples')} | "
                     f"{fmt(accuracy.get('prompt_level_strict'), 4)} | {rest} |")
    return lines


def tuner_tables() -> tuple[list[str], list[dict]]:
    data = load_json(ROOT / "experiments" / "auto_tuner" / "trials.json")
    if not data:
        return ["（AutoTuner 尚未产出结果）"], []
    trials = data.get("trials") or []
    valid = [item for item in trials if item.get("valid") and item.get("score") is not None]
    valid.sort(key=lambda item: item["score"], reverse=True)
    lines = ["| 排名 | 候选 | 维度 | 输出 tok/s | E2E P95(ms) | 峰值显存(GiB) | 质量 | 评分 |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for rank, item in enumerate(valid, 1):
        lines.append(f"| {rank} | {item['candidate']} | {item.get('dimension')} | "
                     f"{fmt(item.get('output_throughput'))} | {fmt(item.get('e2e_p95_ms'), 0)} | "
                     f"{fmt(item.get('vram_peak_gib'))} | {fmt(item.get('quality'), 4)} | "
                     f"{fmt(item.get('score'), 4)} |")
    invalid = [item for item in trials if not item.get("valid")]
    if invalid:
        lines += ["", "无效/失败候选（保留记录）："]
        for item in invalid:
            lines.append(f"- `{item['candidate']}`（{item.get('status')}）："
                         + "；".join(item.get("reasons") or ["未说明"]))
    return lines, trials


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总实验报告")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "实验报告.md")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "experiments.csv")
    args = parser.parse_args()

    measured = facts()
    results = find_results()
    tuner_lines, trials = tuner_tables()

    # 基线必须是"完整"且档位最多的那次；失败/中断的实验单独列，不混进基线表
    def _valid_cells(item: dict) -> int:
        cells = (item.get("perf") or {}).get("cells") or []
        return sum(1 for cell in cells if cell.get("status") == "完整")

    complete = [item for item in results if item.get("status") == "完整" and _valid_cells(item)]
    baseline = next((item for item in complete if "BASELINE" in (item.get("experiment_id") or "")), None)
    if baseline is None and complete:
        baseline = max(complete, key=_valid_cells)
    baseline_cells = ((baseline or {}).get("perf") or {}).get("cells") or []
    baseline_cells = [cell for cell in baseline_cells if cell.get("status") == "完整"]
    curve = concurrency.analyze(baseline_cells) if baseline_cells else {"conclusion": "缺少基线数据"}

    best_trial = next((item for item in sorted(trials, key=lambda x: x.get("score") or 0, reverse=True)
                       if item.get("valid") and item.get("score") is not None), None)
    mtp_trial = next((item for item in trials if item.get("candidate") == "mtp_on"), None)
    mtp_summary = mtp_module.analyze(
        {"output_throughput": (baseline_cells[-1]["output_throughput"] if baseline_cells else None)}
        if baseline_cells else None,
        mtp_trial,
        (baseline or {}).get("accuracy"), (mtp_trial or {}).get("accuracy"))

    lines = ["# Qwen3.8-27B 推理实验报告", "",
             "> 本报告由 `scripts/generate_report.py` 从原始评测产物自动生成，",
             "> 数字均可追溯到 `experiments/` 下的 AIPerf/EvalScope 原生文件。", "",
             "## 1. 实验环境（阶段 0 实测）", ""]
    if measured:
        lines += [f"- 权重：{measured.get('weights_gib')} GiB",
                  f"- KV：{measured.get('kv_per_token_kib')} KiB/token（仅 "
                  f"{measured.get('full_attention_layers')} 个全注意力层计入）",
                  f"- 显存：{measured.get('vram_gib')} GiB",
                  f"- 评测上下文：{measured.get('max_model_len')}"]
    else:
        lines.append("（未找到阶段 0 的 model.json）")

    lines += ["", "## 2. 性能基线（官方 eval/run.sh，100 请求/档）", ""]
    lines += baseline_table(results)
    failed = [item for item in results
              if item.get("status") != "完整" or not _valid_cells(item)]
    if failed:
        lines += ["", "### 失败/中断的实验（保留记录，不参与结论）", ""]
        for item in failed:
            reason = "；".join(item.get("issues") or []) or "未记录原因"
            lines.append(f"- `{item.get('experiment_id')}`：状态 {item.get('status')}；{reason}")
    lines += ["", "### 并发结论", "", f"- {curve.get('conclusion')}", f"- {curve.get('recommendation', '')}"]

    lines += ["", "## 3. 参数搜索（AutoTuner）", ""]
    lines += tuner_lines
    if best_trial:
        lines += ["", f"**当前最优候选**：`{best_trial['candidate']}`，"
                      f"输出吞吐 {fmt(best_trial.get('output_throughput'))} token/s，"
                      f"评分 {fmt(best_trial.get('score'), 4)}。"]

    lines += ["", "## 4. MTP 对照", "", f"- {mtp_summary.get('conclusion')}"]
    if mtp_summary.get("throughput_gain") is not None:
        lines.append(f"- 吞吐变化：{mtp_summary['throughput_gain'] * 100:+.1f}%")
    for risk in mtp_summary.get("risks", []):
        lines.append(f"- 风险：{risk}")

    lines += ["", "## 5. 精度（IFBench，limit=100）", ""]
    lines += accuracy_table(results)
    lines += ["", "> 评分指标为 `prompt_level_strict`；100 样本的二项标准误约 ±4 个百分点，",
              "> 小于该幅度的差异不应作为结论。", ""]

    lines += ["## 6. 自动分析（analyzer/）", ""]
    if not ANALYZERS_OK:
        lines += [f"- 分析模块不可用：`{ANALYZER_ERROR}`", ""]
    else:
        performance = performance_analyzer.analyze(
            [cell.__dict__ if hasattr(cell, "__dict__") else cell for cell in baseline_cells])
        lines += ["### 6.1 性能", "",
                  f"- 结论：{performance.get('conclusion')}",
                  f"- 建议：{performance.get('recommendation')}"]
        best_concurrency = performance.get("best_concurrency")
        if best_concurrency is not None:
            lines.append(f"- 最优并发：{best_concurrency}")

        accuracy_rows = [row for result in results if (row := result.get("accuracy"))]
        if accuracy_rows:
            quality = quality_analyzer.analyze([
                {"dataset": "ifbench", "sample_count": item.get("samples"),
                 "prompt_level_strict": item.get("prompt_level_strict")} for item in accuracy_rows])
            lines += ["", "### 6.2 精度", "",
                      f"- {quality.get('conclusion')}",
                      f"- 95% 置信区间：{quality.get('confidence_interval_95')}"]

        if trials:
            advice = recommendation.recommend(trials)
            lines += ["", "### 6.3 推荐配置", "", f"- {advice.get('conclusion')}"]
            for risk in advice.get("risks", []):
                lines.append(f"- 风险：{risk}")
            if advice.get("quality_unverified"):
                lines.append(f"- 质量未验证的候选：{', '.join(advice['quality_unverified'])}")

        if best_trial and baseline_cells:
            baseline_dicts = [cell.__dict__ if hasattr(cell, "__dict__") else cell
                              for cell in baseline_cells]
            raw_dir = str(best_trial.get("output_dir") or "")
            candidate_dir = Path(raw_dir) if raw_dir and ("/" in raw_dir or "\\" in raw_dir) \
                else ROOT / "experiments" / raw_dir
            candidate_cells = []
            if candidate_dir.is_dir():
                candidate_cells = [
                    {"concurrency": cell.concurrency, "output_throughput": cell.output_throughput,
                     "ttft_p95": cell.ttft_p95, "tpot_p95": cell.tpot_p95, "e2e_p95": cell.e2e_p95}
                    for cell in result_parser.parse_performance(candidate_dir)]
            if candidate_cells:
                delta = comparison_module.summarize(baseline_dicts, candidate_cells)
                lines += ["", "### 6.4 最优候选 vs 基线", "", f"- {delta.get('conclusion')}"]
        lines.append("")

    lines += ["## 7. 复现方式", "", "```bash",
              "bash eval/run.sh --mode check          # 依赖自检（只读）",
              "bash scripts/start_model.sh --profile baseline",
              "bash scripts/run_benchmark.sh --label E00_BASELINE --mode all",
              "python3 backend/optimizer/auto_tuner.py --requests 20",
              "python3 scripts/generate_report.py",
              "```", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for result in results:
        for cell in (result.get("perf") or {}).get("cells") or []:
            rows.append({"experiment": result.get("experiment_id"),
                         "concurrency": cell.get("concurrency"), "status": cell.get("status"),
                         "requests": cell.get("requests"), "ttft_p95_ms": cell.get("ttft_p95"),
                         "tpot_p95_ms": cell.get("tpot_p95"), "e2e_p95_ms": cell.get("e2e_p95"),
                         "request_throughput": cell.get("request_throughput"),
                         "output_throughput": cell.get("output_throughput")})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["experiment", "concurrency", "status", "requests", "ttft_p95_ms",
                                 "tpot_p95_ms", "e2e_p95_ms", "request_throughput", "output_throughput"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"报告：{args.out}")
    print(f"CSV：{args.csv}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
