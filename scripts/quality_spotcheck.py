"""质量抽检：长文本理解 / 逻辑推理 / 代码生成（评分表第 3 项要求的"输出质量达标"）。

与官方 IFBench 的关系：
  - IFBench 是官方裁判，测"指令跟随"，属于正式精度成绩；
  - 本脚本是**补充证据**，专门覆盖评分表点名的三类任务，结果单独存放、
    在报告里标注为"抽检"，不与官方分数混算。

评分方式全部可自动核验（不做"看起来不错"的主观判断）：
  - 长文本理解：答案必须包含给定的关键事实（子串匹配，全中才通过）
  - 逻辑推理：正则抽取最终答案，与标准答案比对
  - 代码生成：ast 语法解析 + 函数存在性 + 在子进程里跑断言（带超时）
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "tests" / "quality_items.json"


def chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    elapsed = time.time() - started
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {"text": content, "seconds": round(elapsed, 2),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason")}


# ---------------------------------------------------------------- 判分
def grade_long_context(answer: str, spec: dict) -> tuple[bool, str]:
    missing = [fact for fact in spec.get("must_include", []) if fact not in answer]
    limit = spec.get("max_chars")
    too_long = limit is not None and len(answer) > limit
    detail = []
    if missing:
        detail.append(f"缺少关键事实 {missing}")
    if too_long:
        detail.append(f"超长 {len(answer)}>{limit}")
    return (not missing and not too_long), ("；".join(detail) or "关键事实齐全且长度合规")


def grade_reasoning(answer: str, spec: dict) -> tuple[bool, str]:
    expected = str(spec.get("answer", "")).strip().lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    letters = re.findall(r"\b([A-Da-d])\b", answer)
    candidates = []
    if numbers:
        candidates.append(numbers[-1].lower())
    if letters:
        candidates.append(letters[-1].lower())
    hit = expected in candidates or (expected and expected in answer.lower()[-200:])
    return hit, f"期望 {expected}；抽到 {candidates[-3:] if candidates else '无'}"


def grade_code(answer: str, spec: dict) -> tuple[bool, str]:
    block = answer
    fences = re.findall(r"```(?:python)?\s*(.*?)```", answer, re.S)
    if fences:
        block = max(fences, key=len)
    try:
        ast.parse(block)
    except SyntaxError as error:
        return False, f"语法错误：{error.msg} (line {error.lineno})"
    name = spec.get("function")
    if name and f"def {name}" not in block:
        return False, f"未定义函数 {name}"
    tests = spec.get("tests") or []
    if not tests:
        return True, "语法通过、函数存在"
    # 注意：不用 tempfile.TemporaryDirectory —— 它建的目录是 0700，
    # 在本项目的受限沙箱里不可写。统一放到工作区 tests/_tmp 下。
    scratch = ROOT / "tests" / "_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    folder = scratch / f"graded_{int(time.time() * 1000)}_{os.getpid()}"
    folder.mkdir(exist_ok=True)
    try:
        script = folder / "case.py"
        script.write_text(block + "\n\n" + "\n".join(tests) + "\n", encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                                  timeout=spec.get("timeout", 10))
        except subprocess.TimeoutExpired:
            return False, "断言执行超时"
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    if done.returncode == 0:
        return True, f"{len(tests)} 条断言全过"
    return False, (done.stderr or "").strip().splitlines()[-1][:160] if done.stderr else "断言失败"


GRADERS = {"long_context": grade_long_context, "reasoning": grade_reasoning, "code": grade_code}


def main() -> int:
    parser = argparse.ArgumentParser(description="三类任务质量抽检")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.8-27b-fp8")
    parser.add_argument("--items", type=Path, default=ITEMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, item in enumerate(items, 1):
        try:
            reply = chat(args.base_url, args.model, item["prompt"], args.max_tokens, args.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            results.append({**item, "passed": False, "detail": f"请求失败：{type(error).__name__}",
                            "answer": "", "seconds": None})
            print(f"[{index}/{len(items)}] {item['id']} 请求失败", flush=True)
            continue
        passed, detail = GRADERS[item["category"]](reply["text"], item)
        results.append({**item, "passed": passed, "detail": detail, "answer": reply["text"],
                        "seconds": reply["seconds"], "completion_tokens": reply["completion_tokens"],
                        "finish_reason": reply["finish_reason"]})
        print(f"[{index}/{len(items)}] {item['id']} {'通过' if passed else '未过'} | {detail}", flush=True)

    summary = {}
    for category in ("long_context", "reasoning", "code"):
        group = [row for row in results if row["category"] == category]
        summary[category] = {"total": len(group),
                             "passed": sum(1 for row in group if row["passed"]),
                             "rate": round(sum(1 for row in group if row["passed"]) / len(group), 4)
                             if group else None}
    overall = {"total": len(results), "passed": sum(1 for row in results if row["passed"]),
               "rate": round(sum(1 for row in results if row["passed"]) / len(results), 4) if results else None}
    payload = {"model": args.model, "base_url": args.base_url, "benchmark": "quality_spotcheck",
               "note": "补充证据，非官方成绩；题目与判据见 tests/quality_items.json",
               "summary": summary, "overall": overall, "results": results,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (args.output_dir / "quality_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 三类任务质量抽检（补充证据）", "",
             f"- 模型：`{args.model}`",
             f"- 题目数：{overall['total']}，通过 {overall['passed']}（{overall['rate']*100:.1f}%）",
             f"- 说明：{payload['note']}", "",
             "| 任务类型 | 题数 | 通过 | 通过率 |", "|---|---:|---:|---:|"]
    names = {"long_context": "长文本理解", "reasoning": "逻辑推理", "code": "代码生成"}
    for category, stat in summary.items():
        rate = f"{stat['rate']*100:.1f}%" if stat["rate"] is not None else "—"
        lines.append(f"| {names[category]} | {stat['total']} | {stat['passed']} | {rate} |")
    lines += ["", "| 编号 | 类型 | 结果 | 判定依据 |", "|---|---|---|---|"]
    for row in results:
        lines.append(f"| {row['id']} | {names[row['category']]} | {'✅' if row['passed'] else '❌'} | "
                     f"{str(row['detail']).replace('|', '/')[:80]} |")
    (args.output_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n总体：{overall['passed']}/{overall['total']} 通过（{overall['rate']*100:.1f}%）")
    for category, stat in summary.items():
        print(f"  {names[category]}：{stat['passed']}/{stat['total']}")
    return 0 if overall["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
