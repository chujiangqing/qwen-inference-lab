"""eval.zip 适配器基类：只调用、不复制、不改口径。

职责（对应需求书 6.1）：
  1. 检查评测环境        4. 启动原有评测程序    7. 读取评测结果
  2. 检查模型服务        5. 记录进程状态        8. 结果导入数据库
  3. 生成评测命令        6. 保存 stdout/stderr  9. 生成实验报告

本模块不含任何评测口径常量：所有口径（ISL/OSL/并发/预热/limit）都由
eval/runner.py 决定，这里只构造它的命令行参数并读取它自己的产物。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import EVAL_ENTRY  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_parser  # noqa: E402


@dataclass
class EvalConfig:
    """一次评测调用的参数。字段与 eval/runner.py 的 CLI 一一对应。"""
    base_url: str
    model: str
    mode: str = "all"                       # all / perf / accuracy
    tokenizer: str = "builtin"
    perf_requests: int = 100
    request_timeout: float = 600
    eval_batch_size: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    env_dir: str | None = None

    def to_argv(self) -> list[str]:
        argv = ["--mode", self.mode, "--base-url", self.base_url, "--model", self.model,
                "--tokenizer", self.tokenizer, "--perf-requests", str(self.perf_requests),
                "--request-timeout", str(self.request_timeout),
                "--eval-batch-size", str(self.eval_batch_size),
                "--api-key-env", self.api_key_env]
        if self.env_dir:
            argv += ["--env-dir", self.env_dir]
        return argv


@dataclass
class RunResult:
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    log_path: str | None = None
    output_dir: str | None = None
    pid: int | None = None
    status: str = "未开始"
    error: str | None = None


class EvalAdapter:
    """通用适配器；性能/精度各自继承并补充参数与前置检查。"""

    profile = "generic"
    modes = ("all",)

    def __init__(self, project_root: Path, config: EvalConfig) -> None:
        self.root = Path(project_root).resolve()
        self.config = config
        self.run_sh = EVAL_ENTRY          # 统一入口，避免各处重复拼路径

    # ------------------------------------------------------------------ 1. 环境
    def environment_status(self) -> dict:
        if not self.run_sh.is_file():
            return {"ready": False, "error": f"缺少评测入口 {self.run_sh}"}
        done = subprocess.run(["bash", str(self.run_sh), "--mode", "check"],
                              capture_output=True, text=True, timeout=600,
                              encoding="utf-8", errors="replace", cwd=str(self.root))
        ready = done.returncode == 0
        payload: dict = {"ready": ready, "exit_code": done.returncode,
                         "raw_tail": (done.stdout or "")[-2000:]}
        try:
            payload["status"] = json.loads(done.stdout)
        except (json.JSONDecodeError, ValueError):
            pass
        return payload

    # ------------------------------------------------------------------ 2. 服务
    def model_status(self) -> dict:
        url = self.config.base_url.rstrip("/") + "/models"
        key = os.environ.get(self.config.api_key_env, "EMPTY")
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError) as error:
            return {"ready": False, "error": f"{type(error).__name__}: {error}"}
        cards = {item.get("id"): item for item in body.get("data", [])}
        card = cards.get(self.config.model)
        return {"ready": card is not None, "models": list(cards),
                "max_model_len": (card or {}).get("max_model_len"), "card": card}

    # ------------------------------------------------------------------ 3. 命令
    def build_command(self, output_dir: Path) -> list[str]:
        return ["bash", str(self.run_sh), *self.config.to_argv(), "--output-dir", str(output_dir)]

    # ------------------------------------------------------- 4/5/6. 启动与留痕
    def run(self, output_dir: Path | None = None) -> RunResult:
        output_dir = Path(output_dir or (self.root / "experiments" / f"{self.profile}_"
                                        f"{time.strftime('%Y%m%d_%H%M%S')}")).resolve()
        logs = output_dir.parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{output_dir.name}.log"
        result = RunResult(command=self.build_command(output_dir), output_dir=str(output_dir),
                           log_path=str(log_path), started_at=_now())

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(result.command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                           errors="replace", env=env, cwd=str(self.root))
                result.pid = process.pid
                for line in process.stdout:          # 实时落盘，中断也保留
                    stream.write(line)
                    stream.flush()
                result.exit_code = process.wait()
            except KeyboardInterrupt:
                result.status = "已中断"
                process.kill()
                raise
            except OSError as error:
                result.error = f"{type(error).__name__}: {error}"
        result.duration_seconds = round(time.time() - started, 1)
        result.ended_at = _now()
        if result.status != "已中断":
            result.status = "完成" if result.exit_code == 0 else "异常"
        return result

    # ------------------------------------------------------------------ 7. 解析
    def parse(self, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        return {"summary": result_parser.parse_summary(output_dir),
                "performance": result_parser.parse_performance(output_dir),
                "accuracy": result_parser.parse_accuracy(output_dir),
                "artifacts": result_parser.collect_artifacts(output_dir)}

    # ------------------------------------------------------------------ 9. 报告
    def write_report(self, output_dir: Path, parsed: dict) -> Path:
        summary = parsed["summary"]
        lines = [f"# 评测报告：{self.profile}", "",
                 f"- 输出目录：`{output_dir}`",
                 f"- 官方总体状态：**{summary.get('status')}**",
                 f"- 开始 / 结束：{summary.get('started_at')} / {summary.get('ended_at')}", ""]
        if parsed["performance"]:
            lines += ["## 性能", "", "| 并发 | 状态 | 成功/请求 | TTFT P95(s) | TPOT P95(ms) | E2E P95(s) | 请求/s | 输出 tok/s |",
                      "|---:|---|---:|---:|---:|---:|---:|---:|"]
            for cell in parsed["performance"]:
                lines.append(
                    f"| {cell.concurrency} | {cell.status} | {cell.success}/{cell.requests} | "
                    f"{_fmt(cell.ttft_p95, .001)} | {_fmt(cell.tpot_p95)} | "
                    f"{_fmt(cell.e2e_p95, .001)} | {_fmt(cell.request_throughput, 1, '.5f')} | "
                    f"{_fmt(cell.output_throughput, 1, '.2f')} |")
        accuracy = parsed.get("accuracy")
        if accuracy:
            lines += ["", "## 精度（IFBench）", "",
                      f"- 状态：**{accuracy.status}**；样本数：{accuracy.samples}",
                      f"- `prompt_level_strict`（评分指标）：{_fmt(accuracy.prompt_level_strict, 1, '.4f')}",
                      f"- 其余三项：inst_strict={_fmt(accuracy.inst_level_strict, 1, '.4f')}, "
                      f"prompt_loose={_fmt(accuracy.prompt_level_loose, 1, '.4f')}, "
                      f"inst_loose={_fmt(accuracy.inst_level_loose, 1, '.4f')}"]
        issues = list(summary.get("issues") or [])
        for cell in parsed["performance"]:
            issues += [f"并发 {cell.concurrency}：{item}" for item in cell.issues]
        if accuracy:
            issues += [f"精度：{item}" for item in accuracy.issues]
        lines += ["", "## 异常与证据", ""]
        lines += [f"- {item}" for item in issues] or ["- 未发现完整性异常。"]
        lines.append("")
        path = Path(output_dir) / "experiment_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def _now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _fmt(value: float | None, factor: float = 1.0, pattern: str = ".3f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value) * factor, pattern)
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    """自查：打印将要执行的命令与当前环境/服务状态。"""
    import argparse
    parser = argparse.ArgumentParser(description="EvalAdapter 自检")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.8-27b-fp8")
    parser.add_argument("--profile", default="generic")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    adapter = EvalAdapter(root, EvalConfig(base_url=args.base_url, model=args.model))
    print(json.dumps({"profile": args.profile,
                      "environment": {k: v for k, v in adapter.environment_status().items() if k != "raw_tail"},
                      "service": adapter.model_status(),
                      "command": adapter.build_command(root / "experiments" / "probe")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
