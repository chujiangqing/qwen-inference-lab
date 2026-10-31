"""实验编排：一次实验 = 串行独占 + 官方评测 + 入库 + 生成报告。

API 层与 AutoTuner 都调用这里，保证"跑一次实验"只有一条实现路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "benchmark"))
sys.path.insert(0, str(BACKEND / "storage"))

from benchmark_manager import BenchmarkManager  # noqa: E402
from core import task_scheduler  # noqa: E402
from core.config import BASE_URL, ROOT, SERVED_MODEL, default_tokenizer, venv_python  # noqa: E402
from eval_adapter import EvalConfig  # noqa: E402

MODES = {"all": ("perf", "accuracy"), "perf": ("perf",), "accuracy": ("accuracy",)}


def make_config(mode: str = "all", tokenizer: str | None = None, perf_requests: int = 100,
                request_timeout: float = 600, eval_batch_size: int = 2) -> EvalConfig:
    return EvalConfig(base_url=BASE_URL, model=SERVED_MODEL, mode=mode,
                      tokenizer=tokenizer or default_tokenizer(),
                      perf_requests=perf_requests, request_timeout=request_timeout,
                      eval_batch_size=eval_batch_size)


def preflight(mode: str = "all", config: EvalConfig | None = None) -> dict:
    """只做检查，不启动评测。"""
    from aiperf_adapter import AIPerfAdapter
    from evalscope_adapter import EvalScopeAdapter

    config = config or make_config(mode)
    report: dict = {"busy": task_scheduler.is_busy(), "busy_holder": task_scheduler.current_holder()}

    # 推理环境自检：vLLM 的 venv 解释器必须存在（缺失时启动必然失败，提前说清楚）
    interpreter = venv_python()
    report["inference_env"] = {"ok": interpreter.is_file(), "python": str(interpreter),
                              "problem": None if interpreter.is_file()
                              else f"找不到 vLLM 解释器：{interpreter}"}

    if mode in ("all", "perf"):
        report["perf"] = AIPerfAdapter(ROOT, config).preflight()
    if mode in ("all", "accuracy"):
        report["accuracy"] = EvalScopeAdapter(ROOT, config).preflight()
    report["ok"] = not report["busy"] and all(
        item.get("ok") for key, item in report.items() if isinstance(item, dict) and "ok" in item)
    return report


def run(label: str, mode: str = "all", config: EvalConfig | None = None,
        wait_seconds: int = 0) -> dict:
    """执行一次实验：抢锁 → 官方评测 → 写库。失败也返回结构化结果，不吞异常。"""
    modes = MODES.get(mode)
    if not modes:
        raise ValueError(f"未知 mode：{mode}")
    config = config or make_config(mode)
    with task_scheduler.exclusive(label=label, wait_seconds=wait_seconds):
        result = BenchmarkManager(ROOT, config, label=label).run(modes, lock=False)
        # 入库由 BenchmarkManager._persist 统一负责（单一入口，避免重复插行）
        return result.to_dict()
