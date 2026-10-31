"""AIPerf 性能评测适配器。

只负责"性能"这一段的参数与前置检查，其余（环境/服务/命令/日志/解析/入库/报告）
全部复用 EvalAdapter，避免出现第二套评测口径。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_adapter import EvalAdapter, EvalConfig  # noqa: E402


class AIPerfAdapter(EvalAdapter):
    profile = "perf"
    modes = ("perf",)

    def __init__(self, project_root: Path, config: EvalConfig) -> None:
        config.mode = "perf"
        super().__init__(project_root, config)

    def preflight(self) -> dict:
        """性能阶段的前置条件：环境就绪、服务在线、能拿到服务端口径。"""
        env = self.environment_status()
        service = self.model_status()
        checks = {
            "environment_ready": env.get("ready", False),
            "service_ready": service.get("ready", False),
            "max_model_len": service.get("max_model_len"),
            "tokenizer": self.config.tokenizer,
            "perf_requests": self.config.perf_requests,
            "request_timeout": self.config.request_timeout,
        }
        checks["ok"] = bool(checks["environment_ready"] and checks["service_ready"])
        if not checks["ok"]:
            reasons = []
            if not checks["environment_ready"]:
                reasons.append("评测依赖未就绪（先跑 bash eval/run.sh --mode check 并按其提示安装）")
            if not checks["service_ready"]:
                reasons.append(f"服务不可用或未返回模型 {self.config.model}")
            checks["reasons"] = reasons
        return checks
