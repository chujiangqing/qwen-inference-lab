"""EvalScope 精度评测适配器。

除了复用 EvalAdapter 的调用/留痕/解析，本模块额外落实需求书 6.6 的
"评测长度冲突处理"：正式评测前检查 输入 + max_tokens(32768) 是否超过上下文，
不满足时直接给出明确错误，不把配置问题包装成模型质量问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_adapter import EvalAdapter, EvalConfig  # noqa: E402

# eval/runner.py 固定给 EvalScope 传 max_tokens=32768；这里只做检查，不改口径。
EVAL_MAX_TOKENS = 32768


class EvalScopeAdapter(EvalAdapter):
    profile = "accuracy"
    modes = ("accuracy",)

    def __init__(self, project_root: Path, config: EvalConfig) -> None:
        config.mode = "accuracy"
        super().__init__(project_root, config)

    def preflight(self) -> dict:
        env = self.environment_status()
        service = self.model_status()
        context = service.get("max_model_len")
        checks = {
            "environment_ready": env.get("ready", False),
            "service_ready": service.get("ready", False),
            "max_model_len": context,
            "accuracy_max_tokens": EVAL_MAX_TOKENS,
            "eval_batch_size": self.config.eval_batch_size,
        }
        reasons = []
        if not checks["environment_ready"]:
            reasons.append("评测依赖未就绪（先跑 bash eval/run.sh --mode check 并按其提示安装）")
        if not checks["service_ready"]:
            reasons.append(f"服务不可用或未返回模型 {self.config.model}")
        # 需求书 6.6 / 审计 P0-1：上下文 <= max_tokens 时官方工具会跳过精度阶段
        if isinstance(context, (int, float)) and context <= EVAL_MAX_TOKENS:
            reasons.append(
                f"服务上下文 max_model_len={context} <= EVAL_MAX_TOKENS={EVAL_MAX_TOKENS}；"
                f"官方 runner.py 会把精度阶段判为配置异常并跳过。"
                f"请用 --max-model-len 更大的 profile（configs/runtime.json 的 accuracy）重启服务")
        checks["ok"] = not reasons
        checks["reasons"] = reasons
        return checks
