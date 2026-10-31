"""前端页面冒烟测试：用假的 streamlit 模块执行 7 个页面，确保不会抛异常。

为什么需要：Streamlit 的报错只出现在浏览器里，命令行看不到。
这个测试把 st.* 全部替换成记录器，任何 ImportError / AttributeError / 空值处理错误都会暴露。

运行：
    python3 -m unittest tests.test_frontend_pages -v
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
PAGES = sorted((FRONTEND / "pages").glob("*.py"))


class _StopPage(Exception):
    """模拟 streamlit 的 st.stop()：正常终止页面渲染，不算错误。"""


class _ColumnConfigStub:
    """streamlit 的 column_config：任何列类型都返回一个占位配置。"""

    def __getattr__(self, name: str):
        def _factory(*args, **kwargs):
            return {"type": name, "label": args[0] if args else kwargs.get("label")}
        return _factory

class _Recorder:
    """替身：任何属性访问都返回可调用的记录器。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "columns":
                count = int(args[0]) if args else 1
                return [_Recorder() for _ in range(count)]
            if name == "selectbox":
                options = args[1] if len(args) > 1 else kwargs.get("options")
                # 真实 Streamlit：有选项返回选中的第一个，没选项返回 None
                return options[0] if options else None
            if name == "number_input":
                return 1
            if name == "text_input":
                return str(args[1]) if len(args) > 1 else ""
            if name == "stop":
                raise _StopPage()
            if name in {"button", "form_submit_button", "download_button"}:
                return False
            if name == "file_uploader":
                return None
            return _Recorder()
        return _call

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_stub() -> _Recorder:
    recorder = _Recorder()
    module = types.ModuleType("streamlit")
    module.__getattr__ = recorder.__getattr__          # 模块级属性
    module.session_state = {}
    module.column_config = _ColumnConfigStub()
    sys.modules["streamlit"] = module
    # plotly 要按"模块 + 子模块"建，否则 `import plotly.express as px` 会把 px 绑成函数
    express = types.ModuleType("plotly.express")
    express.__getattr__ = recorder.__getattr__
    plotly = types.ModuleType("plotly")
    plotly.__getattr__ = recorder.__getattr__
    plotly.express = express
    sys.modules["plotly"] = plotly
    sys.modules["plotly.express"] = express
    return recorder


def _install_api_stub(payload: dict, mode: str = "full") -> None:
    """把 api_client 换成返回固定数据的假模块，避免测试依赖运行中的后端。"""
    module = types.ModuleType("api_client")

    def call(path: str, payload: dict | None = None, method: str | None = None, timeout: float = 60):
        for key, value in payload.items() if False else []:
            pass
        if "experiments/" in path:
            return {
                "experiment": {"id": 1, "experiment_name": "E00_BASELINE_PERF", "status": "完整"},
                "performance": [{"concurrency": 1, "request_count": 100, "ttft_p50": 3400, "ttft_p95": 3440,
                                 "tpot_p50": 31.4, "tpot_p95": 31.44, "e2e_p50": 19400, "e2e_p95": 19481,
                                 "request_throughput": 0.05133, "output_throughput": 26.28, "status": "完整"}],
                "accuracy": [{"dataset": "ifbench", "sample_count": 100, "prompt_level_strict": 0.76,
                              "raw_scores": "{\"inst_level_strict\": 0.8}"}],
                "config": [], "hardware": [], "artifacts": [],
            }
        if mode == "empty":
            if "/api/experiments" in path:
                return {"experiments": []}
            if "/api/reports" in path:
                return {"error": "not found"}
        if "/api/experiments" in path:
            return {"experiments": [{"id": 1, "experiment_name": "E00_BASELINE_PERF",
                                     "experiment_type": "baseline", "status": "完整",
                                     "created_at": "2026-10-31T06:00:00+00:00",
                                     "best_output_throughput": 66.26, "prompt_level_strict": 0.76}]}
        if "/api/hardware" in path:
            return {"gpu": {"available": True, "gpus": [{"index": "0", "name": "RTX 4090",
                                                          "memory_total_mib": 49140, "memory_used_mib": 42584,
                                                          "memory_free_mib": 6556, "utilization": 100,
                                                          "temperature_c": 55, "power_w": 300, "power_limit_w": 450}]},
                    "memory_gib": {"MemTotal": 90, "MemAvailable": 80}, "cpu_count": 20,
                    "disk": {"free_gib": 96.0}}
        if "/api/runtime/status" in path:
            return {"service": {"ready": True, "models": ["qwen3.8-27b-fp8"], "max_model_len": 40960},
                    "profiles": {"baseline": {"description": "32768", "max_model_len": 32768}},
                    "log_dir": "/root/logs/vllm", "pid_file": "/root/logs/vllm/vllm.pid"}
        if "/api/runtime/logs" in path:
            return {"log": "/root/logs/vllm/serve.log", "tail": "Uvicorn running"}
        if "/api/reports" in path:
            return "# 报告\n内容"
        return payload if isinstance(payload, (dict, str)) else {}

    module.call = call
    sys.modules["api_client"] = module


class FrontendPagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_stub()
        _install_api_stub({})
        for path in (FRONTEND, FRONTEND / "components"):
            sys.path.insert(0, str(path))

    def test_pages_execute(self) -> None:
        self.assertTrue(PAGES, "没有找到任何页面文件")
        for page in PAGES:
            with self.subTest(page=page.name):
                code = compile(page.read_text(encoding="utf-8"), str(page), "exec")
                namespace = {"__name__": "__page__", "__file__": str(page)}
                try:
                    exec(code, namespace)          # noqa: S102 - 冒烟测试，故意执行页面脚本
                except _StopPage:
                    continue
                except Exception as error:      # noqa: BLE001 - 任何异常都是失败
                    self.fail(f"{page.name} 执行失败：{type(error).__name__}: {error}")

    def test_app_entry(self) -> None:
        path = FRONTEND / "app.py"
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        try:
            exec(code, {"__name__": "__app__", "__file__": str(path)})   # noqa: S102
        except Exception as error:      # noqa: BLE001
            self.fail(f"app.py 执行失败：{type(error).__name__}: {error}")


if __name__ == "__main__":
    unittest.main()

class FrontendEmptyStateTest(unittest.TestCase):
    """空数据是最容易崩的状态（用户第一次打开时看到的就是它）。"""

    def test_pages_execute_with_no_data(self) -> None:
        _install_stub()
        _install_api_stub({}, mode="empty")
        for path in (FRONTEND, FRONTEND / "components"):
            sys.path.insert(0, str(path))
        for page in PAGES:
            with self.subTest(page=page.name):
                code = compile(page.read_text(encoding="utf-8"), str(page), "exec")
                try:
                    exec(code, {"__name__": "__page__", "__file__": str(page)})   # noqa: S102
                except _StopPage:
                    continue
                except Exception as error:      # noqa: BLE001
                    self.fail(f"{page.name} 在空数据下崩了：{type(error).__name__}: {error}")

class _CompareModeStub(_Recorder):
    """radio 固定选中"对比模式"，用来覆盖 04 页的对比分支。"""

    def __getattr__(self, name: str):
        base = super().__getattr__(name)

        def _call(*args, **kwargs):
            if name == "radio":
                return "基线 vs 优化对比"
            if name == "columns":
                count = int(args[0]) if args else 1
                return [_CompareModeStub() for _ in range(count)]
            if name == "selectbox":
                options = args[1] if len(args) > 1 else kwargs.get("options")
                if not options:
                    return None
                # 两个下拉框：基线选第一个，对比实验必须选不同的那个
                label = str(args[0]) if args else ""
                return options[-1] if "对比" in label else options[0]
            return base(*args, **kwargs)

        return _call


class CompareModeTest(unittest.TestCase):
    """优化前后对比：页面必须把 delta 表和结论真的画出来，而不是空转。"""

    def test_compare_branch_renders(self) -> None:
        _install_stub()                     # 先装好 plotly 等替身（按字母序本类先跑）
        recorder = _CompareModeStub()
        module = types.ModuleType("streamlit")
        module.__getattr__ = recorder.__getattr__
        module.session_state = {}
        module.column_config = _ColumnConfigStub()
        sys.modules["streamlit"] = module

        api = types.ModuleType("api_client")

        def call(path, payload=None, method=None, timeout=60):
            if "/compare" in path:
                return {"baseline": {"id": 1, "name": "E00_BASELINE_PERF"},
                        "candidate": {"id": 2, "name": "FINAL_ctx16384_seq16"},
                        "best_concurrency": 4, "throughput_delta_pct": 12.34,
                        "conclusion": "并发 4 下输出吞吐提升 12.3%",
                        "rows": [{"concurrency": 4,
                                  "output_throughput_base": 66.19, "output_throughput_new": 74.36,
                                  "output_throughput_delta_pct": 12.34,
                                  "ttft_p95_base": 6141, "ttft_p95_new": 6520, "ttft_p95_delta_pct": 6.17,
                                  "tpot_p95_base": 51.78, "tpot_p95_new": 46.1, "tpot_p95_delta_pct": -10.97}]}
            if "/api/experiments" in path:
                return {"experiments": [
                    {"id": 1, "experiment_name": "E00_BASELINE_PERF", "status": "完整"},
                    {"id": 2, "experiment_name": "FINAL_ctx16384_seq16", "status": "完整"}]}
            return {}

        api.call = call
        sys.modules["api_client"] = api

        page = FRONTEND / "pages" / "04_performance.py"
        code = compile(page.read_text(encoding="utf-8"), str(page), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(page)})   # noqa: S102

        names = [item[0] for item in recorder.calls]
        self.assertIn("dataframe", names, "对比表没画出来")
        self.assertTrue("plotly_chart" in names or "bar_chart" in names, "对比图没画出来")
        conclusions = [args[0] for name, args, _ in recorder.calls if name == "success" and args]
        self.assertTrue(any("12.3%" in str(text) for text in conclusions),
                        f"结论没显示：{conclusions}")