"""链路契约测试：锁住"评测 → 落盘 → 入库 → 渲染"这条链上的已知断点。

这三条都是真实发生过的断链（P0-1 路径、P0-2 参数不下发、P1 标量字段），
用测试固定住，避免以后再犯。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "benchmark"))
sys.path.insert(0, str(ROOT / "backend" / "storage"))


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_runtime = _load_script("render_runtime.py")
import result_parser  # noqa: E402

# 测试专用的临时目录（不用 tempfile.mkdtemp：Windows 沙箱下 0700 目录不可写）
SCRATCH = Path(__file__).resolve().parent / "_tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)


def make_case(name: str) -> Path:
    folder = SCRATCH / name
    if folder.exists():
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    return folder


class RenderArgsTest(unittest.TestCase):
    """P0-2：batching 维度必须真的下发到 vLLM。"""

    def setUp(self) -> None:
        self.config = render_runtime.load(ROOT / "configs" / "runtime.json")
        self.profile = dict(self.config["profiles"]["baseline"])

    def _args(self, **overrides) -> list[str]:
        profile = render_runtime.apply_overrides(self.profile, [f"{k}={v}" for k, v in overrides.items()])
        return render_runtime.vllm_arguments(profile, self.config)

    def test_batched_tokens_rendered(self) -> None:
        args = self._args(max_num_batched_tokens=8192)
        self.assertIn("--max-num-batched-tokens", args)
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "8192")

    def test_batched_tokens_absent_by_default(self) -> None:
        """基线不显式设置时不发这个参数，避免改变原始口径。"""
        self.assertNotIn("--max-num-batched-tokens", self._args())

    def test_max_num_seqs_rendered(self) -> None:
        args = self._args(max_num_seqs=8)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "8")

    def test_two_candidates_differ_in_command(self) -> None:
        """不同 batched_tokens 的候选必须产生不同命令行，否则就是重复实验。"""
        a = self._args(max_num_seqs=4, max_num_batched_tokens=2048)
        b = self._args(max_num_seqs=4, max_num_batched_tokens=16384)
        self.assertNotEqual(a, b)


class ScalarFieldTest(unittest.TestCase):
    """P1：AIPerf 把 request_count 写成标量时，解析器不能崩。"""

    def test_scalar_request_count_falls_back(self) -> None:
        folder = make_case("scalar")
        cell = folder / "performance" / "c1"
        cell.mkdir(parents=True)
        (cell / "profile_export_aiperf.json").write_text(
            '{"is_complete": true, "request_count": 100, "error_request_count": 0,'
            ' "request_throughput": 0.05, "output_token_throughput": 26.0,'
            ' "benchmark_duration": 1000.0, "time_to_first_token": {"p50": 3000, "p95": 3400},'
            ' "inter_token_latency": {"p50": 31, "p95": 32},'
            ' "request_latency": {"p50": 19000, "p95": 19500}}',
            encoding="utf-8")
        cells = result_parser.parse_performance(folder)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].status, "未核验")     # 缺官方 summary.json
        self.assertEqual(cells[0].requests, 100.0)      # 标量被接受
        self.assertEqual(cells[0].output_throughput, 26.0)
        self.assertEqual(cells[0].ttft_p95, 3400)

    def test_dict_shape_still_works(self) -> None:
        folder = make_case("dictshape")
        cell = folder / "performance" / "c2"
        cell.mkdir(parents=True)
        (cell / "profile_export_aiperf.json").write_text(
            '{"request_count": {"avg": 24}, "request_throughput": {"avg": 0.08},'
            ' "output_token_throughput": {"avg": 43.0}, "benchmark_duration": {"avg": 900.0}}',
            encoding="utf-8")
        cells = result_parser.parse_performance(folder)
        self.assertEqual(cells[0].requests, 24.0)
        self.assertEqual(cells[0].request_throughput, 0.08)


class IngestContractTest(unittest.TestCase):
    """P0-3：任何一次评测落盘后都必须能进数据库。"""

    def test_persist_ingests(self) -> None:
        import shutil

        import database
        import experiment_repository as repo
        from benchmark_manager import BenchmarkManager, EvalConfig, ExperimentResult

        workdir = make_case("ingest_root")
        (workdir / "backend" / "storage").mkdir(parents=True)
        # 把真实的 storage 模块链接过去，避免复制
        for name in ("database.py", "experiment_repository.py", "artifact_repository.py"):
            (workdir / "backend" / "storage" / name).write_bytes(
                (ROOT / "backend" / "storage" / name).read_bytes())

        manager = BenchmarkManager(workdir, EvalConfig(base_url="http://127.0.0.1:1/v1", model="m"),
                                   label="UNIT")
        result = ExperimentResult(experiment_id="UNIT_INGEST")
        result.status = "完整"
        result.perf = {"cells": [{"concurrency": 1, "status": "完整", "output_throughput": 26.28,
                                  "request_count": 100}], "summary_status": "完整"}
        manager._persist(result)

        db_path = database.default_path(workdir)
        self.assertTrue(db_path.is_file(), "落盘后必须生成数据库")
        rows = repo.list_experiments(db_path, limit=10)
        self.assertTrue(rows, "落盘后必须能在数据库里查到实验")
        self.assertEqual(rows[0]["experiment_name"], "UNIT_INGEST")
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()

class TunerCommandTest(unittest.TestCase):
    """AutoTuner 与基线必须用同一种输入口径（模型 tokenizer，ISL 精确 8192）。"""

    def test_tuner_passes_model_tokenizer(self) -> None:
        sys.path.insert(0, str(ROOT / "backend" / "optimizer"))
        import auto_tuner

        captured = {}

        def fake_run(cmd, timeout=None):
            captured["cmd"] = cmd
            class _Done:
                returncode = 0
                stdout = ""
                stderr = ""
            return _Done()

        original = auto_tuner.run
        auto_tuner.run = fake_run
        try:
            auto_tuner.benchmark({"perf": {"request_timeout": 600}}, "PROBE_LABEL", 4)
        finally:
            auto_tuner.run = original

        cmd = captured.get("cmd") or []
        self.assertIn("--tokenizer", cmd, "AutoTuner 必须显式传 tokenizer，否则和基线口径不一致")
        tokenizer = cmd[cmd.index("--tokenizer") + 1]
        self.assertEqual(tokenizer, auto_tuner.default_tokenizer(),
                         "必须用 default_tokenizer() 的取值，不能各写一套")
        if Path(auto_tuner.default_tokenizer()).is_dir():
            self.assertNotEqual(tokenizer, "builtin",
                                "模型目录存在时必须用模型 tokenizer；builtin 会让服务端 ISL 漂移")

class EncodingHygieneTest(unittest.TestCase):
    """PowerShell 的 `-Encoding utf8` 会写 BOM：bash 脚本带 BOM 直接执行失败，
    Python 的 compile() 也会报 U+FEFF。这里把它固定住。"""

    def test_no_bom_in_sources(self) -> None:
        suffixes = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt", ".toml"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if any(part in {".ssh_helper", "_upload", "reports", "experiments", "__pycache__"}
                   for part in path.parts):
                continue
            if path.read_bytes().startswith("\ufeff".encode("utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [], f"这些文件带 UTF-8 BOM：{offenders}")

class StartupArgumentVerifyTest(unittest.TestCase):
    """候选启动后必须核对参数真的进了命令行，否则实验等于没做。"""

    def _write_record(self, folder: Path, argv: list[str]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "launch_test.json").write_text(
            json.dumps({"vllm_arguments": argv}), encoding="utf-8")

    def test_matching_arguments_pass(self) -> None:
        sys.path.insert(0, str(ROOT / "backend" / "optimizer"))
        import auto_tuner

        folder = make_case("startup_ok")
        self._write_record(folder, ["--max-num-seqs", "8", "--kv-cache-dtype", "auto",
                                    "--max-num-batched-tokens", "16384",
                                    "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'])
        ok, detail = auto_tuner.verify_startup_arguments(
            {"max_num_seqs": 8, "kv_cache_dtype": "auto",
             "max_num_batched_tokens": 16384,
             "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}, folder)
        self.assertTrue(ok, detail)

    def test_missing_flag_fails(self) -> None:
        sys.path.insert(0, str(ROOT / "backend" / "optimizer"))
        import auto_tuner

        folder = make_case("startup_missing")
        self._write_record(folder, ["--max-num-seqs", "4"])
        ok, detail = auto_tuner.verify_startup_arguments({"max_num_seqs": 8}, folder)
        self.assertFalse(ok)
        self.assertIn("未生效", detail)

    def test_no_record_fails(self) -> None:
        sys.path.insert(0, str(ROOT / "backend" / "optimizer"))
        import auto_tuner

        folder = make_case("startup_empty")
        ok, detail = auto_tuner.verify_startup_arguments({"max_num_seqs": 8}, folder)
        self.assertFalse(ok)
        self.assertIn("找不到启动记录", detail)