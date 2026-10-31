"""用真实评测产物联调 result_parser / adapters。

这些断言只在服务器上（有 experiments/ 目录时）才有意义，本地会自动跳过：
    E00_DIR=/root/qwen-lab/experiments/E00_..._PERF_... python3 -m unittest tests.test_result_parser -v

它验证的是"我们的解析器能不能正确读懂官方产物"，不是"官方产物对不对"。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "benchmark"))

import result_parser  # noqa: E402

DEFAULT = sorted((ROOT / "experiments").glob("*PERF*")) if (ROOT / "experiments").is_dir() else []
EXPERIMENT_DIR = Path(os.environ.get("E00_DIR", DEFAULT[-1] if DEFAULT else "")) if DEFAULT or os.environ.get("E00_DIR") else None


@unittest.skipUnless(EXPERIMENT_DIR and EXPERIMENT_DIR.is_dir(), "没有真实评测产物，跳过联调")
class RealArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = result_parser.parse_summary(EXPERIMENT_DIR)
        self.cells = result_parser.parse_performance(EXPERIMENT_DIR)

    def test_summary_readable(self) -> None:
        self.assertIn("summary.json", self.summary["files"])
        self.assertIsNotNone(self.summary["status"])
        self.assertIn(self.summary["status"], {"完整", "异常", "已中断"})

    def test_four_concurrency_cells(self) -> None:
        concurrencies = sorted(cell.concurrency for cell in self.cells)
        self.assertEqual(concurrencies, [1, 2, 4, 8], f"实际解析到 {concurrencies}")

    def test_cells_carry_metrics(self) -> None:
        for cell in self.cells:
            self.assertIsNotNone(cell.output_throughput, f"并发 {cell.concurrency} 缺吞吐")
            self.assertIsNotNone(cell.ttft_p95, f"并发 {cell.concurrency} 缺 TTFT P95")
            self.assertIsNotNone(cell.tpot_p95, f"并发 {cell.concurrency} 缺 TPOT P95")
            self.assertIn(cell.status, {"完整", "异常"})

    def test_input_length_verified(self) -> None:
        """模型 tokenizer 模式下，服务端实际 ISL 必须精确 8192。"""
        lengths = [cell.actual_input_tokens for cell in self.cells if cell.actual_input_tokens]
        self.assertTrue(lengths, "没有实际输入长度记录")
        for item in lengths:
            self.assertEqual(item["min"], 8192)
            self.assertEqual(item["max"], 8192)

    def test_artifacts_registered(self) -> None:
        artifacts = result_parser.collect_artifacts(EXPERIMENT_DIR)
        self.assertGreater(len(artifacts), 10)
        kinds = {item["type"] for item in artifacts}
        self.assertTrue({"json", "jsonl"} & kinds)


if __name__ == "__main__":
    unittest.main()
