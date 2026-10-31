"""报告与搜索空间的显示口径：不能把候选混进基线表，也不能因为找不到实测值就放行候选。"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "optimizer"))

import search_space  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / "_tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FactsFallbackTest(unittest.TestCase):
    """reports/ 下直接没有 model.json，但子目录里有实测结果时必须能找到。"""

    def test_falls_back_to_subdirectory(self) -> None:
        import shutil

        base = SCRATCH / "reports_probe"
        shutil.rmtree(base, ignore_errors=True)
        run_dir = base / "task0_20260101_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "model.json").write_text(json.dumps({
            "weights": {"total_gib": 28.75},
            "kv_budget": {"kv_per_token_kib": 32.0},
            "config_summary": {"full_attention_layers": 16, "max_position_embeddings": 262144},
            "runtime": {"max_model_len": 32768},
        }), encoding="utf-8")
        (run_dir / "gpu.json").write_text(json.dumps({
            "gpus": [{"memory_total_mib": 49140}]
        }), encoding="utf-8")

        facts = search_space.load_model_facts(base)
        self.assertEqual(facts.get("weights_gib"), 28.75)
        self.assertEqual(facts.get("kv_per_token_kib"), 32.0)
        self.assertEqual(facts.get("vram_gib"), 47.99)
        shutil.rmtree(base, ignore_errors=True)


class BaselineTableTest(unittest.TestCase):
    """AutoTuner 候选不能出现在"性能基线"表里。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _load_script("generate_report.py")

    def test_tuner_candidates_excluded(self) -> None:
        results = [
            {"experiment_id": "E00_BASELINE_PERF_20261031_055409", "status": "完整",
             "perf": {"cells": [{"concurrency": 1, "status": "完整", "output_throughput": 26.28,
                                 "request_count": 100}]}},
            {"experiment_id": "AUTOTUNER_batching_seq8_tok8192_1200_20261031_120001",
             "status": "完整",
             "perf": {"cells": [{"concurrency": 1, "status": "完整", "output_throughput": 70.0,
                                 "request_count": 20}]}},
        ]
        text = "\n".join(self.report.baseline_table(results))
        self.assertIn("E00_BASELINE_PERF", text)
        self.assertNotIn("AUTOTUNER_", text, "参数搜索候选不能混进性能基线表")

    def test_failed_runs_excluded(self) -> None:
        results = [
            {"experiment_id": "E00_BASELINE_PERF_x", "status": "异常",
             "perf": {"cells": [{"concurrency": 1, "status": "异常", "output_throughput": 1.0}]}},
        ]
        self.assertEqual(self.report.baseline_table(results), ["（暂无性能数据）"])


if __name__ == "__main__":
    unittest.main()
