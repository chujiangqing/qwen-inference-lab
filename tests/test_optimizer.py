"""优化器单测：搜索空间、显存预算、评分与推荐。

这些是纯计算，不碰 GPU、不碰服务，可在任何机器上跑。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "optimizer"))

import concurrency  # noqa: E402
import search_space  # noqa: E402
from analyzer import recommendation  # noqa: E402
from model.memory_estimator import budget, kv_per_token_bytes  # noqa: E402

# 本模型实测值（阶段 0 报告）：64 层里 16 层全注意力、4 个 KV 头、head_dim 256、FP8
FACTS = {"weights_gib": 28.75, "kv_per_token_kib": 32.0, "vram_gib": 47.99,
         "full_attention_layers": 16, "max_model_len": 32768}

SPACE = {
    "search_space": {"max_num_seqs": [1, 4, 16],
                     "max_num_batched_tokens": [8192],
                     "mtp": [True],
                     "gpu_memory_utilization": [0.9]},
    "dimensions": {"batching": True, "mtp": True, "kv_cache": False},
}


class KvMathTest(unittest.TestCase):
    def test_kv_per_token(self) -> None:
        # 2 × 16 层 × 4 头 × 256 = 32768 字节 = 32 KiB（FP8，每元素 1 字节）
        self.assertEqual(kv_per_token_bytes(16, 4, 256, "fp8"), 32768)

    def test_capacity_matches_measured(self) -> None:
        result = budget(vram_gib=47.99, weights_gib=28.75, kv_per_token_kib=32.0,
                        gpu_memory_utilization=0.9, max_model_len=32768, max_num_seqs=4)
        # 服务器实测 vLLM 报 292,484 token（它扣了更多开销），我们的估算应同量级
        self.assertGreater(result["kv_capacity_tokens"], 350_000)
        self.assertLess(result["kv_capacity_tokens"], 500_000)
        self.assertTrue(result["fits"])

    def test_16_seqs_does_not_fit(self) -> None:
        result = budget(vram_gib=47.99, weights_gib=28.75, kv_per_token_kib=32.0,
                        gpu_memory_utilization=0.9, max_model_len=32768, max_num_seqs=16)
        self.assertFalse(result["fits"])


class SearchSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = search_space.build_candidates(SPACE, FACTS)

    def test_grid_size(self) -> None:
        # 3 个 max_num_seqs × 1 个 batched_tokens + mtp + gpu_util
        batching = [item for item in self.candidates if item.dimension == "batching"]
        self.assertEqual(len(batching), 3)

    def test_16_seqs_invalid(self) -> None:
        sixteen = [item for item in self.candidates
                   if (item.overrides or {}).get("max_num_seqs") == 16
                   and item.dimension == "batching"]
        self.assertTrue(sixteen, "应该生成 max_num_seqs=16 的候选")
        for item in sixteen:
            self.assertFalse(item.valid, "16 并发 × 32768 超过实测 KV 容量，必须判无效")
            self.assertIn("KV 容量", " ".join(item.reasons))

    def test_four_seqs_valid(self) -> None:
        four = [item for item in self.candidates
                if (item.overrides or {}).get("max_num_seqs") == 4]
        self.assertTrue(all(item.valid for item in four))

    def test_mtp_uses_speculative_config(self) -> None:
        mtp = next(item for item in self.candidates if item.name == "mtp_on")
        self.assertIn("speculative_config", mtp.overrides)
        self.assertEqual(mtp.overrides["speculative_config"]["method"], "mtp")


class ConcurrencyTest(unittest.TestCase):
    CELLS = [{"concurrency": 1, "status": "完整", "output_throughput": 26.28, "ttft_p95": 3440},
             {"concurrency": 2, "status": "完整", "output_throughput": 43.14, "ttft_p95": 5043},
             {"concurrency": 4, "status": "完整", "output_throughput": 66.19, "ttft_p95": 6141},
             {"concurrency": 8, "status": "完整", "output_throughput": 66.26, "ttft_p95": 36374}]

    def test_saturation_point(self) -> None:
        result = concurrency.analyze(self.CELLS)
        self.assertEqual(result["saturation"]["from"], 4)
        self.assertEqual(result["saturation"]["to"], 8)
        self.assertAlmostEqual(result["saturation"]["ttft_p95_ratio"], 5.92, places=2)

    def test_recommendation_is_before_saturation(self) -> None:
        result = concurrency.analyze(self.CELLS)
        self.assertEqual(result["best_concurrency"], 4)
        self.assertEqual(result["peak_concurrency"], 8)


class RecommendationTest(unittest.TestCase):
    def test_quality_unverified_is_flagged(self) -> None:
        trials = [{"candidate": "A", "valid": True, "output_throughput": 70, "score": 0.62,
                   "quality": None, "status": "完成"}]
        advice = recommendation.recommend(trials)
        self.assertIn("质量未验证", advice["conclusion"])
        self.assertTrue(advice["risks"])

    def test_quality_below_threshold_rejected(self) -> None:
        trials = [{"candidate": "B", "valid": True, "output_throughput": 70, "score": 0.62,
                   "quality": 0.50, "status": "完成"}]
        advice = recommendation.recommend(trials, quality_threshold=0.70)
        self.assertIsNone(advice["recommended"])

    def test_good_candidate_recommended(self) -> None:
        trials = [{"candidate": "C", "valid": True, "output_throughput": 70, "score": 0.75,
                   "quality": 0.80, "status": "完成"}]
        advice = recommendation.recommend(trials)
        self.assertEqual(advice["recommended"], "C")


if __name__ == "__main__":
    unittest.main()

class ContextDimensionTest(unittest.TestCase):
    """缩上下文换并发：必须生成候选，且剪枝要用候选自己的 max_model_len。"""

    SPACE = {
        "search_space": {"max_num_seqs": [4], "max_num_batched_tokens": [8192],
                         "max_model_len": [16384], "context_max_num_seqs": [16]},
        "dimensions": {"batching": True, "context": True, "mtp": False, "kv_cache": False},
    }

    def test_context_candidate_generated(self) -> None:
        candidates = search_space.build_candidates(self.SPACE, FACTS)
        ctx = [c for c in candidates if c.dimension == "context"]
        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx[0].overrides["max_model_len"], 16384)
        self.assertEqual(ctx[0].overrides["max_num_seqs"], 16)

    def test_pruning_uses_candidate_context(self) -> None:
        # 16 并发 × 16384 = 262,144 token ≤ 实测容量约 42.4 万 → 必须判为可用
        candidates = search_space.build_candidates(self.SPACE, FACTS)
        ctx = next(c for c in candidates if c.dimension == "context")
        self.assertTrue(ctx.valid, ctx.reasons)

    def test_pruning_rejects_too_large(self) -> None:
        space = {
            "search_space": {"max_num_seqs": [4], "max_num_batched_tokens": [8192],
                             "max_model_len": [32768], "context_max_num_seqs": [32]},
            "dimensions": {"batching": True, "context": True, "mtp": False, "kv_cache": False},
        }
        candidates = search_space.build_candidates(space, FACTS)
        ctx = next(c for c in candidates if c.dimension == "context")
        self.assertFalse(ctx.valid)          # 32 × 32768 = 1,048,576 远超容量
        self.assertIn("下调 max_model_len", " ".join(ctx.reasons))