"""搜索空间：把 configs/search_space.yaml 展开成可执行的候选配置，并做合法性/显存检查。

对应需求书 8.1 第 3–5 步、8.2 网格搜索、7.1–7.5 的单因素实验。
合法性判断只用实测数据（scripts/check_model.py 的 model.json 与 check_gpu.py 的 gpu.json），
不在这里猜硬件。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.memory_estimator import capacity_tokens_from_kib  # noqa: E402


@dataclass
class Candidate:
    """一个待验证的参数组合。`overrides` 直接喂给 render_runtime.py --set。"""
    name: str
    overrides: dict = field(default_factory=dict)
    dimension: str = "combo"
    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "overrides": self.overrides, "dimension": self.dimension,
                "valid": self.valid, "reasons": self.reasons}


def load_space(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:      # 延迟导入：没有 PyYAML 时仍可 --help/--dry-run
        raise SystemExit("缺少 PyYAML：在服务器执行 `pip3 install --user pyyaml`") from None
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_model_facts(reports_dir: Path) -> dict:
    """读取阶段 0 的实测结果（模型结构、显存、权重体积）。

    给定目录里没有 model.json 时，自动去 reports/*/model.json 找最新的那份；
    否则前端"展开搜索空间"拿不到实测值，会把本该判无效的候选全放行。
    """
    facts: dict = {}
    reports_dir = Path(reports_dir)

    def newest(name: str) -> Path:
        """先看目录本身，再看子目录，最后看同级目录（reports/task0_*/model.json 这种布局）。"""
        direct = reports_dir / name
        if direct.is_file():
            return direct
        for base in (reports_dir, reports_dir.parent):
            candidates = sorted(base.glob(f"*/{name}"),
                                key=lambda path: path.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
        return direct

    model_json = newest("model.json")
    gpu_json = newest("gpu.json")
    if model_json.is_file():
        data = json.loads(model_json.read_text(encoding="utf-8-sig"))
        facts.update({
            "weights_gib": (data.get("weights") or {}).get("total_gib"),
            "kv_per_token_kib": (data.get("kv_budget") or {}).get("kv_per_token_kib"),
            "full_attention_layers": (data.get("config_summary") or {}).get("full_attention_layers"),
            "max_position_embeddings": (data.get("config_summary") or {}).get("max_position_embeddings"),
            "max_model_len": (data.get("runtime") or {}).get("max_model_len"),
        })
    if gpu_json.is_file():
        data = json.loads(gpu_json.read_text(encoding="utf-8-sig"))
        gpus = data.get("gpus") or [{}]
        facts["vram_gib"] = round((gpus[0].get("memory_total_mib") or 0) / 1024, 2)
    return facts


def kv_capacity_tokens(facts: dict, gpu_memory_utilization: float) -> float | None:
    """按实测权重与 KV/token 估算可容纳 token 数；数据不全时返回 None（不猜）。"""
    weights = facts.get("weights_gib")
    per_token_kib = facts.get("kv_per_token_kib")
    vram = facts.get("vram_gib")
    if not (weights and per_token_kib and vram):
        return None
    return capacity_tokens_from_kib(vram_gib=vram, weights_gib=weights,
                                    kv_per_token_kib=per_token_kib,
                                    gpu_memory_utilization=gpu_memory_utilization)

def build_candidates(space: dict, facts: dict) -> list[Candidate]:
    """展开网格并逐个判定合法性。

    注意：剪枝必须用**候选自己的** max_model_len，而不是全局默认值——
    否则"缩短上下文换并发"这条路会被误判成 OOM 而永远被跳过。
    """
    grid = space.get("search_space") or {}
    dims = space.get("dimensions") or {}
    default_len = facts.get("max_model_len") or 32768
    candidates: list[Candidate] = []

    def add(name: str, dimension: str, **overrides) -> None:
        candidate = Candidate(name=name, overrides={k: v for k, v in overrides.items() if v is not None},
                              dimension=dimension)
        effective_len = int(candidate.overrides.get("max_model_len") or default_len)
        cap = kv_capacity_tokens(facts, float(candidate.overrides.get("gpu_memory_utilization", 0.90)))
        if cap is not None:
            need = effective_len * int(candidate.overrides.get("max_num_seqs", 4))
            if cap < need:
                candidate.valid = False
                candidate.reasons.append(
                    f"KV 容量 {int(cap)} token < 需求 {need}"
                    f"（max_model_len={effective_len} × max_num_seqs={candidate.overrides.get('max_num_seqs', 4)}）"
                    f"；可下调 max_model_len 换取更高并发")
        if candidate.overrides.get("mtp") is True:
            candidate.overrides.pop("mtp", None)
            candidate.overrides["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 3}
            candidate.reasons.append("MTP：用 speculative_config 表达，质量风险需单独复核")
        candidates.append(candidate)

    seqs = (grid.get("max_num_seqs") or [4]) if dims.get("batching", True) else [None]
    batched = (grid.get("max_num_batched_tokens") or [None]) if dims.get("batching", True) else [None]
    for seq, batched_tokens in itertools.product(seqs, batched):
        if seq is None and batched_tokens is None:
            continue
        add(f"batching_seq{seq}_tok{batched_tokens}", "batching",
            max_num_seqs=seq, max_num_batched_tokens=batched_tokens)

    # 上下文 × 并发：短上下文腾出 KV，换更高并发（这是"配置封顶"的正确解法）
    if dims.get("context", False):
        context_lens = grid.get("max_model_len") or [16384]
        context_seqs = grid.get("context_max_num_seqs") or [8, 16]
        for max_len, seq in itertools.product(context_lens, context_seqs):
            add(f"ctx{max_len}_seq{seq}", "context", max_model_len=max_len, max_num_seqs=seq)
            candidates[-1].reasons.append(
                f"缩短上下文到 {max_len} 换取 {seq} 路并发；长文本任务质量需单独复核")

    if dims.get("mtp", True):
        add("mtp_on", "mtp", mtp=True)

    if dims.get("kv_cache", False):
        for dtype in ("auto", "fp8"):
            add(f"kv_{dtype}", "kv_cache", kv_cache_dtype=dtype)

    for util in grid.get("gpu_memory_utilization") or []:
        add(f"gpu_util_{util}", "kv_cache", gpu_memory_utilization=util)

    return candidates


def summarize(candidates: list[Candidate]) -> dict:
    valid = [item for item in candidates if item.valid]
    return {"total": len(candidates), "valid": len(valid), "invalid": len(candidates) - len(valid),
            "by_dimension": {dim: sum(1 for item in valid if item.dimension == dim)
                             for dim in sorted({item.dimension for item in valid})}}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="展开并校验搜索空间")
    parser.add_argument("--space", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs" / "search_space.yaml")
    parser.add_argument("--reports", type=Path,
                        default=Path(__file__).resolve().parents[2] / "reports" / "task0_verify")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    space = load_space(args.space)
    facts = load_model_facts(args.reports)
    candidates = build_candidates(space, facts)
    if args.json:
        print(json.dumps({"facts": facts, "summary": summarize(candidates),
                          "candidates": [item.to_dict() for item in candidates]},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"实测事实：{json.dumps(facts, ensure_ascii=False)}")
    print(f"候选统计：{json.dumps(summarize(candidates), ensure_ascii=False)}")
    for item in candidates:
        flag = "OK  " if item.valid else "跳过"
        print(f"  {flag} {item.name:<34} {json.dumps(item.overrides, ensure_ascii=False)}"
              + (f"  # {'; '.join(item.reasons)}" if item.reasons else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
