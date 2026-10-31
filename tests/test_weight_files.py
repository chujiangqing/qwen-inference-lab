"""权重统计必须只算真实权重：子目录里的 original/ 不能重复计入。

这是评审指出的真问题：check_model.py 原来用 rglob 递归，
ModelScope 目录里若带 original/ 或半成品临时目录，权重会被算成两倍，
进而把"装得下"误判成"装不下"，直接否掉"提并发"这条路线。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRATCH = Path(__file__).resolve().parent / "_tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)


def _load_check_model():
    spec = importlib.util.spec_from_file_location("check_model", ROOT / "scripts" / "check_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


class WeightFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.check_model = _load_check_model()

    def setUp(self) -> None:
        self.root = SCRATCH / "model_layout"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_index(self, names: list[str]) -> None:
        weight_map = {f"layer.{i}": name for i, name in enumerate(names)}
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8")

    def test_nested_original_not_counted(self) -> None:
        """顶层 4 个分片 + original/ 里再放 4 个 → 只能算顶层的 4 个。"""
        shards = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
        for name in shards:
            _touch(self.root / name, 1024)
            _touch(self.root / "original" / name, 1024)
        self._make_index(shards)

        files, nested = self.check_model.weight_files(self.root)
        self.assertEqual(len(files), 4, "必须只统计 index 登记的 4 个分片")
        self.assertEqual(len(nested), 4, "子目录里的 4 个应被识别为已排除")

    def test_without_index_only_top_level(self) -> None:
        """没有 index.json 时只取顶层文件，不递归。"""
        for i in range(1, 3):
            _touch(self.root / f"model-0000{i}-of-00002.safetensors", 512)
            _touch(self.root / "original" / f"model-0000{i}-of-00002.safetensors", 512)

        files, nested = self.check_model.weight_files(self.root)
        self.assertEqual(len(files), 2)
        self.assertEqual(len(nested), 2)

    def test_real_layout_style_single_dir(self) -> None:
        """普通单层布局：index 与顶层一致，不产生"已排除"误报。"""
        shards = ["layers-0.safetensors", "layers-1.safetensors"]
        for name in shards:
            _touch(self.root / name, 2048)
        self._make_index(shards)

        files, nested = self.check_model.weight_files(self.root)
        self.assertEqual(len(files), 2)
        self.assertEqual(nested, [])


if __name__ == "__main__":
    unittest.main()
