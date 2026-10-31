#!/usr/bin/env python3
"""阶段 0 脚本的冒烟测试：确保每个检测脚本都能被调用且 --help 正常。

运行：
    python3 -m unittest discover -s tests -v
不需要 GPU、不需要网络、不安装任何依赖。
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("check_gpu.py", "check_environment.py", "check_eval_tool.py", "check_model.py")
# 测试产物写在 tests/_tmp（默认权限目录），不污染 reports/，也不用 0o700 的 mkdtemp。
TMP = Path(__file__).resolve().parent / "_tmp" / "smoke"


class ScriptCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TMP.mkdir(parents=True, exist_ok=True)

    def test_help_exits_zero(self) -> None:
        for name in SCRIPTS:
            with self.subTest(script=name):
                done = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                                      capture_output=True, text=True, timeout=60,
                                      encoding="utf-8", errors="replace")
                self.assertEqual(done.returncode, 0, msg=done.stderr[:800])
                self.assertIn("usage:", done.stdout.lower())

    def test_model_script_requires_path(self) -> None:
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_model.py")],
                              capture_output=True, text=True, timeout=60,
                              encoding="utf-8", errors="replace")
        self.assertEqual(done.returncode, 2)
        self.assertIn("--model-path", done.stderr)

    def test_missing_model_dir_is_blocking(self) -> None:
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_model.py"),
                               "--model-path", str(ROOT / "does-not-exist"), "--json-only",
                               "--out-dir", str(TMP)],
                              capture_output=True, text=True, timeout=60,
                              encoding="utf-8", errors="replace")
        self.assertEqual(done.returncode, 1)
        self.assertIn("不存在", done.stdout)


class EvalIntegrityTest(unittest.TestCase):
    def test_eval_files_match_zip(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_eval_tool  # noqa: PLC0415

        result = check_eval_tool.compare_integrity(ROOT / "eval", ROOT / "eval" / "eval.zip")
        self.assertEqual(result["modified"], [])
        self.assertEqual(result["missing"], [])
        self.assertTrue(all(entry["match"] for entry in result["members"].values()))


if __name__ == "__main__":
    unittest.main()
