#!/usr/bin/env python3
"""评测工具审计（只读）：文件完整性、CLI 参数、依赖版本、评测口径常量。

用法：
    python3 scripts/check_eval_tool.py [--out-dir reports/task0] [--json-only]

做三件事：
  1. 用 eval.zip 内记录的 SHA-256 校验 eval/ 下三个文件未被修改；
  2. 用 AST 解析 runner.py，列出它接受的参数、真正下发给 AIPerf/EvalScope 的参数；
  3. 若隔离环境已存在，则以只读方式执行 `--help` 与 import 探测，核对版本与参数是否匹配。

任何一步都不会安装依赖、不会请求模型服务。
退出码：0 通过；1 文件缺失或被修改；2 有告警（工具环境尚未准备、参数无法验证）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from _report import ROOT, configure_stdio, header, row, run, save

EVAL_FILES = ("README.md", "runner.py", "run.sh")
DOCUMENTED = {"aiperf": "0.13.0", "evalscope": "1.13.0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_members(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return {info.filename: hashlib.sha256(archive.read(info)).hexdigest()
                for info in archive.infolist() if not info.is_dir()}


def compare_integrity(eval_dir: Path, zip_path: Path) -> dict:
    result = {"zip": str(zip_path), "zip_present": zip_path.is_file(), "members": {}, "modified": [], "missing": []}
    if not zip_path.is_file():
        result["note"] = "未提供 eval.zip，跳过原始包比对"
        for name in EVAL_FILES:
            result["missing"].append(name) if not (eval_dir / name).is_file() else None
        return result
    members = zip_members(zip_path)
    result["zip_sha256"] = sha256(zip_path)
    for name in EVAL_FILES:
        local = eval_dir / name
        expected = members.get(name)
        actual = sha256(local) if local.is_file() else None
        result["members"][name] = {"expected": expected, "actual": actual,
                                   "present": local.is_file(), "match": expected is not None and expected == actual}
        if not local.is_file():
            result["missing"].append(name)
        elif expected and expected != actual:
            result["modified"].append(name)
    return result


def parse_runner(runner_path: Path) -> dict:
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    facts: dict = {"cli_options": [], "aiperf_flags": [], "accuracy_config_keys": [],
                   "constants": {}, "packages": {}}

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"ISL", "OSL", "WARMUP", "LIMIT", "MAX_TOKENS", "CONCURRENCIES", "PACKAGES", "IF_METRICS"}:
                try:
                    facts["constants"][name] = ast.literal_eval(node.value)
                except ValueError:
                    facts["constants"][name] = ast.unparse(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # parser.add_argument("--flag", ...)
            if isinstance(func, ast.Attribute) and func.attr == "add_argument" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
                    default = None
                    for keyword in node.keywords:
                        if keyword.arg == "default":
                            try:
                                default = ast.literal_eval(keyword.value)
                            except ValueError:
                                default = ast.unparse(keyword.value)
                    facts["cli_options"].append({"flag": first.value, "default": default})
        # functions that build tool commands
        if isinstance(node, ast.FunctionDef) and node.name in {"aiperf_command", "accuracy_config"}:
            strings = [item.value for item in ast.walk(node)
                       if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            if node.name == "aiperf_command":
                facts["aiperf_flags"] = sorted({text for text in strings if text.startswith("--")})
            else:
                facts["accuracy_config_keys"] = sorted({text for text in strings if text.endswith("_config")
                                                        or text in {"model", "api_url", "api_key_env", "eval_type",
                                                                    "datasets", "limit", "eval_batch_size", "work_dir",
                                                                    "no_timestamp"}})
    facts["packages"] = facts["constants"].get("PACKAGES", {})
    return facts


def runtime_probe(env_dir: Path, facts: dict) -> dict:
    """只读探测已存在的隔离环境：版本、AIPerf 参数、EvalScope 配置字段。"""
    probe: dict = {"env_dir": str(env_dir), "environments": {}, "flags_verified": None, "note": ""}
    for stage, package in (("perf", DOCUMENTED["aiperf"]), ("accuracy", DOCUMENTED["evalscope"])):
        dist = "aiperf" if stage == "perf" else "evalscope"
        python = env_dir / f"{dist}-{package}" / "bin" / "python"
        entry = {"python": str(python), "present": python.is_file()}
        if python.is_file():
            code = f"import importlib.metadata as m; print(m.version({dist!r}))"
            entry["version"] = run([str(python), "-c", code], timeout=60)["stdout"] or None
        probe["environments"][stage] = entry

    perf_python = probe["environments"]["perf"]["present"]
    aiperf = env_dir / f"aiperf-{DOCUMENTED['aiperf']}" / "bin" / "aiperf"
    if perf_python and aiperf.is_file():
        result = run([str(aiperf), "profile", "--help"], timeout=120)
        if result["ok"]:
            missing = [flag for flag in facts["aiperf_flags"] if flag not in result["stdout"]]
            probe["flags_verified"] = {"checked": len(facts["aiperf_flags"]), "missing": missing}
        else:
            probe["flags_verified"] = {"error": result["stderr"][:500]}
        tokenizer_probe = env_dir / f"aiperf-{DOCUMENTED['aiperf']}" / "bin" / "python"
        check = run([str(tokenizer_probe), "-c",
                     "from aiperf.common.tokenizer import Tokenizer;"
                     "print(type(Tokenizer.from_pretrained('builtin')).__name__)"], timeout=180)
        probe["builtin_tokenizer"] = {"ok": check["ok"],
                                      "detail": (check["stdout"] or check["stderr"])[:300]}

    accuracy_python = probe["environments"]["accuracy"]["present"]
    if accuracy_python:
        python = env_dir / f"evalscope-{DOCUMENTED['evalscope']}" / "bin" / "python"
        code = (
            "import json\n"
            "from evalscope.config import TaskConfig\n"
            "fields = set(getattr(TaskConfig, 'model_fields', {}).keys())\n"
            "wanted = ['model','api_url','api_key','eval_type','datasets','limit','eval_batch_size',"
            "'generation_config','work_dir','no_timestamp']\n"
            "cfg = TaskConfig(model='m', api_url='http://127.0.0.1:1/v1', api_key='EMPTY', eval_type='openai_api',"
            " datasets=['ifbench'], limit=1, eval_batch_size=2, generation_config={'max_tokens': 32768},"
            " work_dir='/tmp/evalscope-probe', no_timestamp=True)\n"
            "print(json.dumps({'missing_fields': [w for w in wanted if w not in fields],"
            " 'generation_config': cfg.generation_config.model_dump(mode='json') if hasattr(cfg.generation_config, 'model_dump') else str(cfg.generation_config)}))\n"
        )
        result = run([str(python), "-c", code], timeout=240)
        probe["task_config"] = {"ok": result["ok"],
                                "detail": (result["stdout"] or result["stderr"])[:800]}
    if not any(entry["present"] for entry in probe["environments"].values()):
        probe["note"] = ("隔离环境尚未准备（任务 0 不允许安装）。请在服务器交互终端执行 "
                         "`bash eval/run.sh --mode check` 并按提示确认安装后，再次运行本脚本。")
    return probe


def dry_run(eval_dir: Path, python: str | None) -> dict:
    """调用 runner.py --dry-run：不安装、不请求模型，只打印真实命令行。"""
    if not python:
        return {"ok": False, "detail": "未找到 Python >= 3.10"}
    result = run([python, str(eval_dir / "runner.py"), "--dry-run",
                  "--base-url", "http://127.0.0.1:9/v1", "--model", "probe-model"], timeout=120)
    return {"ok": result["ok"], "stdout": result["stdout"][:6000], "stderr": result["stderr"][:2000]}


def best_python() -> str | None:
    """runner.py 的入口要求 Python >= 3.10。"""
    import shutil

    for candidate in (shutil.which("python3.12"), shutil.which("python3"), sys.executable):
        if candidate and run([candidate, "-c", "import sys; raise SystemExit(sys.version_info < (3, 10))"],
                             timeout=30)["ok"]:
            return candidate
    return None


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="只读评测工具审计")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "manual")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval")
    parser.add_argument("--zip", type=Path, default=ROOT / "eval" / "eval.zip")
    parser.add_argument("--env-dir", type=Path, default=Path.home() / ".cache/ai-infra-eval/envs")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    warnings: list[str] = []
    payload: dict = {"eval_dir": str(args.eval_dir), "documented_versions": DOCUMENTED, "warnings": warnings}

    runner_path = args.eval_dir / "runner.py"
    if not runner_path.is_file():
        payload["error"] = f"缺少 {runner_path}"
        saved = save(args.out_dir, "eval_tool", payload)
        print(f"缺少 {runner_path}\n已记录：{saved}")
        return 1

    payload["integrity"] = compare_integrity(args.eval_dir, args.zip)
    if payload["integrity"]["modified"]:
        warnings.append(f"评测文件与原始 zip 不一致：{payload['integrity']['modified']}——"
                        f"若确实修改过，必须记录改动原因，且不得改变评测口径")
    if payload["integrity"]["missing"]:
        warnings.append(f"评测文件缺失：{payload['integrity']['missing']}")

    payload["runner_facts"] = parse_runner(runner_path)
    payload["runtime"] = runtime_probe(args.env_dir, payload["runner_facts"])
    payload["dry_run"] = dry_run(args.eval_dir, best_python())

    if payload["runtime"].get("flags_verified") is None:
        warnings.append("未能验证 AIPerf 参数（隔离环境未准备）：参数名是否被 0.13.0 支持，需要在服务器实测确认")
    elif payload["runtime"]["flags_verified"].get("missing"):
        warnings.append(f"AIPerf 0.13.0 不认识这些参数：{payload['runtime']['flags_verified']['missing']}")
    if payload["runtime"].get("task_config", {}).get("ok") is False:
        warnings.append("EvalScope TaskConfig 探测失败，需检查 config 字段名（见 runtime.task_config.detail）")

    if not args.json_only:
        header("评测文件完整性")
        integrity = payload["integrity"]
        row("eval.zip", integrity["zip"], f"sha256={integrity.get('zip_sha256', '—')}")
        for name, info in integrity["members"].items():
            row(f"  {name}", "一致" if info["match"] else ("缺失" if not info["present"] else "已修改"))
        header("runner.py 接受的参数")
        for option in payload["runner_facts"]["cli_options"]:
            row(option["flag"], option["default"])
        header("下发给 AIPerf 的参数")
        for flag in payload["runner_facts"]["aiperf_flags"]:
            row(flag, "")
        header("固定评测口径")
        for name, value in payload["runner_facts"]["constants"].items():
            row(name, value)
        header("隔离环境")
        for stage, entry in payload["runtime"]["environments"].items():
            row(stage, f"{entry['python']} {'存在' if entry['present'] else '不存在'}")
        if payload["runtime"].get("flags_verified"):
            row("AIPerf 参数核对", payload["runtime"]["flags_verified"])
        if payload["runtime"].get("note"):
            row("说明", payload["runtime"]["note"])
        header("dry-run 输出（真实命令行）")
        print(payload["dry_run"].get("stdout") or payload["dry_run"].get("detail"))

    saved = save(args.out_dir, "eval_tool", payload)
    for item in warnings:
        print(f"[告警] {item}")
    print(f"已记录：{saved}")
    return 1 if payload["integrity"]["modified"] or payload["integrity"]["missing"] else (2 if warnings else 0)


if __name__ == "__main__":
    raise SystemExit(main())
