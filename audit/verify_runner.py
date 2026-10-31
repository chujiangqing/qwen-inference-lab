#!/usr/bin/env python3
"""Static + behavioural audit of eval/runner.py (no network, no third party deps)."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else r"eval/runner.py").resolve()
spec = importlib.util.spec_from_file_location("runner", SRC)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# Keep scratch data inside the workspace: the sandbox forbids the system temp dir.
# Never use tempfile.mkdtemp here: mode 0o700 yields a directory this process cannot
# write into under the Windows sandbox (verified), while a default-mode mkdir works.
TMPROOT = Path(__file__).with_name("_tmp")
TMPROOT.mkdir(exist_ok=True)
_CASES = itertools.count()


def newdir(prefix: str) -> Path:
    folder = TMPROOT / f"{prefix}-{os.getpid()}-{next(_CASES)}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    return folder


RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail and not ok else ""))


def expect_raises(name: str, fn, exc) -> None:
    try:
        fn()
    except exc as error:
        check(name, True)
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")
    else:
        check(name, False, "no exception")


# ---------------------------------------------------------------- URL handling
def urls() -> None:
    n = runner.normalize_api_url
    cases_ok = {
        "http://h:8088/v1": "http://h:8088/v1",
        "http://h:8088": "http://h:8088/v1",
        "http://h:8088/v1/": "http://h:8088/v1",
        "https://host/proxy/v1": "https://host/proxy/v1",
        "http://h/openai": "http://h/openai/v1",
    }
    for raw, want in cases_ok.items():
        try:
            got = n(raw)
        except Exception as error:  # noqa: BLE001
            check(f"normalize {raw}", False, f"raised {type(error).__name__}")
            continue
        check(f"normalize {raw} -> {want}", got == want, got)
    for raw in ("http://u:p@h/v1", "http://h/v1?a=1", "http://h/v1#x",
                "http://h/v1/completions", "file:///tmp/v1", "ftp://h/v1"):
        expect_raises(f"reject {raw}", lambda raw=raw: n(raw), Exception)
    # Documented-but-odd: a path that merely contains "v1" gets a second /v1.
    got = n("http://h/v1beta")
    check("note: http://h/v1beta is rewritten to .../v1beta/v1", got == "http://h/v1beta/v1", got)


# ------------------------------------------------------------------- statistics
def stats() -> None:
    p = runner.percentile
    check("p50 of even count interpolates", math.isclose(p([1, 2, 3, 4], 50), 2.5, abs_tol=1e-9))
    check("p95 of single value is that value", p([7.0], 95) == 7.0)
    check("p99 <= max", p([float(i) for i in range(100)], 99) <= 99)
    d = runner.distribution([1.0, 2.0, 3.0])
    check("distribution keys", set(d) == {"mean", "p50", "p95", "p99"}, d)
    # Doc's concurrency-1 cell: 12 samples -> P95 sits between order stats 11 and 12.
    position = (12 - 1) * 95 / 100
    check("n=12 P95 is an extrapolation between 11th/12th order stats",
          math.isclose(position, 10.45), position)
    check("n=12: only ~0.6 samples above P95", math.isclose(12 * 0.05, 0.6), 0.6)


# --------------------------------------------------------- synthetic AIPerf export
def make_perf(folder: Path, *, n: int = 12, conc: int = 1, isl: int = 8192, osl: int = 512,
              unit: str = "ms", warmups: int = 8, complete: bool = True,
              tamper_p95: bool = False, scalar_shape: bool = True, warm_len_wrong: bool = False,
              drop_records: int = 0, error_requests: int = 0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    recs, ttfts, itls, lats = [], [], [], []
    for i in range(n):
        ttft = 3000.0 + 95.0 * i
        itl = 31.4 + 0.33 * i
        lat = ttft + itl * (osl - 1)
        ttfts.append(ttft); itls.append(itl); lats.append(lat)
        recs.append({"metadata": {"benchmark_phase": "profiling"},
                     "metrics": {"time_to_first_token": {"value": ttft, "unit": unit},
                                 "inter_token_latency": {"value": itl, "unit": unit},
                                 "request_latency": {"value": lat, "unit": unit},
                                 "input_sequence_length": {"value": isl, "unit": "tokens"},
                                 "output_sequence_length": {"value": osl, "unit": "tokens"}}})
    for i in range(warmups):
        recs.append({"metadata": {"benchmark_phase": "warmup"},
                     "metrics": {"time_to_first_token": {"value": 2900.0, "unit": "ms"},
                                 "inter_token_latency": {"value": 31.0, "unit": "ms"},
                                 "request_latency": {"value": 18000.0, "unit": "ms"},
                                 "input_sequence_length": {"value": isl if not warm_len_wrong else isl - 1,
                                                           "unit": "tokens"},
                                 "output_sequence_length": {"value": osl, "unit": "tokens"}}})
    if drop_records:
        kept = recs[: n - drop_records] + recs[n:]  # drop profiling records, keep every warmup
    else:
        kept = recs
    with (folder / "profile_export.jsonl").open("w", encoding="utf-8") as stream:
        for r in kept:
            stream.write(json.dumps(r) + "\n")
    duration = 234.32
    native = {
        "is_complete": complete, "was_cancelled": False,
        "request_count": {"avg": n} if scalar_shape else n,
        "error_request_count": {"avg": error_requests},
        "request_error_rate": {"avg": 0},
        "error_summary": {},
        "benchmark_duration": {"avg": duration},
        "request_throughput": {"avg": n / duration},
        "output_token_throughput": {"avg": n * osl / duration},
    }
    if scalar_shape and ttfts:
        for tag, vals in (("time_to_first_token", ttfts), ("inter_token_latency", itls),
                          ("request_latency", lats)):
            native[tag] = {"avg": sum(vals) / len(vals)}
            for q, key in ((50, "p50"), (95, "p95"), (99, "p99")):
                native[tag][key] = runner.percentile(vals, q)
        if tamper_p95:
            native["time_to_first_token"]["p95"] *= 1.10
    (folder / "profile_export_aiperf.json").write_text(json.dumps(native), encoding="utf-8")


def perf_case(name: str, expect_status: str, expect_issue: str = "", *,
              expected: int | None = None, warmup_expected: int | None = None,
              strict_isl: bool = True, **make_kwargs) -> None:
    tmp = newdir("perf")
    try:
        make_perf(tmp, **make_kwargs)
        got = runner.parse_performance(tmp, expected if expected is not None else make_kwargs.get("n", 12),
                                       make_kwargs.get("conc", 1),
                                       warmup_expected=warmup_expected if warmup_expected is not None
                                       else make_kwargs.get("warmups", 8),
                                       strict_isl=strict_isl)
        ok = got["status"] == expect_status and (expect_issue in " ".join(got["issues"]) if expect_issue else True)
        check(f"parse_performance: {name}", ok, f"{got['status']} {got['issues']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def perf_suite() -> None:
    perf_case("clean 12/8192/512 -> 完整", "完整")
    perf_case("one profiling record missing -> 异常", "异常", "期望 12", drop_records=1)
    perf_case("native P95 tampered -> 异常", "异常", "p95 与原生汇总不一致", tamper_p95=True)
    perf_case("TTFT unit seconds -> 异常", "异常", "单位不是 ms", unit="s")
    perf_case("ISL 8191 strict -> 异常", "异常", "8192/512", isl=8191)
    perf_case("OSL 511 -> 异常", "异常", "8192/512", osl=511)
    perf_case("7 warmups -> 异常", "异常", "预热", warmups=7, warmup_expected=8)
    perf_case("warmup length wrong -> 异常", "异常", "预热", warm_len_wrong=True)
    perf_case("run cancelled -> 异常", "异常", "未完整结束", complete=False)
    perf_case("request errors -> 异常", "异常", "请求失败", error_requests=2)
    perf_case("expected 100 but ran 12 -> 异常", "异常", "期望 100", n=12, expected=100, warmups=8)
    perf_case("builtin mode tolerates non-8192 ISL -> 完整", "完整", strict_isl=False, isl=7900)


def perf_schema_probe() -> None:
    """Does a scalar-shaped export crash instead of being reported as one bad cell?"""
    tmp = newdir("perf-schema")
    try:
        make_perf(tmp, scalar_shape=False)
        try:
            runner.parse_performance(tmp, 12, 1, warmup_expected=8, strict_isl=True)
        except Exception as error:  # noqa: BLE001
            escaped = type(error).__name__
        else:
            escaped = "none"
        check("scalar request_count escapes as AttributeError", escaped == "AttributeError", escaped)
        # Show that run_performance() does not catch it.
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        caught: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_performance":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ExceptHandler):
                        names = sub.type.elts if isinstance(sub.type, ast.Tuple) else [sub.type]
                        caught.extend(getattr(x, "id", "?") for x in names)
        check("run_performance() does not catch AttributeError", "AttributeError" not in caught, caught)
        check("run_performance() does not catch IndexError", "IndexError" not in caught, caught)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- synthetic EvalScope export
def make_accuracy(folder: Path, *, n: int = 100, scores: dict | None = None,
                  drop: int = 0, error_sample: bool = False, execution: bool = True,
                  metric_missing: str = "") -> None:
    reports = folder / "native/reports/ifbench"
    reports.mkdir(parents=True, exist_ok=True)
    scores = scores or {"prompt_level_strict": 0.76, "inst_level_strict": 0.81,
                        "prompt_level_loose": 0.79, "inst_level_loose": 0.84}
    metrics = [{"legacy_name": k, "score": v, "num": n} for k, v in scores.items() if k != metric_missing]
    native = {"dataset_name": "ifbench", "metrics": metrics}
    if execution:
        native["execution_summary"] = {"requested": n, "succeeded": n, "errored": 0, "incomplete": 0}
    (reports / "ifbench.json").write_text(json.dumps(native), encoding="utf-8")
    preds = folder / "native/predictions"
    preds.mkdir(parents=True, exist_ok=True)
    with (preds / "ifbench.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(n - drop):
            stream.write(json.dumps({"index": i, "error": True if (error_sample and i == 0) else None,
                                     "model_output": {"choices": [{"message": {"content": "ok"}}]}}) + "\n")


def accuracy_case(name: str, expect_status: str, expect_issue: str = "", **kwargs) -> None:
    tmp = newdir("acc")
    try:
        make_accuracy(tmp, **kwargs)
        got = runner.parse_accuracy(tmp)
        ok = got["status"] == expect_status and (expect_issue in " ".join(got["issues"]) if expect_issue else True)
        check(f"parse_accuracy: {name}", ok, f"{got['status']} {got['issues']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def accuracy_suite() -> None:
    accuracy_case("clean 100 samples -> 完整", "完整")
    accuracy_case("99 predictions -> 异常", "异常", "100", drop=1)
    accuracy_case("sample error flag -> 异常", "异常", "推理错误", error_sample=True)
    accuracy_case("missing execution_summary -> 异常", "异常", "execution_summary", execution=False)
    accuracy_case("missing metric -> 异常", "异常", "inst_level_loose 缺少", metric_missing="inst_level_loose")
    accuracy_case("score 1.5 -> 异常", "异常", "有效原生分数",
                  scores={"prompt_level_strict": 1.5, "inst_level_strict": 0.8,
                          "prompt_level_loose": 0.8, "inst_level_loose": 0.8})
    # The helper sets one num for every metric, so force distinct counts afterwards.
    tmp = newdir("acc-num")
    try:
        make_accuracy(tmp)
        path = tmp / "native/reports/ifbench/ifbench.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for m in data["metrics"]:
            if m["legacy_name"] == "inst_level_strict":
                m["num"] = 512
        path.write_text(json.dumps(data), encoding="utf-8")
        got = runner.parse_accuracy(tmp)
        check("instruction-level num>100 is accepted (per README)", got["status"] == "完整", got["issues"])
        for m in data["metrics"]:
            if m["legacy_name"] == "prompt_level_strict":
                m["num"] = 90
        path.write_text(json.dumps(data), encoding="utf-8")
        got = runner.parse_accuracy(tmp)
        check("prompt-level num=90 is rejected",
              got["status"] == "异常" and "不是 100" in " ".join(got["issues"]), got["issues"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ argument / CLI wiring
def cli() -> None:
    args = runner.parse_args(["--base-url", "http://h:8088/v1", "--model", "m"])
    check("defaults: perf_requests=100, batch=2, timeout=600, tokenizer=builtin",
          (args.perf_requests, args.eval_batch_size, args.request_timeout, args.tokenizer) == (100, 2, 600.0, "builtin"))
    for argv, label in (
        (["--base-url", "http://h/v1", "--model", "m", "--eval-batch-size", "0"], "reject batch 0"),
        (["--base-url", "http://h/v1", "--model", "m", "--perf-requests", "0"], "reject requests 0"),
        (["--base-url", "http://h/v1", "--model", "m", "--request-timeout", "0"], "reject timeout 0"),
        (["--base-url", "http://h/v1", "--model", "m", "--tokenizer-revision", "abc"], "reject revision with builtin"),
        (["--model", "m"], "reject missing base-url"),
    ):
        expect_raises(label, lambda argv=argv: runner.parse_args(argv), SystemExit)

    args = runner.parse_args(["--base-url", "https://host/proxy/v1", "--model", "m"])
    cmd = runner.aiperf_command(args, 4, Path("<OUT>/dataset.jsonl"), Path("<OUT>/c4"), "rid:c4", "sk-x")
    joined = " ".join(cmd)
    checks = {
        "endpoint path is /proxy/v1/completions": "--endpoint /proxy/v1/completions" in joined,
        "url is bare origin": cmd[cmd.index("--url") + 1] == "https://host",
        "no gpu telemetry": "--no-gpu-telemetry" in cmd,
        "2 workers": "--workers-max 2" in joined,
        "export level records": "--export-level records" in joined,
        "extra inputs carry ignore_eos/temperature/cache_salt":
            all(k in joined for k in ("ignore_eos", "temperature", "cache_salt")),
        "legacy max tokens flag": "--use-legacy-max-tokens" in cmd,
    }
    for label, ok in checks.items():
        check("aiperf_command: " + label, ok)
    check("aiperf_command: --tokenizer builtin preserved", "--tokenizer builtin" in joined)


def report_render() -> None:
    tmp = newdir("render")
    try:
        make_perf(tmp / "performance", n=12)
        make_accuracy(tmp)
        perf = runner.parse_performance(tmp / "performance", 12, 1, warmup_expected=8, strict_isl=True)
        acc = runner.parse_accuracy(tmp)
        summary = {"started_at": "2026-10-31T00:00:00+00:00", "ended_at": "2026-10-31T01:00:00+00:00",
                   "status": "完整", "issues": [],
                   "configuration": {"model": "qwen3.8-27b-fp8", "base_url": "http://h:8088/v1",
                                     "tokenizer": "builtin", "eval_batch_size": 2},
                   "performance": [perf], "accuracy": acc}
        runner.render_report(tmp, summary)
        text = (tmp / "report.md").read_text(encoding="utf-8")
        checks = {
            "has performance table row": "| 1 | 完整 |" in text,
            "TTFT converted ms->s": "| 3." in text or "| 4." in text,
            "TPOT stays ms": "TPOT（毫秒/token）" in text,
            "primary metric highlighted": "prompt_level_strict" in text and "0.76" in text,
            "no unexplained exception section": "## 异常与证据" in text,
        }
        for label, ok in checks.items():
            check("render_report: " + label, ok)
        check("render_report: clean run has no issue bullets",
              "未发现完整性异常" in text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def redaction() -> None:
    secret = "sk-abc/def+ghi"
    out = runner.redact(f"key={secret} json={json.dumps(secret)} quoted={secret}", secret)
    check("redact removes literal secret", secret not in out)
    check("redact leaves EMPTY alone", runner.redact("EMPTY", "EMPTY") == "EMPTY")


def doc_arithmetic() -> None:
    """Recompute every derived column of the supplied benchmark document."""
    rows = [(1, 12, 234.32, 3.481, 31.441, 19.527, 0.05121, 26.22),
            (2, 24, 286.01, 5.023, 37.909, 23.826, 0.08391, 42.96),
            (3, 24, 220.13, 7.499, 44.974, 28.865, 0.10902, 55.82),
            (4, 32, 249.08, 9.395, 51.992, 34.914, 0.12847, 65.78),
            (8, 24, 186.76, 41.822, 53.106, 67.417, 0.12850, 65.79),
            (2, 100, 1191.12, 4.608, 37.905, 23.818, 0.08395, 42.98)]
    worst = 0.0
    for conc, n, dur, ttft, tpot, e2e, rps, tps in rows:
        rps_err = abs(n / dur - rps) / rps
        tps_err = abs(n * 512 / dur - tps) / tps
        worst = max(worst, rps_err, tps_err)
    check("document derived columns are self-consistent (<0.1%)", worst < 1e-3, f"max rel err {worst:.2e}")
    # P95s are not additive: only check approximate coherence of the reported cells.
    coherent = all(ttft < e2e and 0.85 <= (ttft + tpot * 0.001 * 511) / e2e <= 1.15
                   for _, _, _, ttft, tpot, e2e, _, _ in rows)
    check("TTFT + TPOT*(OSL-1) is within 15% of end-to-end for every row", coherent)
    counts = [r[1] for r in rows[:5]]
    check("note: per-level request counts are NOT uniform (12/24/24/32/24)", len(set(counts)) > 1, counts)
    c4, c8 = rows[3], rows[4]
    check("concurrency 8 gives no throughput gain over 4 (+0.02%)",
          abs(c8[7] / c4[7] - 1) < 0.001, f"{c8[7] / c4[7]:.5f}")
    check("concurrency 8 TTFT P95 is 4.45x concurrency 4", math.isclose(c8[3] / c4[3], 4.452, abs_tol=0.01))
    check("concurrency 8 E2E P95 is 1.93x concurrency 4", math.isclose(c8[5] / c4[5], 1.9309, abs_tol=0.01))
    # c2 repeat: 24 vs 100 requests must agree if the measurement is stable.
    c2, c2r = rows[1], rows[5]
    check("c2 24-sample vs 100-sample throughput agree within 0.1%",
          abs(c2[6] / c2r[6] - 1) < 1e-3, f"{c2[6] / c2r[6]:.6f}")
    check("c2 24-sample vs 100-sample TTFT P95 differ by 9%",
          math.isclose(c2[3] / c2r[3], 1.0901, abs_tol=0.01), f"{c2[3] / c2r[3]:.4f}")
    # Binomial standard error for a 100-sample accuracy score.
    se = math.sqrt(0.78 * 0.22 / 100)
    check("100-sample score has ~4pp binomial SE", math.isclose(se, 0.0414, abs_tol=1e-3), round(se, 4))
    check("80% vs 76% is inside 1 SE (not a detectable difference)", abs(0.80 - 0.76) < se)
    # eval.zip refuses the accuracy stage when the published context is <= 32768.
    gate = lambda length: isinstance(length, (int, float)) and length <= runner.MAX_TOKENS  # noqa: E731
    check("eval.zip gates accuracy when max_model_len == 32768", gate(32768))
    check("eval.zip allows accuracy when max_model_len == 40960", not gate(40960))
    check("eval.zip allows accuracy when /models reports nothing", not gate(None))


def main() -> int:
    print(f"auditing {SRC}\n")
    urls(); stats(); perf_suite(); perf_schema_probe(); accuracy_suite(); cli(); report_render(); redaction()
    doc_arithmetic()
    total = len(RESULTS)
    failed = [r for r in RESULTS if not r["pass"]]
    print(f"\n{total - len(failed)}/{total} checks passed")
    out = Path(__file__).with_name("verify_results.json")
    out.write_text(json.dumps({"source": str(SRC), "total": total,
                               "failed": failed, "results": RESULTS}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"results -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
