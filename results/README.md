# 实测结果快照

本目录由 `scripts/export_results.py` 自动导出，只包含能说明结论的小文件；
原始日志与逐条预测体积过大，未纳入仓库。

| 文件 | 大小 | 来源 |
|---|---:|---|
| `experiments/AUTOTUNER_ctx16384_seq16_111238_20261031_111238/benchmark_result.json` | 4.5 KB | `experiments/AUTOTUNER_ctx16384_seq16_111238_20261031_111238/benchmark_result.json` |
| `experiments/AUTOTUNER_ctx16384_seq8_104748_20261031_104748/benchmark_result.json` | 4.5 KB | `experiments/AUTOTUNER_ctx16384_seq8_104748_20261031_104748/benchmark_result.json` |
| `experiments/AUTOTUNER_mtp_on_113802_20261031_113802/benchmark_result.json` | 4.5 KB | `experiments/AUTOTUNER_mtp_on_113802_20261031_113802/benchmark_result.json` |
| `experiments/E00_BASELINE_PERF_20261031_053524/performance/c1/summary.json` | 1.2 KB | `experiments/E00_BASELINE_PERF_20261031_053524/performance/c1/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_053524/performance/c2/summary.json` | 0.4 KB | `experiments/E00_BASELINE_PERF_20261031_053524/performance/c2/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_053524/performance/c4/summary.json` | 0.4 KB | `experiments/E00_BASELINE_PERF_20261031_053524/performance/c4/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_053524/performance/c8/summary.json` | 0.4 KB | `experiments/E00_BASELINE_PERF_20261031_053524/performance/c8/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_055409/performance/c1/summary.json` | 1.1 KB | `experiments/E00_BASELINE_PERF_20261031_055409/performance/c1/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_055409/performance/c2/summary.json` | 1.0 KB | `experiments/E00_BASELINE_PERF_20261031_055409/performance/c2/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_055409/performance/c4/summary.json` | 1.1 KB | `experiments/E00_BASELINE_PERF_20261031_055409/performance/c4/summary.json` |
| `experiments/E00_BASELINE_PERF_20261031_055409/performance/c8/summary.json` | 1.0 KB | `experiments/E00_BASELINE_PERF_20261031_055409/performance/c8/summary.json` |
| `experiments/accuracy/accuracy/summary.json` | 0.3 KB | `experiments/E00_BASELINE_ACC2_20261031_081016/accuracy/summary.json` |
| `experiments/accuracy/accuracy/summary.json` | 0.3 KB | `experiments/E00_BASELINE_ACC_20261031_055409/accuracy/summary.json` |
| `auto_tuner/trials.json` | 2.8 KB | `experiments/auto_tuner/trials.json` |
| `auto_tuner/report.md` | 0.9 KB | `experiments/auto_tuner/report.md` |
| `reports/实验报告.md` | 2.3 KB | `docs/实验报告.md` |

校验：`python3 -c "import json;print(json.load(open('results/manifest.json'))['count'])"`

每个 `summary.json` 的 `status` 为「完整」才计入正式成绩。
