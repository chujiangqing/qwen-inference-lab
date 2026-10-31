# eval.zip 评测工具审计报告

> 审计对象：比赛提供的 `eval/eval.zip`（`README.md` / `runner.py` 780 行 / `run.sh` 100 行）
> 审计方式：静态阅读 + AST 提取 + 用合成数据执行 `runner.py` 的内部函数（76 项断言，全部通过）
> 审计结论：**工具本身可用，但有 2 个会直接改变评测结果的口径陷阱，必须在阶段 2 之前决策；另有 3 个解析健壮性问题需要在适配器层兜底。**

---

## 1. 结论摘要

| 编号 | 级别 | 结论 | 影响 |
|---|---|---|---|
| P0-1 | 阻断 | `--max-model-len 32768` 时，eval.zip 会把精度阶段判为「配置异常」并**跳过 IFBench** | 无法用官方工具复现 76%/80% 精度基线 |
| P0-2 | 阻断 | 官方默认口径与基准文档原始协议**不是同一套**（预热 8 vs max(4,N)、并发档位、每档请求数、超时 600 vs 180、tiktoken vs 模型 tokenizer） | 两套数字不可逐项比较，报告中必须标明口径 |
| P1-1 | 高 | `parse_performance` 遇到非预期 JSON 结构会抛 `AttributeError`，且不在捕获列表内 | 单档解析失败会终止后续所有评测，而不是只标该档异常 |
| P1-2 | 中 | 百分位用线性插值重算后与 AIPerf 原生值比对（容差 1e-5） | 若 AIPerf 用别的分位算法或做了舍入，会误报「与原生汇总不一致」 |
| P1-3 | 中 | `execute()` 在子进程正常退出后仍对进程组执行 `SIGKILL` | 清理残留 worker 的副作用，Linux 上需实测确认无截断写盘 |
| P2 | 低 | `http://host/v1beta` 会被补成 `/v1beta/v1`；`--mode` 只认 `all/perf/accuracy/check` | 罕见路径写法，按文档使用不受影响 |

---

## 2. 工具做了什么（读代码得到的事实）

### 2.1 执行链

```
run.sh                      依赖探测（Python>=3.10）→ 有则直接 exec runner.py
  └─ 无 Python 时          uv 引导安装 → uv python install 3.12（必须交互确认 y）
runner.py --mode all        1) GET /v1/models  （失败则不发送任何评测负载）
                            2) 性能阶段 perf:  生成 dataset.jsonl → 探针请求 → 4 档 AIPerf
                            3) 精度阶段 accuracy: EvalScope ifbench limit=100
                            4) report.md / summary.json / environment.json
```

工具环境被固定在 `~/.cache/ai-infra-eval/envs/{aiperf-0.13.0,evalscope-1.13.0}`，
不碰系统 Python，不 sudo，不改 shell 配置。这部分设计良好，且 `--mode check` 可只检查不安装。

### 2.2 命令行参数（`runner.py` 全部 13 个）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `all` | `all` / `perf` / `accuracy` / `check` |
| `--base-url` | 无 | API 根地址，必须带或自动补 `/v1`；拒绝含凭据/查询串/具体接口的 URL |
| `--model` | 无 | 必须与 `/v1/models` 返回的 `id` 完全一致 |
| `--tokenizer` | `builtin` | `builtin`=tiktoken(o200k_base)；传模型目录则改为**模型 token-ID 输入** |
| `--tokenizer-revision` | 无 | 固定远程 tokenizer 版本；与 `builtin` 互斥 |
| `--api-key-env` | `OPENAI_API_KEY` | key 只从环境变量读，禁止写在 URL |
| `--output-dir` | `outputs/时间戳` | 已存在则拒绝，权限 0700 |
| `--env-dir` | `~/.cache/ai-infra-eval/envs` | 工具环境根目录 |
| `--perf-requests` | `100` | **四档共用同一个请求数** |
| `--eval-batch-size` | `2` | EvalScope API 并发 |
| `--request-timeout` | `600` | 秒 |
| `--seed` | `42` | 性能输入随机种子 |
| `--dry-run` | — | 打印配置与命令，不安装不请求 |

### 2.3 不可配置（硬编码）的评测口径

| 常量 | 值 | 位置 |
|---|---|---|
| `CONCURRENCIES` | `(1, 2, 4, 8)` | runner.py:27 |
| `WARMUP` | `8`（固定，不是 `max(4, 并发)`） | runner.py:28 |
| `ISL / OSL` | `8192 / 512` | runner.py:28 |
| `LIMIT` | `100`（IFBench 样本数） | runner.py:28 |
| `MAX_TOKENS` | `32768`（精度阶段生成上限，且**只覆盖这一项**） | runner.py:28 |
| `PACKAGES` | `aiperf==0.13.0`、`evalscope[ifbench]==1.13.0` | runner.py:29 |
| 评分指标 | `prompt_level_strict` | README:50 |

下发给 AIPerf 的参数共 24 个（`--tokenizer`、`--use-server-token-count`、`--use-legacy-max-tokens`、
`--custom-dataset-type single-turn`、`--dataset-sampling-strategy sequential`、`--warmup-request-count`、
`--workers-max 2`、`--request-timeout-seconds`、`--export-level records`、`--ui none`、
`--no-gpu-telemetry`、`--extra-inputs {"ignore_eos":true,"temperature":0,"cache_salt":...}` 等）。
是否都被 AIPerf 0.13.0 支持，本机无法验证——`scripts/check_eval_tool.py` 已实现服务器端只读核对。

---

## 3. P0-1：上下文门禁会让精度阶段被整段跳过

`runner.py:567-571`：

```python
length = model_info.get("max_model_len") or model_info.get("context_length")
if isinstance(length, (int, float)) and length <= MAX_TOKENS:   # MAX_TOKENS = 32768
    result = {"status": "异常", "issues": [f"模型上下文上限 {length} 无法容纳 IFBench 提示词 + "
                                           f"max_tokens={MAX_TOKENS}；未自动缩短输出。"], ...}
    write_json(folder / "summary.json", result)
    return result          # ← 直接返回，不启动 EvalScope
```

而基准文档给出的启动命令是 `--max-model-len 32768`。vLLM 的 `/v1/models` 会回报
`max_model_len=32768`，于是 `32768 <= 32768` 成立 → **精度阶段被跳过，报告里只有一行「配置异常」**。

这意味着：

- 用官方工具 + 文档给的启动参数，**拿不到 IFBench 分数**；
- 文档里的 76% / 80% 必然是在别的条件下产生的（要么 `--max-model-len` 更大，要么该服务的
  `/v1/models` 没有回报 `max_model_len`，要么用的是不同版本的评测脚本）。

**建议**（阶段 2 决策）：

1. 启动服务时把 `--max-model-len` 提到 `40960` 或 `65536`，并实测显存是否够
   （`scripts/check_model.py` 会给出 KV Cache 容量估算）；
2. 若显存不允许，则在报告中明确写「精度基线按 xx 口径采集」，并改用直接调用 EvalScope；
3. 无论哪种，第一次跑完必须检查 `accuracy/summary.json` 的 `status` 是否为 `完整`，
   而不是只看有没有生成文件。

> 该结论已由自动化测试固定：`eval.zip gates accuracy when max_model_len == 32768` → PASS。

---

## 4. P0-2：两套口径，数字不可互换

| 项目 | 口径 A：官方 eval.zip 默认 | 口径 B：基准文档原始协议 |
|---|---|---|
| 工具 | `eval/run.sh` | 直接调 AIPerf |
| 输入计数 | tiktoken `o200k_base` 8192（服务端实际 ISL 会不同） | 模型 tokenizer 抽取 8192 个有效 token-ID |
| 预热 | 固定 8 个 | `max(4, 并发数)` |
| 并发档 | 1 / 2 / 4 / 8 | 1 / 2 / 4 / 8 / 3 |
| 每档正式请求 | 100 | 12 / 24 / 24 / 32 / 24（各档不同） |
| 请求超时 | 600 s | 180 s |
| 随机种子 | 单一 `--seed 42`，四档共用同一份数据 | 逐档不同种子 |
| 缓存隔离 | 每档不同 `cache_salt` | 不添加公共前缀、逐档不同种子 |
| 精度 | ifbench limit=100, batch=2, max_tokens=32768 | 同左，但未说明 batch |

**结论**：口径 A 适合作为"可复现的正式成绩"；口径 B 只能用于"与文档基线对照"。
跨口径**只允许比较趋势**（例如"并发 4→8 吞吐饱和"这类结构性现象），**不允许比较绝对值**。
阶段 2 的 Adapter 必须把口径标识写进每条实验记录（`configs/benchmark.yaml` 已经给出两套模板）。

---

## 5. P1 类问题（适配器层要兜底）

### P1-1 解析崩溃会吃掉整个评测

`run_performance` 的异常捕获是 `(OSError, ValueError, KeyError, TypeError)`，
`main` 的外层是 `(EvaluationError, OSError, ValueError)`，都不含 `AttributeError` / `IndexError`。
而 `parse_performance` 里有 `native.get("request_count", {}).get("avg")` 这类链式取值：
只要 AIPerf 某次导出把 `request_count` 写成标量而不是 `{"avg": ...}`，就会抛 `AttributeError`，
被最外层 `except Exception` 兜住，结果是「执行错误」——**该轮性能结果丢失，精度阶段也不再执行**。

已用合成数据复现（测试项 `scalar request_count escapes as AttributeError`）。

**对策**：阶段 2 的 `result_parser.py` 自己读 `profile_export_aiperf.json` / `profile_export.jsonl`，
对字段缺失做降级（缺就当 `None` 并在报告里写「未导出」），不要依赖 `runner.py` 的解析结果；
同时以 `performance/summary.json` 是否存在、`status` 是否为 `完整` 作为入库前的强制校验。

**落地状态（2026-10-31 补记）**：
首版只写了策略、**没有真正改代码**，被代码评审当场指出（`result_parser.py` 里仍是
`(native.get("request_count") or {}).get("avg")`，标量输入照样抛 `AttributeError`）。现已修复：

- 新增 `_avg()`：同时接受 `{"avg": x}` 与标量 `x`，其它类型一律当缺失返回 `None`；
- 新增 `_dict()`：所有链式取值前先归一化成 dict，避免 `int` 上取 `.get`；
- 回归测试 `tests/test_pipeline_contracts.py::ScalarFieldTest` 用标量输入验证不再抛异常，
  且 `status` 标为「未核验」而不是「完整」——避免把降级数据当权威结果。

### P1-2 百分位一致性校验偏严

`parse_performance` 用线性插值重算 p50/p95/p99，再与 AIPerf 原生导出比对，
`rel_tol=1e-6, abs_tol=1e-5`。AIPerf 若用 nearest-rank 或对导出值做了舍入，就会误报
`xxx p95 与原生汇总不一致`，把整档标记为 `异常`。**第一次真实运行后必须确认有没有这条 issue**，
若有，说明两边分位算法不同，需要在适配器里以「客户端重算值」为准并记录该差异。

### P1-3 子进程组清理

`execute()` 在 `finally` 里对已退出子进程的进程组执行 `SIGKILL`（用于清理残留 worker）。
正常路径下无影响；但若某个 worker 仍在写 artifact，存在截断风险。建议 eval 跑完后校验
`profile_export_aiperf.json` 能 `json.loads` 成功（适配器里做，不要改官方脚本）。

---

## 6. 精度结果的完整性校验（工具做得好的部分）

`parse_accuracy` 的校验比多数自研脚本严格，值得保留并在适配器中复用：

- 预测样本数必须等于 100；
- 每条预测必须有 `model_output.choices`，且不得带 `error`；
- `index` 不得缺失或重复；
- 四项指标分数必须在 `[0,1]`；
- `prompt_level_strict` / `prompt_level_loose` 的 `num` 必须等于 100
  （指令级两项允许 >100，因为按指令加权）；
- `execution_summary.requested/succeeded` 必须等于 100，`errored`/`incomplete` 必须为 0；
- 缺 `execution_summary` 直接判「无法核验完整性」。

这正好满足需求书 6.5「不得用有效的少量评分冒充完整的 100 样本测评」。

---

## 7. 数据自洽性审计（针对基准文档的数字）

用脚本复算了文档表格的每一列（`audit/verify_runner.py`，全部通过）：

| 检查 | 结果 |
|---|---|
| 请求/s、输出 token/s 与测量时长是否自洽 | 6 行全部吻合，最大相对误差 < 0.1% |
| 并发 2：24 请求 vs 100 请求复测 | 吞吐差 0.05%、TPOT P95 差 0.01%、TTFT P95 差 9% → 复现性良好 |
| 并发 4→8 | 吞吐 +0.02%（已饱和）；TTFT P95 ×4.45、E2E P95 ×1.93 → 队列堆积 |
| 每档正式请求数 | 12 / 24 / 24 / 32 / 24，**不一致**，文档未解释原因 |
| P95 的样本量 | n=12 时 P95 落在第 11–12 个次序统计量之间，尾部几乎无分辨率 |
| 精度 80% vs 76% | n=100 的二项标准误 ≈ 4.1pp，差 4pp 在 1 SE 内，**不能声称 FP8 优于非 FP8** |

由此产生的报告写作要求（答辩会被追问）：

1. 并发 8 的 TTFT P95 41.8 s 是**排队结果**，不是单请求能力，必须和吞吐一起讲；
2. 精度只报 `prompt_level_strict`，并标注 `limit=100` 与二项误差，不给"提升 4 个百分点"这类结论；
3. 每档请求数不同这件事，要么在阶段 2 用统一请求数重测，要么在报告里写明这是"按时长预算截断"。

---

## 8. 依赖版本核对

| 依赖 | 固定版本 | 是否真实存在 | 备注 |
|---|---|---|---|
| AIPerf | 0.13.0 | 是 | [NVIDIA AIPerf v0.13.0 文档](https://docs.nvidia.com/aiperf/v0.13.0/)、dynamo 也 pin 了 `aiperf==0.13.0` |
| EvalScope | 1.13.0 | 是 | [PyPI 1.13.0](https://pypi.org/project/evalscope/1.13.0/)（2026-10-13 发布），`ifbench` extra 提供 `emoji` / `nltk>=3.9` / `syllapy>=0.8.0` |
| NLTK 资源 | `punkt_tab` / `stopwords` / `averaged_perceptron_tagger_eng` | 需下载 | 工具会下载到 `envs/nltk_data`，需要网络 |
| uv / Python 3.12 | 安装时获取 | 需确认 | 服务器若无外网，必须先离线准备好 |

版本存在不等于参数兼容。**必须在服务器执行**：

```bash
python3 scripts/check_eval_tool.py --out-dir reports/task0
```

它会在不安装任何东西的前提下，用已存在的隔离环境执行 `aiperf profile --help`，
逐个核对 `runner.py` 实际下发的 24 个参数；并构造一次 `TaskConfig(...)` 验证
`generation_config`、`work_dir`、`no_timestamp` 等字段能否被 1.13.0 接受。

---

## 9. 对阶段 2 的三条硬要求

1. **Adapter 只调用、不复制口径**：命令行参数、`isl/osl/warmup` 一律从 `runner.py` 的
   `--dry-run` 输出解析或直接透传，禁止在适配器里重新实现一遍。
2. **以原生文件为唯一事实来源**：`profile_export_aiperf.json`、`profile_export.jsonl`、
   `accuracy/native/reports/**`、`summary.json` 全部落盘并在数据库里存 hash；
   汇总指标入 SQLite，原始文件按 `experiment_artifacts` 索引。
3. **失败必须保留**：`status != 完整` 的实验照样入库并打标，不得删除；
   OOM / 超时 / 门禁跳过都要有明确 `issues` 文本。
