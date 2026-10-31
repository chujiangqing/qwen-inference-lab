# 评测口径说明（benchmark methodology）

> 用途：阶段 2 的 Adapter 实现依据、实验报告模板、答辩时的口径解释。
> 原则：**口径不可改**。任何优化都必须在这套口径下取得，不能靠改 ISL/OSL/并发定义制造提升。

---

## 1. 两套口径

### 1.1 口径 A：官方 `eval/run.sh`（正式成绩以此为准）

由 `runner.py` 硬编码，**不可配置**的部分：

| 项 | 值 |
|---|---|
| 并发档 | 1 / 2 / 4 / 8 |
| 每档预热 | 8 个请求（固定，不是 `max(4, 并发)`） |
| ISL / OSL | 8192 / 512 |
| 精度 | IFBench，limit=100，eval_batch_size=2，max_tokens=32768 |
| 评分 | `prompt_level_strict` |
| 生成参数 | `temperature=0`、`ignore_eos=true` |

**可配置**：`--perf-requests`（默认 100）、`--request-timeout`（默认 600 s）、
`--seed`（默认 42）、`--tokenizer`（默认 builtin）。

输入构造：

- `--tokenizer builtin`：从 tiktoken `o200k_base` 里挑"单词本身与带前导空格都是单 token"的英文词，
  拼成正好 8192 个 tiktoken 的文本；服务端实际 ISL 会因分词差异而不同，报告单列实际均值/最小/最大。
- `--tokenizer <模型目录>`：从模型词表的非 special token-ID 池里**独立随机抽 8192 个 ID**，
  通过 `extra.prompt` 传给 completions 接口；此时严格核验服务端 ISL=8192。

缓存隔离：每档传不同的 `cache_salt`（`<run_id>:c1` 等）。注意这**只是请求侧声明**，
服务端若忽略该字段，客户端无法证明缓存命中为零——报告不得宣称"缓存命中为 0"。

### 1.2 口径 B：基准文档原始协议（只用于对照）

| 项 | 值 |
|---|---|
| 预热 | `max(4, 并发数)` |
| 并发档 | 1 / 2 / 4 / 8 / 3 |
| 每档正式请求 | 12 / 24 / 24 / 32 / 24 |
| 请求超时 | 180 s |
| 输入 | 模型 tokenizer 有效 token-ID 池抽 8192 |
| 其他 | 逐档不同随机种子、不加公共前缀、GPU 每秒 `nvidia-smi` 采样、任务不重叠 |

口径 B 无法用 `run.sh` 完全复现（预热与并发档硬编码在 `runner.py` 里），
只能直接调用 AIPerf。阶段 2 若要做口径 B 对照，必须自己拼命令并**单独标注为"对照数据"**。

---

## 2. 为什么口径之间不能比较绝对值

1. **输入不同**：tiktoken 的 8192 与模型 tokenizer 的 8192 是两种长度分布，
   prefill 计算量、KV 占用、有效上下文都不同；
2. **预热不同**：8 个 vs `max(4,N)`，影响 CUDA graph 捕获、KV 分配、cache 状态；
3. **请求数不同**：100 vs 12–32，直接影响 P95 的稳定性（见第 4 节）；
4. **超时不同**：600 s vs 180 s，长尾会被截断成"失败"而不是"慢"。

允许跨口径比较的只有**结构性趋势**：例如"并发 4→8 吞吐不再增长且 TTFT 陡增"，
这个现象在两种口径下都成立，可以作为结论；而"26.22 tok/s"这样的绝对值不能跨口径引用。

**但趋势的成因必须单独归因**：在我们的基线里，"4→8 不涨"的直接原因是服务端
`--max-num-seqs 4` 封顶（第 5–8 个请求排队），不是硬件极限。定量依据见
`docs/optimization.md` §3.0（c1 已达峰值带宽 97%，c4 只有 59%，c8 与 c4 相同）。

---

## 3. 指标定义（必须与 AIPerf 原生导出对齐）

| 指标 | 定义 | 单位 |
|---|---|---|
| TTFT | 请求发出 → 首个 token 到达 | ms（报告转秒） |
| TPOT | 每请求 `(端到端延迟 − TTFT) / (输出 token 数 − 1)`，再对请求取分位 | ms/token |
| E2E | 请求发出 → 最后一个 token | ms（报告转秒） |
| 请求吞吐 | 完成请求数 / 测量时长 | req/s |
| 输出吞吐 | 完成请求数 × OSL / 测量时长 | token/s |
| 百分位 | 线性插值（`(n−1)·p`），与 `runner.py` 一致 | — |

**注意**：TPOT 不是"逐块间隔的平均"，而是每请求一个值再取分位。
自己实现统计时若用了别的口径（如全 token 池化），数字会与官方报告不可比。

---

## 4. 样本量与显著性（写报告时必须遵守）

| 现象 | 数学事实 | 报告写法 |
|---|---|---|
| n=12 时的 P95 | 位置 `(12−1)×0.95 = 10.45`，落在第 11、12 个次序统计量之间；只有约 0.6 个样本在 P95 以上 | 不能写成"P95 延迟 3.481 s"这种精确值；应写"约 3.5 s，尾部样本不足以估计 95 分位" |
| n≥100 时的 P95 | 有约 5 个样本在 P95 以上，估计才稳定 | 可作为主要延迟指标 |
| 100 样本精度 80% vs 76% | 二项标准误 `sqrt(0.78×0.22/100) ≈ 4.1pp`，差 4pp < 1 SE | 必须写"差异在统计误差内，不构成结论" |
| 并发 4→8 吞吐 +0.02% | **不是硬件饱和，是服务端 `max_num_seqs=4` 封顶**：第 5–8 个请求只能排队，服务器干的始终是 4 路的活（TTFT ×5.9 是排队时间） | 只能写"在该配置下吞吐不涨"；要下"硬件饱和"结论，必须先把 `max_num_seqs` 提到 ≥8 再测 |

建议：正式成绩每档请求数 ≥ 100（官方默认），调优实验可用小样本，但**小样本数据不得进正式成绩表**。

---

## 5. 精度评测的三个硬约束

1. **上下文**：`max_model_len` 必须 > 32768，否则 `runner.py` 直接把精度阶段判为异常并跳过
   （见 `eval_工具审计.md` P0-1）。这是最容易踩的坑。
2. **只报 `prompt_level_strict`**：另外三项（inst_level_strict/loose）按指令加权，
   计数可能 >100，不能平均后冒充总分。
3. **不完整即无效**：预测样本必须 100 条、`execution_summary` 必须 requested=succeeded=100、
   errored=incomplete=0。缺一项就当"未完成"，不能取部分样本算分。

---

## 6. MTP 的实验与写作方式

文档里的现象：

| 配置 | IFBench（limit=100） |
|---|---|
| FP8 + MTP | 出现无限思考循环，结果无效 |
| FP8，无 MTP | 80% |
| 非 FP8 + MTP | 76% |
| 非 FP8，无 MTP | 76% |

解读要点（答辩会被追问）：

- 这是**一次**失败观测（n=100、单次运行），但现象明确（思考循环 → 输出不终止 → 采样被截断），
  足以作为"稳定性风险"的证据，不足以得出"MTP 一定不可用"的结论；
- 非 FP8 + MTP 是正常的 76%，说明问题与**数值格式叠加投机解码**有关，
  合理怀疑是采样/接受率异常导致重复生成，而不是 MTP 本身；
- 因此正确做法是**默认关闭 MTP**，单独建一组 MTP 对照实验，记录：
  接受率（acceptance rate）、每请求输出长度分布、是否触发 `max_tokens`、IFBench 是否完整。

阶段 3 的 MTP 实验表建议包含列：
`配置 / 并发 / 输出吞吐 / TTFT P95 / TPOT P95 / 接受率 / 平均输出长度 / 触发 max_tokens 比例 / IFBench 状态 / 分数`。

---

## 7. 复现清单（阶段 2 完成后应能一键重放）

```bash
# 1) 检查依赖（只读，退出码 2 = 未就绪）
bash eval/run.sh --mode check

# 2) 看真实命令行（不安装、不请求模型）
bash eval/run.sh --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --dry-run

# 3) 口径 A 正式评测（交互终端，确认安装后）
export OPENAI_API_KEY=...        # 无鉴权则不要设置
bash eval/run.sh --mode all --base-url http://127.0.0.1:8088/v1 \
  --model qwen3.8-27b-fp8 --tokenizer /model/ModelScope/Qwen/Qwen3.8-27B-FP8 \
  --perf-requests 100 --request-timeout 600 \
  --output-dir outputs/E00_baseline_$(date +%Y%m%d_%H%M%S)

# 4) 校验产物完整性（自己写的检查，不属于官方工具）
python3 scripts/check_eval_tool.py --out-dir reports/verify_after_run
```

产物必须齐全才算一次有效实验：
`report.md`、`summary.json`、`configuration.json`、`environment.json`、`model_info.json`、
`performance/{dataset.jsonl,dataset.meta.json,c1,c2,c4,c8}`、`accuracy/{task_config.json,resolved_config.json,native}`。
