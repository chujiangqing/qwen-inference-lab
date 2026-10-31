# Qwen3.8-27B InferenceLab

单卡 RTX 4090（48 GB 变体）上的 Qwen3.8-27B 推理优化与自动化评测平台。
UCloud AI 赛 · 赛道 A（AI Infra）。

> **一句话**：评测口径一行不改（直接用官方 `eval/`），把"起服务 → 跑官方评测 → 参数搜索 →
> 质量复核 → 出报告"做成一条可复现的流水线，并用原始产物 + SHA-256 保证每个数字都能翻到源头。

---

## 1. 已经测出来的东西（全部为服务器实测，非引用）

| 并发 | 状态 | 请求数 | TTFT P95 | TPOT P95 | E2E P95 | 请求/s | 输出 token/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 完整 | 100 | 3440 ms | 31.44 ms | 19.48 s | 0.05133 | 26.28 |
| 2 | 完整 | 100 | 5043 ms | 37.82 ms | 23.73 s | 0.08427 | 43.14 |
| 4 | 完整 | 100 | 6141 ms | 51.78 ms | 30.93 s | 0.12927 | 66.19 |
| 8 | 完整 | 100 | 36374 ms | 52.87 ms | 61.79 s | 0.12941 | 66.26 |

- 输入：模型自身 tokenizer 抽样的 token-ID，**服务端实际 ISL 精确 8192**（min = max = mean）
- 输出：`ignore_eos`，固定 512 token（服务端 usage 核验）
- 与比赛基准文档的偏差：吞吐四档全部在 **0.7% 以内**；延迟在并发 1/2 上偏差 <1%，
  并发 4/8 的 TTFT P95 比文档更低（−35% / −13%），原因见 `docs/实验报告.md` 的口径说明
- 结构性结论与文档一致：**并发 4→8 吞吐只涨 0.1%，TTFT P95 却涨 5.9 倍** → 饱和点在并发 4

精度（IFBench，limit=100）与参数搜索（AutoTuner）结果见 `docs/实验报告.md`。

**硬件**：RTX 4090 49140 MiB，驱动 610.57.04，CUDA UMD 13.3
**软件**：vLLM 0.31.0 + torch 2.13.0+cu130 + flashinfer；官方 AIPerf 0.13.0 / EvalScope 1.13.0
**模型**：`Qwen3.8-27B-FP8`，`qwen3_5` 架构（64 层 = 48 层 Gated-DeltaNet + 16 层全注意力，1 层 MTP，带视觉塔），28.75 GiB

---

## 2. 快速开始

```bash
# 服务器上（只读检查，先看环境是否齐）
bash eval/run.sh --mode check          # 退出码 0 = 依赖就绪

# 起服务（baseline 32768 / accuracy 40960 / mtp）
bash scripts/start_model.sh --profile baseline
bash scripts/check_api.sh              # /v1/models + completions + chat 三项

# 跑一次完整评测（性能 + 精度，小时级，GPU 独占）
bash scripts/run_benchmark.sh --label E00_BASELINE --mode all \
  --tokenizer /model/ModelScope/Qwen/Qwen3.8-27B-FP8

# 参数搜索（每个候选约 26 分钟）
python3 backend/optimizer/auto_tuner.py --requests 20 \
  --include batching_seq8_tok8192 --include batching_seq4_tok16384 --include mtp_on

# 汇总报告
python3 scripts/generate_report.py      # → docs/实验报告.md + data/experiments.csv

# Web 平台
nohup python3 -m uvicorn main:app --host 127.0.0.1 --port 8090 --app-dir backend &
nohup python3 -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 &
# 本机隧道：ssh -L 8501:127.0.0.1:8501 -L 8090:127.0.0.1:8090 root@<服务器>
```

---

### 首次运行必须人工确认一次依赖安装

`eval/run.sh` 在检测到缺依赖时会要求交互确认；**在 nohup / API / cron 这类非交互环境下它会直接
`exit 2`**（`run.sh` 的 `confirm()`：`[[ ! -t 0 ]]` 即拒绝安装）。所以第一次必须先在一个真实终端里跑一遍
`bash eval/run.sh --mode check`，把 `aiperf-0.13.0` / `evalscope-1.13.0` 两个环境装好；
之后 API 和流水线才能非交互调用。这是官方脚本的设计，不是我们的改动。
## 3. 目录

```text
Qwen-Inference-Lab/
├── README.md                本文件
├── requirements.txt         平台层依赖（版本实测；三套环境的分工写在注释里）
├── pyproject.toml           工程元数据与测试配置
├── eval/                    官方评测工具（未修改，SHA-256 可校验）
├── configs/
│   ├── runtime.json         运行时参数：baseline / accuracy / mtp 三个 profile
│   ├── search_space.yaml    AutoTuner 搜索空间与评分权重
│   └── hardware.yaml / model.yaml / benchmark.yaml
├── scripts/
│   ├── start_model.sh / stop_model.sh      起停服务（前置检查、日志、启动记录）
│   ├── render_runtime.py                   把 profile 渲染成 vLLM 参数
│   ├── run_benchmark.sh                    一次实验入口
│   ├── check_api.sh                        接口三项检查
│   ├── check_gpu.py / check_environment.py / check_model.py / check_eval_tool.py
│   ├── task0_audit.sh                      阶段 0 一键环境审计
│   └── generate_report.py                  汇总 Markdown + CSV
├── backend/
│   ├── main.py + api/{hardware,runtime,experiments,benchmark,reports}.py
│   ├── benchmark/{eval_adapter,aiperf_adapter,evalscope_adapter,result_parser,benchmark_manager}.py
│   ├── optimizer/{search_space,auto_tuner,concurrency,batching,kv_cache,mtp}.py
│   ├── monitor/gpu_monitor.py              nvidia-smi 快照 + 峰值采样
│   ├── model/memory_estimator.py           KV/显存预算估算
│   ├── storage/{database,experiment_repository,artifact_repository}.py
│   └── core/config.py
├── frontend/                 app.py + pages/01–07 + components/
├── docs/
│   ├── deployment.md           从零部署 + 8 个实测踩坑
│   ├── architecture.md         分层、数据流、三条设计红线
│   ├── benchmark_methodology.md    口径 A/B、指标定义、显著性
│   ├── eval_工具审计.md         官方工具的 P0/P1 问题（含"精度会被跳过"）
│   ├── defense_guide.md        答辩要点与追问预案
│   └── 环境检测报告.md          阶段 0 实测结果
├── audit/verify_runner.py    对官方 runner.py 的 76 项行为断言
├── tests/                    冒烟测试 + 真实产物联调测试
└── experiments/              全部原始产物（AIPerf/EvalScope 原生文件）
```

---

## 4. 评分项 → 证据（逐条可查，不靠口头描述）

| 评分项 | 证据在哪 | 怎么自己验证 |
|---|---|---|
| 模型固定 / 选型依据 | `docs/optimization.md` §1（5 个版本体积、量化格式、引擎支持、硬件能力对照） | `du -sh /model/ModelScope/*/Qwen3.8-27B*`；`nvidia-smi --query-gpu=compute_cap` |
| 单卡 4090 稳定运行 | `reports/task0_*/gpu.json`、`experiments/*/performance/*/summary.json` | `nvidia-smi`；每个 `summary.json` 的 `status` 必须是 `完整` |
| 推理优化（批处理/并发） | `docs/optimization.md` §3（含带宽定量表）；`experiments/auto_tuner/trials.json` | `python3 backend/optimizer/auto_tuner.py --dry-run` |
| 量化（FP8 / KV fp8） | `docs/optimization.md` §5；`configs/runtime.json` 的 `kv_cache_dtype` | 启动日志里 `--kv-cache-dtype` 的实际值（`/root/logs/vllm/launch_*.json`） |
| 加速（MTP） | `docs/optimization.md` §4；`experiments/MTP_ACC_*` | `summary.json` 的 `status` 与样本数 |
| 输出质量 | `docs/实验报告.md` 精度节；`experiments/E00_BASELINE_ACC2_*/accuracy/summary.json` | 该文件的 `prompt_level_strict` 与 `generated_samples=100` |
| 评测口径未改 | `tests/test_pipeline_contracts.py` 里 `eval/` 与 zip 的 SHA-256 比对 | `python3 -m unittest discover -s tests` |
| 前端可演示 | 7 个页面 + `/api/*` | `curl 127.0.0.1:8090/health`；`curl 127.0.0.1:8501/_stcore/health` |
| 错误案例 | `docs/defense_guide.md` §5；`docs/deployment.md` 踩坑表 1–9 | 对应日志：`/root/logs/*.log` |

> 原则：**报告里出现的每个数字，都必须能追到某个 `summary.json` 或日志行**。
> 追不到的数字不写。

---

## 5. 三条不能碰的红线

1. **模型固定**：只用 Qwen3.8-27B（FP8 为实验主体）。不引入其它模型出成绩。
2. **口径固定**：ISL/OSL/并发/预热/评分指标一律由官方 `eval/runner.py` 决定，本仓库零修改。
   两套口径的差异与不可比性写在 `docs/benchmark_methodology.md`。
3. **数据真实**：没跑过的数字不写；失败、OOM、被门禁跳过的实验都保留并注明原因。

---

## 6. 已知限制（不藏）

- 官方 runner 在 `max_model_len <= 32768` 时会**跳过 IFBench**，所以精度必须用 `accuracy` profile（40960）。
- 并发 8 的吞吐与并发 4 持平（+0.02%），说明瓶颈不在并发数；不要把它当作"扩展性"卖点。
- 100 样本的 IFBench 二项标准误约 ±4 个百分点，小于该幅度的差异不作为结论。
- AutoTuner 的 `--requests 20` 是筛选口径，不能作为正式成绩；正式成绩用 100 请求/档。

---

## 7. 复现验证记录（哪些验过、哪些没验）

评分表要求"结果可复现或可合理解释"。这里照实写清楚边界，不含糊。

### 7.1 已验证（有日志/产物可查）

| 项 | 怎么验的 | 证据 |
|---|---|---|
| 依赖版本 | 从**运行中的环境**用 `pip show` / `import` 取真实值，不是抄文档 | `requirements.txt` 头部注释 + `docs/deployment.md` §3.0 |
| 推理引擎可用 | `vllm --version` = 0.31.0；torch 2.13.0+cu130；`cuda True` | `docs/deployment.md` §3.1 的期望输出 |
| 服务能起 | `scripts/start_model.sh --profile baseline/accuracy` 连续多次成功 | `/root/logs/vllm/launch_*.json` 每次启动记录参数与 PID |
| 评测能跑完 | 4 档 × 100 请求多轮，`status=完整`；IFBench 100 题 | `results/experiments/*/performance/c*/summary.json` |
| 评测未被改动 | `eval/` 三个文件 SHA-256 与原始 zip 一致 | `tests/test_pipeline_contracts.py::test_eval_untouched` |
| 脚本可重建结果 | `scripts/export_results.py` 从 `experiments/` 重新导出 `results/` 并生成 sha256 清单 | `results/manifest.json` |
| 平台可重启 | `restart_ui2.sh` / `hard_restart_api.sh` 反复执行成功 | 健康检查 `/_stcore/health` = ok |

### 7.2 未验证（以及为什么）

| 项 | 状态 | 原因 | 换机器时要额外确认 |
|---|---|---|---|
| 全新机器从零部署 | ❌ 未做 | 三套环境（平台 / vLLM / 官方评测）重建约 40–60 分钟，赛程内优先跑了实验 | 见 7.3 清单 |
| 不同 GPU / 驱动 | ❌ 未做 | 只有一张 4090（compute_cap 8.9，驱动 610.57.04） | FP4/FP8 支持随算力与驱动变化 |
| 依赖源可用性 | ⚠️ 部分 | 使用内网 pip 镜像，未测公网源 | 公网安装时注意 `flashinfer` 的 CUDA 版本匹配 |
| 长时间稳定性（>6 小时） | ⚠️ 部分 | 最长连续约 2.5 小时（精度评测） | 关注显存碎片累积与温度 |

### 7.3 换机器复现的检查清单

1. `nvidia-smi --query-gpu=name,compute_cap,driver_version` —— 确认算力 ≥ 8.9（否则 FP8 不可用）
2. 显存 ≥ 46 GiB —— 权重 28.75 + KV 11.56 + 框架开销
3. `/model` 下有 `Qwen/Qwen3.8-27B-FP8`，且 `config.json` 的 sha256 与
   `configs/model.yaml` 记录一致（本项目记录为 `74227dd6…`）
4. 三套环境分开建，不要混装（见 `requirements.txt` 头部）
5. 先跑 `python -m unittest discover -s tests`（46 个），再跑 `audit/verify_runner.py`（84 条）
6. 首次压测前先确认 GPU 独占：`nvidia-smi --query-compute-apps=pid` 为空

### 7.4 已知会让复现"看起来失败"的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动报"GPU 上已有 1 个计算进程，拒绝启动" | 上一次 vLLM 的 `EngineCore` 子进程残留（它不匹配 `vllm serve` 模式） | `pkill -9 -f EngineCore` |
| 精度评测被跳过 | `max_model_len ≤ 32768` 时官方工具不跑 IFBench | 用 `--profile accuracy`（40960） |
| 端口 8088 连不上但进程还在 | API 进程退出、EngineCore 存活 | `fuser -k 8088/tcp` 后重起 |
| 压测数字明显偏低 | 服务未就绪就开始计时 | 先 `curl /v1/models` 探活，再开跑 |

## 附：`eval/` 目录说明（官方评测入口，未修改）

`eval/` 下三个文件**来自主办方提供的评测包，本项目未做任何修改**，仅作为评测入口调用；
SHA-256 与原始 zip 一致（由 `tests/test_pipeline_contracts.py` 中的断言校验）。

| 文件 | 大小 | SHA-256（前 32 位） |
|---|---:|---|
| `eval/run.sh` | 3.8 KB | `078a2677038dbf3590b0e148546962aa…` |
| `eval/runner.py` | 42.6 KB | `66d30de4f2f4a39b7243226d76f8be95…` |
| `eval/eval.zip` | 20.6 KB | `9075ba2e020b0c93a3c36ff1c1374661…` |

如果你的使用场景不允许再分发该评测包，可以删除本地 `eval/` 目录，
评测流程与文档中的命令仍按原样可用（只需把官方包放回同一位置）。
