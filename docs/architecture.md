# 系统架构

> 一句话：**评测口径只存在于官方 `eval/runner.py` 里，本系统只负责"调度它、收好它的产物、把产物变成可对比的结论"。**

---

## 1. 分层

```text
┌──────────────────────────── 前端（Streamlit, 8501） ────────────────────────────┐
│ 01 总览 │ 02 模型运行 │ 03 实验管理 │ 04 性能 │ 05 精度 │ 06 AutoTuner │ 07 报告 │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │ HTTP (JSON)
┌────────────────────────── 后端 API（FastAPI, 8090） ────────────────────────────┐
│ /api/hardware  /api/runtime  /api/experiments  /api/benchmark  /api/reports     │
└───────┬──────────────────┬───────────────────┬──────────────────┬──────────────┘
        │                  │                   │                  │
┌───────▼───────┐ ┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼──────────┐
│ 硬件监控       │ │ 模型运行管理     │ │ Benchmark 适配   │ │ 分析 / AutoTuner │
│ nvidia-smi     │ │ start/stop_model│ │ 5 个适配器       │ │ 6 个模块          │
└───────┬───────┘ └────────┬────────┘ └────────┬────────┘ └───────┬──────────┘
        │                  │                   │                  │
        │          ┌───────▼────────┐  ┌───────▼────────┐ ┌───────▼──────────┐
        └──────────► 官方 eval/     ◄──┤ AIPerf 0.13.0  │ │ SQLite 6 张表     │
                   │ run.sh         │  │ EvalScope 1.13 │ │ + 原始产物目录     │
                   └───────┬────────┘  └────────────────┘ └──────────────────┘
                           │ 原始产物（JSONL/JSON/MD）
                   ┌───────▼─────────────────────────────────────────────────┐
                   │ experiments/<实验名>/performance/{c1,c2,c4,c8} + accuracy │
                   └─────────────────────────────────────────────────────────┘
```

---

## 2. 八个逻辑模块（对应需求书第三章）

| 模块 | 文件 | 职责 |
|---|---|---|
| 前端可视化 | `frontend/app.py` + `pages/01–07` + `components/` | 7 个页面；只调 API，不直接读文件 |
| 后端 API | `backend/main.py` + `backend/api/*.py` | 5 组路由；长任务同步执行并返回结构化结果 |
| 硬件资源管理 | `backend/api/hardware.py`（+ 阶段 0 的 `scripts/check_gpu.py`） | nvidia-smi 查询、CPU/内存/磁盘 |
| 模型运行管理 | `backend/api/runtime.py` + `scripts/start_model.sh` | profile 启停、前置检查、启动记录、日志尾 |
| Benchmark 评测 | `backend/benchmark/*.py` | 环境检查→服务检查→生成命令→执行→解析→入库→报告 |
| 分析与 AutoTuner | `backend/optimizer/*.py` | 搜索空间、逐候选实验、归一化评分、维度分析 |
| 数据持久化 | `backend/storage/*.py` | 6 张表 + 原始文件哈希登记 |
| 配置 | `configs/runtime.json`、`search_space.yaml`、`hardware/model/benchmark.yaml` | 参数集中管理，脚本不硬编码 |

---

## 3. 三条设计红线（决定了实现方式）

### 3.1 只调用，不复制口径

`backend/benchmark/eval_adapter.py` 里**没有任何** ISL/OSL/并发/预热常量，
它只构造 `bash eval/run.sh ...` 的参数、收 stdout、读产物。
所有口径（8192/512、并发 1/2/4/8、预热 8、limit=100、max_tokens=32768）都来自官方脚本。

**为什么**：一旦在适配器里重算一遍，就有两套口径；报告数字和官方工具对不上时无法自证。

### 3.2 原生产物是唯一事实来源

每个实验目录里保留 AIPerf 的 `profile_export_aiperf.json` / `profile_export.jsonl`、
EvalScope 的 `native/reports` / `predictions`，数据库只存索引、指标和 SHA-256。

**为什么**：评委要查"这个 65.8 token/s 是怎么算的"，必须能一路翻到逐请求记录。

### 3.3 串行独占 + 失败保留

`BenchmarkManager` 用文件锁 `experiments/.benchmark.lock` 保证同一时刻只有一个评测；
启动失败、OOM、完整性异常一律记录到 `reasons` 并入库，不删除、不重试。

**为什么**：单卡上并行评测会互相抢显存，数字全部作废；失败数据恰恰是评分表要的"错误案例"。

---

## 4. 一次实验的完整数据流

```text
用户点"开始评测"
  → POST /api/benchmark/pretend        # 前置检查：依赖? 服务? 上下文够不够?
  → POST /api/benchmark/run
      → BenchmarkManager.run(("perf","accuracy"))
          → 抢文件锁
          → AIPerfAdapter.preflight()      # run.sh --mode check + /v1/models
          → eval/run.sh --mode perf        # 官方工具，小时级
              → experiments/<name>/performance/c{1,2,4,8}/
          → result_parser.parse_performance # 读官方逐档 summary.json
          → eval/run.sh --mode accuracy
          → result_parser.parse_accuracy
          → 写 benchmark_result.json
      → experiment_repository.ingest_*    # 6 张表
      → artifact_repository.register      # 每个文件 + SHA-256
      → EvalAdapter.write_report          # 单实验 Markdown
  → 前端 04/05 页读 /api/experiments/{id} 出曲线
```

---

## 5. 参数是怎么进 vLLM 的

```text
configs/runtime.json  (profile: baseline / accuracy / mtp)
        │
        ├─ scripts/render_runtime.py --profile X --set k=v ...
        │        → 输出 shell 变量 + VLLM_ARGS 数组（不硬编码）
        │
        └─ scripts/start_model.sh  eval 这些变量
                 → 前置检查（GPU 空闲、端口、显存、PID 有效性）
                 → nohup vllm serve <model> "${VLLM_ARGS[@]}"
                 → 轮询 /v1/models 直到就绪，写启动记录 JSON
```

AutoTuner 通过 `--set max_num_seqs=8` 之类覆盖参数，**不修改配置文件**，因此每一次实验的
参数都能从启动记录 JSON 里完整还原。
