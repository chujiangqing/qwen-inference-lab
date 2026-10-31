# 部署与复现

> 目标：**换一台干净的 4090 机器，照这份文档能复现出同一组数字。**
> 本文所有路径、版本、报错都来自比赛服务器实测，不是推测。

---

## 1. 实测环境基线

| 项目 | 实测值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090，**49140 MiB**（48 GB 变体） | 标准版是 24 GB，必须实测确认 |
| 驱动 | 610.57.04，CUDA UMD 13.3 | 向后兼容 cu12x / cu13x wheel |
| CPU / 内存 | AMD EPYC 7B13，20 vCPU / 90 GiB | — |
| 系统 | Ubuntu 22.04.5 LTS，内核 6.12.101-compshare | — |
| 系统 Python | 3.10.12（**无 3.11/3.12、无 conda、无 docker、无 unzip**） | 关键前提 |
| 模型 | `/model/ModelScope/Qwen/Qwen3.8-27B-FP8`，66 分片，**28.75 GiB** | 另有 52 GB 的非 FP8 版本 |
| 模型架构 | `qwen3_5`：64 层 = 48 层 Gated-DeltaNet + 16 层全注意力，1 层 MTP，带视觉塔 | KV 只在 16 层上产生 |
| 磁盘 | `/` 96 GiB 可用；`/model` 为 UPFS 50 TB | venv+权重需 ~40 GiB |

---

## 2. 目录布局

```text
/root/qwen-lab/                 # 项目根（本仓库）
├── eval/                       # 官方评测工具，**不得修改**
├── configs/runtime.json        # 运行时参数（profile: baseline/accuracy/mtp）
├── scripts/
│   ├── start_model.sh          # 起服务（含前置检查、日志、启动记录）
│   ├── stop_model.sh           # 停服务（PID → SIGTERM → SIGKILL）
│   ├── run_benchmark.sh        # 一次实验（性能+精度）
│   └── generate_report.py      # 汇总报告
├── backend/                    # FastAPI + 适配器 + AutoTuner + SQLite
├── frontend/                   # Streamlit（7 页）
├── experiments/                # 全部原始产物（不删）
├── reports/                    # 阶段 0 环境检测结果
└── data/inference_lab.db       # 实验索引库

/root/venvs/vllm031/            # vLLM 0.31.0 + torch 2.13（8.1 GB）
/root/.cache/ai-infra-eval/envs/  # 官方 AIPerf 0.13.0 / EvalScope 1.13.0（1.7 GB）
/root/logs/                     # 服务与评测日志
```

---

## 3. 从零部署（按顺序执行）

### 3.0 依赖清单与环境隔离

本项目有 **三套互不干扰的 Python 环境**，混装会出问题：

| 环境 | 位置 | 装什么 | 谁维护 |
|---|---|---|---|
| 平台层 | 系统 `python3.10` + `~/.local` | `requirements.txt`（fastapi/uvicorn/streamlit/plotly/pandas/PyYAML） | 本仓库 |
| 推理引擎 | `/root/venvs/vllm031` | `vllm==0.31.0`（自带 torch 2.13.0+cu130 / transformers 5.17.0 / flashinfer 0.7.0） | 本仓库 |
| 评测工具 | `/root/.cache/ai-infra-eval/envs/` | `aiperf==0.13.0`、`evalscope[ifbench]==1.13.0` | **官方 eval.zip 自己用 uv 建** |

```bash
pip3 install --user -r requirements.txt      # 平台层（--user 不改系统环境，卸载只需删 ~/.local）
python3 -m venv /root/venvs/vllm031          # 推理引擎
/root/venvs/vllm031/bin/pip install vllm==0.31.0
bash eval/run.sh --mode check                # 评测工具：只检查，缺依赖会提示怎么装
```

版本都来自比赛服务器实测（2026-10-31），不是猜的。

### 3.1 装 vLLM（隔离 venv，不碰系统 Python）

```bash
mkdir -p /root/venvs
python3 -m venv /root/venvs/vllm031
/root/venvs/vllm031/bin/pip install -q --upgrade pip setuptools wheel
/root/venvs/vllm031/bin/pip install vllm==0.31.0        # 约 8.1 GB，含 torch 2.13+cu130
```

版本依据：vLLM 0.31.0 的 `ModelRegistry` 里同时存在
`Qwen3_5ForConditionalGeneration` / `Qwen3_5MTP` / `Qwen3_5MoeForConditionalGeneration`，
与本模型 `config.json` 的 `architectures: ["Qwen3_5ForConditionalGeneration"]` + `mtp.*` 权重完全对应。
**低于这个版本大概率不认识 `qwen3_5`，会报 "Model architecture not supported"。**

验证：

```bash
/root/venvs/vllm031/bin/python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
from vllm.model_executor.models.registry import ModelRegistry
print("Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs())
PY
# 期望：vllm 0.31.0 torch 2.13.0+cu130 cuda True / True
```

### 3.2 起服务

```bash
cd /root/qwen-lab
bash scripts/start_model.sh --profile baseline        # 32768 上下文
# 或： --profile accuracy（40960，精度评测必须）
```

期望输出：

```text
服务已就绪   : 167s
启动记录     : /root/logs/vllm/launch_baseline_YYYYmmdd_HHMMSS.json
接口地址     : http://127.0.0.1:8088/v1
```

首次启动 4–5 分钟（含 flashinfer JIT 编译约 2 分钟）；**第二次起 ~167 秒**（JIT 有缓存）。

### 3.3 装官方评测依赖

```bash
bash eval/run.sh --mode check          # 只检查，退出码 2 = 未就绪
bash eval/run.sh --mode all ...        # 交互终端里确认安装（会提示 y/N）
```

工具链会自己用 `uv` 装 Python 3.12 与两个隔离环境到
`/root/.cache/ai-infra-eval/envs/`。**若 `uv` 直连 PyPI 很慢，先设镜像**：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.4 跑评测

```bash
# 性能（100 请求/档，模型 tokenizer 的严格 token-ID 输入）
bash eval/run.sh --mode perf \
  --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 \
  --tokenizer /model/ModelScope/Qwen/Qwen3.8-27B-FP8 \
  --perf-requests 100 --output-dir experiments/E00_PERF_$(date +%Y%m%d_%H%M%S)

# 精度（必须先把服务切到 accuracy profile，上下文 40960）
bash scripts/start_model.sh --profile accuracy
bash eval/run.sh --mode accuracy \
  --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 \
  --eval-batch-size 2 --output-dir experiments/E00_ACC_$(date +%Y%m%d_%H%M%S)
```

### 3.5 起 Web 平台

```bash
pip3 install --user fastapi uvicorn streamlit plotly pandas    # 一次性
nohup python3 -m uvicorn main:app --host 127.0.0.1 --port 8090 --app-dir backend \
  > /root/logs/api.log 2>&1 &
nohup python3 -m streamlit run frontend/app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  > /root/logs/streamlit.log 2>&1 &
```

本机访问（服务器端口不对公网开放，必须走 SSH 隧道）：

```bash
ssh -p 22209 -L 8501:127.0.0.1:8501 -L 8090:127.0.0.1:8090 root@<服务器>
# 浏览器打开 http://127.0.0.1:8501
```

### 3.6 生成报告

```bash
python3 scripts/generate_report.py           # → docs/实验报告.md + data/experiments.csv
python3 backend/optimizer/auto_tuner.py --requests 20   # 参数搜索（约 26 分钟/候选）
```

---

## 4. 实测踩过的坑（复现时必看）

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `FileNotFoundError: 'ninja'`（启动到 CUDA graph 捕获阶段失败） | flashinfer 运行时 JIT 需要 `ninja`，但 venv 的 `bin` 不在 PATH 上 | `start_model.sh` 里 `export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"` |
| 2 | `unrecognized arguments: --no-enable-prefix-caching` | vLLM 0.31 该参数只有 `--enable-prefix-caching` | 渲染器只发正向参数，不发 `--no-*` |
| 3 | `uv pip install` 卡住 30 分钟不动 | uv **不读 pip.conf**，直连 pypi.org 被墙/极慢 | 设 `UV_INDEX_URL` 指向清华源 |
| 4 | `nltk.downloader -d ...` 报 `AttributeError: 'NoneType' object has no attribute 'id'` | nltk 3.9 的交互式重试在非 TTY 下 EOF | 手动下 zip 并按 `tokenizers/ corpora/ taggers/` 分类解压到 `envs/nltk_data/` |
| 5 | 同上，`unzip: command not found` | 镜像里没有 unzip | `python3 -m zipfile -e x.zip <目标分类目录>` |
| 6 | `已有服务在运行（pid=xxx）` 但 GPU 是空的 | 强杀后 PID 文件残留，PID 被别的进程复用 | `start_model.sh` 现在会校验 `/proc/<pid>/cmdline` 确实是 vLLM，否则清理 PID 文件 |
| 7 | 精度阶段被跳过、只有一行"配置异常" | `max_model_len <= 32768` 时官方 runner 直接跳过 IFBench | 起服务用 `--profile accuracy`（40960）后再跑精度 |
| 8 | SSH 长命令（pip/curl）中途 `Connection closed` | 该 pod 的 SSH 隧道对长连接不友好 | 长任务一律 `nohup ... > log 2>&1 &`，再单独读日志；**不要在 `pkill -f` 里写会匹配到自身命令行的字符串** |
| 9 | 跑了 45 分钟的 IFBench 突然中断，vLLM 被 SIGTERM | 后台"等待 E00 结束"的循环写成了 `for i in $(seq 1 240); do ... done`，**2 小时到点后没有 `break` 就往下走了**，于是把仍在跑精度的服务停掉、直接启动 AutoTuner | 等待循环必须是"只有条件满足才继续"，超时要显式报错退出（`if ! grep ...; then exit 1; fi`）；或者不用长驻等待，改为人工按阶段推进 |
| 10 | 非交互环境跑 `eval/run.sh` 直接 `exit 2` | 官方脚本 `confirm()` 检测到 `stdin` 不是终端就拒绝装依赖 | 首次必须在真实终端跑一次 `bash eval/run.sh --mode check` 完成安装；之后 API/流水线才可无人值守 |

---

## 5. 验证清单（交付前逐条跑）

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv            # 48 GB 变体
bash eval/run.sh --mode check; echo $?                            # 0 = 依赖就绪
bash scripts/start_model.sh --profile baseline --dry-run          # 打印真实命令
curl -s http://127.0.0.1:8088/v1/models | head -c 200             # max_model_len 正确
bash scripts/check_api.sh                                         # models/completions/chat 三项
curl -s http://127.0.0.1:8090/health                              # API
curl -s http://127.0.0.1:8501/_stcore/health                      # 前端
python3 -m unittest discover -s tests -v                          # 冒烟测试
```

---

## 6. 回滚与清理

```bash
bash scripts/stop_model.sh --force
rm -rf /root/venvs/vllm031                 # 8.1 GB
rm -rf /root/.cache/ai-infra-eval          # 1.7 GB（官方工具环境，重装需确认）
```

**不要删** `experiments/` 与 `reports/`：那是全部原始证据。
