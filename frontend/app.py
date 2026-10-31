"""Qwen3.8-27B InferenceLab · 前端入口（Streamlit）。

启动：
    streamlit run frontend/app.py --server.port 8501
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "components"))

import gpu_cards  # noqa: E402
import tables  # noqa: E402
from api_client import call  # noqa: E402

st.set_page_config(page_title="Qwen3.8-27B InferenceLab", page_icon="⚙️", layout="wide")
st.title("Qwen3.8-27B InferenceLab")
st.caption("单卡 4090 上的推理优化与自动化评测平台 · 左侧页面切换")

hardware = call("/api/hardware")
if isinstance(hardware, dict) and "error" not in hardware:
    gpu_cards.render(hardware)
else:
    st.error(f"后端不可达：{hardware}")

status = call("/api/runtime/status")
service = (status or {}).get("service", {})
if service.get("ready"):
    st.success(f"推理服务在线 · 模型 {', '.join(service.get('models', []))} · "
               f"max_model_len={service.get('max_model_len')}")
else:
    st.warning(f"推理服务未就绪：{service.get('error', '未知')}")

st.subheader("最近实验")
experiments = call("/api/experiments?limit=8")
tables.experiments_table((experiments or {}).get("experiments", []) if isinstance(experiments, dict) else [])

st.subheader("使用顺序")
st.markdown("1. **模型运行管理**：选择 profile 启动服务；\n"
            "2. **实验管理**：发起基线或优化实验；\n"
            "3. **性能 / 精度**：查看逐档指标与 IFBench 分数；\n"
            "4. **AutoTuner**：查看候选与评分；\n"
            "5. **实验报告**：导出 Markdown / CSV。")
