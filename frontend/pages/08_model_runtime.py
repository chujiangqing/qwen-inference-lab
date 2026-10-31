"""页面 2：模型运行管理 —— 选择 profile、启停服务、看日志。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

from api_client import call  # noqa: E402

st.header("模型运行管理")
status = call("/api/runtime/status")
service = (status or {}).get("service", {})
profiles = (status or {}).get("profiles", {})

if service.get("ready"):
    st.success(f"服务在线：{service.get('models')} · max_model_len={service.get('max_model_len')}")
else:
    st.error(f"服务未就绪：{service.get('error', '—')}")

with st.form("start_form"):
    profile = st.selectbox("profile", list(profiles) or ["baseline"],
                           help="baseline=32768；accuracy=40960（精度评测用）；mtp=投机解码对照")
    max_num_seqs = st.number_input("覆盖 max_num_seqs（0 = 用 profile 默认）", min_value=0, value=0, step=1)
    submitted = st.form_submit_button("启动服务")

if submitted:
    overrides = {}
    if max_num_seqs:
        overrides["max_num_seqs"] = int(max_num_seqs)
    with st.spinner("正在启动（加载 28.75 GiB 权重，通常 2–5 分钟）…"):
        result = call("/api/runtime/start", {"profile": profile, "overrides": overrides}, timeout=1800)
    if isinstance(result, dict) and result.get("started"):
        st.success(f"已启动 profile={profile}")
        st.json(result.get("status"))
    else:
        st.error(f"启动失败：{result}")

left, right = st.columns(2)
if left.button("停止服务"):
    with st.spinner("正在停止…"):
        result = call("/api/runtime/stop", {})
    st.json(result)
if right.button("刷新状态"):
    st.rerun()

st.subheader("最新服务日志")
logs = call("/api/runtime/logs?lines=80")
st.code((logs or {}).get("tail", "（无日志）"), language="text")
st.caption(f"日志文件：{(logs or {}).get('log')}")
