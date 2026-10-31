"""图表：并发-吞吐 / 并发-延迟曲线，多实验对比。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _frame(detail: dict) -> pd.DataFrame:
    rows = []
    for cell in (detail or {}).get("performance", []) or []:
        rows.append({"并发": cell.get("concurrency"),
                     "请求/s": cell.get("request_throughput"),
                     "输出 tok/s": cell.get("output_throughput"),
                     "TTFT P95(s)": (cell.get("ttft_p95") or 0) / 1000 or None,
                     "TPOT P95(ms)": cell.get("tpot_p95"),
                     "E2E P95(s)": (cell.get("e2e_p95") or 0) / 1000 or None})
    return pd.DataFrame(rows)


def concurrency_curves(detail: dict) -> None:
    frame = _frame(detail)
    if frame.empty:
        st.info("该实验没有性能数据")
        return
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(frame, x="并发", y=["输出 tok/s", "请求/s"], markers=True,
                                title="并发 vs 吞吐"), use_container_width=True)
    with right:
        st.plotly_chart(px.line(frame, x="并发", y=["TTFT P95(s)", "E2E P95(s)"], markers=True,
                                title="并发 vs 延迟"), use_container_width=True)


def compare(experiments: list[dict]) -> None:
    if not experiments:
        st.info("暂无实验可比较")
        return
    frame = pd.DataFrame(experiments)
    st.plotly_chart(px.bar(frame, x="experiment_name", y="best_output_throughput",
                           title="各实验最高输出吞吐（token/s）"), use_container_width=True)
