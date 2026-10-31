"""图表：并发-吞吐 / 并发-延迟曲线。

注意：请求/s（0.05–0.13）与输出 tok/s（26–92）差三个数量级，
画在同一 y 轴上会让前者变成一条贴 0 的直线，因此**不混轴**：
吞吐图只画 tok/s，请求/s 放在逐档明细表里。
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _frame(detail: dict) -> pd.DataFrame:
    rows = []
    for cell in (detail or {}).get("performance", []) or []:
        ttft = cell.get("ttft_p95")
        e2e = cell.get("e2e_p95")
        rows.append({"并发": cell.get("concurrency"),
                     "请求/s": cell.get("request_throughput"),
                     "输出 tok/s": cell.get("output_throughput"),
                     "TTFT P95(s)": round(ttft / 1000, 2) if ttft else None,
                     "TPOT P95(ms)": cell.get("tpot_p95"),
                     "E2E P95(s)": round(e2e / 1000, 2) if e2e else None})
    return pd.DataFrame(rows)


def concurrency_curves(detail: dict) -> None:
    frame = _frame(detail)
    if frame.empty:
        st.info("该实验没有性能数据")
        return
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(frame, x="并发", y=["输出 tok/s"], markers=True,
                                title="并发 vs 输出吞吐（token/s）"), use_container_width=True)
        st.caption("请求/s 量级为 0.05–0.13，与 token/s 差三个数量级，不画在同一轴上；见下方逐档明细。")
    with right:
        st.plotly_chart(px.line(frame, x="并发", y=["TTFT P95(s)", "E2E P95(s)"], markers=True,
                                title="并发 vs 延迟（秒，越低越好）"), use_container_width=True)
        st.caption("TPOT 单位是毫秒/token，量级不同，见逐档明细。")


def compare(experiments: list[dict]) -> None:
    if not experiments:
        st.info("暂无实验可比较")
        return
    frame = pd.DataFrame(experiments)
    st.plotly_chart(px.bar(frame, x="experiment_name", y="best_output_throughput",
                           title="各实验最高输出吞吐（token/s）"), use_container_width=True)

def grouped_bars(frame, x: str, columns: list[str], title: str,
                 names: list[str] | None = None) -> None:
    """并排柱状图。st.bar_chart 默认堆叠（66+66=132），这里必须用 grouped。"""
    if frame is None or frame.empty:
        st.info("无可画的数据")
        return
    labels = names or columns
    data = frame[[x] + columns].copy()
    data.columns = [x] + labels
    figure = px.bar(data, x=x, y=labels, barmode="group", title=title)
    figure.update_layout(legend_title_text="", height=340,
                         margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(figure, use_container_width=True)
