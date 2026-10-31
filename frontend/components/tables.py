"""表格：实验列表、性能明细、精度明细。"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def experiments_table(rows: list[dict]) -> None:
    if not rows:
        st.info("暂无实验记录")
        return
    frame = pd.DataFrame(rows)
    keep = [column for column in ("id", "experiment_name", "experiment_type", "status",
                                  "created_at", "best_output_throughput", "prompt_level_strict")
            if column in frame.columns]
    st.dataframe(frame[keep], use_container_width=True, hide_index=True)


def performance_table(rows: list[dict]) -> None:
    if not rows:
        st.info("无性能数据")
        return
    frame = pd.DataFrame(rows)[["concurrency", "request_count", "ttft_p50", "ttft_p95",
                                "tpot_p50", "tpot_p95", "e2e_p50", "e2e_p95",
                                "request_throughput", "output_throughput", "status"]]
    frame.columns = ["并发", "请求数", "TTFT P50", "TTFT P95", "TPOT P50", "TPOT P95",
                     "E2E P50", "E2E P95", "请求/s", "输出 tok/s", "状态"]
    st.dataframe(frame, use_container_width=True, hide_index=True)


def accuracy_table(rows: list[dict]) -> None:
    if not rows:
        st.info("尚无精度记录")
        return
    frame = pd.DataFrame(rows)[["dataset", "sample_count", "prompt_level_strict"]]
    frame.columns = ["数据集", "样本数", "prompt_level_strict"]
    st.dataframe(frame, use_container_width=True, hide_index=True)
