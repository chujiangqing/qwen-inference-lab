"""页面 7：实验报告 —— Markdown 报告、CSV 导出、原始文件校验。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("实验报告", "Markdown 报告与 CSV 导出")

experiments = call("/api/experiments?limit=50")
rows = (experiments or {}).get("experiments", []) if isinstance(experiments, dict) else []
if not rows:
    st.info("暂无实验")
    st.stop()

names = [row["experiment_name"] for row in rows]
choice = st.selectbox("选择实验", names)
report = call(f"/api/reports/{choice}")
if isinstance(report, str) and report.strip():
    st.markdown(report)
    st.download_button("下载 Markdown", report, file_name=f"{choice}.md")
else:
    st.warning(f"未找到报告：{report}")

st.divider()
if st.button("导出全部实验 CSV"):
    csv_text = call("/api/reports/export/csv")
    if isinstance(csv_text, str):
        st.download_button("下载 CSV", csv_text, file_name="experiments.csv")
    else:
        st.error(csv_text)

st.divider()
experiment_id = int(st.selectbox("校验原始文件哈希", [f"{row['id']} · {row['experiment_name']}" for row in rows])
                    .split("·")[0].strip())
if st.button("开始校验"):
    result = call(f"/api/reports/{experiment_id}/verify")
    if isinstance(result, dict) and result.get("intact"):
        st.success(f"{result['checked']} 个文件全部与登记哈希一致")
    else:
        st.error(f"校验异常：{result}")
