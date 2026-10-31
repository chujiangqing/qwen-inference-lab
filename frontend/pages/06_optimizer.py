"""页面 6：AutoTuner —— 搜索空间、候选合法性、评分排名。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

from api_client import call  # noqa: E402

st.header("AutoTuner")
st.caption("评分 S = α·吞吐 + β·延迟 + γ·质量 + δ·显存；未测质量的配置不得评为最优（需求书 8.3）。")

if st.button("展开搜索空间并做合法性检查"):
    result = call("/api/benchmark/auto-tuner/dry-run", {})
    if isinstance(result, dict) and result.get("output"):
        try:
            data = json.loads(result["output"])
        except json.JSONDecodeError:
            st.code(result["output"], language="text")
        else:
            st.write("实测事实：", data.get("facts"))
            st.write("候选统计：", data.get("summary"))
            frame = pd.DataFrame(data.get("candidates", []))
            if not frame.empty:
                frame["overrides"] = frame["overrides"].apply(lambda item: json.dumps(item, ensure_ascii=False))
                st.dataframe(frame[["name", "dimension", "valid", "overrides", "reasons"]],
                             use_container_width=True, hide_index=True)
    else:
        st.error(f"调用失败：{result}")

st.subheader("已完成的候选结果")
report = call("/api/reports/auto_tuner")
if isinstance(report, str) and report.strip():
    st.markdown(report)
else:
    st.info("尚未产生 AutoTuner 结果（运行 backend/optimizer/auto_tuner.py 后出现在这里）")
