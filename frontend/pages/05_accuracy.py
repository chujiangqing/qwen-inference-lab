"""页面 5：精度 —— IFBench 分数、置信区间、输出长度分布、与环境的关系。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import tables  # noqa: E402
from api_client import call  # noqa: E402

st.header("精度评测（IFBench）")
st.caption("评分指标 prompt_level_strict：一条指令的所有约束全部满足才算通过。"
           "官方 runner 固定 limit=100、max_tokens=32768。")

experiments = call("/api/experiments?limit=50")
rows = (experiments or {}).get("experiments", []) if isinstance(experiments, dict) else []
if not rows:
    st.info("暂无实验记录。先到「实验管理」跑一次 accuracy 模式，或等后台评测结束入库。")
    st.stop()

with_accuracy = [row for row in rows if row.get("prompt_level_strict") is not None]
if not with_accuracy:
    st.info("暂无精度记录（需要以 accuracy 模式跑过，且服务 max_model_len > 32768，否则官方工具会跳过精度）")
    st.stop()

summary = call(f"/api/analysis/accuracy?experiment_id={with_accuracy[0]['id']}")
if not isinstance(summary, dict) or "prompt_level_strict" not in summary:
    st.error(f"精度摘要读取失败：{summary}")
    st.stop()

score = summary.get("prompt_level_strict") or 0
ci = summary.get("ci") or {}
columns = st.columns(4)
columns[0].metric("prompt_level_strict", f"{score * 100:.1f} %")
columns[1].metric("样本数", summary.get("sample_count") or "—")
columns[2].metric("标准误", f"± {ci.get('stderr_pp', '—')} pp",
                  help="二项分布标准误 sqrt(p(1-p)/n)")
columns[3].metric("95% 置信区间",
                  f"{ci.get('ci95_pp', ['—', '—'])[0]} – {ci.get('ci95_pp', ['—', '—'])[1]} %")

st.info("判读规则：只有超过 1 个标准误（约 4pp）的差异才可作为结论；"
        "参考文档里的 76%/80% 与本次结果若在误差内，不能宣称优劣。")

lengths = summary.get("lengths")
if isinstance(lengths, dict):
    st.subheader("模型输出长度分布")
    st.caption(f"来源：{lengths.get('source')}；单位：字符数（含思维链）")
    left, middle, right, fourth = st.columns(4)
    left.metric("最短", f"{lengths.get('min'):,}")
    middle.metric("中位数", f"{lengths.get('median'):,}")
    right.metric("最长", f"{lengths.get('max'):,}")
    fourth.metric("超过 8000 字符", f"{lengths.get('over_8k')} / {lengths.get('count')}")
    near = summary.get("near_truncation")
    if near:
        st.warning(f"有 {near} 条输出接近 max_tokens=32768 上限（被截断），"
                   f"这是单条样本耗时数分钟、整体评测变慢的直接原因。")
    frame = pd.DataFrame([{"统计量": name, "字符数": value} for name, value in
                          (("最短", lengths.get("min")), ("中位数", lengths.get("median")),
                           ("最长", lengths.get("max")))])
    st.bar_chart(frame.set_index("统计量"))

st.subheader("逐实验精度")
tables.accuracy_table([{"dataset": row.get("dataset"), "sample_count": row.get("sample_count"),
                        "prompt_level_strict": row.get("prompt_level_strict")}
                       for row in with_accuracy])

choice = st.selectbox("查看某次实验的原始分数",
                      [f"{row['id']} · {row['experiment_name']}" for row in with_accuracy])
if choice:
    experiment_id = int(str(choice).split("·")[0].strip())
    detail = call(f"/api/experiments/{experiment_id}")
    accuracy = (detail or {}).get("accuracy", []) if isinstance(detail, dict) else []
    if accuracy and accuracy[0].get("raw_scores"):
        with st.expander("原始分数（JSON）"):
            st.json(json.loads(accuracy[0]["raw_scores"]))
