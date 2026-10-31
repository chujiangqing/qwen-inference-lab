"""页面 4：性能评测 —— 单实验曲线 / 基线 vs 优化 并排对比。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import charts  # noqa: E402
import tables  # noqa: E402
from api_client import call  # noqa: E402

st.header("性能评测")

experiments = call("/api/experiments?limit=50")
rows = (experiments or {}).get("experiments", []) if isinstance(experiments, dict) else []
if not rows:
    st.info("暂无实验，请先在「实验管理」发起一次评测")
    st.stop()

options = {f"{row['id']} · {row['experiment_name']}": row["id"] for row in rows}
mode = st.radio("查看方式", ["单实验曲线", "基线 vs 优化对比"], horizontal=True)

if mode == "单实验曲线":
    choice = st.selectbox("选择实验", list(options))
    detail = call(f"/api/experiments/{options[choice]}")
    if not isinstance(detail, dict) or "performance" not in detail:
        st.error(f"读取失败：{detail}")
        st.stop()

    charts.concurrency_curves(detail)
    st.subheader("逐档明细")
    tables.performance_table(detail.get("performance", []))
    st.caption("TTFT/TPOT/E2E 单位为毫秒（AIPerf 原生单位）；TPOT = (E2E − TTFT) / (OSL − 1)。")

else:
    st.caption("选择两个实验按并发档对齐比较；百分比变化为正表示后者更大。"
               "**只有当两次实验用同样的请求数/档时才可直接比大小。**")
    names = list(options)
    left, right = st.columns(2)
    base_name = left.selectbox("基线实验", names, index=0)
    default_index = 1 if len(names) > 1 else 0
    cand_name = right.selectbox("对比实验（优化后）", names, index=default_index)

    base_id, cand_id = options[base_name], options[cand_name]
    if base_id == cand_id:
        st.info("请选择两个不同的实验")
        st.stop()

    data = call(f"/api/experiments/compare?baseline={base_id}&candidate={cand_id}")
    if not isinstance(data, dict) or "rows" not in data:
        st.error(f"对比失败：{data}")
        st.stop()

    st.success(data.get("conclusion") or "已生成对比")

    frame = pd.DataFrame(data["rows"])
    view = frame[["concurrency",
                  "output_throughput_base", "output_throughput_new", "output_throughput_delta_pct",
                  "ttft_p95_base", "ttft_p95_new", "ttft_p95_delta_pct",
                  "tpot_p95_base", "tpot_p95_new", "tpot_p95_delta_pct"]].copy()
    view.columns = ["并发", "吞吐-基线", "吞吐-对比", "吞吐变化%",
                    "TTFT-基线", "TTFT-对比", "TTFT变化%",
                    "TPOT-基线", "TPOT-对比", "TPOT变化%"]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("吞吐对比")
    st.bar_chart(frame.set_index("concurrency")[["output_throughput_base", "output_throughput_new"]])

    st.subheader("TTFT P95 对比（越低越好）")
    st.bar_chart(frame.set_index("concurrency")[["ttft_p95_base", "ttft_p95_new"]])

    with st.expander("原始对比数据（JSON）"):
        st.json(data)

# ---------------------------------------------------------------- 候选排行
st.divider()
st.subheader("参数搜索候选排行")
trials = call("/api/analysis/trials")
if isinstance(trials, dict) and trials.get("trials"):
    import pandas as pd
    frame = pd.DataFrame(trials["trials"])
    keep = [column for column in ("candidate", "dimension", "status", "valid",
                                  "output_throughput", "ttft_p95_ms", "tpot_p95_ms",
                                  "e2e_p95_ms", "vram_peak_gib", "score")
            if column in frame.columns]
    view = frame[keep].copy()
    view.columns = ["候选", "维度", "状态", "有效", "输出吞吐", "TTFT P95", "TPOT P95",
                    "E2E P95", "峰值显存", "得分"][:len(keep)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    ranked = trials.get("ranked") or []
    if ranked:
        st.success(f"当前排名第一：{ranked[0]['candidate']}（得分 {ranked[0].get('score')}）；"
                   "注意候选为 20 请求/档的筛选口径，不能与基线 100 请求/档直接比大小。")
else:
    st.info("参数搜索尚未产出结果（等 AutoTuner 跑完会出现在这里）")

# ---------------------------------------------------------------- 瓶颈核算
st.divider()
st.subheader("瓶颈核算：每生成 1 token 要读多少权重")
st.caption("稠密模型每生成 1 个 token 必须把全部权重读一遍；用实测 TPOT 反推带宽占用，"
           "判断是不是已经撞上显存带宽上限。")
pick = st.selectbox("选择实验做瓶颈核算", list(options))
report = call(f"/api/analysis/bottleneck?experiment_id={options[pick]}")
if isinstance(report, dict) and report.get("rows"):
    import pandas as pd
    frame = pd.DataFrame(report["rows"])
    columns = [column for column in ("concurrency", "output_throughput", "tpot_p95",
                                     "read_gbs", "bandwidth_pct", "status") if column in frame.columns]
    view = frame[columns].copy()
    view.columns = ["并发", "输出吞吐", "TPOT P95 (ms)", "权重读取 (GB/s)", "占峰值带宽 %", "状态"][:len(columns)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "bandwidth_pct" in frame.columns:
        st.bar_chart(frame.set_index("concurrency")[["bandwidth_pct"]])
    st.caption(f"权重 {report.get('weights_gib')} GiB；RTX 4090 标称带宽 "
               f"{report.get('peak_bandwidth_gbs')} GB/s。占比接近 100% 说明该档已被带宽限制。")
else:
    st.info(f"无法核算：{report}")