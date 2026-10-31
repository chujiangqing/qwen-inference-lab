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
import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("性能与对比", "四档并发 · 基线 vs 优化并排对比 · 候选排行 · 瓶颈核算")


experiments = call("/api/experiments?limit=50")
rows = (experiments or {}).get("experiments", []) if isinstance(experiments, dict) else []
if not rows:
    st.info("暂无实验，请先在「实验管理」发起一次评测")
    st.stop()

options = {f"{row['id']} · {row.get('display_name') or row['experiment_name']}（{row.get('request_count') or '?'} 请求/档）": row["id"]
           for row in rows}
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
    st.caption("左侧固定选基线（E00_BASELINE_PERF），右侧选候选，每行才是干净的逐档对比。"
               "百分比变化 = 右 ÷ 左 − 1；只有两次实验请求数/档相同时才可以比大小。")
    # 标注每个实验的"每档请求数"，并自动把左侧配对成同口径的基线
    profile = call("/api/experiments/request-profile") or {}
    profile = profile.get("profile", {}) if isinstance(profile, dict) else {}

    def requests_of(exp_id: int):
        return (profile.get(str(exp_id)) or {}).get("requests")

    labels = {name: f"{name}（{(requests_of(exp_id) or '?')} 请求/档）"
              for name, exp_id in options.items()}
    names = list(labels)

    left, right = st.columns(2)
    # 默认右侧选第一个候选（不是基线本身）
    cand_default = 0
    for index, name in enumerate(names):
        if "BASELINE" not in name.upper() and "BASE20" not in name.upper():
            cand_default = index
            break
    cand_name = right.selectbox("对比实验（右）", names, index=cand_default,
                                format_func=lambda name: labels[name])
    cand_requests = requests_of(options[cand_name])

    # 左侧自动选同请求数的基线；没有就退回"基线"字样或第一项
    base_index = 0
    for index, name in enumerate(names):
        if requests_of(options[name]) != cand_requests:
            continue
        upper = name.upper()
        if "BASE20" in upper or "BASELINE" in upper:
            base_index = index
            break
    base_name = left.selectbox("参照实验（左）", names, index=base_index,
                               format_func=lambda name: labels[name])

    base_id, cand_id = options[base_name], options[cand_name]
    if base_id == cand_id:
        st.info("请选择两个不同的实验")
        st.stop()

    data = call(f"/api/experiments/compare?baseline={base_id}&candidate={cand_id}")
    if not isinstance(data, dict) or "rows" not in data:
        st.error(f"对比失败：{data}")
        st.stop()

    first = (data.get("rows") or [{}])[0]
    req_base, req_new = first.get("requests_base"), first.get("requests_new")
    conclusion = data.get("conclusion") or "已生成对比"
    delta = data.get("throughput_delta_pct")
    if req_base and req_new and req_base != req_new:
        st.warning(f"{conclusion}　｜　口径不同：左 {req_base} 请求/档 vs 右 {req_new} 请求/档，"
                   "此表只用于看趋势；正式结论以 100 请求/档复测为准。")
    elif delta is not None and delta < 0:
        st.error(f"{conclusion}（右侧更慢）")
    else:
        st.success(conclusion)

    frame = pd.DataFrame(data["rows"])
    comparable = bool(req_base and req_new and req_base == req_new)
    if comparable:
        view = frame[["concurrency",
                      "output_throughput_base", "output_throughput_new", "output_throughput_delta_pct",
                      "ttft_p95_base", "ttft_p95_new", "ttft_p95_delta_pct",
                      "tpot_p95_base", "tpot_p95_new", "tpot_p95_delta_pct"]].copy()
    else:
        # 请求数不同：延迟类指标（P95）不可比，只保留吞吐
        view = frame[["concurrency",
                      "output_throughput_base", "output_throughput_new",
                      "output_throughput_delta_pct"]].copy()
    if comparable:
        view.columns = ["并发", "吞吐·基线", "吞吐·优化", "吞吐变化%",
                        "TTFT·基线", "TTFT·优化", "TTFT变化%",
                        "TPOT·基线", "TPOT·优化", "TPOT变化%"]
    else:
        view.columns = ["并发", "吞吐·基线", "吞吐·优化", "吞吐变化%"]
    st.dataframe(theme.colored_delta(view, ["吞吐变化%", "TTFT变化%", "TPOT变化%"],
                                     invert=("TTFT变化%", "TPOT变化%")),
                 use_container_width=True, hide_index=True,
                 column_config={
                     "并发": st.column_config.NumberColumn("并发", format="%d"),
                     "吞吐变化%": st.column_config.NumberColumn(format="%+.2f%%"),
                     "TTFT变化%": st.column_config.NumberColumn(format="%+.2f%%"),
                     "TPOT变化%": st.column_config.NumberColumn(format="%+.2f%%"),
                     "吞吐·基线": st.column_config.NumberColumn(format="%.2f"),
                     "吞吐·优化": st.column_config.NumberColumn(format="%.2f"),
                     "TTFT·基线": st.column_config.NumberColumn(format="%.0f"),
                     "TTFT·优化": st.column_config.NumberColumn(format="%.0f"),
                     "TPOT·基线": st.column_config.NumberColumn(format="%.1f"),
                     "TPOT·优化": st.column_config.NumberColumn(format="%.1f"),
                 })

    st.subheader("吞吐对比")
    charts.grouped_bars(frame, "concurrency",
                        ["output_throughput_base", "output_throughput_new"],
                        "吞吐对比（并排，不是堆叠）", ["基线", "对比"])

    st.subheader("TTFT P95 对比（越低越好）")
    charts.grouped_bars(frame, "concurrency",
                        ["ttft_p95_base", "ttft_p95_new"],
                        "TTFT P95 对比（毫秒，越低越好）", ["基线", "对比"])

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

    # 候选吞吐对比（含基线参考线）
    base_tps = None
    for row in rows:
        if "BASELINE" in str(row.get("experiment_name", "")).upper() and row.get("best_output_throughput"):
            base_tps = max(base_tps or 0, row["best_output_throughput"])
    if "output_throughput" in frame.columns and frame["output_throughput"].notna().any():
        chart_frame = frame.dropna(subset=["output_throughput"]).set_index("candidate")[["output_throughput"]]
        chart_frame.columns = ["输出吞吐 tok/s"]
        st.bar_chart(chart_frame)
        if base_tps:
            st.caption(f"基线（100 请求/档）为 {base_tps:.2f} tok/s；候选为 20 请求/档，仅用于排序。")

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