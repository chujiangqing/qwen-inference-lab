"""页面 1：答辩主页面 —— 一页看懂「选型 → 瓶颈 → 优化 → 质量」。

设计原则：所有结论都在第一屏，其余页面只作为下钻细节。
顺序刻意对齐评分表：选型(10) → 优化(40) → 结果评估(25) → 工程(10)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import gpu_cards  # noqa: E402
import tables  # noqa: E402
from api_client import call  # noqa: E402

st.header("Qwen3.8-27B 单卡推理优化 — 一页看懂")

# ============================================================ 1. 结论
experiments = call("/api/experiments?limit=50")
rows = (experiments or {}).get("experiments", []) if isinstance(experiments, dict) else []
baseline = None
for row in rows:
    if "BASELINE" in str(row.get("experiment_name", "")).upper() and row.get("best_output_throughput"):
        if baseline is None or row["best_output_throughput"] > baseline["best_output_throughput"]:
            baseline = row

trials_payload = call("/api/analysis/trials")
trials = (trials_payload or {}).get("trials", []) if isinstance(trials_payload, dict) else []
ranked = (trials_payload or {}).get("ranked", []) if isinstance(trials_payload, dict) else []
accuracy_rows = [row for row in rows if row.get("prompt_level_strict") is not None]

st.subheader("① 结论")
if ranked:
    best = ranked[0]
    base_tps = (baseline or {}).get("best_output_throughput")
    delta = (best["output_throughput"] / base_tps - 1) * 100 if base_tps and best.get("output_throughput") else None
    line = f"推荐 **{best['candidate']}**：{best.get('output_throughput'):.1f} tok/s"
    if delta is not None:
        line += f"（相对基线 {delta:+.1f}%）"
    if accuracy_rows:
        line += f"；质量 IFBench {accuracy_rows[0]['prompt_level_strict'] * 100:.1f}%"
    st.success(line)
else:
    st.info("参数搜索尚未产出结果；当前已有基线数据，候选实验排队中。")

columns = st.columns(4)
columns[0].metric("基线吞吐（100 请求/档）",
                  f"{baseline['best_output_throughput']:.1f} tok/s" if baseline else "—")
columns[1].metric("候选数 / 有效数", f"{len(trials)} / {len(ranked)}")
columns[2].metric("IFBench",
                  f"{accuracy_rows[0]['prompt_level_strict'] * 100:.1f}%" if accuracy_rows else "—")
columns[3].metric("模型 / 量化", "FP8 权重 + FP8 KV")

# ============================================================ 2. 瓶颈
st.subheader("② 瓶颈在哪（先算再做）")
bottleneck = call(f"/api/analysis/bottleneck?experiment_id={baseline['id']}") if baseline else {}
if isinstance(bottleneck, dict) and bottleneck.get("rows"):
    frame = pd.DataFrame(bottleneck["rows"])
    keep = [column for column in ("concurrency", "output_throughput", "tpot_p95",
                                  "read_gbs", "bandwidth_pct") if column in frame.columns]
    view = frame[keep].copy()
    view.columns = ["并发", "输出吞吐 tok/s", "TPOT P95 ms", "权重读取 GB/s", "占峰值带宽 %"][:len(keep)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(f"权重 {bottleneck.get('weights_gib')} GiB（稠密，每 token 需全量读一遍）；"
               f"4090 标称带宽 {bottleneck.get('peak_bandwidth_gbs')} GB/s。"
               "并发 1 已达带宽上限 → 单请求无法更快；并发 8 与 4 吞吐相同是服务端 max_num_seqs=4 封顶，不是硬件饱和。")
else:
    st.info("还没有可核算的性能数据")

# ============================================================ 3. 优化对比
st.subheader("③ 优化前后对比")
final_rows = [row for row in rows if str(row.get("experiment_name", "")).startswith("FINAL_")]
if baseline and final_rows:
    base_id, cand_id = baseline["id"], final_rows[0]["id"]
    data = call(f"/api/experiments/compare?baseline={base_id}&candidate={cand_id}")
    if isinstance(data, dict) and data.get("rows"):
        st.success(data.get("conclusion", "已生成对比"))
        frame = pd.DataFrame(data["rows"])
        keep = [column for column in ("concurrency", "output_throughput_base", "output_throughput_new",
                                      "output_throughput_delta_pct", "ttft_p95_base", "ttft_p95_new",
                                      "ttft_p95_delta_pct") if column in frame.columns]
        view = frame[keep].copy()
        view.columns = ["并发", "吞吐-基线", "吞吐-优化", "吞吐变化%",
                        "TTFT-基线", "TTFT-优化", "TTFT变化%"][:len(keep)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.bar_chart(frame.set_index("concurrency")[["output_throughput_base", "output_throughput_new"]])
    else:
        st.warning(f"对比失败：{data}")
elif ranked:
    st.info("候选已产出，但还没有 100 请求/档的同口径复测结果（FINAL_*），暂不与基线并排。")
else:
    st.info("等待候选实验结果。")

# ============================================================ 4. 质量
st.subheader("④ 质量代价")
if accuracy_rows:
    summary = call(f"/api/analysis/accuracy?experiment_id={accuracy_rows[0]['id']}")
    if isinstance(summary, dict) and "prompt_level_strict" in summary:
        ci = summary.get("ci") or {}
        left, middle, right, fourth = st.columns(4)
        left.metric("prompt_level_strict", f"{summary['prompt_level_strict'] * 100:.1f}%")
        middle.metric("样本数", summary.get("sample_count") or "—")
        right.metric("标准误", f"± {ci.get('stderr_pp', '—')} pp")
        fourth.metric("95% CI", f"{ci.get('ci95_pp', ['—', '—'])[0]}–{ci.get('ci95_pp', ['—', '—'])[1]}%")
        lengths = summary.get("lengths") or {}
        if lengths:
            st.caption(f"输出长度 中位 {lengths.get('median'):,} 字符 / 最长 {lengths.get('max'):,} 字符；"
                       f"接近 32768 token 上限的样本 {summary.get('near_truncation')} 条"
                       "（这是单条样本耗时数分钟的原因）。")
        st.caption("判读：差异小于 1 个标准误（约 4pp）不能作为结论。")
else:
    st.info("精度评测尚未完成入库")

# ============================================================ 5. 环境
st.subheader("⑤ 运行环境")
hardware = call("/api/hardware")
if isinstance(hardware, dict) and "error" not in hardware:
    gpu_cards.render(hardware)
status = call("/api/runtime/status")
service = (status or {}).get("service", {}) if isinstance(status, dict) else {}
if service.get("ready"):
    st.success(f"模型服务在线：{service.get('models')}，max_model_len={service.get('max_model_len')}")
else:
    st.error(f"模型服务未就绪：{service.get('error', '—')}")

with st.expander("最近实验"):
    tables.experiments_table(rows[:10])
