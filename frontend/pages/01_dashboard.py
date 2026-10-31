"""总览：一屏看清当前状态、最优配置与瓶颈。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import charts  # noqa: E402
import gpu_cards  # noqa: E402
import tables  # noqa: E402
import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("运行总览", "单卡 RTX 4090 · Qwen3.8-27B · FP8 权重 + FP8 KV Cache")

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

with theme.card():
    if ranked:
        best = ranked[0]
        base_tps = (baseline or {}).get("best_output_throughput")
        delta = (best["output_throughput"] / base_tps - 1) * 100 if base_tps and best.get("output_throughput") else None
        headline = f"{best['candidate']} · {best.get('output_throughput'):.1f} tok/s"
        if delta is not None:
            headline += f"（{delta:+.1f}% vs 基线）"
        st.markdown(f"### 当前最优：{headline}")
        theme.metrics([
            ("输出吞吐", theme.num(best.get("output_throughput"), 1, " tok/s"),
             theme.delta_pct(best.get("output_throughput"), base_tps)),
            ("TTFT P95", theme.num(best.get("ttft_p95_ms"), 0, " ms")),
            ("TPOT P95", theme.num(best.get("tpot_p95_ms"), 1, " ms")),
            ("候选 / 有效", f"{len(trials)} / {len(ranked)}"),
        ])
    elif baseline:
        st.markdown(f"### 基线已就绪：{baseline['experiment_name']}")
        theme.metrics([
            ("最高输出吞吐", f"{baseline['best_output_throughput']:.1f} tok/s"),
            ("评测口径", "100 请求/档 × 4 档"),
            ("精度", f"{accuracy_rows[0]['prompt_level_strict'] * 100:.1f}%" if accuracy_rows else "进行中"),
            ("候选/有效", f"{len(trials)} / {len(ranked)}"),
        ])
        st.caption("参数对照实验正在排队执行，完成后这里会显示最优配置与提升幅度。")

final_rows = [row for row in rows if str(row.get("experiment_name", "")).startswith("FINAL_")]
if baseline and final_rows:
    st.markdown("### 优化前后对比")
    data = call(f"/api/experiments/compare?baseline={baseline['id']}&candidate={final_rows[0]['id']}")
    if isinstance(data, dict) and data.get("rows"):
        st.success(data.get("conclusion", "已生成对比"))
        frame = pd.DataFrame(data["rows"])
        keep = [column for column in ("concurrency", "output_throughput_base", "output_throughput_new",
                                      "output_throughput_delta_pct", "ttft_p95_base", "ttft_p95_new",
                                      "ttft_p95_delta_pct") if column in frame.columns]
        view = frame[keep].copy()
        view.columns = ["并发", "吞吐·基线", "吞吐·优化", "吞吐变化", "TTFT·基线", "TTFT·优化", "TTFT 变化"][:len(keep)]
        st.dataframe(theme.colored_delta(view, ["吞吐变化", "TTFT 变化"]),
                     use_container_width=True, hide_index=True)
        charts.grouped_bars(frame, "concurrency",
                            ["output_throughput_base", "output_throughput_new"],
                            "吞吐对比（并排）", ["基线", "优化"])

if isinstance(baseline, dict):
    report = call(f"/api/analysis/bottleneck?experiment_id={baseline['id']}")
    if isinstance(report, dict) and report.get("rows"):
        st.markdown("### 瓶颈定位")
        frame = pd.DataFrame(report["rows"])
        keep = [column for column in ("concurrency", "output_throughput", "tpot_p95",
                                      "read_gbs", "bandwidth_pct") if column in frame.columns]
        view = frame[keep].copy()
        view.columns = ["并发", "输出吞吐 tok/s", "TPOT P95 ms", "权重读取 GB/s", "占峰值带宽"][:len(keep)]
        with theme.card():
            st.dataframe(
                view, use_container_width=True, hide_index=True,
                column_config={
                    "占峰值带宽": st.column_config.ProgressColumn(
                        "占峰值带宽", min_value=0, max_value=100, format="%.1f%%"),
                    "输出吞吐 tok/s": st.column_config.NumberColumn(format="%.2f"),
                    "TPOT P95 ms": st.column_config.NumberColumn(format="%.1f"),
                    "权重读取 GB/s": st.column_config.NumberColumn(format="%.0f"),
                })
            st.caption(f"权重 {report.get('weights_gib')} GiB，每生成 1 个 token 需全量读取一遍；"
                       f"显存带宽峰值 {report.get('peak_bandwidth_gbs')} GB/s。"
                       "并发 1 已接近带宽上限，说明单请求速度由带宽决定，加并发只能摊薄权重读取。")

kv = call("/api/analysis/kv-cache")
if isinstance(kv, dict) and kv.get("kv_capacity_tokens"):
    st.markdown("### 显存与 KV Cache")
    with theme.card():
        theme.metrics([
            ("可用 KV 显存", f"{kv.get('available_kv_gib')} GiB"),
            ("KV 容量", f"{kv.get('kv_capacity_tokens'):,} token"),
            ("单请求上限", f"{kv.get('per_request_tokens'):,} token"),
            ("可容纳并发", f"{kv.get('max_concurrency')}×"),
        ])
        queue = kv.get("queue") or {}
        usage = kv.get("usage_percent") or {}
        st.caption(f"KV 占用 平均 {usage.get('avg')}% / 峰值 {usage.get('max')}%；"
                   f"最大并行 {queue.get('max_running')}、最大排队 {queue.get('max_waiting')}；"
                   f"抢占 {kv.get('preemption_events')} 次。")

st.markdown("### 运行环境")
hardware = call("/api/hardware")
if isinstance(hardware, dict) and "error" not in hardware:
    gpu_cards.render(hardware)
status = call("/api/runtime/status")
service = (status or {}).get("service", {}) if isinstance(status, dict) else {}
if service.get("ready"):
    st.success(f"推理服务在线 · {', '.join(service.get('models') or [])} · max_model_len={service.get('max_model_len')}")
else:
    st.warning(f"推理服务未就绪：{service.get('error', '—')}")

with st.expander("最近实验"):
    tables.experiments_table(rows[:10])
