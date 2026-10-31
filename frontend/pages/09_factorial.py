"""页面：双因子对照（2×2 因子设计）。

两两对比只能回答"这两个实验差多少"；本页把两个变量的四格摆在一起，
给出主效应与交互项，回答"两个优化叠加是相加、饱和还是互相放大"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("双因子对照", "MTP × 并发上限：四格 + 主效应 + 交互项（统一 20 请求/档口径）")

data = call("/api/analysis/factorial")
if not isinstance(data, dict) or "grid" not in data:
    st.error(f"因子分析接口不可达：{data}")
    st.stop()

grid = data["grid"]


def cell_row(key: str, label: str) -> dict:
    cell = grid.get(key)
    if not cell:
        return {"格": label, "实验": "（待补）", "吞吐 tok/s": None,
                "TTFT P95 ms": None, "TPOT P95 ms": None, "最佳并发": None}
    return {"格": label, "实验": cell["experiment"], "吞吐 tok/s": cell["throughput"],
            "TTFT P95 ms": cell["ttft_p95"], "TPOT P95 ms": cell["tpot_p95"],
            "最佳并发": cell["concurrency"]}


rows = [cell_row("off_4", "MTP 关 · 4 路"), cell_row("off_8", "MTP 关 · 8 路"),
        cell_row("on_4", "MTP 开 · 4 路"), cell_row("on_8", "MTP 开 · 8 路")]
frame = pd.DataFrame(rows)

with theme.card():
    st.dataframe(frame, use_container_width=True, hide_index=True,
                 column_config={
                     "吞吐 tok/s": st.column_config.NumberColumn(format="%.2f"),
                     "TTFT P95 ms": st.column_config.NumberColumn(format="%.0f"),
                     "TPOT P95 ms": st.column_config.NumberColumn(format="%.1f"),
                     "最佳并发": st.column_config.NumberColumn(format="%d"),
                 })

effects = data.get("effects") or {}
interaction = data.get("interaction")

st.markdown("### 主效应")
columns = st.columns(2)
columns[0].metric("提并发（MTP 关时）",
                  theme.num(effects.get("seq_at_mtp_off"), 2, " tok/s"),
                  help="MTP 关闭时，把 max_num_seqs 从 4 提到 8 的吞吐增量")
columns[1].metric("开 MTP（4 路时）",
                  theme.num(effects.get("mtp_at_seq4"), 2, " tok/s"),
                  help="4 路并发下，开 MTP 的吞吐增量")

st.markdown("### 交互作用")
if interaction is None:
    st.info(data.get("verdict") or "四格未齐")
else:
    st.metric("交互项 I", f"{interaction:+.2f} tok/s",
              help="I =（开 MTP 后提并发的增量）−（关 MTP 时提并发的增量）")
    verdict = data.get("verdict") or ""
    if interaction > 1:
        st.success(verdict)
    elif interaction < -1:
        st.warning(verdict)
    else:
        st.info(verdict)
    st.caption("判读：I ≈ 0 说明两个优化可加；I < 0 说明它们抢同一个瓶颈（这里是显存带宽），"
               "叠加时后加的那个收益会变小；I > 0 说明互相放大。")

st.markdown("### 四格明细")
st.caption("同一格有多个实验时取吞吐最高的那个；所有格都是 20 请求/档，可直接比较。")
if isinstance(data.get("cells"), list) and data["cells"]:
    st.dataframe(pd.DataFrame(data["cells"])[
        ["experiment", "mtp", "max_num_seqs", "max_model_len",
         "throughput", "ttft_p95", "tpot_p95", "concurrency"]],
        use_container_width=True, hide_index=True)
else:
    st.info("暂无 20 请求/档的实验")
