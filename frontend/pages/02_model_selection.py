"""页面 2：选型 —— 为什么是 FP8，其余版本为什么排除。

对齐评分表「模型与方案选型（10 分）」：不是"我选了 FP8"，而是给出硬约束与逐项证据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from api_client import call  # noqa: E402

st.header("模型与量化方案选型")
st.caption("三条硬约束：① 权重必须装进显存；② 推理框架必须支持该量化格式；③ 硬件架构必须支持该数值类型。")

data = call("/api/analysis/model-options")
if not isinstance(data, dict) or "variants" not in data:
    st.error(f"选型接口不可达：{data}")
    st.stop()

vram = data.get("vram_gib")
cap = data.get("compute_cap")
left, right, third = st.columns(3)
left.metric("本机显存", f"{vram} GiB" if vram else "—")
right.metric("GPU compute capability", cap or "—")
adopted = [item for item in data["variants"] if item["verdict"] == "采用"]
third.metric("满足全部约束的方案", f"{len(adopted)} 个")

frame = pd.DataFrame(data["variants"])
view = frame[["name", "size_gib", "quant_method", "verdict", "reason"]].copy()
view.columns = ["版本", "体积 GiB", "量化格式", "结论", "理由"]
st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("判定依据")
for item in data["variants"]:
    if item["verdict"] == "采用":
        st.success(f"**{item['name']}**：{item['reason']}")
    elif item["verdict"] == "排除":
        st.warning(f"**{item['name']}**：{item['reason']}")

st.markdown("""
### 为什么不自做 PTQ（AWQ / GPTQ）

| 因素 | 实际情况 |
|---|---|
| 量化对象 | 52 GiB 的 BF16 权重，本机装不下，只能 CPU 逐层量化 |
| 工具链 | 该模型是混合架构（48 层 Gated-DeltaNet + 16 层全注意力 + MTP + 视觉塔），AWQ/GPTQ 官方脚本未覆盖 |
| 时间 | 单次量化数小时，赛程内无法完成"量化 → 验证 → 回归"闭环 |
| 收益 | 官方 FP8（29 GiB）已满足显存约束，自做量化的额外收益不明确 |

赛题允许"直接选用开源社区中已量化裁剪好的现成模型版本"，因此采用官方 FP8，
并把它作为基线；KV Cache 的 FP8 量化则作为可对照的变量单独验证（见「性能与对比」页）。
""")
