"""页面 2：选型 —— 为什么是 FP8，其余版本为什么排除。

把每个候选版本的体积、量化格式、引擎支持与硬件能力放在一起，给出可核对的取舍依据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("模型与量化选型", "权重必须装进显存 · 引擎必须支持该格式 · 硬件必须支持该数值类型")




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

st.markdown("### 推理框架对比")
st.dataframe(pd.DataFrame([
    {"框架": "vLLM", "版本": "0.31.0（已装）", "该架构支持": "官方注册表支持",
     "本次实测": "是（4 档 × 100 请求 + IFBench）", "结论": "采用"},
    {"框架": "SGLang", "版本": "0.5.21（索引可达）", "该架构支持": "未验证",
     "本次实测": "否", "结论": "不给结论"},
    {"框架": "TensorRT-LLM", "版本": "1.2.1（索引可达）", "该架构支持": "需构建 engine，混合线性注意力通常不支持",
     "本次实测": "否", "结论": "赛程内无法闭环"},
    {"框架": "llama.cpp", "版本": "llama-cpp-python 0.3.36（可达）", "该架构支持": "仓库内有 Q5_K_M GGUF（19 GB），支持未知",
     "本次实测": "否", "结论": "不给结论"},
]), use_container_width=True, hide_index=True)
st.caption("后三条为「未实测」，不是「不可行」——这是本项目的真实短板：只证明了 vLLM 这条路能跑出数字。")

st.markdown("### 量化路线对比")
st.dataframe(pd.DataFrame([
    {"路线": "社区现成 FP8", "体积": "28.75 GiB", "可行": "✅", "依据": "vLLM 原生支持，已跑完 4 档 × 100 请求与 IFBench"},
    {"路线": "自做 PTQ（AWQ/GPTQ）", "体积": "需先加载 52 GiB BF16", "可行": "❌", "依据": "显存装不下 + 工具链不覆盖混合架构 + 校准需数小时"},
    {"路线": "社区 GGUF（Q5_K_M）", "体积": "19 GB", "可行": "⚠️ 未验证", "依据": "需换 llama.cpp 引擎，架构支持未知"},
    {"路线": "社区 EXL3（3.5bpw）", "体积": "15 GB", "可行": "❌", "依据": "vLLM 不支持 exl3，需 ExLlamaV2"},
    {"路线": "社区 NVFP4", "体积": "21 GB", "可行": "❌", "依据": "FP4 需 Blackwell（sm100），本机 compute_cap = 8.9"},
    {"路线": "DFlash2", "体积": "3.6 GB", "可行": "❌", "依据": "非完整模型，不能独立推理"},
]), use_container_width=True, hide_index=True)
