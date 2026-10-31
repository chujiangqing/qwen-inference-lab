"""GPU 卡片：把硬件快照渲染成指标卡。"""
from __future__ import annotations

import streamlit as st


def render(snapshot: dict) -> None:
    gpu = (snapshot or {}).get("gpu") or {}
    if not gpu.get("available"):
        st.warning(f"GPU 不可用：{gpu.get('error', '未检测到 nvidia-smi')}")
        return
    for card in gpu.get("gpus", []):
        columns = st.columns(4)
        columns[0].metric("型号", card.get("name", "—"))
        columns[1].metric("显存已用", f"{(card.get('memory_used_mib') or 0) / 1024:.1f} GiB",
                          f"共 {(card.get('memory_total_mib') or 0) / 1024:.0f} GiB")
        columns[2].metric("GPU 利用率", f"{card.get('utilization') or 0:.0f} %")
        columns[3].metric("温度 / 功耗", f"{card.get('temperature_c') or 0:.0f} ℃",
                          f"{card.get('power_w') or 0:.0f} W")
