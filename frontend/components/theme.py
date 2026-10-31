"""统一视觉风格：页面配置、样式、标题、卡片、数字格式化。

目标：一眼能看到"结论 + 数字"，表格不拥挤，配色克制。
"""
from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

_CSS = """
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}
  h1 {font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.015em;}
  h2 {font-size: 1.08rem !important; font-weight: 650 !important; margin-top: 1.8rem !important;
      padding-bottom: .35rem; border-bottom: 1px solid rgba(120,140,255,0.18);}
  h3 {font-size: 0.95rem !important; font-weight: 600 !important; opacity: .95;}
  p, li {font-size: 0.90rem; line-height: 1.6;}
  [data-testid="stMetric"] {background: linear-gradient(180deg, rgba(120,140,255,0.10), rgba(120,140,255,0.03));
      border: 1px solid rgba(120,140,255,0.20); border-radius: 12px; padding: 12px 14px;}
  [data-testid="stMetricLabel"] p {font-size: 0.74rem !important; opacity: .72; letter-spacing: .02em;}
  [data-testid="stMetricValue"] {font-size: 1.40rem !important; font-weight: 650; line-height: 1.2;}
  [data-testid="stMetricDelta"] {font-size: 0.76rem !important;}
  [data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
  [data-testid="stExpander"] {border-radius: 10px; border-color: rgba(120,140,255,0.15) !important;}
  [data-testid="stSidebar"] {background: rgba(120,140,255,0.04);}
  [data-testid="stSidebarNav"] span {font-size: 0.88rem;}
  [data-testid="stDecoration"], #MainMenu, footer {visibility: hidden;}
  hr {margin: 1.4rem 0 !important; opacity: .18;}
  code {font-size: .84rem;}
</style>
"""


def setup(title: str, subtitle: str = "", icon: str = "") -> None:
    """页面配置 + 全局样式 + 标题。"""
    try:
        st.set_page_config(page_title=title, page_icon=icon or "⚙️", layout="wide")
    except Exception:            # 同进程重复设置会抛异常
        pass
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f"## {icon} {title}" if icon else f"## {title}")
    if subtitle:
        st.caption(subtitle)


@contextmanager
def card():
    """带边框区块。"""
    try:
        with st.container(border=True):
            yield
    except TypeError:            # 旧版 Streamlit
        yield


def section(title: str, desc: str = "") -> None:
    st.markdown(f"### {title}")
    if desc:
        st.caption(desc)


def metrics(items: list[tuple[str, str]] | list[tuple[str, str, str]],
            helps: list[str] | None = None) -> None:
    """一行指标卡。item 可为 (标签, 值) 或 (标签, 值, delta)。"""
    if not items:
        return
    columns = st.columns(len(items))
    for index, item in enumerate(items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        help_text = (helps or [])[index] if helps and index < len(helps) else None
        if delta is None:
            columns[index].metric(label, value, help=help_text)
        else:
            columns[index].metric(label, value, delta=delta, help=help_text)


# ---------------------------------------------------------------- 数字格式
def num(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def pct(value, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        text = f"{float(value):,.{digits}f}%"
        return f"+{text}" if signed and float(value) > 0 else text
    except (TypeError, ValueError):
        return str(value)


def delta_pct(new, base, digits: int = 1) -> str | None:
    """相对变化，用于 metric 的 delta 参数（正数带 + 号）。"""
    if not base or new is None:
        return None
    try:
        change = (float(new) / float(base) - 1) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f"{change:+.{digits}f}%"


def colored_delta(frame, columns: list[str], threshold: float = 0.0,
                  invert: tuple[str, ...] = ()):
    """给百分比列上色。

    invert 里的列表示"数值越大越差"（延迟类）：正数标红、负数标绿。
    """
    def _style(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if number > threshold:
            return "color: #2e9e5b; font-weight: 600"
        if number < -threshold:
            return "color: #d9534f; font-weight: 600"
        return "opacity: .7"

    def _style_inverted(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if number > threshold:
            return "color: #d9534f; font-weight: 600"
        if number < -threshold:
            return "color: #2e9e5b; font-weight: 600"
        return "opacity: .7"

    try:
        styler = frame.style
        for column in columns:
            if column not in frame.columns:
                continue
            fn = _style_inverted if column in invert else _style
            styler = styler.map(fn, subset=[column])
        return styler
    except (ImportError, AttributeError):
        # 没有 jinja2 时退化为普通表格（功能不受影响，只是没有颜色）
        return frame
