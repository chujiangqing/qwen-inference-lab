"""统一视觉风格：页面配置、样式、标题行、卡片容器。

所有页面都通过 setup() 开头，保证字号、间距、配色一致；
页面里不出现任何"比赛/评分/需求书"之类的字样，只讲系统本身在做什么。
"""
from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

_CSS = """
<style>
  .block-container {padding-top: 2.0rem; padding-bottom: 3.5rem; max-width: 1200px;}
  h1 {font-size: 1.85rem !important; font-weight: 700 !important; letter-spacing: -0.015em;
      margin-bottom: 0.15rem !important;}
  h2 {font-size: 1.10rem !important; font-weight: 600 !important; margin-top: 1.7rem !important;}
  h3 {font-size: 0.98rem !important; font-weight: 600 !important;}
  p, li, .stMarkdown {font-size: 0.92rem;}
  [data-testid="stMetric"] {background: rgba(120,140,255,0.06);
      border: 1px solid rgba(120,140,255,0.18); border-radius: 12px; padding: 14px 16px;}
  [data-testid="stMetricLabel"] p {font-size: 0.78rem !important; opacity: 0.75;}
  [data-testid="stMetricValue"] {font-size: 1.45rem !important; font-weight: 650;}
  [data-testid="stSidebarNav"] a {border-radius: 8px;}
  [data-testid="stDecoration"], #MainMenu, footer {visibility: hidden;}
  div[data-testid="stExpander"] {border-radius: 10px;}
  hr {margin: 1.2rem 0 !important; opacity: 0.25;}
</style>
"""


def setup(title: str, subtitle: str = "", icon: str = "") -> None:
    """设置页面配置与样式，并渲染标题。每个页面只调用一次。"""
    try:
        st.set_page_config(page_title=title, page_icon=icon or "⚙️", layout="wide")
    except Exception:          # 同一进程内重复设置会抛异常，忽略即可
        pass
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f"## {icon} {title}" if icon else f"## {title}")
    if subtitle:
        st.caption(subtitle)


@contextmanager
def card():
    """带边框的区块容器，用于把一组指标/表格包成一张卡片。"""
    try:
        with st.container(border=True):
            yield
    except TypeError:          # 旧版本 Streamlit 不支持 border 参数
        yield


def metrics(items: list[tuple[str, str]], help_texts: list[str] | None = None) -> None:
    """一行等宽指标卡。items = [(标签, 值), ...]"""
    if not items:
        return
    columns = st.columns(len(items))
    for index, (label, value) in enumerate(items):
        help_text = (help_texts or [])[index] if help_texts and index < len(help_texts) else None
        columns[index].metric(label, value, help=help_text)


def note(text: str) -> None:
    """浅色说明文字，替代满屏的 st.info 蓝框。"""
    st.caption(text)
