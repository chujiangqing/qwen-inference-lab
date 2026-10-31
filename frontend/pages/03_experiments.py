"""页面 3：实验管理 —— 发起评测（异步）+ 实时进度 + 历史实验。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(FRONTEND / "components"))

import tables  # noqa: E402
import theme  # noqa: E402
from api_client import call  # noqa: E402

theme.setup("实验管理", "发起评测、跟踪进度、查看历史（评测口径由官方脚本决定，这里只传参数）")




with st.form("run_form"):
    left, right = st.columns(2)
    label = left.text_input("实验名", value="E00_BASELINE")
    mode = left.selectbox("模式", ["all", "perf", "accuracy"])
    tokenizer = right.text_input("tokenizer（留空=模型自身，推荐）", value="",
                                 help="留空则用模型目录（服务端 ISL 精确 8192）；填 builtin 用 tiktoken，服务端长度会漂移")
    requests = right.number_input("perf 请求数/档", min_value=1, value=100, step=1,
                                  help="4 个并发档（1/2/4/8），每档跑这么多请求")
    batch = right.number_input("eval_batch_size（精度）", min_value=1, value=2, step=1)
    submitted = st.form_submit_button("先做前置检查")

if submitted:
    payload = {"label": label, "mode": mode, "tokenizer": tokenizer,
               "perf_requests": int(requests), "eval_batch_size": int(batch)}
    result = call("/api/benchmark/pretend", payload)
    if isinstance(result, dict) and result.get("ok"):
        st.success("前置检查通过，可以开跑")
    else:
        st.error("前置检查未通过，原因如下：")
        problems = (result or {}).get("problems") if isinstance(result, dict) else None
        st.write(problems or result)
    st.json(result)

if st.button("开始评测（后台执行，不阻塞页面）", type="primary"):
    payload = {"label": label, "mode": mode, "tokenizer": tokenizer,
               "perf_requests": int(requests), "eval_batch_size": int(batch)}
    result = call("/api/benchmark/run", payload, timeout=60)
    if isinstance(result, dict) and result.get("job_id"):
        st.session_state["job_id"] = result["job_id"]
        st.session_state["poll_until"] = time.time() + 1800
        st.success(f"已提交：job_id = {result['job_id']}，下面会自动刷新进度")
    else:
        detail = result.get("detail") if isinstance(result, dict) else None
        st.error("提交失败")
        if isinstance(detail, dict):
            st.write(detail.get("message") or detail)
            for problem in detail.get("problems") or []:
                st.write(f"- {problem}")
        else:
            st.write(detail or result)

# ---------------------------------------------------------------- 进度
job_id = st.session_state.get("job_id")
if job_id:
    st.subheader("任务进度")
    placeholder = st.empty()
    deadline = st.session_state.get("poll_until", 0)
    for _ in range(12):                      # 自动刷新约 24 秒
        progress = call(f"/api/benchmark/jobs/{job_id}/progress?lines=25", timeout=15)
        with placeholder.container():
            if not isinstance(progress, dict):
                st.warning(f"查询失败：{progress}")
                break
            status = progress.get("status")
            elapsed = progress.get("elapsed_seconds")
            percent = progress.get("percent")
            done, total = progress.get("cells_done"), progress.get("cells_total")
            st.write(f"状态：**{status}**　已运行：{elapsed if elapsed is not None else '—'} 秒"
                     f"　并发档：{done if done is not None else '—'}/{total or '—'}")
            if percent is not None:
                st.progress(percent / 100)
            if progress.get("output_dir"):
                st.caption(f"产物目录：{progress['output_dir']}")
            if progress.get("error"):
                st.error(progress["error"])
            if progress.get("log_tail"):
                st.code(progress["log_tail"], language="text")
            if status in {"完成", "失败"}:
                st.success("任务已结束，去「性能与对比」「精度评测」页看结果") if status == "完成" \
                    else st.error("任务失败，失败原因见上方")
        if progress.get("status") in {"完成", "失败"}:
            break
        if time.time() > deadline:
            break
        time.sleep(2)
    st.caption("任务在服务端继续跑；离开本页不影响。回来点下面的按钮可再看进度。")
    if st.button("刷新任务进度"):
        st.rerun()

st.subheader("历史实验")
experiments = call("/api/experiments?limit=50")
tables.experiments_table((experiments or {}).get("experiments", []) if isinstance(experiments, dict) else [])
