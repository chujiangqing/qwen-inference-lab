"""后端入口：FastAPI 应用，挂载 6 组路由。

启动：
    uvicorn main:app --host 127.0.0.1 --port 8090   （在 backend/ 目录下）
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "storage"))
sys.path.insert(0, str(BACKEND / "benchmark"))

from api import analysis, benchmark, experiments, hardware, reports, runtime  # noqa: E402
from core.config import API_HOST, API_PORT, ROOT  # noqa: E402

app = FastAPI(title="Qwen3.8-27B InferenceLab API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ROUTERS = (hardware, runtime, experiments, benchmark, analysis, reports)
for module in ROUTERS:
    app.include_router(module.router)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "root": str(ROOT),
            "routers": ["hardware", "runtime", "experiments", "benchmark", "analysis", "reports"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
