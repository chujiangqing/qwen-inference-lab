"""共享 HTTP 客户端：所有页面通过它访问后端 API，不在页面里拼 URL。"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

API_BASE = os.environ.get("LAB_API_BASE", "http://127.0.0.1:8090")


def call(path: str, payload: dict | None = None, method: str | None = None, timeout: float = 60):
    url = f"{API_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}", "detail": error.read().decode("utf-8", "replace")[:800]}
    except (urllib.error.URLError, OSError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
