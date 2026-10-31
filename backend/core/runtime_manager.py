"""模型服务生命周期：起、停、查状态、读日志。

把"怎么起 vLLM"收敛在 core/runtime_manager.py，把"这个模型该怎么配"留在 model/qwen_runtime.py。
API 层不再直接拼 subprocess。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import BASE_URL, ROOT, SERVED_MODEL, runtime_config  # noqa: E402

SCRIPTS = ROOT / "scripts"


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def status() -> dict:
    url = BASE_URL.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {"ready": False, "error": f"{type(error).__name__}: {error}"}
    cards = {item.get("id"): item for item in body.get("data", [])}
    card = cards.get(SERVED_MODEL) or {}
    return {"ready": SERVED_MODEL in cards, "models": list(cards),
            "max_model_len": card.get("max_model_len"), "card": card}


def profiles() -> dict:
    config = runtime_config()
    return {name: {"description": item.get("description"),
                   "max_model_len": item.get("max_model_len"),
                   "max_num_seqs": item.get("max_num_seqs"),
                   "speculative": bool(item.get("speculative_config"))}
            for name, item in (config.get("profiles") or {}).items()}


def start(profile: str = "baseline", overrides: dict | None = None,
          timeout: float = 1800) -> dict:
    command = ["bash", str(SCRIPTS / "start_model.sh"), "--profile", profile]
    for key, value in (overrides or {}).items():
        command += ["--set", f"{key}={json.dumps(value, separators=(',', ':'))}"]
    done = _run(command, timeout)
    return {"ok": done.returncode == 0, "profile": profile,
            "output": (done.stdout or done.stderr or "")[-1500:], "status": status()}


def stop(timeout: float = 180) -> dict:
    done = _run(["bash", str(SCRIPTS / "stop_model.sh"), "--force"], timeout)
    return {"ok": done.returncode == 0,
            "output": ((done.stdout or "") + (done.stderr or ""))[-800:], "status": status()}


def logs(lines: int = 60) -> dict:
    log_dir = Path(runtime_config()["server"]["log_dir"])
    candidates = sorted(log_dir.glob("serve_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {"log": None, "tail": ""}
    newest = candidates[0]
    text = newest.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return {"log": str(newest), "tail": "".join(text[-lines:])}


def wait_ready(timeout: float = 900, interval: float = 3.0) -> dict:
    deadline = time.time() + timeout
    started = time.time()
    while time.time() < deadline:
        current = status()
        if current.get("ready"):
            return {"ready": True, "seconds": round(time.time() - started, 1), "status": current}
        time.sleep(interval)
    return {"ready": False, "seconds": round(time.time() - started, 1), "status": status()}
