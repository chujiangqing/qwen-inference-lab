"""GPU 监控：快照查询 + 后台采样。

为什么单独成文件：`api/hardware.py`（实时快照）与 `optimizer/auto_tuner.py`（实验期间峰值）
都需要 nvidia-smi，但一个要"当前值"、一个要"整段的峰值"。收敛在这里，避免两份实现。
AIPerf 自带的 GPU telemetry 在校准中出现心跳问题，所以统一用独立 nvidia-smi 轮询。
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

FIELDS = ("index,name,memory.total,memory.used,memory.free,utilization.gpu,"
          "temperature.gpu,power.draw,power.limit")


@dataclass
class GpuSample:
    index: str
    name: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    memory_free_mib: float | None
    utilization: float | None
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None

    def to_dict(self) -> dict:
        return self.__dict__ | {}


def _num(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def snapshot() -> dict:
    try:
        done = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if done.returncode != 0:
        return {"available": False, "error": done.stderr.strip()[:300]}
    samples = []
    for line in done.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 9:
            continue
        samples.append(GpuSample(parts[0], parts[1], _num(parts[2]), _num(parts[3]), _num(parts[4]),
                                 _num(parts[5]), _num(parts[6]), _num(parts[7]), _num(parts[8])))
    return {"available": True, "gpus": [sample.to_dict() for sample in samples]}


class PeakSampler:
    """实验期间后台采样，记录显存峰值（GiB）与最高利用率。"""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.vram_peak_gib = 0.0
        self.util_peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            data = snapshot()
            if data.get("available"):
                for gpu in data["gpus"]:
                    used = gpu.get("memory_used_mib") or 0
                    self.vram_peak_gib = max(self.vram_peak_gib, used / 1024)
                    self.util_peak = max(self.util_peak, gpu.get("utilization") or 0)
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
