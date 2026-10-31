"""显存监控：把 GPU 采样变成"趋势 + 峰值"，用于判断 OOM 风险与碎片。

和 gpu_monitor 的分工：gpu_monitor 只负责"读一次 nvidia-smi"，
这里负责"连续读、算峰值/均值、判定是否贴着上限跑"。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from monitor.gpu_monitor import snapshot


@dataclass
class VramTrace:
    samples: list[float] = field(default_factory=list)
    utilization: list[float] = field(default_factory=list)
    interval: float = 2.0

    @property
    def peak_gib(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def mean_gib(self) -> float:
        return round(sum(self.samples) / len(self.samples), 2) if self.samples else 0.0

    @property
    def floor_gib(self) -> float:
        return min(self.samples) if self.samples else 0.0

    def summary(self, total_gib: float | None = None) -> dict:
        headroom = round(total_gib - self.peak_gib, 2) if total_gib else None
        return {"samples": len(self.samples), "peak_gib": round(self.peak_gib, 2),
                "mean_gib": self.mean_gib, "floor_gib": round(self.floor_gib, 2),
                "peak_utilization": max(self.utilization) if self.utilization else None,
                "headroom_gib": headroom,
                "tight": bool(headroom is not None and headroom < 2.0)}


class VramMonitor:
    """上下文管理器：`with VramMonitor() as mon: ... ; mon.trace.summary()`"""

    def __init__(self, interval: float = 2.0) -> None:
        self.trace = VramTrace(interval=interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_gib: float | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            data = snapshot()
            if data.get("available"):
                for gpu in data["gpus"]:
                    self.total_gib = round((gpu.get("memory_total_mib") or 0) / 1024, 2)
                    self.trace.samples.append((gpu.get("memory_used_mib") or 0) / 1024)
                    if gpu.get("utilization") is not None:
                        self.trace.utilization.append(float(gpu["utilization"]))
            self._stop.wait(self.trace.interval)

    def __enter__(self) -> "VramMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        time.sleep(0)


if __name__ == "__main__":
    import json

    with VramMonitor(interval=1.0) as monitor:
        time.sleep(3)
    print(json.dumps(monitor.trace.summary(monitor.total_gib), ensure_ascii=False, indent=2))
