"""系统监控：CPU / 内存 / 磁盘。纯标准库，Linux 与 Windows 都能跑。"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def memory_gib() -> dict:
    if sys.platform.startswith("linux"):
        info = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith(("MemTotal", "MemAvailable", "MemFree")):
                    key, value = line.split(":", 1)
                    info[key] = round(int(value.split()[0]) / 1024 ** 2, 2)
        except OSError:
            pass
        return {"total": info.get("MemTotal"), "available": info.get("MemAvailable"),
                "free": info.get("MemFree")}
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return {"total": round(status.ullTotalPhys / 1024 ** 3, 2),
                "available": round(status.ullAvailPhys / 1024 ** 3, 2), "free": None}
    return {"total": None, "available": None, "free": None}


def disk_gib(path: Path | str) -> dict:
    target = Path(path)
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as error:
        return {"path": str(target), "error": str(error)}
    return {"path": str(target),
            "total_gib": round(usage.total / 1024 ** 3, 1),
            "free_gib": round(usage.free / 1024 ** 3, 1),
            "used_percent": round(usage.used / usage.total * 100, 1)}


def cpu() -> dict:
    info = {"count": os.cpu_count()}
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        try:
            info["loadavg"] = [round(value, 2) for value in os.getloadavg()]
        except (OSError, AttributeError):
            pass
    return info


def snapshot(root: Path) -> dict:
    return {"cpu": cpu(), "memory_gib": memory_gib(), "disk": disk_gib(root)}


if __name__ == "__main__":
    import json

    print(json.dumps(snapshot(Path(__file__).resolve().parents[2]), ensure_ascii=False, indent=2))
