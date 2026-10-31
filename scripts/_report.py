"""Small shared helpers for the Task-0 detection scripts.

Read-only by design: nothing here installs packages, edits files outside the
report directory, or talks to the model server.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - older/embedded interpreters
            pass


def run(command: list[str], timeout: float = 60, cwd: str | None = None) -> dict:
    """Run a command and return its outcome instead of raising."""
    shown = " ".join(str(part) for part in command)
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              cwd=cwd, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {"command": shown, "ok": False, "code": None, "stdout": "", "stderr": "命令不存在"}
    except subprocess.TimeoutExpired:
        return {"command": shown, "ok": False, "code": None, "stdout": "", "stderr": f"超时 >{timeout}s"}
    except OSError as error:
        return {"command": shown, "ok": False, "code": None, "stdout": "", "stderr": str(error)}
    return {"command": shown, "ok": done.returncode == 0, "code": done.returncode,
            "stdout": done.stdout.strip(), "stderr": done.stderr.strip()}


def find_executable(names: tuple[str, ...], extra: tuple[str, ...] = ()) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for candidate in extra:
        path = Path(os.path.expandvars(candidate))
        if path.is_file():
            return str(path)
    return None


def executable_version(path: str | None, flag: str = "--version") -> str | None:
    if not path:
        return None
    result = run([path, flag], timeout=30)
    if result["ok"] and result["stdout"]:
        return result["stdout"].splitlines()[0].strip()
    return None


def working_python(path: str | None) -> str | None:
    """Windows 的 WindowsApps 存根会以 9009 退出，必须实际执行一次才算可用。"""
    if not path:
        return None
    result = run([path, "-c", "import sys; print(sys.version_info[0])"], timeout=30)
    return path if result["ok"] and result["stdout"].strip().isdigit() else None


def read_os_release() -> dict:
    data = {}
    for candidate in ("/etc/os-release", "/usr/lib/os-release"):
        path = Path(candidate)
        if path.is_file():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"')
            break
    return data


def memory_bytes() -> dict:
    """Total and available RAM without third-party packages."""
    if sys.platform.startswith("linux"):
        info = {}
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
        return {"total": info.get("MemTotal"), "available": info.get("MemAvailable")}
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return {"total": status.ullTotalPhys, "available": status.ullAvailPhys}
    if sys.platform == "darwin":
        page = int(run(["sysctl", "-n", "hw.pagesize"])["stdout"] or 4096)
        total = run(["sysctl", "-n", "hw.memsize"])["stdout"]
        return {"total": int(total) if total.isdigit() else None, "available": None}
    return {"total": None, "available": None}


def gib(value: float | int | None) -> float | None:
    return None if value is None else round(value / 1024 ** 3, 2)


def disk_report(path: str | Path | None) -> dict:
    if not path:
        return {"path": None, "exists": False}
    resolved = Path(path).expanduser()
    exists = resolved.exists()
    entry = {"path": str(resolved), "exists": exists}
    probe = resolved if exists else resolved.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        entry.update(total_gib=gib(usage.total), free_gib=gib(usage.free),
                     used_percent=round(usage.used / usage.total * 100, 1))
    except OSError as error:
        entry["error"] = str(error)
    return entry


def python_package_versions(names: tuple[str, ...]) -> dict:
    from importlib import metadata

    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def save(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {"generated_at": utc_now(), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def row(label: str, value: object, note: str = "") -> None:
    text = "—" if value is None else str(value)
    suffix = f"   # {note}" if note else ""
    print(f"{label:<28} {text}{suffix}")


def platform_summary() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "executable": sys.executable}
