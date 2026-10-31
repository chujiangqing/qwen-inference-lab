"""任务调度：保证同一时刻只有一个评测/一个服务变动。

单卡上并行评测会互相抢显存，数字全部作废（需求书 原则 5）。
这里用 O_EXCL 文件锁 + 存活校验实现"串行独占"，陈旧锁自动回收。
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / "experiments" / ".benchmark.lock"


class BusyError(RuntimeError):
    """已有任务在跑。"""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def current_holder() -> int | None:
    if not LOCK.is_file():
        return None
    try:
        pid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def is_busy() -> bool:
    return current_holder() is not None


@contextmanager
def exclusive(label: str = "task", wait_seconds: int = 0):
    """独占执行。wait_seconds=0 表示不等待，直接抛 BusyError。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0, wait_seconds)
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n{label}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            holder = current_holder()
            if holder is None:            # 陈旧锁：进程已死，回收
                LOCK.unlink(missing_ok=True)
                continue
            if time.time() >= deadline:
                raise BusyError(f"已有任务在运行（pid={holder}）；单卡评测必须串行")
            time.sleep(2)
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    print({"busy": is_busy(), "holder": current_holder(), "lock": str(LOCK)})
