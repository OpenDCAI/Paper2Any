from __future__ import annotations

import asyncio
import fcntl
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from dataflow_agent.utils import get_project_root

PROJECT_ROOT = get_project_root()
LOCK_ROOT = (PROJECT_ROOT / "outputs" / ".locks").resolve()


@dataclass(frozen=True)
class _HeldLockSlot:
    fd: int


class AsyncInterProcessSemaphore:
    """A small file-lock based semaphore that works across uvicorn workers."""

    def __init__(self, name: str, limit: int = 1, poll_interval: float = 0.2) -> None:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._")
        self._name = safe_name or "lock"
        self._limit = max(1, int(limit))
        self._poll_interval = max(0.05, float(poll_interval))

    def _slot_path(self, index: int) -> Path:
        return LOCK_ROOT / f"{self._name}.{index}.lock"

    def _try_acquire_once(self) -> _HeldLockSlot | None:
        LOCK_ROOT.mkdir(parents=True, exist_ok=True)
        for index in range(self._limit):
            fd = os.open(self._slot_path(index), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue

            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} slot={index}\n".encode("utf-8"))
            return _HeldLockSlot(fd=fd)
        return None

    async def acquire(self) -> _HeldLockSlot:
        while True:
            held = self._try_acquire_once()
            if held is not None:
                return held
            await asyncio.sleep(self._poll_interval)

    @staticmethod
    def release(held: _HeldLockSlot) -> None:
        try:
            fcntl.flock(held.fd, fcntl.LOCK_UN)
        finally:
            os.close(held.fd)

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        held = await self.acquire()
        try:
            yield
        finally:
            self.release(held)
