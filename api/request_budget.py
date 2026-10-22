"""Relative read-only request budgets and explicit disconnect ownership."""

from __future__ import annotations

import asyncio
import math
from typing import Awaitable, Callable, TypeVar

from fastapi import HTTPException, Request
from utils.deadline import RequestDeadline, deadline_scope


T = TypeVar("T")
REQUEST_TIMEOUT_HEADER = "X-Request-Timeout-Ms"


async def run_read_only_request(
    request: Request, local_seconds: float, job: Callable[[], Awaitable[T]],
) -> T:
    """A caller duration may only reduce the local bound, never extend it."""
    raw = request.headers.get(REQUEST_TIMEOUT_HEADER)
    seconds = local_seconds
    if raw is not None:
        try:
            if len(raw) > 32:
                raise ValueError
            duration = float(raw)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(400, f"{REQUEST_TIMEOUT_HEADER} must be finite and positive") from exc
        seconds = min(seconds, duration / 1000)
        if seconds <= 0:
            raise HTTPException(400, f"{REQUEST_TIMEOUT_HEADER} is too small")
    deadline = RequestDeadline(seconds)
    with deadline_scope(deadline):
        task = asyncio.create_task(job())

    async def watch_disconnect() -> None:
        while not task.done():
            if await request.is_disconnected():
                deadline.cancel()
                task.cancel()
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await task
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if not task.done():
            deadline.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
