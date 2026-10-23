"""Synchronous provider facade with cancellable total HTTP I/O deadlines."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import requests


class BoundedSummarySession:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    def post(self, url: str, *, json: Any, headers: dict[str, str], timeout: float):
        async def request():
            # Cancellation closes the actual async stream; no detached request
            # keeps running after its probe's timeout+1 ownership lease expires.
            async with (
                httpx.AsyncClient(transport=self.transport, timeout=timeout) as client,
                client.stream("POST", url, json=json, headers=headers) as response,
            ):
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise requests.RequestException(
                            "summary response exceeded its byte budget"
                        )
                # aiter_bytes has already removed compression/framing. Keep
                # status/retry metadata without decoding or framing this body twice.
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower()
                    not in {"content-encoding", "content-length", "transfer-encoding"}
                }
                return httpx.Response(
                    response.status_code, headers=response_headers, content=bytes(body)
                )

        async def bounded_request():
            return await asyncio.wait_for(request(), timeout=timeout)

        try:
            return asyncio.run(bounded_request())
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            raise requests.RequestException(
                "summary transport deadline or I/O failure"
            ) from exc

    def close(self):
        pass  # Each request exits and closes its actual client/stream synchronously.
