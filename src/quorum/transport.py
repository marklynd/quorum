"""HTTP transport for council members, with a timeout that actually bounds elapsed time.

The bug this file exists to avoid
--------------------------------
Every HTTP client's ``timeout=`` is a set of PER-OPERATION timeouts. In httpx it is four of
them (connect, read, write, pool), and ``read`` is the maximum gap BETWEEN chunks, not the
total time. A model that keeps the socket fed, by streaming tokens slowly or emitting
processing heartbeats, never trips it. The request runs unbounded and the process never exits.

This is not hypothetical. It is the single most common defect in the open-source LLM council
projects surveyed while building this, including the most popular one. See CREDITS in the
README.

``asyncio.wait_for`` is the only thing here that bounds real elapsed time. Everything else is
belt and braces.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class Reply:
    """One model's answer, or an honest account of why there isn't one."""

    model: str
    text: str = ""
    error: str | None = None
    elapsed: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass
class Transport:
    """Calls models. Bounded, retryable, and honest about failure.

    :param api_key: OpenRouter key. Falls back to ``OPENROUTER_API_KEY``.
    :param per_call_timeout: hard wall-clock ceiling for one model call, in seconds.
    :param max_attempts: total attempts per call, including the first.
    :param referer: sent as ``HTTP-Referer``, which OpenRouter uses for attribution.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = OPENROUTER_URL
    per_call_timeout: float = 120.0
    max_attempts: int = 2
    temperature: float = 0.2
    referer: str = "https://github.com/marklynd/quorum"
    title: str = "quorum"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    async def ask(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2000,
        correction: str | None = None,
    ) -> Reply:
        """Ask one model once, bounded by ``per_call_timeout`` of real elapsed time.

        ``correction`` appends a repair turn to the conversation. It is how a model that
        returned unparseable output gets a second chance without the caller having to
        reconstruct the whole exchange.
        """
        if not self.api_key:
            return Reply(model=model, error="no API key")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if correction:
            messages.append({"role": "assistant", "content": correction["previous"]})
            messages.append({"role": "user", "content": correction["instruction"]})

        payload = {
            "model": model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        loop = asyncio.get_running_loop()
        started = loop.time()
        last_error = "not attempted"

        for attempt in range(1, self.max_attempts + 1):
            try:
                # The nested bound is deliberate. httpx's own timeout handles a dead socket
                # quickly; asyncio.wait_for handles a live socket that never finishes, which
                # is the failure mode that actually hangs a run.
                async with httpx.AsyncClient(timeout=self.per_call_timeout) as client:
                    resp = await asyncio.wait_for(
                        client.post(self.base_url, headers=self._headers(), json=payload),
                        timeout=self.per_call_timeout,
                    )
                if resp.status_code != 200:
                    last_error = f"http {resp.status_code}: {resp.text[:160]}"
                    retryable = resp.status_code in (429, 500, 502, 503, 504)
                    if retryable and attempt < self.max_attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        continue
                    break
                body: dict[str, Any] = resp.json()
                text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                return Reply(model=model, text=text,
                             elapsed=loop.time() - started, attempts=attempt)
            except asyncio.TimeoutError:
                last_error = f"exceeded {self.per_call_timeout:.0f}s wall clock"
                break  # a model that blew the deadline will blow it again
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue

        return Reply(model=model, error=last_error, elapsed=loop.time() - started,
                     attempts=self.max_attempts)


async def gather_bounded(coros: list, total_deadline: float) -> list:
    """Run everything concurrently under one overall deadline.

    Anything unfinished when the deadline passes is cancelled and returned as a
    ``TimeoutError`` in its slot, so the caller still gets a full-length list and can report
    which member failed to answer. A partial council is a result. A hung process is not.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    _done, pending = await asyncio.wait(tasks, timeout=total_deadline)
    for task in pending:
        task.cancel()
    out = []
    for task in tasks:
        if task in pending:
            out.append(asyncio.TimeoutError("run deadline exceeded"))
        else:
            try:
                out.append(task.result())
            except Exception as exc:
                out.append(exc)
    return out


def extract_json(text: str) -> dict | None:
    """Recover a JSON object from a model response.

    Handles the two ways models actually break this in production: wrapping the object in a
    ``` fence, and running out of output tokens partway through, which leaves braces unclosed.
    A greedy regex handles neither. This walks the string tracking depth while ignoring braces
    inside string literals, then, if the object was truncated, closes what is still open.
    Scores are emitted first in our schema, so a truncated response usually still yields them.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    body = t[start:end] if end > 0 else t[start:]
    candidates = [body]
    if depth > 0:
        trimmed = body.rstrip().rstrip(",")
        candidates.append(trimmed + ('"' if in_str else "") + "}" * depth)
        if "," in trimmed:
            candidates.append(trimmed.rsplit(",", 1)[0] + "}" * depth)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None
