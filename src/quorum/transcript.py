"""Dated, hash-verified transcripts.

A council run is evidence, not a cache. If you want to ask later whether the council was right,
you need what it said *before* the answer was known, in a form nobody could quietly edit
afterwards. That is the whole reason this module exists.

Layout::

    .quorum/2026-07-26T21-04-11Z-superhero-movies-are-dead/
        round1.json      each member's independent position, before deliberation
        final.json       positions after deliberation
        result.json      consensus, spread, opinion, dissent
        manifest.json    sha256 of each file, plus the models and rubric used

``verify()`` recomputes the hashes. A mismatch means the record was changed after the fact,
which for a published score is the thing you most want to be able to detect.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 1


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "run").lower()).strip("-")
    return (s[:limit].rstrip("-")) or "run"


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: Any) -> str:
    """Write JSON atomically and return its hash.

    Atomic because a half-written transcript that still parses is worse than none: it looks
    like a record and is not one.
    """
    body = json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return _sha256(body)


@dataclass
class Transcript:
    """Reads and writes council transcripts under a root directory."""

    root: Path = Path(".quorum")

    def save(self, result: Any, *, models: list[str], rubric_name: str,
             label: str | None = None, extra: dict | None = None) -> Path:
        """Persist one run. Returns the run directory."""
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        run_dir = self.root / f"{stamp}-{_slug(label or data.get('claim', ''))}"
        run_dir.mkdir(parents=True, exist_ok=True)

        round1 = data.pop("round1", [])
        final = data.pop("final", [])

        hashes = {
            "round1.json": _atomic_write(run_dir / "round1.json", round1),
            "final.json": _atomic_write(run_dir / "final.json", final),
            "result.json": _atomic_write(run_dir / "result.json", data),
        }
        manifest = {
            "schema": SCHEMA,
            "written_utc": stamp,
            "claim": data.get("claim", ""),
            "rubric": rubric_name,
            "models": models,
            "consensus_total": data.get("consensus_total"),
            "consensus_verdict": data.get("consensus_verdict"),
            "quorum_met": data.get("quorum_met"),
            "sha256": hashes,
            "note": ("round1.json is the council's position before deliberation. It is the "
                     "evidence for grading the council against real outcomes later, and it is "
                     "not rewritten by subsequent runs."),
        }
        if extra:
            manifest["extra"] = extra
        _atomic_write(run_dir / "manifest.json", manifest)
        return run_dir

    def verify(self, run_dir: str | Path) -> tuple[bool, list[str]]:
        """Recompute hashes. Returns ``(ok, problems)``."""
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return False, ["no manifest.json"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = []
        for name, expected in (manifest.get("sha256") or {}).items():
            path = run_dir / name
            if not path.exists():
                problems.append(f"{name} is missing")
                continue
            actual = _sha256(path.read_text(encoding="utf-8"))
            if actual != expected:
                problems.append(f"{name} has been modified since it was written")
        return (not problems), problems

    def runs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.iterdir() if (p / "manifest.json").exists())

    def load(self, run_dir: str | Path) -> dict:
        run_dir = Path(run_dir)
        out: dict[str, Any] = {}
        for name in ("manifest", "result", "round1", "final"):
            path = run_dir / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        return out
