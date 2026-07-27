"""Rubrics: the thing that makes this a scoring council rather than an opinion poll.

Most council projects ask several models the same open question and rank the prose that comes
back. That produces a nice answer and nothing you can audit later.

A rubric turns the same machinery into a measurement. Each dimension has a fixed point range
and named bands, so a score is not just a number, it is a claim that the evidence fell in a
specific band. That is checkable by a reader, comparable across runs, and gradeable against
reality once the outcome is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Band:
    """One named range within a dimension. ``lo`` and ``hi`` are inclusive."""

    lo: int
    hi: int
    label: str

    def contains(self, score: int) -> bool:
        return self.lo <= score <= self.hi


@dataclass
class Dimension:
    """One axis of judgement."""

    name: str
    max_points: int
    question: str = ""
    bands: list[Band] = field(default_factory=list)

    def band_for(self, score: int) -> Band | None:
        return next((b for b in self.bands if b.contains(score)), None)


@dataclass
class Verdict:
    """A named range on the total, so the headline is derived and never hand-typed."""

    label: str
    ceiling: int
    meaning: str = ""


@dataclass
class Rubric:
    """A complete scoring scheme, loadable from YAML so it is data rather than code.

    Every other rubric-scoring council found in the survey hard-coded its dimension names in
    several files, which meant changing the rubric required a fork. This does not.
    """

    name: str
    dimensions: list[Dimension]
    verdicts: list[Verdict] = field(default_factory=list)
    direction: str = "higher_is_worse"
    description: str = ""
    scale_note: str = ""

    @property
    def max_total(self) -> int:
        return sum(d.max_points for d in self.dimensions)

    @property
    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    def verdict_for(self, total: int) -> str:
        for v in sorted(self.verdicts, key=lambda x: x.ceiling):
            if total <= v.ceiling:
                return v.label
        return self.verdicts[-1].label if self.verdicts else ""

    def validate_scores(self, scores: dict[str, Any]) -> tuple[dict[str, int] | None, str]:
        """Coerce and check a model's scores. Returns ``(scores, "")`` or ``(None, reason)``.

        This is a hard gate, and it is deliberately unforgiving. Several surveyed projects fall
        back to a mid-range default for a missing dimension, which quietly manufactures a
        number nobody produced. A score that cannot be validated is a failure to be reported,
        never a value to be invented.
        """
        if not isinstance(scores, dict):
            return None, "scores is not an object"
        out: dict[str, int] = {}
        for d in self.dimensions:
            if d.name not in scores:
                return None, f"missing dimension: {d.name}"
            raw = scores[d.name]
            try:
                val = round(float(raw))
            except (TypeError, ValueError):
                return None, f"{d.name} is not a number: {raw!r}"
            if not 0 <= val <= d.max_points:
                return None, f"{d.name} out of range 0-{d.max_points}: {val}"
            out[d.name] = val
        extra = set(scores) - set(self.dimension_names)
        if extra:
            return None, f"unknown dimensions: {sorted(extra)}"
        return out, ""

    def prompt_block(self) -> str:
        """The rubric as the members see it."""
        lines = [f"RUBRIC: {self.name}"]
        if self.description:
            lines.append(self.description)
        if self.scale_note:
            lines.append(self.scale_note)
        lines.append("")
        for d in self.dimensions:
            lines.append(f"{d.name} (0 to {d.max_points}). {d.question}".rstrip())
            for b in d.bands:
                lines.append(f"    {b.lo}-{b.hi}: {b.label}")
        if self.verdicts:
            lines.append("")
            lines.append(f"Total is the sum, 0 to {self.max_total}. Verdict bands:")
            low = 0
            for v in sorted(self.verdicts, key=lambda x: x.ceiling):
                lines.append(f"    {low}-{v.ceiling}: {v.label}"
                             + (f" — {v.meaning}" if v.meaning else ""))
                low = v.ceiling + 1
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> Rubric:
        dims = []
        for d in data["dimensions"]:
            bands = [Band(int(b["lo"]), int(b["hi"]), b["label"]) for b in d.get("bands", [])]
            dims.append(Dimension(name=d["name"], max_points=int(d.get("max_points", 20)),
                                  question=d.get("question", ""), bands=bands))
        verdicts = [Verdict(v["label"], int(v["ceiling"]), v.get("meaning", ""))
                    for v in data.get("verdicts", [])]
        r = cls(name=data.get("name", "rubric"), dimensions=dims, verdicts=verdicts,
                direction=data.get("direction", "higher_is_worse"),
                description=data.get("description", ""),
                scale_note=data.get("scale_note", ""))
        r.check()
        return r

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def check(self) -> None:
        """Fail loudly at load time rather than subtly at scoring time."""
        if not self.dimensions:
            raise ValueError("rubric has no dimensions")
        names = self.dimension_names
        if len(set(names)) != len(names):
            raise ValueError("duplicate dimension names")
        for d in self.dimensions:
            if d.max_points <= 0:
                raise ValueError(f"{d.name}: max_points must be positive")
            for b in d.bands:
                if b.lo > b.hi:
                    raise ValueError(f"{d.name}: band {b.label} has lo > hi")
                if b.hi > d.max_points:
                    raise ValueError(f"{d.name}: band {b.label} exceeds max_points")
            if d.bands:
                covered = set()
                for b in d.bands:
                    covered |= set(range(b.lo, b.hi + 1))
                missing = set(range(0, d.max_points + 1)) - covered
                if missing:
                    raise ValueError(
                        f"{d.name}: bands do not cover {sorted(missing)[:5]}. "
                        "Every attainable score must fall in a named band, or a member can "
                        "return a number the rubric cannot explain.")
        if self.verdicts:
            ceilings = [v.ceiling for v in self.verdicts]
            if max(ceilings) < self.max_total:
                raise ValueError("top verdict ceiling is below the maximum total")
