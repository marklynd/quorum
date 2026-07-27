"""The council: independent scoring, then deliberation, then a majority opinion and a dissent.

Design notes that are load-bearing
----------------------------------
**Consensus measures agreement, not accuracy.** Models trained on overlapping corpora
correlate, including on their mistakes. Five models handed the same wrong summary produce five
confident agreeing wrong answers, and the agreement makes the error *more* persuasive. So this
library never reports a consensus without also reporting the spread, and it keeps the
pre-deliberation scores forever so agreement and correctness can be measured separately once
the real outcome is known.

**Deliberation raises agreement whether or not it raises accuracy.** That is the reason round
one is preserved verbatim, revisions must name the evidence that moved them, and holding out is
explicitly legitimate.

**Peer identities are hidden and order is shuffled.** Otherwise members defer to a brand rather
than to an argument, and a fixed order leaks identity across runs.

**A failure is never a number.** No mid-range default, no filling a gap from a model's memory.
If a member cannot produce a valid score, that is recorded as a failure with its reason.
"""
from __future__ import annotations

import asyncio
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .rubric import Rubric
from .transport import Reply, Transport, extract_json, gather_bounded

SCORE_SYSTEM = """You are one of {n} independent members of a scoring council.

You will score ONE claim against a fixed rubric. You are scoring the CLAIM AS IT CIRCULATES,
not the quality of the research behind it. Strong research being misused scores badly.

{rubric}

Absolute rules:
- Score ONLY from the EVIDENCE provided below. It is the fetched text of the primary sources.
- If the evidence does not contain something you need, say so in "missing" and score
  conservatively. Do NOT fill the gap from your own knowledge. Your recollection may be wrong
  and it is not auditable.
- Never invent, estimate or round a figure. Quote figures only if they appear in the evidence.
- You are one of {n}. Do not guess what the others will say and do not split the difference.
  Your independent judgement is the whole value you add. Disagreement is a useful result.

Return ONLY a JSON object, no prose around it, with exactly these keys:
{{"scores": {{{score_keys}}},
  "reasoning": {{"<dimension>": "one sentence citing the evidence"}},
  "one_line": "one sentence on where this claim actually stands",
  "dissent": "the strongest argument AGAINST your own score",
  "confidence": "high|medium|low",
  "missing": "what you needed and did not find, or an empty string"}}"""

DELIBERATE_EXTRA = """

DELIBERATION. Below are the other members' scores and reasoning, anonymised and shuffled. You
do not know which model produced which, and that is deliberate: judge the argument, not the
author.

You may revise your scores or hold them. Both are respectable. What is NOT acceptable is moving
toward the group to reduce friction. Revise only if another member pointed at something in the
EVIDENCE you had missed or misread, and say exactly what it was.

Add two keys to your JSON:
  "revised": true or false
  "revision_reason": if revised, the specific evidence that changed your mind. If you held, why
    the others did not persuade you. Never write that you are aligning with the consensus. That
    is not a reason."""

REPAIR = ("Your previous response could not be parsed as JSON. Return ONLY the JSON object, "
          "with no markdown fence, no commentary, and every rubric dimension present.")


@dataclass
class MemberOpinion:
    """One member's position at one point in the process."""

    model: str
    scores: dict[str, int] | None = None
    total: int | None = None
    verdict: str = ""
    reasoning: dict[str, str] = field(default_factory=dict)
    one_line: str = ""
    dissent: str = ""
    confidence: str = ""
    missing: str = ""
    error: str | None = None
    elapsed: float = 0.0
    attempts: int = 1
    revised: bool | None = None
    revision_reason: str = ""
    round1_total: int | None = None
    round1_scores: dict[str, int] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.total is not None

    @property
    def moved(self) -> int | None:
        if self.round1_total is None or self.total is None:
            return None
        return self.total - self.round1_total


@dataclass
class CouncilResult:
    """Everything the council produced, including what went wrong."""

    claim: str
    rubric: str
    members_ok: int
    members_failed: int
    quorum: int
    quorum_met: bool
    consensus_scores: dict[str, int] = field(default_factory=dict)
    consensus_total: int | None = None
    consensus_verdict: str = ""
    spread: dict[str, dict[str, int]] = field(default_factory=dict)
    total_low: int | None = None
    total_high: int | None = None
    total_range: int | None = None
    verdicts_named: list[str] = field(default_factory=list)
    agreement: str = ""
    opinion_of_the_council: dict[str, Any] = field(default_factory=dict)
    dissent: dict[str, Any] = field(default_factory=dict)
    deliberation: dict[str, Any] = field(default_factory=dict)
    round1: list[dict] = field(default_factory=list)
    final: list[dict] = field(default_factory=list)
    evidence_chars: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Council:
    """Runs a rubric-scored council over one claim and one body of evidence.

    :param models: model identifiers, as OpenRouter names.
    :param rubric: a :class:`~quorum.rubric.Rubric`.
    :param transport: a configured :class:`~quorum.transport.Transport`.
    :param quorum: minimum valid members for the run to count. Defaults to a majority.
    :param deliberate: run the second round.
    :param run_deadline: overall wall-clock ceiling per round, in seconds.
    :param seed: shuffle seed, for reproducible anonymisation in tests.
    """

    def __init__(
        self,
        models: Sequence[str],
        rubric: Rubric,
        transport: Transport | None = None,
        quorum: int | None = None,
        deliberate: bool = True,
        run_deadline: float = 240.0,
        seed: int | None = None,
    ) -> None:
        if len(models) < 2:
            raise ValueError("a council needs at least two members")
        self.models = list(models)
        self.rubric = rubric
        self.transport = transport or Transport()
        self.quorum = quorum if quorum is not None else (len(models) // 2 + 1)
        self.deliberate = deliberate
        self.run_deadline = run_deadline
        self._rng = random.Random(seed)

    # ---------------------------------------------------------------- prompts
    def _system(self, deliberating: bool = False) -> str:
        keys = ", ".join(f'"{d.name}": <0-{d.max_points}>' for d in self.rubric.dimensions)
        base = SCORE_SYSTEM.format(n=len(self.models), rubric=self.rubric.prompt_block(),
                                   score_keys=keys)
        return base + (DELIBERATE_EXTRA if deliberating else "")

    def _user(self, claim: str, evidence: str, context: str = "") -> str:
        parts = [f"CLAIM AS IT CIRCULATES: {claim}"]
        if context:
            parts.append(f"HOW IT IS USED: {context}")
        parts.append("EVIDENCE FOLLOWS. This is everything you may score from.\n\n" + evidence)
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- one member
    async def _score_one(self, model: str, system: str, user: str) -> MemberOpinion:
        reply: Reply = await self.transport.ask(model, system, user)
        if not reply.ok:
            return MemberOpinion(model=model, error=reply.error or "empty response",
                                 elapsed=reply.elapsed, attempts=reply.attempts)

        obj = extract_json(reply.text)
        if obj is None or "scores" not in obj:
            # One repair turn. Borrowed from MAGI: re-ask with the failure named, rather than
            # discarding a member for a formatting problem. See CREDITS.
            repair = await self.transport.ask(
                model, system, user,
                correction={"previous": reply.text[:4000], "instruction": REPAIR})
            obj = extract_json(repair.text) if repair.ok else None
            if obj is None or "scores" not in obj:
                return MemberOpinion(
                    model=model,
                    error="unparseable after repair attempt: " + (reply.text[:120] or "empty"),
                    elapsed=reply.elapsed + repair.elapsed, attempts=reply.attempts + 1)

        scores, why = self.rubric.validate_scores(obj.get("scores"))
        if scores is None:
            return MemberOpinion(model=model, error=f"invalid scores: {why}",
                                 elapsed=reply.elapsed, attempts=reply.attempts)

        total = sum(scores.values())
        return MemberOpinion(
            model=model, scores=scores, total=total, verdict=self.rubric.verdict_for(total),
            reasoning={k: str(v) for k, v in (obj.get("reasoning") or {}).items()},
            one_line=str(obj.get("one_line", ""))[:600],
            dissent=str(obj.get("dissent", ""))[:800],
            confidence=str(obj.get("confidence", "")).lower()[:8],
            missing=str(obj.get("missing", ""))[:600],
            elapsed=reply.elapsed, attempts=reply.attempts,
            revised=bool(obj.get("revised")) if "revised" in obj else None,
            revision_reason=str(obj.get("revision_reason", ""))[:800])

    # ---------------------------------------------------------------- rounds
    async def _round(self, system: str, users: dict[str, str]) -> list[MemberOpinion]:
        coros = [self._score_one(m, system, users[m]) for m in self.models]
        results = await gather_bounded(coros, self.run_deadline)
        out: list[MemberOpinion] = []
        for model, res in zip(self.models, results, strict=False):
            if isinstance(res, MemberOpinion):
                out.append(res)
            elif isinstance(res, asyncio.TimeoutError):
                out.append(MemberOpinion(
                    model=model, error="did not answer within the round deadline"))
            else:
                out.append(MemberOpinion(
                    model=model, error=f"{type(res).__name__}: {str(res)[:120]}"))
        return out

    def _anon_panel(self, opinions: list[MemberOpinion], exclude: str) -> str:
        """The other members' positions, anonymised and shuffled.

        Identities are replaced with neutral letters and the order is randomised on every call,
        so nothing about position or label carries information about which model spoke.
        """
        others = [o for o in opinions if o.ok and o.model != exclude]
        self._rng.shuffle(others)
        blocks = []
        for i, o in enumerate(others):
            letter = chr(ord("A") + i)
            blocks.append(
                f"Member {letter} scored {o.total} of {self.rubric.max_total} ({o.verdict}).\n"
                f"  summary: {o.one_line}\n"
                f"  per dimension: {o.scores}\n"
                f"  their own strongest counter-argument: {o.dissent}")
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- public
    async def run(self, claim: str, evidence: str, context: str = "") -> CouncilResult:
        """Score one claim. Never raises on member failure; failures are reported."""
        base_user = self._user(claim, evidence, context)
        notes: list[str] = []

        r1 = await self._round(self._system(False), {m: base_user for m in self.models})
        good1 = [o for o in r1 if o.ok]

        if len(good1) < self.quorum:
            return CouncilResult(
                claim=claim, rubric=self.rubric.name, members_ok=len(good1),
                members_failed=len(r1) - len(good1), quorum=self.quorum, quorum_met=False,
                round1=[asdict(o) for o in r1], final=[asdict(o) for o in r1],
                evidence_chars=len(evidence),
                notes=[f"quorum not met: {len(good1)} of {self.quorum} required members "
                       "returned a valid score. No consensus is reported, deliberately."])

        final = r1
        if self.deliberate and len(good1) >= 2:
            users = {}
            for o in good1:
                users[o.model] = (
                    base_user
                    + "\n\n=== THE OTHER MEMBERS, ANONYMISED AND SHUFFLED ===\n"
                    + self._anon_panel(good1, exclude=o.model)
                    + f"\n\nYour own independent score was {o.total}. Reconsider, then return "
                      "the same JSON with the two extra keys.")
            r2 = await self._round(self._system(True), users) if users else []
            merged: dict[str, MemberOpinion] = {o.model: o for o in r1}
            for o2 in r2:
                prior = merged.get(o2.model)
                if prior is None or not prior.ok:
                    continue
                if o2.ok:
                    o2.round1_total, o2.round1_scores = prior.total, prior.scores
                    merged[o2.model] = o2
                else:
                    # Silence is not assent. A member that cannot deliberate keeps its opinion.
                    prior.revised = False
                    prior.revision_reason = f"no deliberation response ({o2.error}); round 1 stands"
                    prior.round1_total, prior.round1_scores = prior.total, prior.scores
                    notes.append(f"{prior.model} did not deliberate; round 1 score stands")
            final = [merged[m] for m in self.models if m in merged]

        return self._aggregate(claim, evidence, r1, final, notes)

    def _aggregate(self, claim: str, evidence: str, r1: list[MemberOpinion],
                   final: list[MemberOpinion], notes: list[str]) -> CouncilResult:
        good = [o for o in final if o.ok]
        totals = sorted(o.total for o in good)  # type: ignore[misc]
        per_dim = {d.name: sorted(o.scores[d.name] for o in good)  # type: ignore[index]
                   for d in self.rubric.dimensions}
        consensus = {k: round(statistics.median(v)) for k, v in per_dim.items()}
        total = sum(consensus.values())
        rng = totals[-1] - totals[0]

        # Who writes for the council and who dissents is decided arithmetically. No synthesis
        # model gets to smooth the disagreement into something that offends nobody.
        author = min(good, key=lambda o: abs(o.total - total))  # type: ignore[operator]
        outlier = max(good, key=lambda o: abs(o.total - total))  # type: ignore[operator]

        band = self.rubric.max_total or 100
        agreement = ("tight" if rng <= band * 0.10 else
                     "workable" if rng <= band * 0.20 else "wide")
        if agreement == "wide":
            notes.append(
                f"members disagreed by {rng} of {band} points. A consensus this wide is a "
                "finding, not a number to quote on its own.")

        moved = [o for o in good if o.moved not in (None, 0)]
        r1_totals = [o.total for o in r1 if o.ok]

        return CouncilResult(
            claim=claim, rubric=self.rubric.name,
            members_ok=len(good), members_failed=len(final) - len(good),
            quorum=self.quorum, quorum_met=True,
            consensus_scores=consensus, consensus_total=total,
            consensus_verdict=self.rubric.verdict_for(total),
            spread={k: {"low": v[0], "high": v[-1], "range": v[-1] - v[0]}
                    for k, v in per_dim.items()},
            total_low=totals[0], total_high=totals[-1], total_range=rng,
            verdicts_named=sorted({o.verdict for o in good}),
            agreement=agreement,
            opinion_of_the_council={"model": author.model, "total": author.total,
                                    "verdict": author.verdict, "text": author.one_line,
                                    "reasoning": author.reasoning},
            dissent={"model": outlier.model, "total": outlier.total, "verdict": outlier.verdict,
                     "text": outlier.dissent, "one_line": outlier.one_line,
                     "held_after_deliberation": outlier.revised is False,
                     "reason": outlier.revision_reason},
            deliberation={
                "ran": any(o.round1_total is not None for o in good),
                "moved": [{"model": o.model, "from": o.round1_total, "to": o.total,
                           "delta": o.moved, "reason": o.revision_reason} for o in moved],
                "held": [o.model for o in good if o.revised is False],
                "round1_spread": (max(r1_totals) - min(r1_totals)) if r1_totals else None,
                "round2_spread": rng,
            },
            round1=[asdict(o) for o in r1], final=[asdict(o) for o in final],
            evidence_chars=len(evidence), notes=notes)


def run_sync(council: Council, claim: str, evidence: str, context: str = "") -> CouncilResult:
    """Convenience wrapper for callers that are not async."""
    return asyncio.run(council.run(claim, evidence, context))
