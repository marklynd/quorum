"""Repeat runs, because one council run is a sample of size one.

Why this module exists
----------------------
Scoring the same claim, from the same evidence, against the same rubric, three times, produced
consensus totals that differed by more than twenty points. Members swung across two verdict
bands. Nothing about the inputs changed.

That is not a bug to patch away. It is the honest variance of the method, and a publication
that prints a single number without it is doing exactly what this kind of tool is supposed to
catch other people doing.

So: run the council N times and publish the median with the observed spread. A claim whose
score is stable across runs and a claim whose score swings twenty points are different objects
and the reader deserves to know which one they are looking at.

Cost note. N runs cost N times as much. Three is usually enough to distinguish stable from
unstable; five is better if the decision matters.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from .council import Council, CouncilResult


@dataclass
class StabilityReport:
    """The council's answer, plus how much that answer moves when you ask again."""

    claim: str
    rubric: str
    runs_requested: int
    runs_valid: int
    median_total: int | None = None
    median_verdict: str = ""
    median_scores: dict[str, int] = field(default_factory=dict)
    run_totals: list[int] = field(default_factory=list)
    run_verdicts: list[str] = field(default_factory=list)
    run_range: int | None = None
    run_stdev: float | None = None
    stability: str = ""
    verdict_stable: bool = False
    dimension_volatility: dict[str, int] = field(default_factory=dict)
    dissents: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def publishable(self) -> bool:
        """Whether this is safe to print as a headline number.

        Deliberately conservative. An unstable score can still be published, but it must be
        published *as* unstable, with the range shown. This flag is what a caller checks before
        printing a bare number.
        """
        return self.runs_valid >= 2 and self.verdict_stable and self.stability != "unstable"


async def run_repeated(council: Council, claim: str, evidence: str, context: str = "",
                       runs: int = 3, on_run=None) -> StabilityReport:
    """Run the council ``runs`` times and report the median with its variance.

    Runs are sequential rather than concurrent. Firing fifteen simultaneous calls at one
    provider invites rate limiting, and a rate-limited run is not an independent sample of
    anything.

    :param on_run: optional callback ``(index, total, CouncilResult)`` invoked after each run.
        Repeated runs take minutes. A caller that cannot see progress cannot tell a slow run
        from a hung one, which is exactly the confusion this library exists to remove, so the
        hook is here rather than left to the caller to invent.
    """
    results: list[CouncilResult] = []
    for i in range(runs):
        result = await council.run(claim, evidence, context)
        results.append(result)
        if on_run is not None:
            try:
                on_run(i + 1, runs, result)
            except Exception:  # noqa: BLE001 - a broken callback must not lose the run
                pass

    valid = [r for r in results if r.quorum_met and r.consensus_total is not None]
    report = StabilityReport(
        claim=claim, rubric=council.rubric.name,
        runs_requested=runs, runs_valid=len(valid),
        results=[r.to_dict() for r in results])

    if not valid:
        report.notes.append("no run reached quorum. Nothing is reported, deliberately.")
        return report

    totals = sorted(r.consensus_total for r in valid)  # type: ignore[misc]
    report.run_totals = [r.consensus_total for r in valid]  # type: ignore[misc]
    report.run_verdicts = [r.consensus_verdict for r in valid]
    report.median_total = round(statistics.median(totals))
    report.median_verdict = council.rubric.verdict_for(report.median_total)
    report.run_range = totals[-1] - totals[0]
    report.run_stdev = round(statistics.stdev(totals), 1) if len(totals) > 1 else 0.0

    report.median_scores = {
        d.name: round(statistics.median([r.consensus_scores[d.name] for r in valid]))
        for d in council.rubric.dimensions if all(d.name in r.consensus_scores for r in valid)}

    # Which dimension is doing the moving. Usually one or two carry most of the instability,
    # and naming them is more useful to a reader than a single overall figure.
    report.dimension_volatility = {
        d.name: max(r.consensus_scores[d.name] for r in valid)
                - min(r.consensus_scores[d.name] for r in valid)
        for d in council.rubric.dimensions if all(d.name in r.consensus_scores for r in valid)}

    report.verdict_stable = len(set(report.run_verdicts)) == 1
    band = council.rubric.max_total or 100
    rng = report.run_range or 0
    report.stability = ("stable" if rng <= band * 0.05 else
                        "usable" if rng <= band * 0.12 else "unstable")

    report.dissents = [
        {"run": i + 1, "model": r.dissent.get("model"), "total": r.dissent.get("total"),
         "text": r.dissent.get("text")} for i, r in enumerate(valid)]

    if not report.verdict_stable:
        report.notes.append(
            "the verdict changed between runs: " + ", ".join(sorted(set(report.run_verdicts)))
            + ". Publish the range, not the median alone.")
    if report.stability == "unstable":
        worst = max(report.dimension_volatility.items(), key=lambda kv: kv[1], default=None)
        note = f"consensus moved {rng} of {band} points across identical runs."
        if worst and worst[1]:
            note += f" Most of it is in {worst[0]}, which moved {worst[1]}."
        report.notes.append(note)
    if len(valid) < runs:
        report.notes.append(f"{runs - len(valid)} of {runs} runs failed to reach quorum")
    return report
