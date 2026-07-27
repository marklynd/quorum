"""Stability tests. The point of the module is honesty about variance, so test that."""
import json

import pytest

from quorum.council import Council
from quorum.rubric import Rubric
from quorum.stability import run_repeated
from quorum.transport import Reply

RUBRIC = Rubric.from_dict({
    "name": "t",
    "dimensions": [
        {"name": "A", "max_points": 20, "bands": [{"lo": 0, "hi": 20, "label": "any"}]},
        {"name": "B", "max_points": 20, "bands": [{"lo": 0, "hi": 20, "label": "any"}]},
    ],
    "verdicts": [{"label": "Low", "ceiling": 19}, {"label": "Mid", "ceiling": 29},
                 {"label": "High", "ceiling": 40}],
})


def payload(a, b):
    return json.dumps({"scores": {"A": a, "B": b}, "reasoning": {"A": "x", "B": "y"},
                       "one_line": "l", "dissent": "d", "confidence": "medium", "missing": ""})


class ScriptedTransport:
    """Returns a different score on each successive call, to simulate real drift."""

    def __init__(self, sequences):
        self.sequences = {k: list(v) for k, v in sequences.items()}

    async def ask(self, model, system, user, max_tokens=2000, correction=None):
        seq = self.sequences[model]
        val = seq.pop(0) if len(seq) > 1 else seq[0]
        return Reply(model=model, text=payload(*val))


@pytest.mark.asyncio
async def test_stable_scores_are_reported_as_stable():
    t = ScriptedTransport({"m1": [(5, 5)], "m2": [(5, 5)], "m3": [(6, 6)]})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    rep = await run_repeated(c, "claim", "evidence " * 100, runs=3)

    assert rep.runs_valid == 3
    assert rep.run_range == 0
    assert rep.stability == "stable"
    assert rep.verdict_stable
    assert rep.publishable


@pytest.mark.asyncio
async def test_swinging_scores_are_flagged_and_not_publishable():
    """The real finding from production: same inputs, very different answers."""
    t = ScriptedTransport({
        "m1": [(2, 2), (18, 18), (10, 10)],
        "m2": [(3, 3), (17, 17), (9, 9)],
        "m3": [(2, 2), (19, 19), (11, 11)],
    })
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    rep = await run_repeated(c, "claim", "evidence " * 100, runs=3)

    assert rep.stability == "unstable"
    assert not rep.verdict_stable
    assert not rep.publishable, "an unstable score must not pass as a headline number"
    assert any("verdict changed" in n for n in rep.notes)
    assert any("moved" in n for n in rep.notes)


@pytest.mark.asyncio
async def test_names_the_dimension_doing_the_moving():
    t = ScriptedTransport({
        "m1": [(2, 10), (18, 10), (10, 10)],
        "m2": [(2, 10), (18, 10), (10, 10)],
        "m3": [(2, 10), (18, 10), (10, 10)],
    })
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    rep = await run_repeated(c, "claim", "evidence " * 100, runs=3)

    assert rep.dimension_volatility["A"] > rep.dimension_volatility["B"]
    assert rep.dimension_volatility["B"] == 0


@pytest.mark.asyncio
async def test_the_published_total_is_the_sum_of_its_parts():
    """A rubric total must decompose, or disputing one component means nothing.

    Median-of-run-totals and sum-of-dimension-medians are different statistics. Production hit a
    case where they differed by a point, which would have printed a headline score that the five
    components underneath it did not add up to. The sum is what gets published.
    """
    t = ScriptedTransport({
        "m1": [(4, 9), (5, 10), (6, 8)],
        "m2": [(5, 8), (6, 9), (4, 10)],
        "m3": [(6, 10), (4, 8), (5, 9)],
    })
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    rep = await run_repeated(c, "claim", "evidence " * 100, runs=3)

    assert rep.median_total == sum(rep.median_scores.values())
    assert rep.median_verdict == RUBRIC.verdict_for(rep.median_total)
    assert rep.median_of_run_totals is not None
