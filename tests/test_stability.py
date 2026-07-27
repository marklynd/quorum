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
