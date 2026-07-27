"""Council tests with a fake transport, so no network and no spend."""
import json

import pytest

from quorum.council import Council
from quorum.rubric import Rubric
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


class FakeTransport:
    """Returns scripted replies keyed by model."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def ask(self, model, system, user, max_tokens=2000, correction=None):
        self.calls.append((model, bool(correction)))
        item = self.script.get(model)
        if isinstance(item, list):
            item = item.pop(0) if item else ""
        if item is None:
            return Reply(model=model, error="simulated failure")
        return Reply(model=model, text=item)


def payload(a, b, **extra):
    return json.dumps({"scores": {"A": a, "B": b}, "reasoning": {"A": "x", "B": "y"},
                       "one_line": "a line", "dissent": "a counter", "confidence": "medium",
                       "missing": "", **extra})


@pytest.mark.asyncio
async def test_consensus_spread_and_dissent():
    t = FakeTransport({"m1": payload(5, 5), "m2": payload(10, 10), "m3": payload(6, 6)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False, seed=1)
    r = await c.run("claim", "evidence " * 100)

    assert r.quorum_met
    assert r.consensus_total == 12          # medians: A=6, B=6
    assert (r.total_low, r.total_high) == (10, 20)
    assert r.dissent["model"] == "m2"       # furthest from consensus
    assert r.opinion_of_the_council["model"] == "m3"   # closest to it


@pytest.mark.asyncio
async def test_a_failed_member_never_becomes_a_number():
    t = FakeTransport({"m1": payload(5, 5), "m2": None, "m3": payload(7, 7)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    r = await c.run("claim", "evidence " * 100)

    assert r.members_ok == 2 and r.members_failed == 1
    failed = next(m for m in r.final if m["model"] == "m2")
    assert failed["total"] is None and failed["scores"] is None
    assert "simulated failure" in failed["error"]


@pytest.mark.asyncio
async def test_quorum_not_met_reports_no_consensus():
    t = FakeTransport({"m1": payload(5, 5), "m2": None, "m3": None})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    r = await c.run("claim", "evidence " * 100)

    assert not r.quorum_met
    assert r.consensus_total is None
    assert "quorum not met" in r.notes[0]


@pytest.mark.asyncio
async def test_unparseable_response_triggers_one_repair_attempt():
    t = FakeTransport({"m1": ["not json at all", payload(4, 4)],
                       "m2": payload(6, 6), "m3": payload(6, 6)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    r = await c.run("claim", "evidence " * 100)

    assert r.members_ok == 3
    assert ("m1", True) in t.calls, "the repair turn was never sent"


@pytest.mark.asyncio
async def test_invalid_scores_are_rejected_not_coerced():
    t = FakeTransport({"m1": json.dumps({"scores": {"A": 5}}),      # missing B
                       "m2": payload(6, 6), "m3": payload(6, 6)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    r = await c.run("claim", "evidence " * 100)

    m1 = next(m for m in r.final if m["model"] == "m1")
    assert m1["total"] is None
    assert "missing dimension" in m1["error"]


@pytest.mark.asyncio
async def test_deliberation_preserves_round_one():
    t = FakeTransport({
        "m1": [payload(5, 5), payload(8, 8, revised=True, revision_reason="saw table 3")],
        "m2": [payload(10, 10), payload(10, 10, revised=False, revision_reason="held")],
        "m3": [payload(6, 6), payload(7, 7, revised=True, revision_reason="recount")],
    })
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=True, seed=1)
    r = await c.run("claim", "evidence " * 100)

    m1 = next(m for m in r.final if m["model"] == "m1")
    assert m1["round1_total"] == 10 and m1["total"] == 16
    assert r.deliberation["ran"] is True
    assert "m2" in r.deliberation["held"]
    assert r.round1[0]["total"] == 10, "round 1 must survive deliberation untouched"


@pytest.mark.asyncio
async def test_wide_disagreement_is_flagged_in_the_notes():
    t = FakeTransport({"m1": payload(1, 1), "m2": payload(19, 19), "m3": payload(10, 10)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, deliberate=False)
    r = await c.run("claim", "evidence " * 100)

    assert r.agreement == "wide"
    assert any("disagreed by" in n for n in r.notes)


@pytest.mark.asyncio
async def test_peer_panel_is_anonymised():
    """Members must not see who said what. Judge the argument, not the brand."""
    t = FakeTransport({"m1": payload(5, 5), "m2": payload(9, 9), "m3": payload(7, 7)})
    c = Council(["m1", "m2", "m3"], RUBRIC, transport=t, seed=1, deliberate=False)
    await c.run("claim", "evidence " * 100)
    from quorum.council import MemberOpinion
    ops = [MemberOpinion(model="anthropic/claude-x", scores={"A": 5, "B": 5}, total=10,
                         verdict="Low", one_line="l", dissent="d"),
           MemberOpinion(model="openai/gpt-y", scores={"A": 9, "B": 9}, total=18,
                         verdict="Low", one_line="l", dissent="d")]
    panel = c._anon_panel(ops, exclude="none")
    assert "claude" not in panel.lower()
    assert "gpt" not in panel.lower()
    assert "Member A" in panel
