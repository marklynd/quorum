"""Rubric tests. The validator is a gate, so its strictness is the thing under test."""
import pytest

from quorum.rubric import Rubric

GOOD = {
    "name": "t",
    "dimensions": [
        {"name": "A", "max_points": 20,
         "bands": [{"lo": 0, "hi": 10, "label": "low"}, {"lo": 11, "hi": 20, "label": "high"}]},
        {"name": "B", "max_points": 20,
         "bands": [{"lo": 0, "hi": 20, "label": "any"}]},
    ],
    "verdicts": [{"label": "Fine", "ceiling": 20}, {"label": "Bad", "ceiling": 40}],
}


def test_loads_and_totals():
    r = Rubric.from_dict(GOOD)
    assert r.max_total == 40
    assert r.dimension_names == ["A", "B"]
    assert r.verdict_for(15) == "Fine"
    assert r.verdict_for(30) == "Bad"


def test_bands_must_cover_every_attainable_score():
    bad = {**GOOD, "dimensions": [
        {"name": "A", "max_points": 20, "bands": [{"lo": 0, "hi": 10, "label": "low"}]}]}
    with pytest.raises(ValueError, match="do not cover"):
        Rubric.from_dict(bad)


def test_duplicate_dimensions_rejected():
    bad = {**GOOD, "dimensions": GOOD["dimensions"] + [GOOD["dimensions"][0]]}
    with pytest.raises(ValueError, match="duplicate"):
        Rubric.from_dict(bad)


class TestValidateScores:
    def setup_method(self):
        self.r = Rubric.from_dict(GOOD)

    def test_accepts_valid(self):
        scores, why = self.r.validate_scores({"A": 5, "B": 7})
        assert scores == {"A": 5, "B": 7} and why == ""

    def test_missing_dimension_is_a_failure_not_a_default(self):
        """The important one.

        Several surveyed projects substitute a mid-range default for a missing dimension, which
        invents a number nobody produced. A gap must be reported, never filled.
        """
        scores, why = self.r.validate_scores({"A": 5})
        assert scores is None
        assert "missing dimension: B" in why

    def test_out_of_range_rejected(self):
        assert self.r.validate_scores({"A": 25, "B": 1})[0] is None
        assert self.r.validate_scores({"A": -1, "B": 1})[0] is None

    def test_non_numeric_rejected(self):
        assert self.r.validate_scores({"A": "high", "B": 1})[0] is None

    def test_unknown_dimension_rejected(self):
        assert self.r.validate_scores({"A": 1, "B": 2, "C": 3})[0] is None

    def test_float_is_rounded_not_refused(self):
        assert self.r.validate_scores({"A": 4.6, "B": 2.2})[0] == {"A": 5, "B": 2}


def test_shipped_example_rubric_is_valid():
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "hype-index.yaml"
    r = Rubric.load(path)
    assert r.max_total == 100
    assert len(r.dimensions) == 5
    assert r.verdict_for(53) == "Half True"
    assert r.verdict_for(100) == "Unsupported"
