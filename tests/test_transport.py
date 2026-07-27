"""Transport tests, with the hang regression front and centre."""
import asyncio
import time

import pytest

from quorum.transport import Reply, Transport, extract_json, gather_bounded


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_it(self):
        assert extract_json('Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_truncated_object_still_yields_leading_keys(self):
        """A model that runs out of tokens mid-object must not cost us the whole response.

        Scores are emitted first in the schema precisely so this recovers them.
        """
        truncated = '{"scores": {"A": 3, "B": 4}, "reasoning": {"A": "because it wa'
        got = extract_json(truncated)
        assert got is not None
        assert got["scores"] == {"A": 3, "B": 4}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"note": "a } inside a string", "n": 2}')["n"] == 2

    def test_garbage(self):
        assert extract_json("no json here") is None
        assert extract_json("") is None


class TestBoundedGather:
    @pytest.mark.asyncio
    async def test_a_hung_coroutine_cannot_hold_up_the_run(self):
        """THE regression test.

        A member that never returns is the failure that hangs naive implementations, including
        the most popular open-source council. The run must finish on the deadline and report
        the straggler as a timeout, not wait for it.
        """
        async def quick():
            return "done"

        async def never():
            await asyncio.sleep(3600)
            return "never"

        started = time.monotonic()
        results = await gather_bounded([quick(), never(), quick()], total_deadline=0.5)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, "the deadline did not bound the run"
        assert results[0] == "done"
        assert isinstance(results[1], asyncio.TimeoutError)
        assert results[2] == "done"

    @pytest.mark.asyncio
    async def test_returns_one_slot_per_input(self):
        async def ok():
            return 1

        async def boom():
            raise ValueError("nope")

        results = await gather_bounded([ok(), boom()], total_deadline=5)
        assert len(results) == 2
        assert results[0] == 1
        assert isinstance(results[1], ValueError)


class TestTransport:
    @pytest.mark.asyncio
    async def test_missing_key_is_reported_not_raised(self):
        reply = await Transport(api_key="").ask("some/model", "sys", "user")
        assert not reply.ok
        assert "no API key" in reply.error

    def test_reply_ok_requires_text(self):
        assert not Reply(model="m", text="   ").ok
        assert Reply(model="m", text="hi").ok
        assert not Reply(model="m", text="hi", error="boom").ok
