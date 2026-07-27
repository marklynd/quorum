# Contributing

Pull requests welcome. Two things this project will not trade away:

1. **A failure never becomes a number.** No default score for a missing dimension, no silent
   fallback to the first model in the config, no synthesised value standing in for a member that
   did not answer. If you cannot report a real score, report the failure.
2. **Every network call is bounded by real elapsed time.** Not per-read, not per-chunk. If you add
   a call, wrap it in `asyncio.wait_for` and add a test that proves a hung response cannot hold up
   a run.

Run the tests with `pytest -q`. They use a fake transport, so they need no API key and cost
nothing.
