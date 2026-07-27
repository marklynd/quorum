# quorum

**A council of language models that scores a claim against a rubric, using only evidence you supply.**

```bash
pip install quorum-council

quorum score \
  --claim "Superhero movies are dead." \
  --evidence sources.txt \
  --rubric examples/hype-index.yaml
```

```
The Hype Index: Superhero movies are dead.

  openai/gpt-5.6-luna-pro           64  Overstated   (round 1: 60)
  anthropic/claude-opus-4.8         36  Holds Up
  google/gemini-2.5-pro             41  Half True    (round 1: 33)
  perplexity/sonar-pro              38  Holds Up
  deepseek/deepseek-chat            46  Half True

  consensus 41 Half True   spread 36 to 64   agreement wide
  opinion   google/gemini-2.5-pro: a real post-2019 decline, applied to a whole genre
  dissent   openai/gpt-5.6-luna-pro at 64: the 2026 release calendar is not audience rejection
  note: members disagreed by 28 of 100 points. A consensus this wide is a finding, not a
        number to quote on its own.

  transcript .quorum/2026-07-26T21-04-11Z-superhero-movies-are-dead
```

---

## Why this exists

There are a lot of good LLM council projects. They ask several models the same open question,
have them rank each other's prose, and synthesise an answer. That produces a nice paragraph.

It does not produce anything you can audit six months later.

`quorum` is built for the case where you need a **measurement** instead of an opinion:

- **Fixed rubric.** Named dimensions, fixed point ranges, named bands. A score is not just a
  number, it is a claim that the evidence fell in a specific band, which a reader can check.
- **Evidence only.** Members are given the fetched text of your sources and told, in the system
  prompt and enforced at the API boundary, to score from that and report what is missing rather
  than filling gaps from memory. A score derived from a model's recollection is not auditable.
- **Round one is preserved forever.** Deliberation raises agreement whether or not it raises
  accuracy. The pre-deliberation positions are written to disk and never rewritten, so when the
  real outcome is known you can measure whether arguing made the council *more right* or merely
  *more unanimous*.
- **Failures stay visible.** No mid-range default for a missing dimension, no silent fallback to
  the first model in the config. If a member cannot produce a valid score, that is reported as a
  failure with its reason.

### Consensus is not accuracy

Models trained on overlapping corpora correlate, including on their mistakes. Five models handed
the same wrong summary produce five confident agreeing wrong answers, and the agreement makes the
error *more* persuasive, not less.

So `quorum` never reports a consensus without also reporting the spread, labels agreement as
`tight`, `workable` or `wide`, and adds an explicit note when the members disagreed badly. A wide
spread is not a bug in the run. It is the most useful thing the run found.

---

## The bug at the centre of this library

Every HTTP client's `timeout=` is a set of **per-operation** timeouts. In `httpx` there are four
(connect, read, write, pool), and `read` is the maximum gap *between chunks*, not total elapsed.
A model that keeps the socket fed, by streaming tokens slowly or emitting processing heartbeats,
never trips it. The request runs unbounded and the process never exits.

This is the single most common defect in the open-source council projects surveyed while building
this, including the most popular one. `quorum` bounds it in two places:

```python
async with httpx.AsyncClient(timeout=self.per_call_timeout) as client:
    resp = await asyncio.wait_for(                      # the real elapsed-time bound
        client.post(url, headers=headers, json=payload),
        timeout=self.per_call_timeout,
    )
```

and again around the whole round, so a straggler is cancelled and reported rather than waited on:

```python
done, pending = await asyncio.wait(tasks, timeout=total_deadline)
for task in pending:
    task.cancel()
```

`tests/test_transport.py::test_a_hung_coroutine_cannot_hold_up_the_run` is the regression test. It
puts a coroutine that sleeps for an hour next to two that return immediately and asserts the run
finishes in under two seconds.

---

## How a run works

**Round one, independent.** Every member scores the claim alone. No member sees another's answer.
A response that will not parse gets exactly one repair turn, with the failure named, before the
member is recorded as failed.

**Quorum check.** If fewer than `quorum` members returned a valid score, the run reports no
consensus at all. A number derived from two of five members is not a council result.

**Round two, deliberation.** Each member sees the others' positions, **anonymised and shuffled**,
and may revise or hold. Identities are hidden so members judge the argument rather than the brand;
order is shuffled on every call so position carries no information either. A revision must name
the specific evidence that changed it. Moving toward the group to reduce friction is explicitly
called out as unacceptable. A member that fails to respond keeps its round-one score, because
silence is not assent.

**Aggregation.** Consensus is the per-dimension median. The **opinion of the council** is written
by the member closest to that consensus, the **dissent** by the member furthest from it. Both are
chosen arithmetically, so no synthesis model gets to smooth the disagreement into a paragraph that
offends nobody.

**Transcript.** A dated directory with `round1.json`, `final.json`, `result.json` and a
`manifest.json` carrying a SHA-256 of each. `quorum verify` recomputes them, so a record altered
after the fact is detectable.

---

## Use it as a library

```python
import asyncio
from quorum import Council, Rubric, Transport, Transcript

rubric = Rubric.load("examples/hype-index.yaml")
council = Council(
    models=["openai/gpt-5.6-luna-pro", "anthropic/claude-opus-4.8",
            "google/gemini-2.5-pro", "perplexity/sonar-pro", "deepseek/deepseek-chat"],
    rubric=rubric,
    transport=Transport(per_call_timeout=120),
    quorum=3,
)

result = asyncio.run(council.run(claim="...", evidence=fetched_source_text))

print(result.consensus_total, result.consensus_verdict, result.agreement)
print(result.dissent["text"])

Transcript().save(result, models=council.models, rubric_name=rubric.name)
```

## Use it over MCP

```bash
pip install 'quorum-council[mcp]'
quorum serve --rubric examples/hype-index.yaml
```

Tools exposed: `score_claim`, `describe_rubric`, `verify_transcript`, `list_transcripts`,
`health`.

`score_claim` **requires** the caller to pass evidence and refuses anything under 200 characters.
An agent that wants a score has to show its sources first. That is enforced in the tool, not
suggested in a docstring.

## Write your own rubric

Rubrics are YAML, so changing one is data rather than a fork. See
[`examples/hype-index.yaml`](examples/hype-index.yaml).

```yaml
name: My rubric
direction: higher_is_worse
dimensions:
  - name: Evidence quality
    max_points: 20
    question: What is the artifact, really?
    bands:
      - { lo: 0,  hi: 9,  label: "Peer reviewed or official statistics." }
      - { lo: 10, hi: 20, label: "Marketing material or no identifiable source." }
verdicts:
  - { label: Sound,  ceiling: 9 }
  - { label: Shaky,  ceiling: 20 }
```

Bands must cover every attainable score. The loader refuses a rubric where they do not, because
otherwise a member can return a number the rubric cannot explain.

---

## Case study: The Hype Index

[The Hype Index](https://thehypeindex.com) is a newsletter that takes one loud public claim per
edition, traces it to its primary source, and scores it 0 to 100 across five components. `quorum`
is the scoring engine, and `examples/hype-index.yaml` is the production rubric.

The publication does three things with the output that the library is designed to support:

1. **Publishes the spread, not just the number.** Where the five models disagreed tells the reader
   which component is genuinely contested.
2. **Publishes the dissent verbatim.** The member furthest from consensus gets its objection
   printed, unedited.
3. **Grades the council later.** Every edition carries a dated, falsifiable prediction. When one
   resolves, the pre-deliberation transcript shows how each model scored it *before* anyone knew
   the answer. Over time that produces a public record of how well AI judgement performs on
   contested claims, which is a different and more interesting question than any single score.

Point three is why transcripts are hashed and why round one is never rewritten.

---

## Credits

This library learned from work that came before it, and the debts are specific.

- **[karpathy/llm-council](https://github.com/karpathy/llm-council)** established the pattern:
  parallel first opinions, peer review with identities hidden, then synthesis. `quorum`'s shape
  comes from there. No code was copied: that repository carries no licence.
- **[amiable-dev/llm-council](https://github.com/amiable-dev/llm-council)** (MIT) diagnosed the
  per-operation timeout problem and fixed it with `asyncio.wait_for`, documented with an ADR and a
  regression test. Independently implemented here, but they found it first and the approach is
  theirs.
- **[jason-chao/MAGI](https://github.com/jason-chao/MAGI)** (MIT) contributed two ideas: re-asking
  a model with the parse failure named rather than discarding it, and a quorum gate so a
  permanently broken member drops out instead of blocking the run.
- **[danielrosehill/Awesome-LLM-Council-Projects](https://github.com/danielrosehill/Awesome-LLM-Council-Projects)**
  made the survey possible.

What is different here: rubric-based numeric scoring rather than ranking prose; evidence supplied
by the caller and enforced; per-dimension spread as a first-class output; pre-deliberation
positions persisted as hash-verified evidence for later grading; and failure that stays visible
instead of becoming a default score.

## Licence

MIT. See [LICENSE](LICENSE).
