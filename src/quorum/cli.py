"""Command line interface. ``quorum score`` and ``quorum serve``."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .council import Council
from .rubric import Rubric
from .transcript import Transcript
from .transport import Transport

DEFAULT_MODELS = [
    "openai/gpt-5.6-luna-pro",
    "anthropic/claude-opus-4.8",
    "x-ai/grok-4.5",
    "perplexity/sonar-pro",
    "deepseek/deepseek-chat",
]


def _score(args: argparse.Namespace) -> int:
    rubric = Rubric.load(args.rubric)
    evidence = Path(args.evidence).read_text(encoding="utf-8")
    if len(evidence.strip()) < 200:
        print("evidence file is too short. This council scores documents, not recollections.",
              file=sys.stderr)
        return 2
    models = args.model or DEFAULT_MODELS
    council = Council(models, rubric,
                      transport=Transport(per_call_timeout=args.timeout),
                      deliberate=not args.no_deliberate,
                      run_deadline=args.deadline, quorum=args.quorum)
    result = asyncio.run(council.run(args.claim, evidence, args.context or ""))

    if not result.quorum_met:
        print(f"QUORUM NOT MET: {result.members_ok} of {result.quorum} required members "
              "returned a valid score. No consensus reported.", file=sys.stderr)
        for m in result.final:
            if m["error"]:
                print(f"  {m['model']}: {m['error']}", file=sys.stderr)
        return 1

    print(f"\n{rubric.name}: {args.claim}\n")
    for m in result.final:
        if m["total"] is None:
            print(f"  {m['model']:<32} FAILED  {m['error']}")
            continue
        delta = "" if m.get("round1_total") in (None, m["total"]) \
            else f"  (round 1: {m['round1_total']})"
        print(f"  {m['model']:<32} {m['total']:>3}  {m['verdict']}{delta}")
    print(f"\n  consensus {result.consensus_total} {result.consensus_verdict}"
          f"   spread {result.total_low} to {result.total_high}"
          f"   agreement {result.agreement}")
    print(f"  opinion   {result.opinion_of_the_council['model']}: "
          f"{result.opinion_of_the_council['text'][:110]}")
    print(f"  dissent   {result.dissent['model']} at {result.dissent['total']}: "
          f"{result.dissent['text'][:110]}")
    for n in result.notes:
        print(f"  note: {n}")

    if not args.no_save:
        path = Transcript(root=Path(args.transcripts)).save(
            result, models=council.models, rubric_name=rubric.name, label=args.claim)
        print(f"\n  transcript {path}")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from . import mcp_server
    mcp_server.configure(rubric_path=args.rubric, models=args.model or None,
                         transcript_root=args.transcripts)
    mcp_server.main()
    return 0


def _verify(args: argparse.Namespace) -> int:
    ok, problems = Transcript().verify(args.run_dir)
    print("intact" if ok else "ALTERED")
    for p in problems:
        print("  " + p)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="quorum", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one claim with the council")
    s.add_argument("--claim", required=True)
    s.add_argument("--evidence", required=True, help="file containing the source text")
    s.add_argument("--rubric", required=True, help="rubric YAML")
    s.add_argument("--context", default="")
    s.add_argument("--model", action="append", help="repeat for each member")
    s.add_argument("--quorum", type=int, default=None)
    s.add_argument("--timeout", type=float, default=120.0, help="per member, seconds")
    s.add_argument("--deadline", type=float, default=240.0, help="per round, seconds")
    s.add_argument("--no-deliberate", action="store_true")
    s.add_argument("--no-save", action="store_true")
    s.add_argument("--transcripts", default=".quorum")
    s.add_argument("--json", help="also write the full result to this file")
    s.set_defaults(func=_score)

    v = sub.add_parser("verify", help="check a saved transcript has not been altered")
    v.add_argument("run_dir")
    v.set_defaults(func=_verify)

    m = sub.add_parser("serve", help="run the MCP server")
    m.add_argument("--rubric")
    m.add_argument("--model", action="append")
    m.add_argument("--transcripts", default=".quorum")
    m.set_defaults(func=_serve)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
