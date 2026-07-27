"""MCP server: expose the council to any agent that speaks Model Context Protocol.

Run it::

    quorum serve --rubric examples/hype-index.yaml

Or point an MCP client at ``python -m quorum.mcp_server``.

Design choice worth stating: ``score_claim`` requires the caller to pass the evidence. It will
not fetch, and it will not let a model answer from memory. An agent that wants a score has to
show its sources first. That is the whole point of the tool and it is enforced here rather than
suggested in a docstring.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .council import Council
from .rubric import Rubric
from .transcript import Transcript
from .transport import Transport

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The MCP extra is not installed. Install it with:\n"
        "    pip install 'quorum-council[mcp]'") from exc

DEFAULT_MODELS = [
    "openai/gpt-5.6-luna-pro",
    "anthropic/claude-opus-4.8",
    "google/gemini-2.5-pro",
    "perplexity/sonar-pro",
    "deepseek/deepseek-chat",
]

mcp = FastMCP("quorum")
_state: dict[str, Any] = {"rubric": None, "models": DEFAULT_MODELS, "transcripts": Transcript()}


def configure(rubric_path: str | None = None, models: list[str] | None = None,
              transcript_root: str | None = None) -> None:
    if rubric_path:
        _state["rubric"] = Rubric.load(rubric_path)
    if models:
        _state["models"] = models
    if transcript_root:
        _state["transcripts"] = Transcript(root=Path(transcript_root))


@mcp.tool()
async def score_claim(
    claim: str,
    evidence: str,
    context: str = "",
    rubric_yaml: str = "",
    models: list[str] | None = None,
    deliberate: bool = True,
    save: bool = True,
) -> str:
    """Score one claim with a council of models, against a rubric, using only the evidence given.

    :param claim: the claim exactly as it circulates in public.
    :param evidence: the fetched text of the primary sources. Required. The council is
        instructed to score only from this and to report what is missing rather than filling
        gaps from memory.
    :param context: optional note on how the claim is being used.
    :param rubric_yaml: inline rubric YAML. Falls back to the server's configured rubric.
    :param models: override the council membership for this call.
    :param deliberate: run the second round, where members see each other anonymised.
    :param save: write a dated, hash-verified transcript.
    :returns: JSON with the consensus, the spread, the opinion of the council, and the dissent.
    """
    if not evidence or len(evidence.strip()) < 200:
        return json.dumps({
            "error": "evidence is required and must be substantial",
            "why": ("This council scores documents, not recollections. Fetch the primary "
                    "sources and pass their text. A score produced from model memory is not "
                    "auditable and this tool will not produce one.")})

    rubric = Rubric.from_dict(yaml.safe_load(rubric_yaml)) if rubric_yaml \
        else _state["rubric"]
    if rubric is None:
        return json.dumps({"error": "no rubric configured. Pass rubric_yaml or start the "
                                    "server with --rubric."})

    council = Council(models or _state["models"], rubric,
                      transport=Transport(), deliberate=deliberate)
    result = await council.run(claim, evidence, context)

    saved = None
    if save and result.quorum_met:
        saved = str(_state["transcripts"].save(
            result, models=council.models, rubric_name=rubric.name, label=claim))

    out = result.to_dict()
    out.pop("round1", None)
    out.pop("final", None)
    out["members"] = [
        {"model": m["model"], "total": m["total"], "verdict": m["verdict"],
         "one_line": m["one_line"], "error": m["error"],
         "moved": (None if m.get("round1_total") is None or m.get("total") is None
                   else m["total"] - m["round1_total"])}
        for m in result.final]
    out["transcript"] = saved
    return json.dumps(out, indent=1, ensure_ascii=False)


@mcp.tool()
async def describe_rubric(rubric_yaml: str = "") -> str:
    """Show the rubric the council will use, exactly as the members will see it."""
    rubric = Rubric.from_dict(yaml.safe_load(rubric_yaml)) if rubric_yaml \
        else _state["rubric"]
    if rubric is None:
        return "No rubric configured."
    return rubric.prompt_block()


@mcp.tool()
async def verify_transcript(run_dir: str) -> str:
    """Recompute the hashes on a saved run and report whether it was altered."""
    ok, problems = _state["transcripts"].verify(run_dir)
    return json.dumps({"intact": ok, "problems": problems})


@mcp.tool()
async def list_transcripts(limit: int = 20) -> str:
    """List saved council runs, newest last."""
    runs = _state["transcripts"].runs()[-limit:]
    out = []
    for r in runs:
        try:
            manifest = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
            out.append({"dir": str(r), "claim": manifest.get("claim"),
                        "total": manifest.get("consensus_total"),
                        "verdict": manifest.get("consensus_verdict"),
                        "written": manifest.get("written_utc")})
        except Exception:
            out.append({"dir": str(r), "error": "unreadable manifest"})
    return json.dumps(out, indent=1)


@mcp.tool()
async def health() -> str:
    """Check that the server has what it needs before anyone depends on it."""
    return json.dumps({
        "api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
        "rubric": getattr(_state["rubric"], "name", None),
        "dimensions": getattr(_state["rubric"], "dimension_names", []),
        "models": _state["models"],
        "transcript_root": str(_state["transcripts"].root),
    })


def main() -> None:  # pragma: no cover
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
