"""quorum: a council of language models that scores a claim against a rubric.

The difference from a general LLM council: this one measures rather than opines. Members score
fixed dimensions with fixed point ranges, reading only evidence you supply, and every
pre-deliberation position is preserved so agreement and accuracy can be told apart later.
"""
from .council import Council, CouncilResult, MemberOpinion, run_sync
from .rubric import Band, Dimension, Rubric, Verdict
from .transcript import Transcript
from .transport import Reply, Transport, extract_json

__version__ = "0.1.0"
__all__ = [
    "Band",
    "Council",
    "CouncilResult",
    "Dimension",
    "MemberOpinion",
    "Reply",
    "Rubric",
    "Transcript",
    "Transport",
    "Verdict",
    "extract_json",
    "run_sync",
]
