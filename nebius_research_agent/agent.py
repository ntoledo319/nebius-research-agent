"""Resilient research agent: Tavily search -> Nemotron synthesis -> self-check.

Pipeline for one question:

1. search   — Tavily returns ranked sources, assigned citation ids [1..n].
2. synthesize — the LLM (NVIDIA Nemotron via Nebius Token Factory) writes a
   brief with inline [n] citations.
3. self-check — a second LLM call audits citation coverage and answers with
   JSON; a deterministic regex check runs alongside it so coverage is always
   computed even if the model's JSON is malformed.

The agent is transport-agnostic: ``llm`` is any ``callable(messages) -> str``
and ``searcher`` any ``callable(query) -> list[Source]``, which keeps tests
fully offline and lets the CLI plug in the real resilient clients.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .tavily_client import Source

SYNTHESIS_SYSTEM = """\
You are a research analyst. Using only the numbered sources provided, write a \
concise brief answering the user's question. Rules:
- Cite every factual claim inline as [n] using only the source numbers given.
- Never invent sources, URLs, or numbers. If the sources cannot answer a \
part of the question, say so explicitly.
- Structure: 1-sentence answer first, then 3-6 short paragraphs, then a \
"Gaps" line listing what the sources did not cover."""

SELFCHECK_SYSTEM = """\
You audit citation coverage of a research brief. Given the brief and the list \
of valid source ids, respond with a JSON object only, no prose:
{"score": <0.0-1.0>, "uncited": [<ids never cited>], "hallucinated": [<cited \
ids that do not exist>], "notes": "<one sentence>"}"""

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def citation_coverage(brief: str, n_sources: int) -> dict:
    """Deterministic [n]-citation audit. Pure and testable."""
    found = {int(m) for m in _CITATION_RE.findall(brief)}
    valid = set(range(1, n_sources + 1))
    cited = sorted(found & valid)
    return {
        "total": n_sources,
        "cited": cited,
        "uncited": sorted(valid - found),
        "unknown": sorted(found - valid),
        "coverage": (len(cited) / n_sources) if n_sources else 0.0,
    }


def parse_selfcheck(text: str) -> dict | None:
    """Extract the self-check JSON object from a model reply, tolerating prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class ResearchResult:
    question: str
    brief: str
    sources: list[Source]
    coverage: dict
    self_check: dict | None = None
    trace: list[str] = field(default_factory=list)


class ResearchAgent:
    def __init__(self, llm, searcher, *, max_results: int = 5, self_check: bool = True) -> None:
        self.llm = llm          # callable(list[dict]) -> str
        self.searcher = searcher  # callable(str) -> list[Source]
        self.max_results = max_results
        self.self_check_enabled = self_check

    def research(self, question: str) -> ResearchResult:
        trace: list[str] = []

        t0 = time.monotonic()
        sources = self.searcher(question)
        trace.append(f"search: {len(sources)} sources via Tavily "
                     f"({time.monotonic() - t0:.2f}s)")
        if not sources:
            return ResearchResult(
                question=question,
                brief="No sources found; cannot write an honest brief.",
                sources=[],
                coverage=citation_coverage("", 0),
                trace=trace,
            )

        numbered = "\n\n".join(
            f"[{i}] {s.title} — {s.url}\n{s.content}"
            for i, s in enumerate(sources, start=1)
        )
        t0 = time.monotonic()
        brief = self.llm([
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content":
                f"Question: {question}\n\nSources:\n{numbered}"},
        ])
        trace.append(f"synthesize: brief drafted via Token Factory "
                     f"({time.monotonic() - t0:.2f}s)")

        coverage = citation_coverage(brief, len(sources))
        trace.append(
            f"coverage: {len(coverage['cited'])}/{coverage['total']} sources cited"
            + (f", uncited={coverage['uncited']}" if coverage["uncited"] else "")
            + (f", unknown={coverage['unknown']}" if coverage["unknown"] else "")
        )

        check = None
        if self.self_check_enabled:
            t0 = time.monotonic()
            raw = self.llm([
                {"role": "system", "content": SELFCHECK_SYSTEM},
                {"role": "user", "content":
                    f"Valid source ids: 1..{len(sources)}\n\nBrief:\n{brief}"},
            ])
            check = parse_selfcheck(raw)
            trace.append(
                "self-check: " + (
                    f"model score={check.get('score')} ({time.monotonic() - t0:.2f}s)"
                    if check is not None
                    else "model returned unparseable JSON; deterministic coverage stands"
                )
            )

        return ResearchResult(
            question=question,
            brief=brief,
            sources=sources,
            coverage=coverage,
            self_check=check,
            trace=trace,
        )
