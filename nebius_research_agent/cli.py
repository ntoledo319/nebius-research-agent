"""CLI: python -m nebius_research_agent "your question" """

from __future__ import annotations

import argparse
import json
import sys

from .agent import ResearchAgent
from .tavily_client import TavilyClient, TavilyNotConfiguredError
from .tokenfactory import (
    ChainExhaustedError,
    NebiusNotConfiguredError,
    ResilientChatClient,
)


def build_agent(args: argparse.Namespace) -> tuple[ResearchAgent, ResilientChatClient, TavilyClient]:
    llm_client = ResilientChatClient.from_env()
    searcher_client = TavilyClient.from_env()
    agent = ResearchAgent(
        llm=lambda messages: llm_client.chat(messages).content,
        searcher=lambda q: searcher_client.search(q, max_results=args.max_results),
        max_results=args.max_results,
        self_check=not args.no_self_check,
    )
    return agent, llm_client, searcher_client


def main(argv: list[str] | None = None, *, agent: ResearchAgent | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nebius_research_agent",
        description="Resilient research agent: Tavily search + NVIDIA Nemotron "
                    "reasoning on Nebius Token Factory, with model failover.",
    )
    parser.add_argument("question", nargs="+", help="research question")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--no-self-check", action="store_true")
    args = parser.parse_args(argv)
    question = " ".join(args.question)

    clients: list = []
    if agent is None:
        try:
            agent, llm_client, searcher_client = build_agent(args)
        except (NebiusNotConfiguredError, TavilyNotConfiguredError) as exc:
            print(f"not configured: {exc}", file=sys.stderr)
            return 2
        clients = [llm_client, searcher_client]

    try:
        result = agent.research(question)
    except ChainExhaustedError as exc:
        print(f"all model rails failed: {exc}", file=sys.stderr)
        return 3
    finally:
        for c in clients:
            c.close()

    if args.json:
        print(json.dumps({
            "question": result.question,
            "brief": result.brief,
            "sources": [vars(s) for s in result.sources],
            "coverage": result.coverage,
            "self_check": result.self_check,
            "trace": result.trace,
        }, indent=2))
    else:
        print(f"# Brief: {result.question}\n")
        print(result.brief)
        print("\n## Sources")
        for i, s in enumerate(result.sources, start=1):
            print(f"[{i}] {s.title} — {s.url}")
        print("\n## Citation coverage")
        cov = result.coverage
        print(f"{len(cov['cited'])}/{cov['total']} cited; "
              f"uncited={cov['uncited']}; unknown={cov['unknown']}")
        if result.self_check is not None:
            print(f"model self-check score: {result.self_check.get('score')}")
        print("\n## Trace")
        for line in result.trace:
            print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
