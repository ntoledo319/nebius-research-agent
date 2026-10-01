# Resilient Research Agent

**Nebius x NVIDIA Global AI Hackathon** — *Best Apps and Agents* track, with a
*Best Use of Tavily* bonus entry.

Ask a question, get a cited brief. The agent searches the web with **Tavily**,
reasons over the sources with an **NVIDIA Nemotron** open model served by
**Nebius Token Factory**, writes a brief with inline `[n]` citations, then
audits its own citation coverage with a second model call. Every model call
runs through a failover chain with per-rail retries, exponential backoff with
jitter, and full hop logging — a provider hiccup rotates the call to the next
Nemotron rail instead of killing the run.

## Architecture

```
            ┌─────────────────── ResearchAgent ───────────────────┐
 question ─▶│ 1. search      Tavily Search API ──▶ ranked sources │
            │ 2. synthesize  Nemotron (Token Factory) ──▶ brief   │
            │ 3. self-check  Nemotron audits citations ──▶ JSON   │
            └───────┬──────────────────────┬──────────────────────┘
                    │                      │
        tavily_client.py         tokenfactory.py
        TavilyClient             ResilientChatClient
        POST /search             POST /chat/completions (OpenAI-compatible)
                                 ├─ rail 0: nvidia/Nemotron-3_5-Lightning
                                 └─ rail 1: nvidia/Nemotron-3-Super-120B-A12B
                                   (retries → backoff+jitter → rotate →
                                    ChainExhaustedError, never a fake answer)
```

A deterministic regex audit (`citation_coverage`) runs alongside the model
self-check, so coverage numbers are always real even if the model's JSON is
unparseable.

## Hackathon mapping

| Requirement | Where |
|---|---|
| Runtime call to Token Factory inference API | `tokenfactory.py` — `POST {NEBIUS_BASE_URL}/chat/completions`, OpenAI-compatible, default `https://api.tokenfactory.nebius.com/v1` |
| At least one NVIDIA open-source model | Default chain is all-NVIDIA: `nvidia/Nemotron-3_5-Lightning` → `nvidia/Nemotron-3-Super-120B-A12B` (see [Token Factory Nemotron docs](https://github.com/nebius/token-factory-cookbook/tree/main/models/nemotron)) |
| Best Use of Tavily | `tavily_client.py` drives the search stage; every claim in the brief is grounded in a Tavily-ranked source |
| Working demo | `python -m nebius_research_agent "question"` (CLI); web demo planned |
| Public repo README | this file |

## Setup

Requires Python 3.11+ and `httpx`. Copy `.env.example` to `.env` (or export):

```bash
export NEBIUS_API_KEY=...    # tokenfactory.nebius.com
export TAVILY_API_KEY=...    # app.tavily.com (free tier works)
```

Optional overrides:

| Var | Default |
|---|---|
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` |
| `NEBIUS_MODEL` | `nvidia/Nemotron-3_5-Lightning` |
| `NEBIUS_MODELS` | comma-separated failover chain (overrides `NEBIUS_MODEL`) |
| `TAVILY_BASE_URL` | `https://api.tavily.com` |

## Demo

```bash
python -m nebius_research_agent "What is NVIDIA Nemotron 3 and who serves it?"
python -m nebius_research_agent --json "..."        # machine-readable
python -m nebius_research_agent --max-results 8 --no-self-check "..."
```

Output: the brief with inline citations, a numbered source list, citation
coverage stats (deterministic + model self-check), and a trace of every stage
and model hop. To watch the failover in action, set
`NEBIUS_MODELS=nvidia/Nemotron-3_5-Lightning,nvidia/Nemotron-3-Super-120B-A12B`
and run with logging at INFO — each rail attempt, backoff, and rotation is
logged.

## Honesty notes

- **No fake responses.** Without `NEBIUS_API_KEY` / `TAVILY_API_KEY` the
  clients raise `NebiusNotConfiguredError` / `TavilyNotConfiguredError` (CLI
  exits 2 with instructions). If every model rail fails, the agent raises
  `ChainExhaustedError` (CLI exits 3) rather than inventing an answer.
- If Tavily returns zero sources, the agent says so instead of hallucinating
  a brief.
- All 26 tests run fully offline against mocked HTTP — nothing in the test
  suite touches a real API or fabricates demo output.

## Tests

```bash
python -m pytest        # 26 passed — protocol shapes, failover order,
                        # backoff, citation coverage, self-check parsing, CLI
```

## Roadmap

- Web demo (Gradio/FastAPI front end on the same `ResearchAgent`)
- Streaming synthesis output
- Optional second Tavily pass ("deepen gaps" from the self-check report)
