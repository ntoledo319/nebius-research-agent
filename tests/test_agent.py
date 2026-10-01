from nebius_research_agent.agent import (
    ResearchAgent,
    citation_coverage,
    parse_selfcheck,
)
from nebius_research_agent.tavily_client import Source

SOURCES = [
    Source("Alpha", "https://a.example", "fact one"),
    Source("Beta", "https://b.example", "fact two"),
    Source("Gamma", "https://c.example", "fact three"),
]


def test_citation_coverage_counts():
    cov = citation_coverage("Uses [1] and [2]. Also [2] again.", 3)
    assert cov["cited"] == [1, 2]
    assert cov["uncited"] == [3]
    assert cov["unknown"] == []
    assert cov["coverage"] == 2 / 3


def test_citation_coverage_flags_unknown_ids():
    cov = citation_coverage("Claims [1] and [9].", 2)
    assert cov["cited"] == [1]
    assert cov["unknown"] == [9]


def test_citation_coverage_no_sources():
    cov = citation_coverage("Nothing cited.", 0)
    assert cov["coverage"] == 0.0
    assert cov["cited"] == []


def test_parse_selfcheck_tolerates_prose():
    raw = 'Here is the audit:\n{"score": 0.75, "uncited": [3], "notes": "ok"}\nDone.'
    parsed = parse_selfcheck(raw)
    assert parsed["score"] == 0.75
    assert parsed["uncited"] == [3]


def test_parse_selfcheck_malformed_returns_none():
    assert parse_selfcheck("no json at all") is None
    assert parse_selfcheck("{broken: json}") is None
    assert parse_selfcheck("[1, 2, 3]") is None  # not an object


def test_agent_pipeline_end_to_end():
    llm_calls = []

    def fake_llm(messages):
        llm_calls.append(messages)
        if len(llm_calls) == 1:
            return "Answer first. Detail from [1] and [3]. Gaps: none."
        return '{"score": 0.67, "uncited": [2], "hallucinated": [], "notes": "fine"}'

    agent = ResearchAgent(fake_llm, lambda q: list(SOURCES))
    result = agent.research("what is nemotron?")

    assert result.brief.startswith("Answer first.")
    assert result.coverage["cited"] == [1, 3]
    assert result.coverage["uncited"] == [2]
    assert result.self_check["score"] == 0.67
    assert len(llm_calls) == 2
    # synthesis prompt carries numbered sources
    user_msg = llm_calls[0][1]["content"]
    assert "[1] Alpha — https://a.example" in user_msg
    assert "[3] Gamma — https://c.example" in user_msg
    assert "what is nemotron?" in user_msg
    # trace records each stage
    assert any(t.startswith("search:") for t in result.trace)
    assert any(t.startswith("synthesize:") for t in result.trace)
    assert any(t.startswith("coverage:") for t in result.trace)
    assert any(t.startswith("self-check:") for t in result.trace)


def test_agent_malformed_selfcheck_does_not_crash():
    replies = iter(["Brief with [1].", "sorry, cannot comply"])

    agent = ResearchAgent(lambda m: next(replies), lambda q: list(SOURCES))
    result = agent.research("q")
    assert result.self_check is None
    assert "unparseable" in result.trace[-1]
    # deterministic coverage still computed
    assert result.coverage["cited"] == [1]


def test_agent_no_sources_stays_honest():
    agent = ResearchAgent(lambda m: "should not be called", lambda q: [])
    result = agent.research("obscure question")
    assert "No sources" in result.brief
    assert result.coverage["total"] == 0


def test_agent_self_check_can_be_disabled():
    calls = []
    agent = ResearchAgent(
        lambda m: calls.append(m) or "Brief [1].",
        lambda q: list(SOURCES),
        self_check=False,
    )
    result = agent.research("q")
    assert len(calls) == 1
    assert result.self_check is None
