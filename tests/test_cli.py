import json

import pytest

from nebius_research_agent.agent import ResearchAgent
from nebius_research_agent.cli import main
from nebius_research_agent.tavily_client import Source


@pytest.fixture
def fake_agent():
    sources = [Source("Alpha", "https://a.example", "fact one", 0.9)]
    replies = iter(["Answer with [1].", '{"score": 1.0, "uncited": []}'])
    return ResearchAgent(lambda m: next(replies), lambda q: sources)


def test_cli_smoke_text_output(fake_agent, capsys):
    rc = main(["what", "is", "nemotron?"], agent=fake_agent)
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Brief: what is nemotron?" in out
    assert "Answer with [1]." in out
    assert "[1] Alpha — https://a.example" in out
    assert "1/1 cited" in out
    assert "model self-check score: 1.0" in out
    assert "## Trace" in out


def test_cli_json_output(fake_agent, capsys):
    rc = main(["q", "--json"], agent=fake_agent)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["question"] == "q"
    assert payload["brief"] == "Answer with [1]."
    assert payload["sources"][0]["url"] == "https://a.example"
    assert payload["coverage"]["cited"] == [1]
    assert payload["self_check"]["score"] == 1.0
    assert payload["trace"]


def test_cli_missing_keys_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    rc = main(["anything"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "NEBIUS_API_KEY" in err
