import httpx
import pytest

from nebius_research_agent.tavily_client import (
    TavilyClient,
    TavilyNotConfiguredError,
    TavilySearchError,
)


def make_client(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return TavilyClient("tvly-test", http_client=http, **kwargs)


def test_from_env_without_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyNotConfiguredError):
        TavilyClient.from_env()


def test_request_shape_and_result_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "results": [
                {"title": "T1", "url": "https://a.example", "content": "c1", "score": 0.9},
                {"title": "T2", "url": "https://b.example", "content": "c2"},
                {"url": "https://c.example"},
            ]
        })

    sources = make_client(handler).search("nemotron", max_results=3)

    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["body"]["api_key"] == "tvly-test"
    assert seen["body"]["query"] == "nemotron"
    assert seen["body"]["max_results"] == 3
    assert [s.title for s in sources] == ["T1", "T2", "https://c.example"]
    assert sources[0].score == 0.9
    assert sources[1].score == 0.0
    assert sources[2].content == ""


def test_http_error_raises():
    client = make_client(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(TavilySearchError):
        client.search("anything")


def test_empty_results():
    client = make_client(lambda r: httpx.Response(200, json={"results": []}))
    assert client.search("nothing") == []
