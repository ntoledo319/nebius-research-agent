import httpx
import pytest

from nebius_research_agent.tokenfactory import (
    ChainExhaustedError,
    NebiusNotConfiguredError,
    ResilientChatClient,
)


def make_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_payload(model, content="answer"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": model,
    }


MESSAGES = [{"role": "user", "content": "hello"}]


def test_from_env_without_key_raises(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(NebiusNotConfiguredError):
        ResilientChatClient.from_env()


def test_empty_key_raises():
    with pytest.raises(NebiusNotConfiguredError):
        ResilientChatClient(["m"], "")


def test_request_shape_and_response_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_payload(seen["body"]["model"], "hi there"))

    client = ResilientChatClient(
        ["nvidia/Nemotron-3_5-Lightning"], "test-key",
        http_client=make_transport(handler),
    )
    result = client.chat(MESSAGES, temperature=0.1, max_tokens=128)

    assert seen["url"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert seen["body"]["messages"] == MESSAGES
    assert seen["body"]["temperature"] == 0.1
    assert seen["body"]["max_tokens"] == 128
    assert result.content == "hi there"
    assert result.usage["prompt_tokens"] == 10
    assert result.rail_index == 0
    assert result.attempts == 1


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.test/v9")
    monkeypatch.setenv("NEBIUS_MODELS", "a,b, c")
    client = ResilientChatClient.from_env(http_client=make_transport(
        lambda r: httpx.Response(200, json=ok_payload("a"))))
    assert client.base_url == "https://example.test/v9"
    assert client.models == ["a", "b", "c"]


def test_failover_rotates_to_second_model():
    calls = []
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "first":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=ok_payload(model, "from second"))

    client = ResilientChatClient(
        ["first", "second"], "k",
        rail_tries=2, backoff_base=1.0,
        http_client=make_transport(handler), sleep_fn=sleeps.append,
    )
    result = client.chat(MESSAGES)

    assert result.model == "second"
    assert result.rail_index == 1
    # first rail got its 2 tries, then exactly one call on rail 2
    assert calls == ["first", "first", "second"]
    # one backoff sleep between the two retries on rail 1, jittered
    assert len(sleeps) == 1
    assert 0.5 <= sleeps[0] <= 1.5
    assert client.health["first"]["consecutive_failures"] == 2
    assert client.health["second"]["consecutive_failures"] == 0


def test_non_retryable_400_skips_remaining_tries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "bad":
            return httpx.Response(400, text="invalid model")
        return httpx.Response(200, json=ok_payload(model, "ok"))

    client = ResilientChatClient(
        ["bad", "good"], "k", rail_tries=3,
        http_client=make_transport(handler), sleep_fn=lambda s: None,
    )
    result = client.chat(MESSAGES)
    assert result.model == "good"
    assert calls == ["bad", "good"]  # no wasted retries on a 400


def test_retry_within_rail_on_429():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=ok_payload("m", "recovered"))

    client = ResilientChatClient(
        ["m"], "k", rail_tries=2,
        http_client=make_transport(handler), sleep_fn=lambda s: None,
    )
    result = client.chat(MESSAGES)
    assert result.content == "recovered"
    assert result.attempts == 2


def test_transport_error_is_retryable_then_fails_over():
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        model = json.loads(request.content)["model"]
        if model == "down":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=ok_payload(model, "up"))

    client = ResilientChatClient(
        ["down", "up"], "k", rail_tries=2,
        http_client=make_transport(handler), sleep_fn=lambda s: None,
    )
    assert client.chat(MESSAGES).model == "up"


def test_chain_exhausted_reports_all_rails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = ResilientChatClient(
        ["a", "b"], "k", rail_tries=1,
        http_client=make_transport(handler), sleep_fn=lambda s: None,
    )
    with pytest.raises(ChainExhaustedError) as excinfo:
        client.chat(MESSAGES)
    assert [m for m, _ in excinfo.value.attempts] == ["a", "b"]


def test_malformed_200_body_treated_as_rail_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = ResilientChatClient(
        ["x"], "k", rail_tries=1,
        http_client=make_transport(handler), sleep_fn=lambda s: None,
    )
    with pytest.raises(ChainExhaustedError):
        client.chat(MESSAGES)
