"""Tavily search client.

Wraps the Tavily Search API (``POST https://api.tavily.com/search``). The
network call is lazy: without ``TAVILY_API_KEY`` the client raises
``TavilyNotConfiguredError`` at call time — no fabricated results, ever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "https://api.tavily.com"


class TavilyNotConfiguredError(RuntimeError):
    """TAVILY_API_KEY is missing; the client refuses to fake search results."""


class TavilySearchError(RuntimeError):
    pass


@dataclass
class Source:
    title: str
    url: str
    content: str
    score: float = 0.0


class TavilyClient:
    """``http_client`` is injectable so tests run fully offline."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, **kwargs) -> "TavilyClient":
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise TavilyNotConfiguredError(
                "TAVILY_API_KEY is not set. Get a free key from "
                "app.tavily.com and export it; without it this client "
                "raises instead of returning invented sources."
            )
        base_url = os.environ.get("TAVILY_BASE_URL", DEFAULT_BASE_URL)
        return cls(api_key, base_url, **kwargs)

    def close(self) -> None:
        self._http.close()

    def search(self, query: str, *, max_results: int = 5) -> list[Source]:
        resp = self._http.post(
            f"{self.base_url}/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
        if resp.status_code != 200:
            raise TavilySearchError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        sources = []
        for item in data.get("results") or []:
            sources.append(
                Source(
                    title=str(item.get("title") or item.get("url") or "untitled"),
                    url=str(item.get("url") or ""),
                    content=str(item.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                )
            )
        return sources
