"""Web search tool with Brave API primary search and HTML fallback."""

import json
import logging
import os
import re
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from hello_agents.tools import Tool, ToolParameter, ToolResponse, tool_action


logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Search the web with Brave first, then fall back to public search pages."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: int = 10,
    ):
        super().__init__(
            name="web_search",
            description="使用搜索引擎搜索网络信息，必要时自动退化到网页抓取模式",
            expandable=True,
        )

        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self.max_results = max_results
        self.timeout = timeout
        self._brave_url = "https://api.search.brave.com/res/v1/web/search"
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = parameters.get("query", "")
        count = parameters.get("count", self.max_results)
        return self._search(query, count)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词",
                required=True,
            ),
            ToolParameter(
                name="count",
                type="integer",
                description=f"返回结果数量，默认 {self.max_results}",
                required=False,
            ),
        ]

    def _search(self, query: str, count: Optional[int] = None) -> ToolResponse:
        if not query:
            return ToolResponse.error(
                code="INVALID_INPUT",
                message="搜索查询不能为空",
            )

        limit = self._normalize_count(count)

        if self.api_key:
            brave_response = self._search_with_brave(query, limit)
            if brave_response is not None:
                return brave_response
        else:
            logger.warning(
                "Brave search skipped because BRAVE_API_KEY is not configured; using HTML fallback. query=%r",
                query,
            )

        fallback_response = self._search_with_html_fallback(query, limit)
        if fallback_response is not None:
            return fallback_response

        return ToolResponse.error(
            code="SEARCH_UNAVAILABLE",
            message=(
                "Brave Search API 不可用，且网页抓取 fallback 也未能返回结果。"
                "请稍后重试，或配置 BRAVE_API_KEY 以提升稳定性。"
            ),
        )

    def _search_with_brave(self, query: str, count: int) -> Optional[ToolResponse]:
        try:
            url = f"{self._brave_url}?q={quote_plus(query)}&count={count}"
            request = Request(url)
            request.add_header("Accept", "application/json")
            request.add_header("Accept-Encoding", "gzip")
            request.add_header("X-Subscription-Token", self.api_key)

            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = self._parse_brave_results(data)
            if not results:
                logger.warning(
                    "Brave search returned no results; falling back to HTML search. query=%r",
                    query,
                )
                return None

            return ToolResponse.success(
                text=self._format_results(results, source="Brave Search API"),
                data={
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "source": "brave",
                    "fallback_used": False,
                },
            )
        except HTTPError as exc:
            logger.warning(
                "Brave search failed with HTTP error; falling back to HTML search. query=%r status=%s reason=%s",
                query,
                exc.code,
                exc.reason,
            )
            return None
        except URLError as exc:
            logger.warning(
                "Brave search failed with network error; falling back to HTML search. query=%r error=%s",
                query,
                exc,
            )
            return None
        except Exception:
            logger.exception(
                "Brave search failed unexpectedly; falling back to HTML search. query=%r",
                query,
            )
            return None

    def _search_with_html_fallback(self, query: str, count: int) -> Optional[ToolResponse]:
        providers = (
            ("bing", self._search_bing_html),
            ("duckduckgo", self._search_duckduckgo_html),
        )

        for provider_name, provider in providers:
            try:
                results = provider(query, count)
            except Exception:
                logger.exception(
                    "HTML fallback provider failed. provider=%s query=%r",
                    provider_name,
                    query,
                )
                results = []

            if results:
                return ToolResponse.success(
                    text=self._format_results(
                        results,
                        source=f"{provider_name} HTML fallback",
                        fallback_note="当前结果来自网页抓取 fallback，稳定性可能低于正式搜索 API。",
                    ),
                    data={
                        "query": query,
                        "results": results,
                        "count": len(results),
                        "source": provider_name,
                        "fallback_used": True,
                    },
                )

        return None

    def _search_bing_html(self, query: str, count: int) -> List[dict]:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={count}"
        html = self._fetch_html(url)

        matches = re.finditer(
            r'<li class="b_algo".*?<h2><a href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a></h2>.*?'
            r'(?:<div class="b_caption"[^>]*>.*?<p>(?P<desc>.*?)</p>.*?</div>)?',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: List[dict] = []
        for match in matches:
            result = self._build_result(
                title=match.group("title"),
                url=match.group("url"),
                description=match.group("desc") or "",
            )
            if result:
                results.append(result)
            if len(results) >= count:
                break

        return results

    def _search_duckduckgo_html(self, query: str, count: int) -> List[dict]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        html = self._fetch_html(url)

        matches = re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'(?:<a[^>]*class="result__snippet"[^>]*>(?P<desc>.*?)</a>|'
            r'<div[^>]*class="result__snippet"[^>]*>(?P<desc2>.*?)</div>)?',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: List[dict] = []
        for match in matches:
            raw_url = match.group("url")
            result = self._build_result(
                title=match.group("title"),
                url=self._normalize_duckduckgo_url(raw_url),
                description=match.group("desc") or match.group("desc2") or "",
            )
            if result:
                results.append(result)
            if len(results) >= count:
                break

        return results

    def _fetch_html(self, url: str) -> str:
        request = Request(url)
        request.add_header("User-Agent", self.user_agent)
        request.add_header("Accept", "text/html,application/xhtml+xml")
        request.add_header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")

        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _parse_brave_results(self, data: dict) -> List[dict]:
        results = []
        for item in data.get("web", {}).get("results", []):
            result = self._build_result(
                title=item.get("title", ""),
                url=item.get("url", ""),
                description=item.get("description", ""),
            )
            if result:
                results.append(result)
        return results

    def _build_result(self, title: str, url: str, description: str) -> Optional[dict]:
        clean_title = self._clean_text(title)
        clean_url = self._clean_url(url)
        clean_description = self._clean_text(description)

        if not clean_title or not clean_url:
            return None

        return {
            "title": clean_title,
            "url": clean_url,
            "description": clean_description,
        }

    def _normalize_duckduckgo_url(self, url: str) -> str:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return query["uddg"][0]
        return url

    def _clean_url(self, url: str) -> str:
        text = unescape((url or "").strip())
        if text.startswith("//"):
            return f"https:{text}"
        return text

    def _clean_text(self, text: str) -> str:
        text = unescape(text or "")
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_count(self, count: Optional[int]) -> int:
        try:
            value = int(count) if count is not None else self.max_results
        except (TypeError, ValueError):
            value = self.max_results
        return max(1, min(value, 10))

    def _format_results(
        self,
        results: List[dict],
        source: str,
        fallback_note: Optional[str] = None,
    ) -> str:
        lines = [f"找到 {len(results)} 个结果", f"来源: {source}", ""]
        if fallback_note:
            lines.extend([fallback_note, ""])

        for index, result in enumerate(results, 1):
            lines.append(f"{index}. **{result['title']}**")
            lines.append(f"   URL: {result['url']}")
            if result["description"]:
                lines.append(f"   {result['description'][:200]}")
            lines.append("")

        return "\n".join(lines).strip()

    @tool_action("search_web", "搜索网络信息")
    def _search_action(self, query: str, count: int = None) -> str:
        response = self._search(query, count)
        return response.text
