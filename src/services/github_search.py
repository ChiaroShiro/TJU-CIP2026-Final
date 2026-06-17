"""
GitHub 代码关联搜索。

目标：
- 为论文自动补充公开代码链接
- 优先保留 GitHub 仓库
- 尽量使用标题 + 关键词弱假设，避免把噪声结果误当代码
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

GITHUB_REPO_RE = re.compile(r"https?://github\.com/[^/\s]+/[^/\s#?]+", re.IGNORECASE)
NON_REPO_OWNERS = {
    "about",
    "apps",
    "blog",
    "collections",
    "customer-stories",
    "enterprise",
    "events",
    "explore",
    "features",
    "marketplace",
    "new",
    "organizations",
    "orgs",
    "pricing",
    "search",
    "settings",
    "showcases",
    "sponsors",
    "topics",
    "trending",
}


@dataclass
class GitHubCodeHit:
    title: str
    url: str
    snippet: str = ""
    confidence: float = 0.0
    source: str = "web"


class GitHubCodeSearcher:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[GitHubCodeHit]:
        query = (query or "").strip()
        if not query:
            return []

        hits: List[GitHubCodeHit] = []
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{query} github code", max_results=max_results * 3))
        except Exception:
            results = []

        seen = set()
        for item in results:
            url = item.get("href") or ""
            title = item.get("title") or ""
            snippet = item.get("body") or ""
            for repo in self._extract_github_urls(f"{url} {snippet}"):
                if repo in seen:
                    continue
                seen.add(repo)
                hits.append(
                    GitHubCodeHit(
                        title=title or repo,
                        url=repo,
                        snippet=snippet[:280],
                        confidence=self._score_repo(query, repo, title, snippet),
                    )
                )
                if len(hits) >= max_results:
                    return hits
        return hits

    def search_repo_for_paper(self, paper_title: str, keywords: Optional[List[str]] = None, max_results: int = 5) -> List[GitHubCodeHit]:
        parts = [paper_title] + list(keywords or [])
        query = " ".join(p for p in parts if p).strip()
        return self.search(query, max_results=max_results)

    def extract_repos_from_text(self, text: str) -> List[str]:
        return self._extract_github_urls(text)

    @staticmethod
    def _extract_github_urls(text: str) -> List[str]:
        if not text:
            return []
        urls = GITHUB_REPO_RE.findall(text)
        cleaned = []
        seen = set()
        for url in urls:
            url = url.rstrip(".,)\"'").rstrip("/")
            if url in seen or "/blob/" in url or "/tree/" in url:
                continue
            if not GitHubCodeSearcher._looks_like_repo_url(url):
                continue
            seen.add(url)
            cleaned.append(url)
        return cleaned

    @staticmethod
    def _looks_like_repo_url(url: str) -> bool:
        match = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)$", url, re.IGNORECASE)
        if not match:
            return False
        owner, repo = match.group(1).lower(), match.group(2).lower()
        if owner in NON_REPO_OWNERS:
            return False
        if repo in {"", "repositories", "stars", "followers", "following"}:
            return False
        return True

    @staticmethod
    def _score_repo(query: str, repo: str, title: str, snippet: str) -> float:
        score = 0.1
        q = query.lower()
        hay = f"{repo} {title} {snippet}".lower()
        for token in re.split(r"\s+", q):
            if len(token) >= 4 and token in hay:
                score += 0.1
        if "github.com" in repo:
            score += 0.2
        if repo.count("/") >= 2:
            score += 0.1
        if "code" in hay or "implementation" in hay or "repo" in hay:
            score += 0.1
        return min(score, 1.0)
