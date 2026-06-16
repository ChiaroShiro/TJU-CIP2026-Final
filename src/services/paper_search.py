import time
import arxiv
import requests
from typing import List, Optional

from ..core.models import PaperItem
from .github_search import GitHubCodeSearcher


class ArxivSearcher:
    """
    arXiv搜索器，含有限制和容错：
    - query 超长自动截断到 200 字符
    - HTTP 429/其他异常返回空列表而非抛出
    - num_retries 由 arxiv.Client 内部处理，这里再加一层保护
    """
    MAX_QUERY_LEN = 200

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        # page_size=小一些，num_retries 多一点，delay 适当拉长避免 429
        self.client = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)

    def search(self, query: str, max_results: Optional[int] = None) -> List[PaperItem]:
        limit = max_results or self.max_results
        safe_query = (query or "")[: self.MAX_QUERY_LEN].strip()
        if not safe_query:
            return []

        try:
            search = arxiv.Search(
                query=safe_query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in self.client.results(search):
                papers.append(PaperItem(
                    paper_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    published=result.published.isoformat() if result.published else "",
                    updated=result.updated.isoformat() if result.updated else "",
                    categories=result.categories,
                ))
            return papers
        except Exception as e:
            err_str = str(e)[:200]
            is_rate_limit = "429" in err_str or "503" in err_str
            print(f"[ArxivSearcher] search failed ({type(e).__name__}): {err_str[:120]}")
            if is_rate_limit:
                # 抛出特定异常让 tool 层能告知 Agent "arxiv 被限流"
                raise RuntimeError(f"arXiv rate-limited (429/503). Try semantic_scholar or web_search instead.") from e
            return []


class SemanticScholarSearcher:
    """
    Semantic Scholar 搜索器，含容错和重试。
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    MAX_QUERY_LEN = 300

    def search(self, query: str, max_results: int = 10) -> List[PaperItem]:
        safe_query = (query or "")[: self.MAX_QUERY_LEN].strip()
        if not safe_query:
            return []

        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/paper/search",
                    params={
                        "query": safe_query,
                        "limit": max_results,
                        "fields": "title,authors,abstract,year,externalIds,url",
                    },
                    timeout=15,
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    return []
                papers = []
                for item in resp.json().get("data", []):
                    papers.append(PaperItem(
                        paper_id=item.get("paperId", ""),
                        title=item.get("title", ""),
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        abstract=item.get("abstract") or "",
                        url=item.get("url")
                             or f"https://www.semanticscholar.org/paper/{item.get('paperId','')}",
                        published=str(item.get("year", "")),
                    ))
                return papers
            except Exception as e:
                print(f"[S2Searcher] search failed ({type(e).__name__}): {str(e)[:120]}")
                return []
        return []


class PaperDiscoveryService:
    """
    聚合 arXiv / Semantic Scholar / GitHub 代码线索的论文发现服务。
    """

    def __init__(self, arxiv_max_results: int = 10):
        self.arxiv = ArxivSearcher(max_results=arxiv_max_results)
        self.s2 = SemanticScholarSearcher()
        self.github = GitHubCodeSearcher()

    def search_topic(self, query: str, max_results: int = 10) -> List[PaperItem]:
        papers = self.arxiv.search(query, max_results=max_results)
        if len(papers) < max_results:
            papers.extend(self.s2.search(query, max_results=max_results - len(papers)))
        return self._dedupe_and_rank(query, papers)

    def enrich_with_code(self, papers: List[PaperItem], max_code_hits: int = 3) -> List[PaperItem]:
        enriched: List[PaperItem] = []
        for paper in papers:
            code_urls = self._collect_code_urls(paper)
            if not code_urls:
                code_hits = self.github.search_repo_for_paper(
                    paper.title,
                    keywords=paper.categories[:3],
                    max_results=max_code_hits,
                )
                code_urls = [hit.url for hit in code_hits]

            paper.code_urls = code_urls
            paper.code_url = code_urls[0] if code_urls else ""
            paper.code_repos = code_urls
            paper.has_code = bool(code_urls)
            paper.code_confidence = self._code_confidence(paper)
            enriched.append(paper)

        return sorted(
            enriched,
            key=lambda p: (
                1 if p.has_code else 0,
                p.code_confidence,
                self._published_year(p.published),
            ),
            reverse=True,
        )

    def _dedupe_and_rank(self, query: str, papers: List[PaperItem]) -> List[PaperItem]:
        seen = set()
        deduped: List[PaperItem] = []
        for paper in papers:
            key = paper.paper_id or paper.title
            if key in seen:
                continue
            seen.add(key)
            deduped.append(paper)
        return sorted(
            deduped,
            key=lambda p: (
                self._query_match_score(query, p),
                self._published_year(p.published),
            ),
            reverse=True,
        )

    def _collect_code_urls(self, paper: PaperItem) -> List[str]:
        candidates = []
        for text in [paper.url, paper.abstract, paper.title]:
            candidates.extend(self.github.extract_repos_from_text(text))
        seen = set()
        cleaned = []
        for url in candidates:
            if url not in seen:
                seen.add(url)
                cleaned.append(url)
        return cleaned

    @staticmethod
    def _published_year(value: str) -> int:
        try:
            return int(str(value)[:4])
        except Exception:
            return 0

    @staticmethod
    def _query_match_score(query: str, paper: PaperItem) -> float:
        q = (query or "").lower()
        text = f"{paper.title} {paper.abstract} {' '.join(paper.authors)}".lower()
        score = 0.0
        for token in [t for t in q.split() if len(t) >= 4]:
            if token in text:
                score += 1.0
        return score + (0.25 if paper.has_code else 0.0)

    @staticmethod
    def _code_confidence(paper: PaperItem) -> float:
        if paper.code_urls:
            return min(1.0, 0.65 + 0.1 * len(paper.code_urls))
        return 0.0
