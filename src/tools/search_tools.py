"""
搜索相关工具：arXiv、Semantic Scholar、web（web_search 留作 TODO）

所有工具遵循统一接口：run(args: dict) -> ToolResult
"""

from typing import List

from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..services.github_search import GitHubCodeSearcher
from .tool import Tool, ToolResult


def _papers_to_dicts(papers: List) -> List[dict]:
    """把 PaperItem 列表序列化为易读 dict，控制 abstract 长度。"""
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "authors": p.authors[:3],
            "abstract": (p.abstract or "")[:400],
            "url": p.url,
            "published": p.published,
            "updated": p.updated,
            "categories": p.categories,
            "code_urls": p.code_urls,
            "code_url": p.code_url,
            "has_code": p.has_code,
            "code_confidence": p.code_confidence,
        }
        for p in papers
    ]


def build_arxiv_search_tool(searcher: ArxivSearcher) -> Tool:
    def run(args):
        query = (args.get("query") or "").strip()
        max_results = int(args.get("max_results", 5))
        if not query:
            return ToolResult(success=False, error="query is required")
        papers = searcher.search(query, max_results=max_results)
        return ToolResult(
            success=True,
            content={"count": len(papers), "papers": _papers_to_dicts(papers)},
            metadata={"source": "arxiv"},
        )

    return Tool(
        name="search_arxiv",
        description=(
            "Search arXiv for academic papers. Returns paper_id, title, authors, "
            "abstract (first 400 chars), and URL. Best for established research."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "English keyword query, 3-8 words. NO full sentences, NO Chinese.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max papers to return (default 5, recommend 3-8)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        run=run,
    )


def build_s2_search_tool(searcher: SemanticScholarSearcher) -> Tool:
    def run(args):
        query = (args.get("query") or "").strip()
        max_results = int(args.get("max_results", 5))
        if not query:
            return ToolResult(success=False, error="query is required")
        papers = searcher.search(query, max_results=max_results)
        return ToolResult(
            success=True,
            content={"count": len(papers), "papers": _papers_to_dicts(papers)},
            metadata={"source": "semantic_scholar"},
        )

    return Tool(
        name="search_semantic_scholar",
        description=(
            "Search Semantic Scholar for academic papers. Broader coverage than arXiv "
            "(includes journals, workshops). Use as a second opinion or when arXiv "
            "gives too few results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "English keyword query, 3-8 words.",
                },
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        run=run,
    )


def build_github_code_search_tool(searcher: GitHubCodeSearcher) -> Tool:
    def run(args):
        query = (args.get("query") or "").strip()
        max_results = int(args.get("max_results", 5))
        if not query:
            return ToolResult(success=False, error="query is required")
        hits = searcher.search(query, max_results=max_results)
        return ToolResult(
            success=True,
            content={
                "count": len(hits),
                "repos": [
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "snippet": hit.snippet,
                        "confidence": hit.confidence,
                    }
                    for hit in hits
                ],
            },
            metadata={"source": "github"},
        )

    return Tool(
        name="search_github_code",
        description=(
            "Search GitHub repositories related to a paper or method. Use this after paper search "
            "to prioritize papers with public implementations and to collect code links for the report."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Paper title or method name plus optional keywords.",
                },
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        run=run,
    )
