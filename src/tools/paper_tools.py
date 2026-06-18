"""
论文阅读/分析工具：
  - fetch_paper_fulltext  : 下载 PDF 并提取结构化文本
  - analyze_paper         : 深度分析（贡献/方法/局限/数据集）
"""

from typing import Optional

from ..agents.paper_analyzer import PaperAnalyzer
from ..services.paper_fetcher import PaperFetcher
from ..services.paper_search import ArxivSearcher
from .tool import Tool, ToolResult


def build_fetch_fulltext_tool(fetcher: PaperFetcher) -> Tool:
    def run(args):
        arxiv_id = (args.get("arxiv_id") or "").strip()
        if not arxiv_id:
            return ToolResult(success=False, error="arxiv_id is required")

        ft = fetcher.fetch_fulltext(arxiv_id)
        if ft is None:
            return ToolResult(
                success=False,
                error=f"Unable to fetch or parse PDF for {arxiv_id}",
            )

        sections_preview = {
            k: (v[:1500] + ("..." if len(v) > 1500 else ""))
            for k, v in ft.sections.items()
        }
        return ToolResult(
            success=True,
            content={
                "arxiv_id": ft.arxiv_id,
                "num_pages": ft.num_pages,
                "num_chars": ft.num_chars,
                "sections": sections_preview,
            },
        )

    return Tool(
        name="fetch_paper_fulltext",
        description=(
            "Download an arXiv paper PDF and extract its sections (abstract, introduction, "
            "method, experiments, conclusion). Section text is truncated to 1500 chars each. "
            "Use when abstract alone is insufficient to judge a paper's contribution."
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arXiv id such as '2301.00234' (with or without version).",
                },
            },
            "required": ["arxiv_id"],
        },
        run=run,
    )


def build_analyze_paper_tool(
    analyzer: PaperAnalyzer,
    arxiv_searcher: ArxivSearcher,
) -> Tool:
    """
    需要传入 arxiv_searcher。
    持久化由 analyzer 负责：若 analyzer 注入了 memory，analyze() 会经
    save_paper_note 把笔记摘要写入向量库（paper:{id}）并把节点/关系并入论文图谱。
    工具层不再重复写入，避免对同一 paper:{id} 的重复 add。
    """
    def run(args):
        arxiv_id = (args.get("arxiv_id") or "").strip()
        focus: Optional[str] = args.get("focus") or None
        if not arxiv_id:
            return ToolResult(success=False, error="arxiv_id is required")

        metas = arxiv_searcher.search(f"id:{arxiv_id}", max_results=1)
        if not metas:
            return ToolResult(
                success=False,
                error=f"Paper metadata not found for {arxiv_id}",
            )

        result = analyzer.analyze(metas[0], focus=focus, use_fulltext=True)
        return ToolResult(success=True, content=result)

    return Tool(
        name="analyze_paper",
        description=(
            "Deeply analyze one paper in conservative mode (downloads full text, does section-level map-reduce). "
            "This path prioritizes note quality and does not rely on image understanding during main synthesis. "
            "Produces a structured note following paper-reader quality standards: "
            "TL;DR, contributions, method summary, formulas with symbol legends, "
            "datasets, results with numbers, critical view, reproducibility checklist. "
            "Also saves the note to workspace/paper_notes/ and indexes it in vector store. "
            "Use sparingly on the 2-3 most important papers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv id such as '2301.00234'",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional focus area, e.g. 'methodology' or 'experiments'",
                },
            },
            "required": ["arxiv_id"],
        },
        run=run,
    )
