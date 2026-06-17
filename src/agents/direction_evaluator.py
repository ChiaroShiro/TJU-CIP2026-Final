from typing import Dict, List, Optional
import json

from ..core.llm import LLMClient
from ..core.config import Settings
from ..core.models import PaperItem
from ..core.utils import extract_json_object
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher


class DirectionEvaluator:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()

    def evaluate_direction(
        self,
        direction: str,
        queries: Optional[List[str]] = None,
        on_event=None,
    ) -> Dict:
        """
        评估研究方向。

        Args:
            direction: 用户的原始研究方向描述（任意语言）
            queries:   可选，预先提炼好的英文检索关键词（由上游路由Agent提供）。
                       若为空，本方法会自己提炼。
            on_event:  可选进度回调（phase/log 事件）。
        """
        emit = on_event if on_event is not None else (lambda e: None)

        # 1. 获取检索关键词
        if queries is None or not queries:
            emit({"type": "phase", "label": "提炼英文检索词", "pct": 10})
            queries = self._extract_queries(direction)
        emit({"type": "log", "level": "info", "text": "检索词：" + " / ".join(queries)})

        # 2. 用每个查询分别搜索，合并去重
        all_papers: List[PaperItem] = []
        seen_titles = set()
        for qi, q in enumerate(queries):
            emit({"type": "phase", "label": f"检索论文：{q}",
                  "pct": 20 + int(50 * (qi + 1) / max(1, len(queries)))})
            papers_arxiv = self.arxiv.search(q, max_results=5)
            papers_s2 = self.s2.search(q, max_results=3)
            for p in papers_arxiv + papers_s2:
                if p.title and p.title not in seen_titles:
                    seen_titles.add(p.title)
                    all_papers.append(p)
        emit({"type": "log", "level": "info", "text": f"共检索到 {len(all_papers)} 篇相关论文"})

        # 3. 调 LLM 评估
        emit({"type": "phase", "label": "LLM 多维评分中", "pct": 80})
        papers_summary = self._format_papers(all_papers[:12])
        prompt = self._build_evaluation_prompt(direction, queries, papers_summary)
        response = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.3)

        result = extract_json_object(response) or {}
        if not result:
            result = {
                "feasibility": 0.5, "novelty": 0.5, "impact": 0.5,
                "analysis": response,
            }

        result["papers"] = all_papers[:12]
        result["search_queries"] = queries
        return result

    # ------------------------------------------------------------------ #
    # 查询提炼
    # ------------------------------------------------------------------ #

    def _extract_queries(self, direction: str) -> List[str]:
        """
        用 LLM 把用户的长描述（可能是中文）提炼为 3-5 个简短英文检索关键词。
        这样对 arXiv / Semantic Scholar 更友好，也避免 URL 过长触发 429。
        """
        prompt = f"""
Extract 3-5 concise English keyword queries for academic paper search from the user's research idea below.

Requirements:
- each query: 3-8 English keywords, no full sentences
- cover different angles of the idea (core topic, key technique, benchmark, etc.)
- prefer canonical academic terms
- NO Chinese characters, NO quotes, NO punctuation except spaces

User's research idea:
{direction}

Return JSON only:
{{"queries": ["query1", "query2", "query3"]}}
""".strip()
        try:
            raw = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.1)
            data = extract_json_object(raw)
            queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
            if queries:
                return queries[:5]
        except Exception as e:
            print(f"[DirectionEvaluator] query extraction failed: {e}")

        # 降级：截断原始输入到 100 字符作为唯一 query
        return [direction[:100].strip()]

    # ------------------------------------------------------------------ #
    # 格式化与提示词
    # ------------------------------------------------------------------ #

    def _format_papers(self, papers: List[PaperItem]) -> str:
        if not papers:
            return "(no papers retrieved — external search may be rate-limited)"
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.title}")
            lines.append(f"   Authors: {', '.join(p.authors[:3])}")
            lines.append(f"   Abstract: {(p.abstract or '')[:200]}...")
            lines.append(f"   URL: {p.url}\n")
        return "\n".join(lines)

    def _build_evaluation_prompt(
        self, direction: str, queries: List[str], papers_summary: str
    ) -> str:
        queries_text = "\n".join(f"  - {q}" for q in queries)
        return f"""
你是一位资深的学术研究顾问。请评估以下研究方向的可行性和价值。

研究方向:
{direction}

用于检索的英文关键词:
{queries_text}

相关已有论文:
{papers_summary}

请从以下维度评估(0-1分):
1. feasibility (可行性): 技术上是否可实现，资源需求是否合理
2. novelty (新颖性): 是否有创新点，与现有工作的区别
3. impact (影响力): 潜在的学术/应用价值

返回JSON格式:
{{
  "feasibility": 0.0-1.0,
  "novelty": 0.0-1.0,
  "impact": 0.0-1.0,
  "analysis": "详细分析文本，包括: 1)现有研究现状总结 2)该方向的优势和挑战 3)建议的研究切入点 4)是否已有相关benchmark",
  "recommendations": ["建议1", "建议2", "建议3"],
  "related_topics": ["相关主题1", "相关主题2"],
  "benchmarks": ["已知相关benchmark名称1", "名称2"]
}}
""".strip()
