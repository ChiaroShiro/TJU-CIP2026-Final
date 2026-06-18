"""
PaperAnalyzer - paper reading and note generation.

Modes:
- analyze(): conservative arXiv reading, text first
- analyze_multimodal(): arXiv reading with optional figure interpretation
- analyze_local_pdf(): conservative local PDF reading
"""

import base64
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI

from ..core.config import Settings
from ..core.llm import LLMClient
from ..core.models import PaperItem
from ..core.utils import extract_json_object
from ..memory.memory_manager import MemoryManager
from ..services.paper_fetcher import PaperFetcher, PaperFullText
from ..services.paper_figure_fetcher import FigureRef, PaperFigureFetcher


logger = logging.getLogger(__name__)

SECTION_ORDER = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "discussion",
    "conclusion",
]

CHUNK_CHAR_LIMIT = 6000
NOTES_SUBDIR = "paper_notes"
ASSETS_SUBDIR = "assets"
MAX_VISION_FIGURES = 6
MAX_REFERENCE_CHARS = 18000
MIN_MEMORY_CONTEXT_CONFIDENCE = 0.58
MIN_MEMORY_CONNECTION_CONFIDENCE = 0.62


class PaperAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        settings: Settings,
        memory: Optional[MemoryManager] = None,
    ):
        self.llm = llm_client
        self.settings = settings
        self.memory = memory
        self.fetcher = PaperFetcher(cache_dir=settings.workspace_dir / "pdf_cache")
        self.notes_dir = settings.workspace_dir / NOTES_SUBDIR
        self.assets_dir = self.notes_dir / ASSETS_SUBDIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        paper: PaperItem,
        focus: Optional[str] = None,
        use_fulltext: bool = True,
        on_event=None,
    ) -> Dict:
        """Conservative arXiv reading mode for research and survey use."""
        emit = on_event if on_event is not None else (lambda e: None)
        if use_fulltext:
            emit({"type": "phase", "label": "下载并解析 PDF 全文", "pct": 8})
            fulltext = self.fetcher.fetch_fulltext(paper.paper_id)
            if fulltext and fulltext.num_chars > 500:
                result = self._analyze_fulltext(
                    paper,
                    fulltext,
                    focus,
                    figures=[],
                    output_language="zh",
                    include_cited_similar_work=False,
                    include_memory_context=False,
                    on_event=on_event,
                )
                emit({"type": "phase", "label": "抓取关键图表", "pct": 88})
                figures = self._fetch_figures(paper)
                result["_num_figures"] = len(figures)
                result["_analysis_mode"] = "conservative"
                result["_focus"] = focus or ""
                emit({"type": "phase", "label": "保存笔记与记忆", "pct": 96})
                self._save_note_and_memory(paper, result, figures)
                return result

        emit({"type": "log", "level": "warn", "text": "全文不可用，降级为基于摘要的保守分析"})
        result = self._analyze_abstract(paper, focus, output_language="zh")
        result["_analysis_mode"] = "conservative"
        result["_focus"] = focus or ""
        self._save_note_and_memory(paper, result, figures=[])
        return result

    def analyze_multimodal(
        self,
        paper: PaperItem,
        focus: Optional[str] = None,
        use_fulltext: bool = True,
        on_event=None,
    ) -> Dict:
        """Detailed reading mode for arXiv papers, with optional figure interpretation."""
        emit = on_event if on_event is not None else (lambda e: None)
        if use_fulltext:
            emit({"type": "phase", "label": "下载并解析 PDF 全文", "pct": 8})
            fulltext = self.fetcher.fetch_fulltext(paper.paper_id)
            if fulltext and fulltext.num_chars > 500:
                result = self._analyze_fulltext(
                    paper,
                    fulltext,
                    focus,
                    figures=[],
                    output_language="zh",
                    include_cited_similar_work=True,
                    include_memory_context=True,
                    on_event=on_event,
                )
                emit({"type": "phase", "label": "抓取关键图表", "pct": 82})
                figures = self._fetch_figures(paper)
                result["_num_figures"] = len(figures)
                result["_analysis_mode"] = "multimodal"
                result["_focus"] = focus or ""
                if figures:
                    emit({"type": "phase", "label": f"多模态解读 {len(figures)} 张图", "pct": 90})
                    result["figures_interpretation"] = self._describe_figures_with_vision(
                        paper=paper,
                        analysis=result,
                        figures=figures,
                        focus=focus,
                    )
                emit({"type": "phase", "label": "保存笔记与记忆", "pct": 96})
                self._save_note_and_memory(paper, result, figures)
                return result

        emit({"type": "log", "level": "warn", "text": "全文不可用，降级为基于摘要的分析"})
        result = self._analyze_abstract(paper, focus, output_language="zh")
        result["_analysis_mode"] = "multimodal"
        result["_focus"] = focus or ""
        self._save_note_and_memory(paper, result, figures=[])
        return result

    def analyze_local_pdf(
        self,
        pdf_path: Union[str, Path],
        title: Optional[str] = None,
        focus: Optional[str] = None,
        on_event=None,
    ) -> Dict:
        """Conservative local PDF reading mode."""
        emit = on_event if on_event is not None else (lambda e: None)
        path = Path(pdf_path).expanduser().resolve()
        emit({"type": "phase", "label": "解析本地 PDF", "pct": 8})
        inferred_title = (title or "").strip() or self.fetcher.infer_title_from_pdf(path)
        paper = PaperItem(
            paper_id=path.stem,
            title=inferred_title,
            authors=[],
            abstract="",
            url=str(path),
            published="",
        )

        fulltext = self.fetcher.fetch_local_fulltext(path, paper_id=paper.paper_id)
        if fulltext and fulltext.num_chars > 500:
            result = self._analyze_fulltext(
                paper,
                fulltext,
                focus,
                figures=[],
                output_language="zh",
                include_cited_similar_work=True,
                include_memory_context=True,
                on_event=on_event,
            )
            result["_source"] = "local_pdf"
            result["_analysis_mode"] = "conservative"
            result["_focus"] = focus or ""
            emit({"type": "phase", "label": "保存笔记与记忆", "pct": 96})
            self._save_note_and_memory(paper, result, figures=[])
            return result

        result = {
            "tldr": "",
            "problem": "",
            "contributions": [],
            "method_summary": "",
            "formulas": [],
            "datasets": [],
            "results": "",
            "strengths": [],
            "weaknesses": ["无法从本地 PDF 中提取足够文本。"],
            "tags": [],
            "relevance_score": 0.0,
            "cited_similar_work": [],
            "memory_connections": [],
            "_source": "local_pdf_failed",
            "_analysis_mode": "conservative",
            "_focus": focus or "",
        }
        self._save_note_and_memory(paper, result, figures=[])
        return result

    def _save_note_and_memory(
        self,
        paper: PaperItem,
        analysis: Dict,
        figures: List[FigureRef],
    ) -> None:
        self._save_note(paper, analysis, figures)
        if self.memory is not None:
            try:
                self.memory.save_paper_note(paper.paper_id, paper.title, analysis)
            except Exception as exc:
                # 保持 best-effort（不中断笔记产出），但不再静默吞掉
                logger.warning("保存论文图谱失败 (%s): %s", paper.paper_id, exc)

    def _fetch_figures(self, paper: PaperItem) -> List[FigureRef]:
        method_name = self._extract_method_name_from_title(paper.title)
        fig_fetcher = PaperFigureFetcher(self.assets_dir, method_name=method_name)
        figures = fig_fetcher.extract_figures(paper.paper_id)
        if figures:
            fig_fetcher.localize_unreachable(figures)
        return figures

    def _analyze_fulltext(
        self,
        paper: PaperItem,
        fulltext: PaperFullText,
        focus: Optional[str],
        figures: List[FigureRef],
        output_language: str,
        include_cited_similar_work: bool,
        include_memory_context: bool,
        on_event=None,
    ) -> Dict:
        emit = on_event if on_event is not None else (lambda e: None)
        chunks = self._build_chunks(fulltext)
        total_chunks = len(chunks)
        emit({"type": "log", "level": "info",
              "text": f"全文 {fulltext.num_pages} 页，切分为 {total_chunks} 个章节块"})
        chunk_summaries: List[Dict] = []
        for ci, (chunk_name, chunk_text) in enumerate(chunks):
            emit({"type": "phase",
                  "label": f"逐章分析 ({ci + 1}/{total_chunks})：{chunk_name}",
                  "pct": 12 + int(58 * (ci + 1) / max(1, total_chunks))})
            summary = self._analyze_chunk(paper, chunk_name, chunk_text, focus, output_language)
            chunk_summaries.append({"section": chunk_name, "summary": summary})

        emit({"type": "phase", "label": "综合提炼结构化笔记", "pct": 74})
        memory_context = self._collect_memory_context(paper, focus) if include_memory_context else {}
        final = self._synthesize(
            paper,
            chunk_summaries,
            focus,
            figures,
            output_language,
            memory_context=memory_context,
        )
        final["_source"] = "fulltext"
        final["_num_pages"] = fulltext.num_pages
        final["_num_chunks"] = len(chunk_summaries)
        final["_num_figures"] = len(figures)
        final["_section_summaries"] = chunk_summaries

        references_text = (fulltext.sections or {}).get("references", "").strip()
        if include_cited_similar_work and references_text:
            emit({"type": "phase", "label": "梳理引用中的相似工作", "pct": 79})
            final["cited_similar_work"] = self._extract_cited_similar_work(
                paper=paper,
                analysis=final,
                references_text=references_text,
                focus=focus,
                output_language=output_language,
            )
        else:
            final["cited_similar_work"] = []

        if include_memory_context:
            final["memory_connections"] = self._interpret_memory_connections(
                paper=paper,
                analysis=final,
                memory_context=memory_context,
                output_language=output_language,
            )
        else:
            final["memory_connections"] = []
        final["_memory_context_confidence"] = memory_context.get("aggregate_confidence", 0.0) if memory_context else 0.0

        return final

    def _collect_memory_context(self, paper: PaperItem, focus: Optional[str]) -> Dict:
        if self.memory is None:
            return {"papers": [], "edges": [], "aggregate_confidence": 0.0, "has_confident_context": False}
        query = focus or paper.title or paper.paper_id
        try:
            context = self.memory.get_paper_graph_context(query, top_k=5)
            if not context.get("has_confident_context", False):
                return {
                    "papers": [],
                    "edges": [],
                    "aggregate_confidence": context.get("aggregate_confidence", 0.0),
                    "has_confident_context": False,
                }
            return context
        except Exception:
            return {"papers": [], "edges": [], "aggregate_confidence": 0.0, "has_confident_context": False}

    def _build_chunks(self, ft: PaperFullText) -> List[Tuple[str, str]]:
        chunks: List[Tuple[str, str]] = []
        if ft.sections:
            for key in SECTION_ORDER:
                text = ft.sections.get(key)
                if not text:
                    continue
                if len(text) <= CHUNK_CHAR_LIMIT:
                    chunks.append((key, text))
                else:
                    for i, sub in enumerate(self._split_by_chars(text, CHUNK_CHAR_LIMIT)):
                        chunks.append((f"{key}_part{i + 1}", sub))
        if not chunks:
            for i, sub in enumerate(self._split_by_chars(ft.raw_text, CHUNK_CHAR_LIMIT)):
                chunks.append((f"chunk_{i + 1}", sub))
        return chunks

    @staticmethod
    def _split_by_chars(text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]
        parts: List[str] = []
        paragraphs = text.split("\n\n")
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 > limit and buf:
                parts.append(buf.strip())
                buf = para
            else:
                buf += ("\n\n" if buf else "") + para
        if buf:
            parts.append(buf.strip())
        return parts

    def _analyze_chunk(
        self,
        paper: PaperItem,
        section_name: str,
        chunk_text: str,
        focus: Optional[str],
        output_language: str,
    ) -> str:
        focus_line = f"\nUser focus: {focus}" if focus else ""
        language_line = (
            "\nWrite the extraction in simplified Chinese."
            if output_language == "zh"
            else "\nWrite the extraction in English."
        )
        prompt = f"""You are reading the {section_name} section of "{paper.title}".
Extract ALL key information following these zero-omission rules.{focus_line}{language_line}

ZERO-OMISSION RULES:
- Every formula in this section with full LaTeX, not pseudocode
- Every important table mentioned
- Every figure reference with what it shows
- Use specific numbers, not vague claims
- Keep technical terms as plain text, without wiki-link markup

FORMULA QUALITY CHECKS:
1. No variable name collision
2. Keep every operator mentioned in prose
3. Match the paper's symbol convention
4. Keep exact sum/integral bounds
5. Do not silently drop operators

Section content:
{chunk_text}

Output:
- Structured Markdown extraction
- 200-500 words
- Preserve quantitative details verbatim
- Use concise but information-dense phrasing""".strip()

        return self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        ).strip()

    def _synthesize(
        self,
        paper: PaperItem,
        chunk_summaries: List[Dict],
        focus: Optional[str],
        figures: List[FigureRef],
        output_language: str,
        memory_context: Optional[Dict] = None,
    ) -> Dict:
        combined = "\n\n".join(
            f"## {item['section']}\n{item['summary']}" for item in chunk_summaries
        )
        focus_line = f"\nUser focus: {focus}" if focus else ""
        figure_count = len(figures)
        figure_hint = (
            f"\n\nThe paper has {figure_count} extracted figures. "
            "If useful, summarize them in 'figures_interpretation'."
            if figure_count
            else ""
        )
        language_line = (
            "\nWrite all natural-language fields in simplified Chinese."
            if output_language == "zh"
            else "\nWrite all natural-language fields in English."
        )
        memory_block = ""
        if (
            memory_context
            and memory_context.get("has_confident_context", False)
            and memory_context.get("aggregate_confidence", 0.0) >= MIN_MEMORY_CONTEXT_CONFIDENCE
            and memory_context.get("papers")
        ):
            paper_lines = []
            for item in memory_context.get("papers", [])[:5]:
                paper_lines.append(
                    f"- {item.get('title', '')}: {item.get('tldr', '') or item.get('problem', '')} "
                    f"[confidence={item.get('confidence', 0.0):.2f}]"
                )
            edge_lines = []
            for edge in memory_context.get("edges", [])[:8]:
                edge_lines.append(
                    f"- {edge.get('src_paper_id', '')} -> {edge.get('dst_paper_id', '')}: "
                    f"{edge.get('relation_type', '')} ({edge.get('evidence', '')}) "
                    f"[confidence={edge.get('confidence', 0.0):.2f}]"
                )
            memory_block = (
                "\n\nAlready-read related papers from memory:\n"
                + "\n".join(paper_lines)
                + ("\nKnown relations:\n" + "\n".join(edge_lines) if edge_lines else "")
            )

        prompt = f"""Based on the section-by-section analysis of "{paper.title}", produce a final structured analysis.{focus_line}{figure_hint}{language_line}{memory_block}

Section analyses:
{combined}

Return JSON with this exact schema:
{{
  "tldr": "Single sentence core contribution",
  "problem": "What concrete problem does the paper solve?",
  "prior_limitations": "What was broken or missing in prior work?",
  "motivation": "Why the authors believe their approach addresses the gap",
  "contributions": ["Contribution 1 headline - what + why it matters", "..."],
  "method_summary": "Architecture or approach overview (200-400 words, plain text only)",
  "modules": [
    {{"name": "Module name", "motivation": "...", "design": "..."}}
  ],
  "formulas": [
    {{"name": "Formula short name", "latex": "exact LaTeX", "meaning": "one-line description", "symbols": {{"x": "meaning"}}}}
  ],
  "datasets": [
    {{"name": "...", "size": "...", "used_for": "train/eval/both", "notes": "..."}}
  ],
  "implementation": {{"backbone": "...", "optimizer": "...", "lr": "...", "batch_size": "...", "epochs": "...", "hardware": "..."}},
  "results": "Main results with specific numbers",
  "ablations": "Key ablation findings with numbers",
  "figures_interpretation": [
    {{"index": 1, "what_it_shows": "description", "why_it_matters": "..."}}
  ],
  "strengths": ["Specific strength citing evidence"],
  "weaknesses": ["Specific weakness citing evidence"],
  "reproducibility": {{"code": true, "weights": false, "details_sufficient": true, "data_public": true}},
  "related_work": ["Prior Work 1 - relationship", "Prior Work 2 - relationship"],
  "future_work": "What the authors propose next",
  "tags": ["tag1", "tag2", "tag3"],
  "relevance_score": 0.0
}}

Critical rules:
- Use plain text only. Never use wiki links like [[...]].
- Formulas must include all important formulas from the section analyses.
- Numbers, dataset names, and baselines must be accurate.
- Strengths and weaknesses must be concrete.
- Modules: list 2-5 most important sub-modules only.""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = extract_json_object(response)
        if not result:
            return {
                "tldr": "",
                "problem": "",
                "contributions": [],
                "method_summary": response,
                "formulas": [],
                "datasets": [],
                "results": "",
                "strengths": [],
                "weaknesses": [],
                "tags": [],
                "relevance_score": 0.0,
            }
        return result

    def _extract_cited_similar_work(
        self,
        paper: PaperItem,
        analysis: Dict,
        references_text: str,
        focus: Optional[str],
        output_language: str,
    ) -> List[Dict]:
        focus_line = f"\nUser focus: {focus}" if focus else ""
        language_line = (
            "\nWrite all natural-language fields in simplified Chinese."
            if output_language == "zh"
            else "\nWrite all natural-language fields in English."
        )
        clipped_references = references_text[:MAX_REFERENCE_CHARS]

        prompt = f"""You are building a literature-survey helper section for the paper "{paper.title}".{focus_line}{language_line}

Current paper summary:
- Problem: {analysis.get('problem', '')}
- Prior limitations: {analysis.get('prior_limitations', '')}
- Motivation: {analysis.get('motivation', '')}
- Method summary: {analysis.get('method_summary', '')[:1800]}
- Main contributions: {analysis.get('contributions', [])}
- Related work summary: {analysis.get('related_work', [])}

References section text:
{clipped_references}

Task:
Identify the 4-8 cited works that are most likely to be directly similar, foundational, or strongest baselines for this paper's method direction.

Return JSON only:
{{
  "cited_similar_work": [
    {{
      "title": "paper title or short citation if exact title unavailable",
      "category": "baseline/foundation/competing method/closest prior work",
      "why_related": "why it is relevant to the current paper",
      "difference_vs_this_paper": "how the current paper differs or advances",
      "priority": 1
    }}
  ]
}}

Rules:
- Prefer truly related works, not generic references.
- If the exact title is unclear, keep the shortest identifiable citation text.
- Be conservative and avoid hallucinating details absent from the references and summary.
- Rank with priority 1 as most important.
- Keep each item concise but useful for fast survey.""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = extract_json_object(response) or {}
        items = result.get("cited_similar_work", [])
        if not isinstance(items, list):
            return []

        cleaned: List[Dict] = []
        for idx, item in enumerate(items, 1):
            if isinstance(item, str):
                cleaned.append(
                    {
                        "title": item,
                        "category": "",
                        "why_related": "",
                        "difference_vs_this_paper": "",
                        "priority": idx,
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "category": str(item.get("category", "")).strip(),
                    "why_related": str(item.get("why_related", "")).strip(),
                    "difference_vs_this_paper": str(item.get("difference_vs_this_paper", "")).strip(),
                    "priority": item.get("priority", idx),
                }
            )

        cleaned = [item for item in cleaned if item.get("title")]
        cleaned.sort(key=lambda x: int(x.get("priority", 999)))
        return cleaned[:8]

    def _interpret_memory_connections(
        self,
        paper: PaperItem,
        analysis: Dict,
        memory_context: Dict,
        output_language: str,
    ) -> List[Dict]:
        if not memory_context:
            return []
        if not memory_context.get("has_confident_context", False):
            return []
        if memory_context.get("aggregate_confidence", 0.0) < MIN_MEMORY_CONTEXT_CONFIDENCE:
            return []

        papers = memory_context.get("papers", [])
        if not papers:
            return []

        language_line = (
            "Write all natural-language fields in simplified Chinese."
            if output_language == "zh"
            else "Write all natural-language fields in English."
        )
        related_summary = "\n".join(
            f"- {item.get('title', '')}: {item.get('tldr', '') or item.get('problem', '')}"
            for item in papers[:5]
        )
        prompt = f"""You are relating a new paper to already-read papers in memory.
{language_line}

Current paper:
- Title: {paper.title}
- Problem: {analysis.get('problem', '')}
- Method: {analysis.get('method_summary', '')[:1600]}
- Contributions: {analysis.get('contributions', [])}

Already-read papers:
{related_summary}

Return JSON only:
{{
  "memory_connections": [
    {{
      "title": "already-read paper title",
      "relation": "same line / extends / contrasts with / baseline family / nearby topic",
      "why_relevant": "why this remembered paper helps interpret the current one",
      "difference": "main difference from the current paper",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Only use papers listed above.
- Keep 2-5 most helpful links.
- Be conservative and concise.
- Confidence must be between 0 and 1.
- If a link is weak or speculative, set confidence below 0.60.""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = extract_json_object(response) or {}
        items = result.get("memory_connections", [])
        if not isinstance(items, list):
            return []

        cleaned: List[Dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            cleaned.append(
                {
                    "title": title,
                    "relation": str(item.get("relation", "")).strip(),
                    "why_relevant": str(item.get("why_relevant", "")).strip(),
                    "difference": str(item.get("difference", "")).strip(),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
            )
        cleaned = [
            item for item in cleaned
            if item.get("confidence", 0.0) >= MIN_MEMORY_CONNECTION_CONFIDENCE
        ]
        cleaned.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return cleaned[:5]

    def _analyze_abstract(
        self,
        paper: PaperItem,
        focus: Optional[str],
        output_language: str,
    ) -> Dict:
        focus_line = f"\nFocus: {focus}" if focus else ""
        language_line = (
            "\nWrite all natural-language fields in simplified Chinese."
            if output_language == "zh"
            else "\nWrite all natural-language fields in English."
        )
        prompt = f"""Analyze this paper based on its abstract only (full text unavailable).{focus_line}{language_line}

Title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Published: {paper.published}
Abstract: {paper.abstract}

Return JSON:
{{
  "tldr": "Single sentence core contribution",
  "problem": "Core problem solved",
  "contributions": ["contribution 1", "contribution 2"],
  "method_summary": "Method overview based on abstract",
  "formulas": [],
  "datasets": [],
  "results": "Results mentioned in abstract",
  "strengths": ["Based on abstract"],
  "weaknesses": ["Cannot fully assess without full text"],
  "tags": ["tag1", "tag2"],
  "relevance_score": 0.0
}}""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = extract_json_object(response)
        if not result:
            result = {"tldr": response, "contributions": [], "tags": []}
        result["_source"] = "abstract_only"
        result["cited_similar_work"] = []
        result["memory_connections"] = []
        return result

    def _describe_figures_with_vision(
        self,
        paper: PaperItem,
        analysis: Dict,
        figures: List[FigureRef],
        focus: Optional[str],
    ) -> List[Dict]:
        usable_figures = [fig for fig in figures if fig.local_path or fig.url][:MAX_VISION_FIGURES]
        if not usable_figures:
            return []

        try:
            client = OpenAI(
                api_key=self.settings.vision_api_key,
                base_url=self.settings.vision_base_url,
            )
        except Exception:
            return []

        results: List[Dict] = []
        for fig in usable_figures:
            content: List[Dict] = [
                {
                    "type": "text",
                    "text": self._build_vision_prompt(
                        paper=paper,
                        analysis=analysis,
                        fig=fig,
                        focus=focus,
                    ),
                }
            ]
            image_part = self._build_image_message_part(fig)
            if not image_part:
                continue
            content.append(image_part)

            try:
                response = client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                )
                raw = self.llm._extract_text_response(response)
                data = extract_json_object(raw) or {}
                if data:
                    results.append(
                        {
                            "index": fig.index,
                            "what_it_shows": data.get("what_it_shows", ""),
                            "why_it_matters": data.get("why_it_matters", ""),
                        }
                    )
            except Exception:
                continue
        return results

    def _build_vision_prompt(
        self,
        paper: PaperItem,
        analysis: Dict,
        fig: FigureRef,
        focus: Optional[str],
    ) -> str:
        focus_line = f"\n用户关注点: {focus}" if focus else ""
        return f"""你正在为论文笔记补充图表解读。
论文标题: {paper.title}
图编号: Figure {fig.index}
图注: {fig.caption}{focus_line}

已生成的正文摘要:
- 核心问题: {analysis.get('problem', '')}
- 方法概述: {analysis.get('method_summary', '')[:1200]}
- 结果概述: {analysis.get('results', '')[:800]}

请直接观察图片内容，并返回 JSON:
{{
  "what_it_shows": "这张图具体展示了什么",
  "why_it_matters": "它为什么重要，与论文主线有什么关系"
}}

要求:
- 用简体中文
- 只说看得见且与论文主线有关的内容
- 不要臆测图中无法确认的数值
- 只返回 JSON，不要返回别的文字""".strip()

    def _build_image_message_part(self, fig: FigureRef) -> Optional[Dict]:
        if fig.local_path:
            path = Path(fig.local_path)
            if not path.exists():
                return None
            mime = self._guess_mime_type(path)
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }

        if fig.url:
            return {
                "type": "image_url",
                "image_url": {"url": fig.url},
            }
        return None

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".gif":
            return "image/gif"
        return "image/png"

    def _save_note(
        self,
        paper: PaperItem,
        analysis: Dict,
        figures: List[FigureRef],
    ) -> Optional[Path]:
        method_name = self._extract_method_name(paper, analysis)
        path = self.notes_dir / f"{method_name}.md"

        now = datetime.utcnow().strftime("%Y-%m-%d")
        tags = analysis.get("tags", [])
        image_source = self._compute_image_source(figures)
        read_mode = "full" if analysis.get("_source") in {"fulltext", "local_pdf"} else "abstract"

        author_text = ", ".join(author for author in paper.authors[:5] if author)
        frontmatter = (
            f"---\n"
            f"title: \"{paper.title}\"\n"
            f"method_name: \"{method_name}\"\n"
            f"arxiv_id: \"{paper.paper_id}\"\n"
            f"authors: [{author_text}]\n"
            f"year: {paper.published[:4] if paper.published else 'unknown'}\n"
            f"tags: [{', '.join(tags)}]\n"
            f"read_mode: {read_mode}\n"
            f"image_source: {image_source}\n"
            f"analysis_mode: {analysis.get('_analysis_mode', 'conservative')}\n"
            f"created: {now}\n"
            f"---\n\n"
        )

        body = self._render_note_body(paper, analysis, figures)
        path.write_text(frontmatter + body, encoding="utf-8")

        analysis["_note_path"] = str(path)
        analysis["_method_name"] = method_name
        analysis["_image_source"] = image_source
        return path

    def _render_note_body(
        self,
        paper: PaperItem,
        analysis: Dict,
        figures: List[FigureRef],
    ) -> str:
        parts = [f"# {paper.title}\n"]

        if analysis.get("tldr"):
            parts.append(f"## 一句话总结\n\n> {analysis['tldr']}\n")

        if analysis.get("problem") or analysis.get("prior_limitations") or analysis.get("motivation"):
            parts.append("## 问题背景\n")
            if analysis.get("problem"):
                parts.append(f"### 要解决的问题\n{analysis['problem']}\n")
            if analysis.get("prior_limitations"):
                parts.append(f"### 现有方法的不足\n{analysis['prior_limitations']}\n")
            if analysis.get("motivation"):
                parts.append(f"### 本文的动机\n{analysis['motivation']}\n")

        if analysis.get("contributions"):
            parts.append("## 核心贡献\n")
            for i, item in enumerate(analysis["contributions"], 1):
                parts.append(f"{i}. {item}")
            parts.append("")

        if analysis.get("method_summary"):
            parts.append("## 方法详解\n")
            parts.append(f"### 整体思路\n\n{analysis['method_summary']}\n")
            modules = analysis.get("modules") or []
            if modules:
                parts.append("### 关键模块\n")
                for i, module in enumerate(modules, 1):
                    parts.append(f"#### 模块{i}: {module.get('name', '')}")
                    if module.get("motivation"):
                        parts.append(f"- 设计动机: {module['motivation']}")
                    if module.get("design"):
                        parts.append(f"- 具体设计: {module['design']}")
                parts.append("")

        memory_connections = analysis.get("memory_connections") or []
        if memory_connections:
            parts.append("## 与记忆库中已读论文的关系\n")
            parts.append("这部分用于把当前论文放回你已经读过的论文网络里，帮助判断它属于哪条方法线、补的是哪类空白。\n")
            for item in memory_connections:
                parts.append(f"### {item.get('title', '')}")
                if item.get("relation"):
                    parts.append(f"- 关系: {item.get('relation', '')}")
                if item.get("why_relevant"):
                    parts.append(f"- 为什么值得对照: {item.get('why_relevant', '')}")
                if item.get("difference"):
                    parts.append(f"- 主要差异: {item.get('difference', '')}")
                if item.get("confidence", 0.0):
                    parts.append(f"- 置信度: {item.get('confidence', 0.0):.2f}")
                parts.append("")

        if analysis.get("formulas"):
            parts.append("## 关键公式\n")
            for i, formula in enumerate(analysis["formulas"], 1):
                parts.append(f"### 公式{i}: {formula.get('name', '')}\n")
                parts.append(f"$$\n{formula.get('latex', '')}\n$$\n")
                if formula.get("meaning"):
                    parts.append(f"含义: {formula.get('meaning', '')}\n")
                symbols = formula.get("symbols") or {}
                if symbols:
                    parts.append("符号说明:")
                    for symbol, meaning in symbols.items():
                        parts.append(f"- `{symbol}`: {meaning}")
                    parts.append("")

        if analysis.get("datasets"):
            parts.append("## 数据集与实验设置\n")
            parts.append("| 数据集 | 规模 | 用途 | 备注 |")
            parts.append("| --- | --- | --- | --- |")
            for dataset in analysis["datasets"]:
                if isinstance(dataset, dict):
                    parts.append(
                        f"| {dataset.get('name', '')} | {dataset.get('size', '')} | "
                        f"{dataset.get('used_for', '')} | {dataset.get('notes', '')} |"
                    )
                else:
                    parts.append(f"| {dataset} |  |  |  |")
            parts.append("")

        implementation = analysis.get("implementation") or {}
        if implementation:
            parts.append("### 实现细节\n")
            for key, value in implementation.items():
                if value:
                    parts.append(f"- {key}: {value}")
            parts.append("")

        if analysis.get("results"):
            parts.append(f"## 主要结果\n\n{analysis['results']}\n")

        if analysis.get("ablations"):
            parts.append(f"## 消融与分析\n\n{analysis['ablations']}\n")

        cited_similar_work = analysis.get("cited_similar_work") or []
        if cited_similar_work:
            parts.append("## 引用中的相似工作\n")
            parts.append("这部分用于快速建立这个方向的最小文献地图，优先列出与本文最接近、最基础或最强对比的工作。\n")
            for item in cited_similar_work:
                title = item.get("title", "")
                category = item.get("category", "")
                why_related = item.get("why_related", "")
                difference = item.get("difference_vs_this_paper", "")
                header = f"### {title}"
                if category:
                    header += f" [{category}]"
                parts.append(header)
                if why_related:
                    parts.append(f"- 相关性: {why_related}")
                if difference:
                    parts.append(f"- 与本文差异: {difference}")
                parts.append("")

        if figures:
            parts.append("## 关键图表\n")
            interpretations = {
                item.get("index"): item
                for item in (analysis.get("figures_interpretation") or [])
                if isinstance(item, dict)
            }
            for fig in figures:
                parts.append(f"### Figure {fig.index}: {fig.caption}\n")
                if fig.local_path:
                    image_path = Path(fig.local_path).name
                    parts.append(f"![Figure {fig.index}](assets/{image_path})\n")
                elif fig.url:
                    parts.append(f"![Figure {fig.index}]({fig.url})\n")
                interp = interpretations.get(fig.index, {})
                if interp.get("what_it_shows"):
                    parts.append(f"- 图像内容: {interp.get('what_it_shows', '')}")
                if interp.get("why_it_matters"):
                    parts.append(f"- 作用: {interp.get('why_it_matters', '')}")
                parts.append("")

        if analysis.get("strengths") or analysis.get("weaknesses"):
            parts.append("## 批判性分析\n")
            if analysis.get("strengths"):
                parts.append("### 优点")
                for item in analysis["strengths"]:
                    parts.append(f"- {item}")
                parts.append("")
            if analysis.get("weaknesses"):
                parts.append("### 局限")
                for item in analysis["weaknesses"]:
                    parts.append(f"- {item}")
                parts.append("")

        reproducibility = analysis.get("reproducibility") or {}
        if reproducibility:
            parts.append("## 可复现性评估\n")
            parts.append(f"- [{'x' if reproducibility.get('code') else ' '}] 提供代码")
            parts.append(f"- [{'x' if reproducibility.get('weights') else ' '}] 提供权重")
            parts.append(f"- [{'x' if reproducibility.get('details_sufficient') else ' '}] 训练细节充分")
            parts.append(f"- [{'x' if reproducibility.get('data_public') else ' '}] 数据可公开获取")
            parts.append("")

        if analysis.get("related_work"):
            parts.append("## 论文中提到的相关工作\n")
            for item in analysis["related_work"]:
                parts.append(f"- {item}")
            parts.append("")

        if analysis.get("future_work"):
            parts.append(f"## 未来方向\n\n{analysis['future_work']}\n")

        if analysis.get("_section_summaries"):
            parts.append("## 分章节摘录\n")
            for item in analysis["_section_summaries"]:
                parts.append(f"### {item.get('section', '')}\n")
                parts.append(f"{item.get('summary', '')}\n")

        parts.append("## 速查卡片\n")
        parts.append(f"- 论文: {paper.title}")
        parts.append(f"- 核心结论: {analysis.get('tldr', '')}")
        parts.append(f"- 来源: {paper.url}")
        parts.append(f"- 阅读模式: {analysis.get('_analysis_mode', 'conservative')}")
        parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _compute_image_source(figures: List[FigureRef]) -> str:
        if not figures:
            return "none"
        has_local = any(fig.local_path for fig in figures)
        has_online = any(fig.url and not fig.local_path for fig in figures)
        if has_local and has_online:
            return "mixed"
        if has_local:
            return "local"
        return "online"

    def _extract_method_name(self, paper: PaperItem, analysis: Dict) -> str:
        title = paper.title or ""
        if ":" in title:
            candidate = title.split(":")[0].strip()
            if 2 <= len(candidate) <= 40 and not self._looks_like_id(candidate):
                return self._sanitize_filename(candidate)
        if analysis.get("contributions"):
            first = str(analysis["contributions"][0]).strip()
            if first and 2 <= len(first) <= 40 and not self._looks_like_id(first):
                return self._sanitize_filename(first)
        if title and not self._looks_like_id(title):
            return self._sanitize_filename(title[:60])
        return paper.paper_id.replace("/", "_")

    @staticmethod
    def _extract_method_name_from_title(title: str) -> str:
        if not title:
            return "paper"
        if ":" in title:
            candidate = title.split(":")[0].strip()
            if 2 <= len(candidate) <= 40 and not PaperAnalyzer._looks_like_id(candidate):
                return PaperAnalyzer._sanitize_filename(candidate)
        return PaperAnalyzer._sanitize_filename(title[:60])

    @staticmethod
    def _looks_like_id(text: str) -> bool:
        normalized = (text or "").strip().lower().replace("arxiv:", "").replace(" ", "")
        return bool(normalized) and (
            normalized.startswith("http://arxiv.org/abs/")
            or normalized.startswith("https://arxiv.org/abs/")
            or normalized.startswith("http://arxiv.org/pdf/")
            or normalized.startswith("https://arxiv.org/pdf/")
            or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", normalized) is not None
        )

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        replacements = {
            "π": "Pi",
            "Σ": "Sigma",
            "α": "Alpha",
            "β": "Beta",
            "γ": "Gamma",
            "λ": "Lambda",
            "μ": "Mu",
            "θ": "Theta",
        }
        for greek, ascii_name in replacements.items():
            name = name.replace(greek, ascii_name)
        safe = "".join(char if char.isalnum() or char in "-_ " else "" for char in name)
        return safe.strip().replace(" ", "_") or "unnamed"
