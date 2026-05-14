"""
PaperAnalyzer - 深度论文分析 Agent

参考 dailypaper-skills/paper-reader 的质量标准升级，包括：
1. 零遗漏原则：所有 figures / formulas / tables 必须出现在笔记
2. 多源图片 fallback：arXiv HTML → 项目主页 → PDF
3. 图片可达性检查 + 选择性本地化
4. 公式 5 类必检（变量冲突/文本不一致/符号不一致/求和范围错/缺算子）
5. 概念库联动：[[Concept]] 链接自动创建 stub 文件
6. 输出符合 paper-note-template 格式的结构化笔记
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.llm import LLMClient
from ..core.config import Settings
from ..core.models import PaperItem
from ..core.utils import extract_json_object
from ..services.paper_fetcher import PaperFetcher, PaperFullText
from ..services.paper_figure_fetcher import PaperFigureFetcher, FigureRef
from ..services.concept_library import ConceptLibrary


SECTION_ORDER = [
    "abstract", "introduction", "related_work",
    "method", "experiments", "discussion", "conclusion",
]

CHUNK_CHAR_LIMIT = 6000

NOTES_SUBDIR = "paper_notes"
ASSETS_SUBDIR = "assets"
CONCEPTS_SUBDIR = "concepts"


class PaperAnalyzer:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.fetcher = PaperFetcher(
            cache_dir=settings.workspace_dir / "pdf_cache"
        )
        self.notes_dir = settings.workspace_dir / NOTES_SUBDIR
        self.assets_dir = self.notes_dir / ASSETS_SUBDIR
        self.concepts_dir = self.notes_dir / CONCEPTS_SUBDIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

        self.concept_library = ConceptLibrary(self.concepts_dir)

    def analyze(
        self,
        paper: PaperItem,
        focus: Optional[str] = None,
        use_fulltext: bool = True,
    ) -> Dict:
        if use_fulltext:
            fulltext = self.fetcher.fetch_fulltext(paper.paper_id)
            if fulltext and fulltext.num_chars > 500:
                # 1. 图片提取（多源 fallback）
                figures = self._fetch_figures(paper)
                # 2. 文本分析
                result = self._analyze_fulltext(paper, fulltext, focus, figures)
                # 3. 保存笔记 + 概念库联动
                self._save_note(paper, result, figures)
                return result

        # 降级：仅 abstract
        result = self._analyze_abstract(paper, focus)
        self._save_note(paper, result, figures=[])
        return result

    # ------------------------------------------------------------------ #
    # 图片提取
    # ------------------------------------------------------------------ #

    def _fetch_figures(self, paper: PaperItem) -> List[FigureRef]:
        """提取论文图片 + 可达性检查。"""
        method_name = self._extract_method_name_from_title(paper.title)
        fig_fetcher = PaperFigureFetcher(self.assets_dir, method_name=method_name)
        figures = fig_fetcher.extract_figures(paper.paper_id)
        if figures:
            fig_fetcher.localize_unreachable(figures)
        return figures

    # ------------------------------------------------------------------ #
    # 全文分析（Map-Reduce）
    # ------------------------------------------------------------------ #

    def _analyze_fulltext(
        self,
        paper: PaperItem,
        fulltext: PaperFullText,
        focus: Optional[str],
        figures: List[FigureRef],
    ) -> Dict:
        chunks = self._build_chunks(fulltext)
        chunk_summaries: List[Dict] = []
        for chunk_name, chunk_text in chunks:
            summary = self._analyze_chunk(paper, chunk_name, chunk_text, focus)
            chunk_summaries.append({"section": chunk_name, "summary": summary})

        final = self._synthesize(paper, chunk_summaries, focus, figures)
        final["_source"] = "fulltext"
        final["_num_pages"] = fulltext.num_pages
        final["_num_chunks"] = len(chunk_summaries)
        final["_num_figures"] = len(figures)
        final["_section_summaries"] = chunk_summaries
        return final

    def _build_chunks(self, ft: PaperFullText) -> List[tuple]:
        chunks: List[tuple] = []
        if ft.sections:
            for key in SECTION_ORDER:
                text = ft.sections.get(key)
                if not text:
                    continue
                if len(text) <= CHUNK_CHAR_LIMIT:
                    chunks.append((key, text))
                else:
                    for i, sub in enumerate(self._split_by_chars(text, CHUNK_CHAR_LIMIT)):
                        chunks.append((f"{key}_part{i+1}", sub))
        if not chunks:
            for i, sub in enumerate(self._split_by_chars(ft.raw_text, CHUNK_CHAR_LIMIT)):
                chunks.append((f"chunk_{i+1}", sub))
        return chunks

    @staticmethod
    def _split_by_chars(text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]
        parts = []
        paragraphs = text.split("\n\n")
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 2 > limit and buf:
                parts.append(buf.strip())
                buf = p
            else:
                buf += ("\n\n" if buf else "") + p
        if buf:
            parts.append(buf.strip())
        return parts

    def _analyze_chunk(
        self, paper: PaperItem, section_name: str, chunk_text: str, focus: Optional[str],
    ) -> str:
        focus_line = f"\nUser focus: {focus}" if focus else ""
        prompt = f"""You are reading the **{section_name}** section of "{paper.title}".
Extract ALL key information following these zero-omission rules:{focus_line}

ZERO-OMISSION RULES:
- EVERY formula in this section (with full LaTeX, NOT pseudocode)
- EVERY table mentioned (preserve all rows/columns)
- EVERY figure reference with what it shows
- Use specific numbers, not vague claims ("achieves 82.3 F1" not "achieves good results")
- Wrap key technical terms in [[Concept]] on first appearance

FORMULA QUALITY (5 must-checks):
1. No variable name collision (bound vs free vars must differ)
2. Formula must include EVERY operator described in prose ($\\mathbb{{E}}$, $\\nabla$, stop-gradient, etc.)
3. Symbol convention must match the paper (don't substitute $p$ with $q$)
4. Sum/integral bounds must match exactly
5. No silent dropping of operators

Section content:
{chunk_text}

Output: structured Markdown extraction (200-500 words). Preserve all quantitative details verbatim.""".strip()

        return self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        ).strip()

    def _synthesize(
        self,
        paper: PaperItem,
        chunk_summaries: List[Dict],
        focus: Optional[str],
        figures: List[FigureRef],
    ) -> Dict:
        combined = "\n\n".join(
            f"## {c['section']}\n{c['summary']}" for c in chunk_summaries
        )
        focus_line = f"\nUser focus: {focus}" if focus else ""
        figure_count = len(figures)
        figure_hint = (
            f"\n\nThe paper has {figure_count} figures (already extracted). "
            f"In your output, list each figure's interpretation in 'figures_interpretation'."
        ) if figure_count else ""

        prompt = f"""Based on the section-by-section analysis of "{paper.title}", produce a final structured analysis.{focus_line}{figure_hint}

Section analyses:
{combined}

Return JSON with this exact schema:
{{
  "tldr": "Single sentence, ≤50 words, capturing the core contribution",
  "problem": "What concrete problem does the paper solve?",
  "prior_limitations": "What was broken/missing in prior work?",
  "motivation": "Why the authors believe their approach addresses the gap",
  "contributions": ["Contribution 1 headline — what + why it matters", "..."],
  "method_summary": "Architecture/approach overview (200-400 words, use [[Concept]] for key terms; NO ASCII diagrams)",
  "modules": [
    {{"name": "Module name with [[link]]", "motivation": "...", "design": "..."}}
  ],
  "formulas": [
    {{"name": "[[Concept|short name]]", "latex": "exact LaTeX", "meaning": "one-line description", "symbols": {{"x": "meaning"}}}}
  ],
  "datasets": [{{"name": "...", "size": "...", "used_for": "train/eval/both", "notes": "..."}}],
  "implementation": {{"backbone": "...", "optimizer": "...", "lr": "...", "batch_size": "...", "epochs": "...", "hardware": "..."}},
  "results": "Main results with specific numbers",
  "ablations": "Key ablation findings with numbers",
  "figures_interpretation": [
    {{"index": 1, "what_it_shows": "description", "why_it_matters": "..."}}
  ],
  "strengths": ["Specific strength citing a number or design choice"],
  "weaknesses": ["Specific weakness — missing experiment, unvalidated assumption"],
  "reproducibility": {{"code": true/false, "weights": true/false, "details_sufficient": true/false, "data_public": true/false}},
  "related_work": ["[[Prior Work 1]] — relationship", "[[Prior Work 2]] — relationship"],
  "future_work": "What the authors propose next",
  "tags": ["tag1", "tag2", "tag3"],
  "relevance_score": 0.0
}}

Critical rules:
- Wrap ALL key technical terms in [[Concept]] on first appearance (used for concept library auto-linking)
- formulas array MUST include every formula from the section analyses (zero-omission)
- Numbers, dataset names, baseline names must be accurate (no hallucination)
- Strengths/weaknesses cite specific evidence, not vibes
- modules: only list 2-5 most important sub-modules, each with motivation + design""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            return {
                "tldr": "", "problem": "", "contributions": [],
                "method_summary": response, "formulas": [],
                "datasets": [], "results": "", "strengths": [],
                "weaknesses": [], "tags": [], "relevance_score": 0.0,
            }
        return result

    # ------------------------------------------------------------------ #
    # 降级：仅 abstract
    # ------------------------------------------------------------------ #

    def _analyze_abstract(self, paper: PaperItem, focus: Optional[str]) -> Dict:
        focus_line = f"\nFocus: {focus}" if focus else ""
        prompt = f"""Analyze this paper based on its abstract only (full text unavailable).{focus_line}

Title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Published: {paper.published}
Abstract: {paper.abstract}

Return JSON (same schema as fulltext analysis but with empty arrays for fields not derivable from abstract):
{{
  "tldr": "Single sentence core contribution",
  "problem": "Core problem solved",
  "contributions": ["contribution 1", "contribution 2"],
  "method_summary": "Method overview based on abstract (use [[Concept]] for key terms)",
  "formulas": [],
  "datasets": [],
  "results": "Results mentioned in abstract (if any)",
  "strengths": ["Based on abstract"],
  "weaknesses": ["Cannot fully assess without full text"],
  "tags": ["tag1", "tag2"],
  "relevance_score": 0.0
}}""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            result = {"tldr": response, "contributions": [], "tags": []}
        result["_source"] = "abstract_only"
        return result

    # ------------------------------------------------------------------ #
    # 笔记保存（对齐 paper-note-template + 概念库联动）
    # ------------------------------------------------------------------ #

    def _save_note(self, paper: PaperItem, analysis: Dict, figures: List[FigureRef]) -> Optional[Path]:
        method_name = self._extract_method_name(paper, analysis)
        filename = f"{method_name}.md"
        path = self.notes_dir / filename

        now = datetime.utcnow().strftime("%Y-%m-%d")
        tags = analysis.get("tags", [])
        image_source = self._compute_image_source(figures)

        # YAML frontmatter（对齐 paper-note-template）
        frontmatter = (
            f"---\n"
            f"title: \"{paper.title}\"\n"
            f"method_name: \"{method_name}\"\n"
            f"arxiv_id: \"{paper.paper_id}\"\n"
            f"authors: [{', '.join(paper.authors[:5])}]\n"
            f"year: {paper.published[:4] if paper.published else 'unknown'}\n"
            f"tags: [{', '.join(tags)}]\n"
            f"read_mode: {'full' if analysis.get('_source') == 'fulltext' else 'abstract'}\n"
            f"image_source: {image_source}\n"
            f"created: {now}\n"
            f"---\n\n"
        )

        body = self._render_note_body(paper, analysis, figures)
        content = frontmatter + body
        path.write_text(content, encoding="utf-8")

        # 概念库联动：扫描笔记中的 [[Concept]] 链接，创建缺失的 stub
        new_concepts = self.concept_library.scan_and_create(content, paper_method_name=method_name)
        analysis["_note_path"] = str(path)
        analysis["_method_name"] = method_name
        analysis["_new_concepts"] = new_concepts
        analysis["_image_source"] = image_source
        return path

    def _render_note_body(
        self, paper: PaperItem, analysis: Dict, figures: List[FigureRef],
    ) -> str:
        parts = [f"# {paper.title}\n"]

        # TL;DR
        if analysis.get("tldr"):
            parts.append(f"## 一句话总结\n\n> {analysis['tldr']}\n")

        # 核心贡献
        if analysis.get("contributions"):
            parts.append("## 核心贡献\n")
            for i, c in enumerate(analysis["contributions"], 1):
                parts.append(f"{i}. **{c}**")
            parts.append("")

        # 问题背景
        if analysis.get("problem") or analysis.get("prior_limitations") or analysis.get("motivation"):
            parts.append("## 问题背景\n")
            if analysis.get("problem"):
                parts.append(f"### 要解决的问题\n{analysis['problem']}\n")
            if analysis.get("prior_limitations"):
                parts.append(f"### 现有方法的局限\n{analysis['prior_limitations']}\n")
            if analysis.get("motivation"):
                parts.append(f"### 本文的动机\n{analysis['motivation']}\n")

        # 方法详解
        if analysis.get("method_summary"):
            parts.append("## 方法详解\n")
            parts.append(f"### 模型架构\n\n{analysis['method_summary']}\n")
            modules = analysis.get("modules") or []
            for i, m in enumerate(modules, 1):
                parts.append(f"### 模块{i}: {m.get('name','')}\n")
                if m.get("motivation"):
                    parts.append(f"**设计动机**: {m['motivation']}\n")
                if m.get("design"):
                    parts.append(f"**具体实现**: {m['design']}\n")

        # 关键公式
        if analysis.get("formulas"):
            parts.append("## 关键公式\n")
            for i, f in enumerate(analysis["formulas"], 1):
                parts.append(f"### 公式{i}: {f.get('name', '')}\n")
                parts.append(f"$$\n{f.get('latex', '')}\n$$\n")
                parts.append(f"**含义**: {f.get('meaning', '')}\n")
                syms = f.get("symbols") or {}
                if syms:
                    parts.append("**符号说明**:")
                    for sym, meaning in syms.items():
                        parts.append(f"- ${sym}$: {meaning}")
                    parts.append("")

        # 关键图表
        if figures:
            parts.append("## 关键图表\n")
            interpretations = {
                fi.get("index"): fi
                for fi in (analysis.get("figures_interpretation") or [])
            }
            for fig in figures:
                parts.append(f"### Figure {fig.index}: {fig.caption}\n")
                # 优先用本地路径（Obsidian wikilink），其次外链
                if fig.local_path:
                    fname = Path(fig.local_path).name
                    parts.append(f"![[{fname}]]\n")
                elif fig.url:
                    parts.append(f"![Figure {fig.index}]({fig.url})\n")
                interp = interpretations.get(fig.index, {})
                what = interp.get("what_it_shows", "")
                why = interp.get("why_it_matters", "")
                if what or why:
                    parts.append(f"**说明**: {what} {why}\n".strip())

        # 数据集
        if analysis.get("datasets"):
            parts.append("## 数据集\n")
            parts.append("| 数据集 | 规模 | 用途 | 备注 |")
            parts.append("|--------|------|------|------|")
            for d in analysis["datasets"]:
                parts.append(
                    f"| {d.get('name','')} | {d.get('size','')} | {d.get('used_for','')} | {d.get('notes','')} |"
                )
            parts.append("")

        # 实现细节
        impl = analysis.get("implementation") or {}
        if impl:
            parts.append("## 实现细节\n")
            for k, v in impl.items():
                if v:
                    parts.append(f"- **{k}**: {v}")
            parts.append("")

        # 实验结果
        if analysis.get("results"):
            parts.append(f"## 主要结果\n\n{analysis['results']}\n")

        if analysis.get("ablations"):
            parts.append(f"## 消融实验\n\n{analysis['ablations']}\n")

        # 批判性思考
        if analysis.get("strengths") or analysis.get("weaknesses"):
            parts.append("## 批判性思考\n")
            if analysis.get("strengths"):
                parts.append("### 优点")
                for s in analysis["strengths"]:
                    parts.append(f"- {s}")
                parts.append("")
            if analysis.get("weaknesses"):
                parts.append("### 局限性")
                for w in analysis["weaknesses"]:
                    parts.append(f"- {w}")
                parts.append("")

        # 可复现性
        repro = analysis.get("reproducibility") or {}
        if repro:
            parts.append("### 可复现性评估\n")
            parts.append(f"- [{'x' if repro.get('code') else ' '}] 代码开源")
            parts.append(f"- [{'x' if repro.get('weights') else ' '}] 预训练权重")
            parts.append(f"- [{'x' if repro.get('details_sufficient') else ' '}] 训练细节完整")
            parts.append(f"- [{'x' if repro.get('data_public') else ' '}] 数据集可获取")
            parts.append("")

        # 关联笔记
        if analysis.get("related_work"):
            parts.append("## 关联笔记\n")
            for rw in analysis["related_work"]:
                parts.append(f"- {rw}")
            parts.append("")

        # 未来方向
        if analysis.get("future_work"):
            parts.append(f"## 未来方向\n\n{analysis['future_work']}\n")

        # 速查卡片
        if analysis.get("tldr"):
            parts.append("## 速查卡片\n")
            parts.append(f"> [!summary] {paper.title}")
            parts.append(f"> - **核心**: {analysis.get('tldr','')}")
            parts.append(f"> - **代码**: {paper.url}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _compute_image_source(figures: List[FigureRef]) -> str:
        """根据 figures 来源计算 frontmatter image_source 字段。"""
        if not figures:
            return "none"
        has_local = any(f.local_path for f in figures)
        has_online = any(f.url and not f.local_path for f in figures)
        if has_local and has_online:
            return "mixed"
        if has_local:
            return "local"
        return "online"

    # ------------------------------------------------------------------ #
    # 方法名提取
    # ------------------------------------------------------------------ #

    def _extract_method_name(self, paper: PaperItem, analysis: Dict) -> str:
        """从论文标题或贡献提取方法名作为文件名。"""
        title = paper.title or ""
        if ":" in title:
            candidate = title.split(":")[0].strip()
            if len(candidate) <= 30:
                return self._sanitize_filename(candidate)
        if analysis.get("contributions"):
            first = analysis["contributions"][0]
            if "**" in first:
                parts = first.split("**")
                if len(parts) >= 2 and len(parts[1]) <= 30:
                    return self._sanitize_filename(parts[1])
        return paper.paper_id.replace("/", "_")

    @staticmethod
    def _extract_method_name_from_title(title: str) -> str:
        """简化版方法名（用于图片文件命名）。"""
        if not title:
            return "paper"
        if ":" in title:
            candidate = title.split(":")[0].strip()
            if len(candidate) <= 30:
                return PaperAnalyzer._sanitize_filename(candidate)
        return PaperAnalyzer._sanitize_filename(title[:30])

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        replacements = {"π": "Pi", "σ": "Sigma", "α": "Alpha",
                        "β": "Beta", "γ": "Gamma", "λ": "Lambda"}
        for greek, ascii_name in replacements.items():
            name = name.replace(greek, ascii_name)
        safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)
        return safe.strip().replace(" ", "_") or "unnamed"
