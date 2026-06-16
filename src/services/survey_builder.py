"""
主题文献综述与演进海报生成。

该模块故意保持低耦合：
- 没有 LLM API key 时，也能基于检索证据生成保守综述和图表
- 有 LLM key 时，上层可以再加入更强的自然语言综合
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import PaperItem, SurveyArtifact
from .paper_search import PaperDiscoveryService


class SurveyBuilder:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.discovery = PaperDiscoveryService()
        self.output_root = self.workspace_dir / "surveys"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        topic: str,
        max_papers: int = 12,
        output_name: Optional[str] = None,
    ) -> SurveyArtifact:
        safe_name = output_name or self._safe_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_root / f"{timestamp}_{safe_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        papers = self.discovery.search_topic(topic, max_results=max_papers)
        papers = self.discovery.enrich_with_code(papers, max_code_hits=3)
        papers = papers[:max_papers]

        raw_data_path = out_dir / "papers.json"
        raw_data_path.write_text(
            json.dumps([self._paper_to_dict(p) for p in papers], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        timeline_path = out_dir / "algorithm_timeline.svg"
        timeline_path.write_text(self._render_timeline_svg(topic, papers), encoding="utf-8")

        poster_path = out_dir / "survey_poster.svg"
        poster_path.write_text(self._render_poster_svg(topic, papers), encoding="utf-8")

        report_path = out_dir / "survey_report.md"
        report_path.write_text(
            self._render_report(topic, papers, timeline_path.name, poster_path.name),
            encoding="utf-8",
        )

        return SurveyArtifact(
            topic=topic,
            output_dir=str(out_dir),
            papers=papers,
            report_file=str(report_path),
            poster_file=str(poster_path),
            timeline_file=str(timeline_path),
            raw_data_file=str(raw_data_path),
        )

    def _render_report(self, topic: str, papers: List[PaperItem], timeline_name: str, poster_name: str) -> str:
        has_code = [p for p in papers if p.has_code]
        no_code = [p for p in papers if not p.has_code]
        years = sorted({self._year(p) for p in papers if self._year(p)})

        lines = [
            f"# {topic} 文献综述草稿",
            "",
            "## 生成说明",
            "",
            "- 本综述只基于自动检索到的 arXiv / Semantic Scholar 元数据与 GitHub 公开代码线索。",
            "- 未配置 LLM API key 时，不生成超出摘要证据的细节性结论，避免明显幻觉。",
            "- 论文排序优先考虑公开代码，其次考虑时间新近性和主题匹配。",
            "",
            "## 主题覆盖与置信度",
            "",
            f"- 检索主题: `{topic}`",
            f"- 入选论文数: {len(papers)}",
            f"- 带公开代码论文数: {len(has_code)}",
            f"- 覆盖年份: {min(years) if years else 'unknown'} - {max(years) if years else 'unknown'}",
            f"- 主题准确率目标: >=85%。当前自动阶段采用标题/摘要关键词匹配与人工可复核链接，不声称已完成最终准确率评估。",
            "",
            "## 自动综述",
            "",
            self._build_conservative_survey(topic, papers),
            "",
            "## 优先阅读论文（公开代码优先）",
            "",
            "| 序号 | 年份 | 论文 | 代码 | 摘要证据 |",
            "| --- | --- | --- | --- | --- |",
        ]

        for idx, paper in enumerate(papers, 1):
            title = self._md_link(paper.title, paper.url)
            code = self._md_link("GitHub", paper.code_url) if paper.code_url else "未发现"
            abstract = self._clean_inline(paper.abstract)[:180]
            lines.append(f"| {idx} | {self._year(paper) or 'unknown'} | {title} | {code} | {abstract}... |")

        lines.extend([
            "",
            "## 代码复现优先级",
            "",
        ])
        if has_code:
            for paper in has_code:
                lines.append(f"- {self._md_link(paper.title, paper.url)}: {self._md_link(paper.code_url, paper.code_url)}")
        else:
            lines.append("- 暂未发现可信 GitHub 仓库，建议补充 Papers with Code 或人工核验。")

        lines.extend([
            "",
            "## 算法发展演进图",
            "",
            f"![Algorithm timeline]({timeline_name})",
            "",
            "## 综述海报",
            "",
            f"![Survey poster]({poster_name})",
            "",
            "## 风险与不确定性",
            "",
            "- 仅靠标题/摘要无法保证方法细节完全准确；正式报告应在 API key 配好后调用 `analyze` 或 `read-paper` 精读核心论文。",
            "- GitHub 关联来自公开网页检索，存在同名项目误匹配风险；代码置信度低于 0.75 的链接建议人工核对。",
            "- 演进图按论文发布日期排序，若综述主题包含早期非 arXiv 工作，需补充手工种子论文以避免时间线缺失。",
        ])

        if no_code:
            lines.extend([
                "",
                "## 暂未发现代码的论文",
                "",
            ])
            for paper in no_code:
                lines.append(f"- {self._md_link(paper.title, paper.url)}")

        return "\n".join(lines)

    def _build_conservative_survey(self, topic: str, papers: List[PaperItem]) -> str:
        if not papers:
            return "未检索到足够论文，无法生成可靠综述。"

        latest = sorted(papers, key=lambda p: self._year(p), reverse=True)[:5]
        coded = [p for p in papers if p.has_code][:5]
        lines = [
            f"围绕 `{topic}`，系统优先检索最新论文，并将带 GitHub 代码的论文置于前列。",
            "从当前证据看，入选论文主要可以作为后续精读和复现实验的候选集合。",
            "",
            "最新论文线索：",
        ]
        for paper in latest:
            lines.append(f"- {self._year(paper) or 'unknown'}: {paper.title}")
        lines.append("")
        lines.append("公开代码线索：")
        if coded:
            for paper in coded:
                lines.append(f"- {paper.title}: {paper.code_url}")
        else:
            lines.append("- 当前检索未发现高置信 GitHub 仓库。")
        return "\n".join(lines)

    def _render_timeline_svg(self, topic: str, papers: List[PaperItem]) -> str:
        sorted_papers = sorted(papers, key=lambda p: (self._year(p), p.published or p.updated or ""))
        width = 1400
        height = max(520, 160 + len(sorted_papers) * 88)
        x0 = 110
        y0 = 110
        row_gap = 86
        escaped_topic = html.escape(topic)
        nodes = []
        for idx, paper in enumerate(sorted_papers):
            y = y0 + idx * row_gap
            year = self._year(paper) or "?"
            title = html.escape(self._shorten(paper.title, 82))
            code_badge = "CODE" if paper.has_code else "NO CODE"
            color = "#0f766e" if paper.has_code else "#64748b"
            nodes.append(f"""
  <g>
    <circle cx="{x0}" cy="{y}" r="13" fill="{color}" />
    <text x="{x0 - 72}" y="{y + 5}" text-anchor="start" font-size="18" font-weight="700" fill="#1f2937">{year}</text>
    <rect x="{x0 + 34}" y="{y - 30}" width="1110" height="60" rx="8" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="{x0 + 54}" y="{y - 5}" font-size="18" font-weight="700" fill="#1f2937">{title}</text>
    <text x="{x0 + 54}" y="{y + 20}" font-size="13" fill="{color}">{code_badge}</text>
  </g>""")
        line_y2 = y0 + max(0, len(sorted_papers) - 1) * row_gap
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="70" y="55" font-size="34" font-weight="800" fill="#0f172a">Algorithm Evolution: {escaped_topic}</text>
  <text x="70" y="84" font-size="15" fill="#475569">Ordered by publication date; green nodes indicate discovered GitHub code.</text>
  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{line_y2}" stroke="#94a3b8" stroke-width="4"/>
  {''.join(nodes)}
</svg>
"""

    def _render_poster_svg(self, topic: str, papers: List[PaperItem]) -> str:
        width, height = 1600, 2200
        top = papers[:6]
        years = sorted({self._year(p) for p in papers if self._year(p)})
        code_count = sum(1 for p in papers if p.has_code)
        escaped_topic = html.escape(topic)

        rows = []
        for idx, paper in enumerate(top):
            y = 790 + idx * 190
            color = "#0f766e" if paper.has_code else "#475569"
            title = html.escape(self._shorten(paper.title, 86))
            abstract = html.escape(self._shorten(self._clean_inline(paper.abstract), 170))
            code = "GitHub code found" if paper.has_code else "No code found"
            rows.append(f"""
  <g>
    <rect x="110" y="{y}" width="1380" height="150" rx="8" fill="#ffffff" stroke="#d9e2e1"/>
    <circle cx="155" cy="{y + 45}" r="18" fill="{color}"/>
    <text x="190" y="{y + 38}" font-size="18" font-weight="800" fill="#0f172a">{self._year(paper) or 'unknown'} · {title}</text>
    <text x="190" y="{y + 70}" font-size="15" fill="{color}">{code}</text>
    <text x="190" y="{y + 105}" font-size="15" fill="#475569">{abstract}</text>
  </g>""")

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <rect x="0" y="0" width="1600" height="420" fill="#0f766e"/>
  <text x="110" y="150" font-size="58" font-weight="900" fill="#ffffff">{escaped_topic}</text>
  <text x="110" y="215" font-size="28" fill="#d1fae5">Literature Survey Agent Output</text>
  <text x="110" y="285" font-size="22" fill="#ecfeff">Evidence-first survey · latest papers · GitHub code prioritized · timeline checked by publication dates</text>

  <g>
    <rect x="110" y="500" width="380" height="170" rx="8" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="145" y="560" font-size="22" fill="#475569">Papers</text>
    <text x="145" y="625" font-size="58" font-weight="900" fill="#0f172a">{len(papers)}</text>
  </g>
  <g>
    <rect x="540" y="500" width="380" height="170" rx="8" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="575" y="560" font-size="22" fill="#475569">With Code</text>
    <text x="575" y="625" font-size="58" font-weight="900" fill="#0f766e">{code_count}</text>
  </g>
  <g>
    <rect x="970" y="500" width="520" height="170" rx="8" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="1005" y="560" font-size="22" fill="#475569">Year Span</text>
    <text x="1005" y="625" font-size="50" font-weight="900" fill="#0f172a">{min(years) if years else '?'} - {max(years) if years else '?'}</text>
  </g>

  <text x="110" y="745" font-size="34" font-weight="900" fill="#0f172a">Representative Papers</text>
  {''.join(rows)}

  <rect x="110" y="1960" width="1380" height="130" rx="8" fill="#ecfeff" stroke="#bae6fd"/>
  <text x="145" y="2010" font-size="24" font-weight="800" fill="#164e63">Quality Guardrails</text>
  <text x="145" y="2050" font-size="18" fill="#164e63">Topic accuracy target >=85%; claims are limited to retrieved evidence; timeline uses publication dates to avoid obvious chronological errors.</text>
</svg>
"""

    @staticmethod
    def _paper_to_dict(paper: PaperItem) -> Dict:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "url": paper.url,
            "published": paper.published,
            "updated": paper.updated,
            "categories": paper.categories,
            "code_urls": paper.code_urls,
            "code_url": paper.code_url,
            "has_code": paper.has_code,
            "code_confidence": paper.code_confidence,
        }

    @staticmethod
    def _safe_filename(text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in text)
        return safe.strip().replace(" ", "_")[:60] or "survey"

    @staticmethod
    def _year(paper: PaperItem):
        for value in (paper.published, paper.updated):
            try:
                return int(str(value)[:4])
            except Exception:
                continue
        return 0

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _clean_inline(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").replace("|", "/").strip()

    @staticmethod
    def _md_link(label: str, url: str) -> str:
        if not url:
            return label
        return f"[{label}]({url})"
