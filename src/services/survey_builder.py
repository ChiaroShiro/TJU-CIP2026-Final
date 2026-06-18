"""主题文献综述与演进海报生成。

该模块故意保持低耦合：
- 没有 LLM API key 时，也能基于检索证据生成保守综述、算法演进图（确定性分泳道）和海报
- 有 LLM key 时，算法演进图会升级为「LLM 划分技术分支 + 判定继承/对比关系」的版本
"""

from __future__ import annotations

import base64
import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import PaperItem, SurveyArtifact
from ..core.utils import extract_json_object
from .evolution_graph import EvolutionGraphBuilder
from .paper_figure_fetcher import PaperFigureFetcher
from .paper_search import PaperDiscoveryService


logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_CN_QUERY_TERMS = [
    ("大语言模型", "large language model"),
    ("语言模型", "language model"),
    ("多智能体", "multi-agent"),
    ("智能体", "agent"),
    ("代理", "agent"),
    ("世界模型", "world model"),
    ("扩散模型", "diffusion model"),
    ("扩散语言模型", "diffusion language model"),
    ("强化学习", "reinforcement learning"),
    ("自监督", "self-supervised"),
    ("多模态", "multimodal"),
    ("视觉语言", "vision language"),
    ("图神经网络", "graph neural network"),
    ("语音表示", "speech representation"),
    ("文献综述", "literature review"),
    ("推理", "reasoning"),
    ("规划", "planning"),
    ("反思", "reflection"),
    ("优化", "optimization"),
    ("加速", "acceleration"),
    ("对齐", "alignment"),
    ("检索增强", "retrieval augmented generation"),
    ("演化", "evolution"),
    ("演进", "evolution"),
    ("技术", "techniques"),
    ("技巧", "techniques"),
]


class SurveyBuilder:
    def __init__(self, workspace_dir: Path, llm: Any = None, memory: Any = None):
        self.workspace_dir = Path(workspace_dir)
        self.llm = llm
        self.memory = memory  # 可选：传入则把演进图节点/关系并入论文图谱
        self.discovery = PaperDiscoveryService()
        self.output_root = self.workspace_dir / "surveys"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        topic: str,
        max_papers: int = 12,
        output_name: Optional[str] = None,
        with_figures: bool = True,
        origin: str = "survey",
        seed_papers: Optional[List[PaperItem]] = None,
        search_queries: Optional[List[str]] = None,
    ) -> SurveyArtifact:
        safe_name = output_name or self._safe_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_root / f"{timestamp}_{safe_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        papers = self._collect_papers(
            topic=topic,
            max_papers=max_papers,
            seed_papers=seed_papers,
            search_queries=search_queries,
        )

        raw_data_path = out_dir / "papers.json"
        raw_data_path.write_text(
            json.dumps([self._paper_to_dict(p) for p in papers], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 算法演进图（多泳道 + 演进关系；有 key 时 LLM 升级，无 key 走确定性启发式）
        evo = EvolutionGraphBuilder(self.llm).build(topic, papers)
        evolution_path = out_dir / "algorithm_evolution.svg"
        evolution_path.write_text(evo["svg"], encoding="utf-8")

        # 可选：把演进图的节点/关系并入论文图谱（"检索发现"层），失败不影响产物
        if self.memory is not None:
            try:
                self.memory.merge_evolution_graph(papers, evo, origin=origin)
            except Exception as exc:
                # 合并失败不影响综述/演进图/海报产物，但不再静默吞掉
                logger.warning("演进图合并到论文图谱失败: %s", exc)

        # 海报（尽力嵌入论文原图）
        figures_map = self._collect_poster_figures(papers, out_dir) if with_figures else {}
        poster_path = out_dir / "survey_poster.svg"
        poster_path.write_text(self._render_poster_svg(topic, papers, figures_map), encoding="utf-8")

        report_path = out_dir / "survey_report.md"
        report_path.write_text(
            self._render_report(topic, papers, evolution_path.name, poster_path.name, evo),
            encoding="utf-8",
        )

        return SurveyArtifact(
            topic=topic,
            output_dir=str(out_dir),
            papers=papers,
            report_file=str(report_path),
            poster_file=str(poster_path),
            evolution_file=str(evolution_path),
            raw_data_file=str(raw_data_path),
        )

    def _collect_papers(
        self,
        topic: str,
        max_papers: int,
        seed_papers: Optional[List[PaperItem]] = None,
        search_queries: Optional[List[str]] = None,
    ) -> List[PaperItem]:
        """收集图/综述用论文。

        GUI 深度研究已经检索过英文论文；这里优先复用这些结果，避免研究主题是中文时
        又拿中文直接查 arXiv/S2 导致 papers.json 为空。直达 survey 路径会为中文主题
        生成英文检索兜底，再尝试原主题。
        """
        papers = self._dedupe_papers(seed_papers or [])
        queries = self._build_search_queries(topic, search_queries)
        searched_queries = set()

        for query in queries:
            if len(papers) >= max_papers:
                break
            searched_queries.add(query)
            try:
                found = self.discovery.search_topic(query, max_results=max_papers - len(papers))
            except Exception as exc:
                logger.warning("论文检索失败 query=%r: %s", query, exc)
                continue
            papers = self._dedupe_papers([*papers, *found])

        if not papers and self._contains_cjk(topic):
            for query in self._llm_search_queries(topic):
                if len(papers) >= max_papers or query in searched_queries:
                    continue
                searched_queries.add(query)
                try:
                    found = self.discovery.search_topic(query, max_results=max_papers - len(papers))
                except Exception as exc:
                    logger.warning("LLM 兜底论文检索失败 query=%r: %s", query, exc)
                    continue
                papers = self._dedupe_papers([*papers, *found])

        if not papers:
            return []

        enriched = self.discovery.enrich_with_code(papers[:max_papers], max_code_hits=3)
        return self._dedupe_papers(enriched)[:max_papers]

    @staticmethod
    def _dedupe_papers(papers: List[PaperItem]) -> List[PaperItem]:
        deduped: List[PaperItem] = []
        seen = set()
        for paper in papers or []:
            if not paper or not paper.title:
                continue
            keys = {
                (paper.paper_id or "").strip().lower(),
                (paper.url or "").strip().lower(),
                paper.title.strip().lower(),
            }
            keys.discard("")
            if not keys or seen.intersection(keys):
                continue
            seen.update(keys)
            deduped.append(paper)
        return deduped

    def _build_search_queries(
        self,
        topic: str,
        search_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """生成论文检索式，中文主题优先用便宜的英文兜底。

        arXiv / Semantic Scholar 对中文 query 覆盖很差。深度研究路径会传入 Agent
        已经用过的英文 query；Survey 直达路径没有这些 query 时，这里做轻量改写。
        LLM 改写只在这些检索式完全失败后才触发，避免图生成依赖额外 API 调用。
        """
        queries: List[str] = []

        def add(query: str) -> None:
            query = (query or "").strip()
            if query and query not in queries:
                queries.append(query)

        for query in search_queries or []:
            add(query)

        topic_text = (topic or "").strip()
        if topic_text and self._contains_cjk(topic_text):
            add(self._heuristic_english_query(topic_text))

        add(topic_text)
        return queries

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(_CJK_RE.search(text or ""))

    def _llm_search_queries(self, topic: str) -> List[str]:
        if self.llm is None or not getattr(self.llm, "available", False):
            return []
        try:
            text = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite Chinese research topics into concise English academic "
                            "paper search queries. Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\n"
                            "Return {\"queries\": [..]} with 2-3 English queries for arXiv "
                            "and Semantic Scholar. Each query should be under 12 words."
                        ),
                    },
                ],
                temperature=0.0,
            )
            data = extract_json_object(text)
            raw_queries = data.get("queries") if isinstance(data, dict) else []
            queries = []
            for query in raw_queries or []:
                query = re.sub(r"\s+", " ", str(query or "")).strip()
                if 3 <= len(query) <= 140 and query not in queries:
                    queries.append(query)
            return queries[:3]
        except Exception as exc:
            logger.warning("中文主题英文检索式生成失败: %s", exc)
            return []

    @staticmethod
    def _heuristic_english_query(topic: str) -> str:
        ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9+_.-]*", topic or "")
        terms: List[str] = []
        for term in ascii_terms:
            if term not in terms:
                terms.append(term)
        for cn, en in _CN_QUERY_TERMS:
            if cn in (topic or "") and en not in terms:
                terms.append(en)
        return " ".join(terms[:10])

    # ------------------------------------------------------------------ #
    # 综述报告
    # ------------------------------------------------------------------ #
    def _render_report(
        self,
        topic: str,
        papers: List[PaperItem],
        evolution_name: str,
        poster_name: str,
        evo: Dict[str, Any],
    ) -> str:
        has_code = [p for p in papers if p.has_code]
        no_code = [p for p in papers if not p.has_code]
        years = sorted({self._year(p) for p in papers if self._year(p)})
        evo_mode_text = {
            "llm": "LLM 划分技术分支并判定继承/对比关系",
            "heuristic": "按 arXiv 主类目分泳道、按发布时间排列（无 LLM key，确定性）",
            "empty": "论文不足，未能生成",
        }.get(evo.get("mode", ""), "")

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
            f"> 生成方式：{evo_mode_text}。{evo.get('note', '')}",
            "",
            f"![Algorithm evolution]({evolution_name})",
            "",
        ])
        branches = evo.get("branches") or []
        if branches:
            lines.append("识别到的技术分支：")
            for b in branches:
                lines.append(f"- {b.get('name', '')}")
            lines.append("")

        lines.extend([
            "## 综述海报",
            "",
            f"![Survey poster]({poster_name})",
            "",
            "## 风险与不确定性",
            "",
            "- 仅靠标题/摘要无法保证方法细节完全准确；正式报告应在 API key 配好后调用 `analyze` 或 `read-paper` 精读核心论文。",
            "- GitHub 关联来自公开网页检索，存在同名项目误匹配风险；代码置信度低于 0.75 的链接建议人工核对。",
            "- 演进图的技术分支与继承/对比关系：无 key 时按类目确定性分组、不臆造关系；有 key 时由 LLM 在论文集合内部判定并强制时间方向，仍建议人工复核关键连线。",
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

    # ------------------------------------------------------------------ #
    # 海报原图采集（尽力而为）
    # ------------------------------------------------------------------ #
    def _collect_poster_figures(
        self, papers: List[PaperItem], out_dir: Path, limit: int = 3
    ) -> Dict[str, Dict[str, str]]:
        """为前若干篇 arXiv 论文各取一张代表图，base64 内联进海报。

        网络/解析失败均静默跳过，海报退化为纯文字卡片（不影响主流程）。
        """
        result: Dict[str, Dict[str, str]] = {}
        assets_dir = out_dir / "assets"
        for paper in papers:
            if len(result) >= limit:
                break
            arxiv_id = self._arxiv_id(paper)
            if not arxiv_id:
                continue
            try:
                fetcher = PaperFigureFetcher(assets_dir, method_name=arxiv_id.replace("/", "_"))
                figures = fetcher.extract_figures(arxiv_id)
                if not figures:
                    continue
                fig = figures[0]
                data_uri = self._figure_to_data_uri(fetcher, fig)
                if data_uri:
                    result[paper.paper_id] = {"data_uri": data_uri}
            except Exception:
                continue
        return result

    @staticmethod
    def _figure_to_data_uri(fetcher: PaperFigureFetcher, fig) -> str:
        try:
            local = fig.local_path
            if not local and fig.url:
                # 海报内嵌需要图片字节：无论远程图可不可达，都下载到本地再读
                downloaded = fetcher._download_image(fig.url, fig.index)
                local = str(downloaded) if downloaded else ""
            if not local:
                return ""
            path = Path(local)
            if not path.exists():
                return ""
            data = path.read_bytes()
            if len(data) < 800 or len(data) > 2_500_000:  # 太小多半是图标，太大不宜内联
                return ""
            mime = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/png")
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return ""

    def _arxiv_id(self, paper: PaperItem) -> str:
        pid = (getattr(paper, "paper_id", "") or "").strip()
        if _ARXIV_ID_RE.match(pid):
            return pid
        url = getattr(paper, "url", "") or ""
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url)
        return m.group(1) if m else ""

    # ------------------------------------------------------------------ #
    # 海报 SVG
    # ------------------------------------------------------------------ #
    def _render_poster_svg(
        self, topic: str, papers: List[PaperItem], figures_map: Dict[str, Dict[str, str]]
    ) -> str:
        width, height = 1600, 2200
        top = papers[:6]
        years = sorted({self._year(p) for p in papers if self._year(p)})
        code_count = sum(1 for p in papers if p.has_code)
        fig_count = len(figures_map)
        escaped_topic = html.escape(topic)

        rows = []
        for idx, paper in enumerate(top):
            y = 790 + idx * 190
            color = "#0f766e" if paper.has_code else "#475569"
            title = html.escape(self._shorten(paper.title, 78))
            abstract = html.escape(self._shorten(self._clean_inline(paper.abstract), 150))
            code = "GitHub code found" if paper.has_code else "No code found"
            fig = figures_map.get(paper.paper_id)
            has_fig = bool(fig and fig.get("data_uri"))
            text_w = 1130 if has_fig else 1360
            rows.append(f"""
  <g>
    <rect x="110" y="{y}" width="1380" height="150" rx="10" fill="#ffffff" stroke="#d9e2e1"/>
    <circle cx="155" cy="{y + 45}" r="18" fill="{color}"/>
    <text x="190" y="{y + 38}" font-size="18" font-weight="800" fill="#0f172a">{self._year(paper) or 'unknown'} · {title}</text>
    <text x="190" y="{y + 70}" font-size="15" fill="{color}">{code}</text>
    <text x="190" y="{y + 105}" font-size="15" fill="#475569">{abstract}</text>""")
            if has_fig:
                rows.append(
                    f'    <clipPath id="figclip{idx}"><rect x="1270" y="{y + 16}" width="200" height="118" rx="8"/></clipPath>\n'
                    f'    <rect x="1270" y="{y + 16}" width="200" height="118" rx="8" fill="#f1f5f9" stroke="#d9e2e1"/>\n'
                    f'    <image x="1270" y="{y + 16}" width="200" height="118" clip-path="url(#figclip{idx})" '
                    f'preserveAspectRatio="xMidYMid meet" href="{fig["data_uri"]}"/>'
                )
            rows.append("  </g>")

        fig_note = (
            f"Figures: {fig_count} original paper figure(s) embedded"
            if fig_count else "Figures: none auto-fetched (text-only cards)"
        )
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Inter, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs><linearGradient id="posterHead" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#0f766e"/><stop offset="1" stop-color="#4f46e5"/>
  </linearGradient></defs>
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <rect x="0" y="0" width="1600" height="420" fill="url(#posterHead)"/>
  <text x="110" y="150" font-size="56" font-weight="900" fill="#ffffff">{escaped_topic}</text>
  <text x="110" y="215" font-size="28" fill="#d1fae5">Literature Survey Agent Output</text>
  <text x="110" y="285" font-size="22" fill="#ecfeff">Evidence-first survey · latest papers · GitHub code prioritized · timeline checked by publication dates</text>
  <text x="110" y="330" font-size="19" fill="#a7f3d0">{fig_note}</text>

  <g>
    <rect x="110" y="500" width="380" height="170" rx="10" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="145" y="560" font-size="22" fill="#475569">Papers</text>
    <text x="145" y="625" font-size="58" font-weight="900" fill="#0f172a">{len(papers)}</text>
  </g>
  <g>
    <rect x="540" y="500" width="380" height="170" rx="10" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="575" y="560" font-size="22" fill="#475569">With Code</text>
    <text x="575" y="625" font-size="58" font-weight="900" fill="#0f766e">{code_count}</text>
  </g>
  <g>
    <rect x="970" y="500" width="520" height="170" rx="10" fill="#ffffff" stroke="#d9e2e1"/>
    <text x="1005" y="560" font-size="22" fill="#475569">Year Span</text>
    <text x="1005" y="625" font-size="50" font-weight="900" fill="#0f172a">{min(years) if years else '?'} - {max(years) if years else '?'}</text>
  </g>

  <text x="110" y="745" font-size="34" font-weight="900" fill="#0f172a">Representative Papers</text>
  {''.join(rows)}

  <rect x="110" y="1960" width="1380" height="130" rx="10" fill="#ecfeff" stroke="#bae6fd"/>
  <text x="145" y="2010" font-size="24" font-weight="800" fill="#164e63">Quality Guardrails</text>
  <text x="145" y="2050" font-size="18" fill="#164e63">Topic accuracy target >=85%; claims are limited to retrieved evidence; timeline uses publication dates to avoid obvious chronological errors.</text>
</svg>
"""

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
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
