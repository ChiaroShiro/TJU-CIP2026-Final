"""
评估框架 - 论文分析质量（零遗漏原则验证）

评估维度：
1. 公式覆盖率：笔记中公式数 / 论文实际公式数
2. 图片覆盖率：笔记中图片数 / 论文实际图片数
3. 表格覆盖率：笔记中表格数 / 论文实际表格数
4. 概念链接密度：[[Concept]] 数量 / 笔记总词数
5. 笔记完整性：各必需章节是否存在

使用方式：
    python eval/eval_paper_analysis.py <arxiv_id>

会分析一篇论文并输出质量指标。
"""

import re
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.core.llm import LLMClient
from src.agents.paper_analyzer import PaperAnalyzer
from src.services.paper_search import ArxivSearcher
from src.services.paper_fetcher import PaperFetcher


def count_formulas_in_pdf(fulltext_raw: str) -> int:
    """估算论文原文中的公式数量（基于常见 LaTeX 模式）。"""
    # 匹配 \begin{equation} / \begin{align} / $$ ... $$ / 编号公式
    patterns = [
        r"\\begin\{equation\}",
        r"\\begin\{align",
        r"\\begin\{gather",
        r"\$\$[^$]+\$\$",
    ]
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, fulltext_raw))
    # 也统计独立行的数学表达式（如 arXiv HTML 提取后的格式）
    # 简单启发式：包含 = 且包含 \sum / \int / \frac 的行
    for line in fulltext_raw.split("\n"):
        if "=" in line and any(op in line for op in ["\\sum", "\\int", "\\frac", "\\mathbb"]):
            count += 1
    return max(count, 1)  # 至少 1 避免除零


def count_figures_in_pdf(fulltext_raw: str) -> int:
    """估算论文原文中的 Figure 数量。"""
    # 匹配 "Figure X" / "Fig. X" / "Fig X"
    matches = re.findall(r"(?:Figure|Fig\.?)\s*(\d+)", fulltext_raw, re.IGNORECASE)
    if matches:
        return max(int(m) for m in matches)
    return 0


def count_tables_in_pdf(fulltext_raw: str) -> int:
    """估算论文原文中的 Table 数量。"""
    matches = re.findall(r"(?:Table)\s*(\d+)", fulltext_raw, re.IGNORECASE)
    if matches:
        return max(int(m) for m in matches)
    return 0


def evaluate_note_quality(note_path: Path, fulltext_raw: str) -> Dict:
    """评估生成的笔记质量。"""
    if not note_path.exists():
        return {"error": "note file not found"}

    note_text = note_path.read_text(encoding="utf-8")

    # 论文中的实际数量
    pdf_formulas = count_formulas_in_pdf(fulltext_raw)
    pdf_figures = count_figures_in_pdf(fulltext_raw)
    pdf_tables = count_tables_in_pdf(fulltext_raw)

    # 笔记中的数量
    note_formulas = len(re.findall(r"\$\$\n.*?\n\$\$", note_text, re.DOTALL))
    note_figures = len(re.findall(r"!\[.*?\]\(.*?\)|!\[\[.*?\]\]", note_text))
    note_tables = note_text.count("| ") // 3  # 粗略：每个表格至少 3 行含 |
    note_concepts = len(re.findall(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]+)?\]\]", note_text))
    note_words = len(note_text.split())
    note_lines = len(note_text.split("\n"))

    # 必需章节检查
    required_sections = [
        "一句话总结",
        "核心贡献",
        "问题背景",
        "方法详解",
        "关键公式",
        "关键图表",
        "批判性思考",
    ]
    sections_present = {s: s in note_text for s in required_sections}
    section_coverage = sum(sections_present.values()) / len(required_sections)

    # 覆盖率计算
    formula_coverage = min(note_formulas / max(pdf_formulas, 1), 1.0)
    figure_coverage = min(note_figures / max(pdf_figures, 1), 1.0)
    table_coverage = min(note_tables / max(pdf_tables, 1), 1.0)
    concept_density = note_concepts / max(note_words, 1) * 100  # 每 100 词多少个概念链接

    return {
        "note_path": str(note_path),
        "note_stats": {
            "lines": note_lines,
            "words": note_words,
            "chars": len(note_text),
        },
        "coverage": {
            "formulas": {
                "in_paper": pdf_formulas,
                "in_note": note_formulas,
                "coverage_rate": round(formula_coverage, 3),
            },
            "figures": {
                "in_paper": pdf_figures,
                "in_note": note_figures,
                "coverage_rate": round(figure_coverage, 3),
            },
            "tables": {
                "in_paper": pdf_tables,
                "in_note": note_tables,
                "coverage_rate": round(table_coverage, 3),
            },
        },
        "quality": {
            "concept_links": note_concepts,
            "concept_density_per_100_words": round(concept_density, 2),
            "section_coverage": round(section_coverage, 3),
            "sections_present": sections_present,
        },
        "zero_omission_score": round(
            (formula_coverage + figure_coverage + table_coverage + section_coverage) / 4, 3
        ),
    }


def print_eval(metrics: Dict):
    """格式化打印。"""
    print("=" * 60)
    print("论文分析质量评估")
    print("=" * 60)

    ns = metrics["note_stats"]
    print(f"\n📄 笔记统计:")
    print(f"  行数: {ns['lines']}  词数: {ns['words']}  字符: {ns['chars']}")

    cov = metrics["coverage"]
    print(f"\n📊 覆盖率（零遗漏原则）:")
    for item in ["formulas", "figures", "tables"]:
        d = cov[item]
        rate = d["coverage_rate"]
        bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {item:<10} {d['in_note']}/{d['in_paper']}  {rate:.1%}  {bar}")

    q = metrics["quality"]
    print(f"\n🔗 概念链接:")
    print(f"  [[Concept]] 数量: {q['concept_links']}")
    print(f"  密度: {q['concept_density_per_100_words']:.2f} / 100 词")

    print(f"\n📋 章节完整性 ({q['section_coverage']:.0%}):")
    for section, present in q["sections_present"].items():
        status = "✅" if present else "❌"
        print(f"  {status} {section}")

    print(f"\n🎯 零遗漏综合分: {metrics['zero_omission_score']:.3f}")
    print(f"   (公式覆盖 + 图片覆盖 + 表格覆盖 + 章节完整) / 4")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    settings = Settings.from_env(root)

    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2305.18290"
    print(f"🚀 分析论文: {arxiv_id}\n")

    # 1. 获取论文元数据
    arxiv = ArxivSearcher(max_results=1)
    papers = arxiv.search(f"id:{arxiv_id}", max_results=1)
    if not papers:
        print(f"❌ 未找到论文 {arxiv_id}")
        sys.exit(1)

    paper = papers[0]
    print(f"📖 {paper.title}\n")

    # 2. 获取全文（用于统计实际公式/图/表数量）
    fetcher = PaperFetcher(settings.workspace_dir / "pdf_cache")
    fulltext = fetcher.fetch_fulltext(arxiv_id)
    if not fulltext:
        print("⚠️  无法获取全文，覆盖率统计将不准确")
        raw_text = paper.abstract
    else:
        raw_text = fulltext.raw_text
        print(f"  全文: {fulltext.num_pages} 页, {fulltext.num_chars} 字符\n")

    # 3. 执行分析
    analyzer = PaperAnalyzer(LLMClient(settings), settings)
    print("⏳ 正在分析（Map-Reduce + 图片提取 + 概念库）...\n")
    result = analyzer.analyze(paper, use_fulltext=True)

    # 4. 评估笔记质量
    note_path = Path(result.get("_note_path", ""))
    if not note_path.exists():
        # 尝试在 paper_notes 目录找
        notes_dir = settings.workspace_dir / "paper_notes"
        candidates = list(notes_dir.glob("*.md"))
        if candidates:
            note_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if note_path.exists():
        metrics = evaluate_note_quality(note_path, raw_text)
        print_eval(metrics)

        # 保存
        import json
        output_path = root / "eval" / "paper_eval_results.json"
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 详细结果已保存: {output_path}")
    else:
        print("❌ 未找到生成的笔记文件")
        print(f"   分析结果: {json.dumps(result, ensure_ascii=False, default=str)[:500]}")
