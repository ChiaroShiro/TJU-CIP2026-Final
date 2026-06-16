#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import sys
import time

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agents.conversational_agent import ConversationalAgent
from src.agents.manager import ResearchManager
from src.core.config import Settings
from src.core.llm import LLMClient
from src.gui_app import launch_gui
from src.orchestrator import ResearchOrchestrator
from src.services.survey_builder import SurveyBuilder


load_dotenv()

app = typer.Typer()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
console = Console()


def get_orchestrator() -> ResearchOrchestrator:
    root = Path(__file__).parent
    settings = Settings.from_env(root)
    return ResearchOrchestrator(settings)


def _format_dataset_brief(dataset) -> str:
    if isinstance(dataset, str):
        return dataset
    if not isinstance(dataset, dict):
        return str(dataset)

    name = dataset.get("name", "") or "Unknown"
    extras = []
    if dataset.get("used_for"):
        extras.append(str(dataset["used_for"]))
    if dataset.get("size"):
        extras.append(str(dataset["size"]))
    return f"{name} ({', '.join(extras)})" if extras else name


def _format_module_brief(module) -> str:
    if isinstance(module, str):
        return module
    if not isinstance(module, dict):
        return str(module)

    name = module.get("name", "") or "Unnamed module"
    details = module.get("motivation") or module.get("design") or ""
    return f"{name}: {details}" if details else name


def _print_analysis_result(title: str, result: dict, local: bool = False):
    console.print(Panel(f"[bold]{title}[/bold]"))
    source = result.get("_source", "unknown")
    if source in {"fulltext", "local_pdf"}:
        label = "本地 PDF" if local else "全文 PDF"
        console.print(
            f"[dim]分析来源: {label} ({result.get('_num_pages', '?')} 页, "
            f"{result.get('_num_chunks', '?')} 个章节块)[/dim]"
        )
    else:
        console.print("[yellow]分析来源: 仅摘要（全文不可用，已自动降级）[/yellow]")

    mode = result.get("_analysis_mode")
    if mode == "multimodal":
        console.print("[dim]分析模式: 多模态细读版[/dim]")
    elif mode == "conservative":
        console.print("[dim]分析模式: 保守版[/dim]")

    console.print(f"\n[bold]核心问题:[/bold] {result.get('problem', '')}")

    contributions = result.get("contributions", [])
    if contributions:
        console.print("\n[bold]主要贡献:[/bold]")
        for item in contributions:
            console.print(f"  • {item}")

    method_summary = result.get("method_summary", "")
    modules = result.get("modules", [])
    if method_summary or modules:
        console.print("\n[bold]方法:[/bold]")
        if method_summary:
            console.print(method_summary)
        for module in modules[:5]:
            console.print(f"  • {_format_module_brief(module)}")

    datasets = result.get("datasets", [])
    if datasets:
        console.print("\n[bold]数据集:[/bold]")
        for dataset in datasets:
            console.print(f"  • {_format_dataset_brief(dataset)}")

    cited_similar_work = result.get("cited_similar_work", [])
    if cited_similar_work:
        console.print("\n[bold]引用中的相似工作:[/bold]")
        for item in cited_similar_work[:5]:
            if isinstance(item, str):
                console.print(f"  • {item}")
                continue
            title_text = item.get("title", "") or "Unknown"
            category = item.get("category", "")
            why_related = item.get("why_related", "")
            suffix = f" [{category}]" if category else ""
            if why_related:
                console.print(f"  • {title_text}{suffix}: {why_related}")
            else:
                console.print(f"  • {title_text}{suffix}")

    if result.get("results"):
        console.print(f"\n[bold]实验结果:[/bold] {result['results']}")

    weaknesses = result.get("weaknesses", []) or result.get("limitations", [])
    if weaknesses:
        console.print("\n[bold]局限性:[/bold]")
        for item in weaknesses:
            console.print(f"  • {item}")

    if result.get("future_work"):
        console.print(f"\n[bold]未来方向:[/bold] {result['future_work']}")

    note_path = result.get("_note_path")
    if note_path:
        console.print(f"\n[green]笔记已保存至: {note_path}[/green]")


@app.command()
def evaluate(direction: str = typer.Argument(..., help="研究方向描述")):
    """评估研究方向的可行性和价值。"""
    console.print(Panel(f"[bold]评估研究方向:[/bold] {direction}", style="blue"))
    orch = get_orchestrator()

    with console.status("正在检索相关论文并评估..."):
        result = orch.evaluate_direction(direction)

    table = Table(title="评估结果")
    table.add_column("维度", style="cyan")
    table.add_column("得分", style="green")
    table.add_row("可行性", f"{result.get('feasibility', 0):.2f}")
    table.add_row("新颖性", f"{result.get('novelty', 0):.2f}")
    table.add_row("影响力", f"{result.get('impact', 0):.2f}")
    console.print(table)
    console.print(Panel(result.get("analysis", ""), title="详细分析"))

    papers = result.get("papers", [])
    if papers:
        console.print(f"\n[bold]找到 {len(papers)} 篇相关文章[/bold]")
        for i, paper in enumerate(papers[:5], 1):
            console.print(f"  {i}. {paper.title}")
            console.print(f"     {paper.url}")


@app.command()
def search(query: str = typer.Argument(..., help="搜索关键词")):
    """搜索相关论文。"""
    orch = get_orchestrator()
    with console.status("搜索中..."):
        papers = orch.search_papers(query)

    console.print(f"\n[bold]找到 {len(papers)} 篇论文[/bold]")
    for i, paper in enumerate(papers, 1):
        console.print(f"\n[cyan]{i}. {paper.title}[/cyan]")
        console.print(f"   作者: {', '.join(paper.authors[:3])}")
        console.print(f"   摘要: {paper.abstract[:150]}...")
        console.print(f"   链接: {paper.url}")


@app.command()
def discover(
    topic: str = typer.Argument(..., help="研究主题或英文关键词"),
    max_papers: int = typer.Option(10, "--max-papers", "-n", help="最多返回论文数"),
):
    """搜索最新论文并自动关联 GitHub 代码，公开代码论文优先。"""
    orch = get_orchestrator()
    with console.status("正在检索论文并关联 GitHub 代码..."):
        papers = orch.discover_papers(topic, max_results=max_papers)

    table = Table(title=f"论文发现结果: {topic}")
    table.add_column("#", style="cyan", width=4)
    table.add_column("年份", width=8)
    table.add_column("论文")
    table.add_column("公开代码", style="green")
    table.add_column("链接")
    for i, paper in enumerate(papers, 1):
        year = (paper.published or paper.updated or "")[:4] or "?"
        table.add_row(str(i), year, paper.title, paper.code_url or "-", paper.url)
    console.print(table)


@app.command()
def survey(
    topic: str = typer.Argument(..., help="研究主题"),
    max_papers: int = typer.Option(12, "--max-papers", "-n", help="综述纳入论文数"),
):
    """生成文献综述草稿、算法演进图和 SVG 海报。无需 LLM API key。"""
    root = Path(__file__).parent
    settings = Settings.from_env(root)
    builder = SurveyBuilder(settings.workspace_dir)

    with console.status("正在生成综述、演进图和海报..."):
        artifact = builder.build(topic, max_papers=max_papers)

    console.print(Panel("[bold green]综述产物已生成[/bold green]", border_style="green"))
    console.print(f"报告: {artifact.report_file}")
    console.print(f"演进图: {artifact.timeline_file}")
    console.print(f"海报: {artifact.poster_file}")
    console.print(f"原始数据: {artifact.raw_data_file}")


@app.command()
def analyze(
    url: str = typer.Argument(..., help="论文 arXiv ID 或 URL"),
    focus: str = typer.Option(None, "--focus", "-f", help="关注点"),
):
    """保守版 arXiv 论文分析。调研默认使用这一版。"""
    orch = get_orchestrator()
    paper_id = url.rstrip("/").split("/")[-1].replace(".pdf", "")

    with console.status("分析论文中..."):
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            console.print("[red]未找到该论文[/red]")
            raise typer.Exit(1)
        result = orch.analyze_paper(papers[0], focus)

    _print_analysis_result(papers[0].title, result, local=False)


@app.command("read-paper")
def read_paper(
    source: str = typer.Argument(..., help="本地 PDF 路径，或 arXiv ID / arXiv URL"),
    title: str = typer.Option(None, "--title", "-t", help="本地 PDF 可选手动指定标题"),
    focus: str = typer.Option(None, "--focus", "-f", help="关注点"),
):
    """细读论文。arXiv 走多模态细读版，本地 PDF 走保守版。"""
    orch = get_orchestrator()

    possible_file = Path(source).expanduser()
    if possible_file.exists() and possible_file.is_file():
        pdf_file = possible_file.resolve()
        if pdf_file.suffix.lower() != ".pdf":
            console.print(f"[red]文件不是 PDF: {pdf_file}[/red]")
            raise typer.Exit(1)

        display_title = title or orch.analyzer.fetcher.infer_title_from_pdf(pdf_file)
        console.print(Panel(f"[bold]本地论文阅读（保守版）[/bold] {display_title}", style="blue"))
        console.print(f"[dim]{pdf_file}[/dim]")

        with console.status("正在解析本地 PDF 并分析..."):
            result = orch.analyzer.analyze_local_pdf(pdf_file, title=title, focus=focus)

        _print_analysis_result(display_title, result, local=True)
        return

    paper_id = source.rstrip("/").split("/")[-1].replace(".pdf", "")
    console.print(Panel(f"[bold]arXiv 论文阅读（多模态细读版）[/bold] {paper_id}", style="blue"))
    with console.status("正在下载 arXiv 论文并进行多模态分析..."):
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            console.print("[red]未找到该 arXiv 论文[/red]")
            raise typer.Exit(1)
        result = orch.analyze_paper_multimodal(papers[0], focus)

    _print_analysis_result(papers[0].title, result, local=False)


@app.command("paper-reader")
def paper_reader(
    source: str = typer.Argument(..., help="本地 PDF 路径，或 arXiv ID / arXiv URL"),
    title: str = typer.Option(None, "--title", "-t", help="本地 PDF 可选手动指定标题"),
    focus: str = typer.Option(None, "--focus", "-f", help="关注点"),
):
    """read-paper 的别名。"""
    read_paper(source=source, title=title, focus=focus)


@app.command()
def research(
    topic: str = typer.Argument(..., help="研究主题"),
    max_steps: int = typer.Option(30, "--max-steps", "-s", help="最大决策步数"),
    max_tokens: int = typer.Option(200000, "--max-tokens", "-t", help="总 token 上限"),
    legacy: bool = typer.Option(False, "--legacy", help="使用旧的编排式流程"),
):
    """对主题进行深度研究并生成报告。默认走保守版论文分析链路。"""
    root = Path(__file__).parent
    settings = Settings.from_env(root)

    if legacy:
        console.print(Panel(f"[bold]深度研究（编排模式）:[/bold] {topic}", style="yellow"))
        orch = get_orchestrator()
        with console.status("研究中，请稍候..."):
            result = orch.run_deep_research(topic)
        console.print(f"\n[green]报告已保存至: {result.report_file}[/green]")
        stats = orch.memory_stats()
        console.print(
            f"[dim]记忆库: {stats['episodes']} 情节 | {stats['skills']} 技能 | {stats['vectors']} 向量[/dim]"
        )
        return

    console.print(
        Panel(
            f"[bold cyan]自主研究模式[/bold cyan]\n"
            f"主题: {topic}\n"
            f"预算: {max_steps} 步 / {max_tokens:,} tokens",
            border_style="cyan",
        )
    )

    manager = ResearchManager(settings, max_steps=max_steps, max_total_tokens=max_tokens)
    console.print("[dim]Agent 开始自主研究...[/dim]\n")
    result = manager.run(topic)

    console.print(f"\n{'─' * 60}")
    if result.finished:
        console.print(f"[green]研究完成[/green] ({result.finish_reason})")
        output = result.final_output
        report = output.get("output", str(output)) if isinstance(output, dict) else str(output)

        reports_dir = settings.workspace_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:40].strip()
        report_path = reports_dir / f"{timestamp}_{safe_topic}.md"
        report_path.write_text(report, encoding="utf-8")

        console.print(f"[green]报告已保存至: {report_path}[/green]")
        console.print(Panel(report[:3000], title="最终报告（前 3000 字）", border_style="green"))
    else:
        console.print(f"[yellow]研究未完成[/yellow] (原因: {result.finish_reason})")
        if result.steps:
            last = result.steps[-1]
            last_status = last.tool_result.error if last.tool_result and not last.tool_result.success else "ok"
            console.print(f"[dim]最后一步: {last.tool_name} -> {last_status}[/dim]")

    console.print(
        f"\n[dim]总步数: {len(result.steps)} | 总 tokens: {result.total_tokens:,} | "
        f"耗时: {result.total_elapsed_ms / 1000:.1f}s[/dim]"
    )

    if result.steps:
        console.print("\n[bold]执行轨迹:[/bold]")
        for step in result.steps:
            success = bool(step.tool_result and step.tool_result.success)
            status = "OK" if success else "ERR"
            name = step.tool_name or "(thinking)"
            console.print(f"  {step.step_idx + 1}. [{status}] {name} ({step.tokens_used} tok, {step.elapsed_ms}ms)")


@app.command()
def chat():
    """对话式交互模式。"""
    orch = get_orchestrator()
    agent = ConversationalAgent(LLMClient(orch.settings), orch.memory)

    console.print(
        Panel.fit(
            "[bold cyan]Deep Research Agent - 对话模式[/bold cyan]\n"
            "直接用自然语言描述你的研究想法或问题。\n"
            "命令: [dim]/quit 退出 | /clear 清空历史 | /stats 查看记忆统计[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            console.print("[yellow]再见[/yellow]")
            break
        if user_input.lower() == "/clear":
            agent.clear()
            console.print("[dim]对话历史已清空[/dim]")
            continue
        if user_input.lower() == "/stats":
            stats = orch.memory_stats()
            console.print(f"[dim]情节: {stats['episodes']} | 技能: {stats['skills']} | 向量: {stats['vectors']}[/dim]")
            continue

        agent.add_user_message(user_input)

        with console.status("[cyan]思考中...[/cyan]"):
            action = agent.decide()

        console.print(f"[dim]-> 意图: {action.raw_intent} | 动作: {action.action}[/dim]")
        if action.queries:
            console.print(f"[dim]-> 检索词: {' / '.join(action.queries)}[/dim]")

        try:
            reply = _dispatch(orch, agent, action)
        except Exception as exc:
            reply = f"[red]执行出错: {type(exc).__name__}: {exc}[/red]"

        agent.add_assistant_message(reply)
        console.print(Panel(reply, title="[bold magenta]Agent[/bold magenta]", border_style="magenta"))


@app.command()
def gui(
    no_browser: bool = typer.Option(False, "--no-browser", help="只启动服务，不自动打开浏览器"),
):
    """启动本地 GUI 页面。"""
    orch = get_orchestrator()
    url = launch_gui(orch, open_browser=not no_browser)
    console.print(
        Panel.fit(
            f"[bold cyan]GUI 已启动[/bold cyan]\n"
            f"访问地址: {url}\n"
            f"按 [bold]Ctrl+C[/bold] 关闭服务",
            border_style="cyan",
        )
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]GUI 已关闭[/yellow]")


def _dispatch(orch, agent, action) -> str:
    """将路由动作翻译成具体后端调用。"""
    current_action = action.action

    if current_action in ("ask_user", "chitchat"):
        return action.reply or "我在听，请继续说。"

    if current_action == "memory_query":
        stats = orch.memory_stats()
        ctx = orch.memory.format_context_for_prompt(action.topic or "research")
        paper_graph = orch.memory.get_paper_graph_context(action.topic or "research", top_k=5)
        recent = [
            {
                "id": ep.id,
                "topic": ep.topic,
                "insights": ep.insights[:300],
                "created_at": ep.created_at,
            }
            for ep in orch.memory.get_recent_episodes(limit=5)
        ]
        skills = [
            {
                "id": sk.id,
                "name": sk.name,
                "description": sk.description[:300],
                "usage_count": sk.usage_count,
            }
            for sk in orch.memory.get_relevant_skills(action.topic or "research", limit=5)
        ]
        papers = [
            {
                "doc_id": hit.doc_id,
                "preview": hit.content[:300],
                "metadata": hit.metadata,
            }
            for hit in orch.memory.get_related_papers(action.topic or "research", top_k=5)
        ]
        brief = {
            "stats": stats,
            "context": ctx[:2000],
            "paper_graph": paper_graph,
            "recent_episodes": recent,
            "relevant_skills": skills,
            "related_papers": papers,
        }
        agent.add_tool_result(f"memory query done. {brief}")
        return agent.summarize_result(action, brief)

    if current_action == "paper_note_query":
        notes = orch.memory.find_paper_notes(action.topic or "", top_k=5)
        brief = {
            "query": action.topic,
            "note_count": len(notes),
            "notes": notes,
        }
        agent.add_tool_result(f"paper note query done. {brief}")
        return agent.summarize_result(action, brief)

    if current_action == "evaluate":
        topic = action.topic or (action.queries[0] if action.queries else "")
        result = orch.evaluate_direction(topic, queries=action.queries or None)
        brief = {
            "feasibility": result.get("feasibility"),
            "novelty": result.get("novelty"),
            "impact": result.get("impact"),
            "analysis": result.get("analysis", "")[:1500],
            "recommendations": result.get("recommendations", []),
            "benchmarks": result.get("benchmarks", []),
            "paper_count": len(result.get("papers", [])),
            "sample_papers": [{"title": p.title, "url": p.url} for p in result.get("papers", [])[:5]],
        }
        agent.add_tool_result(f"evaluate done. {brief}")
        return agent.summarize_result(action, brief)

    if current_action == "search":
        papers = orch.search_papers_multi(action.queries or [action.topic], per_query=4)
        brief = {
            "count": len(papers),
            "papers": [
                {
                    "title": p.title,
                    "authors": p.authors[:3],
                    "abstract": (p.abstract or "")[:200],
                    "url": p.url,
                }
                for p in papers[:8]
            ],
        }
        agent.add_tool_result(f"search done. got {len(papers)} papers.")
        return agent.summarize_result(action, brief)

    if current_action == "analyze":
        paper_id = action.paper_id.strip().split("/")[-1].replace(".pdf", "")
        if not paper_id:
            return "要分析论文的话，请给我 arXiv ID 或链接。"
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            return f"没找到 arXiv ID 为 {paper_id} 的论文，确认一下？"
        result = orch.analyze_paper(papers[0], action.focus or None)
        result["_title"] = papers[0].title
        agent.add_tool_result(f"analyze done for {papers[0].title}")
        return agent.summarize_result(action, result)

    if current_action == "research":
        topic = action.topic or (action.queries[0] if action.queries else "")
        console.print("[yellow]深度研究通常需要几分钟，请稍候...[/yellow]")
        result = orch.run_deep_research(topic)
        brief = {
            "topic": result.topic,
            "report_file": result.report_file,
            "plan_size": len(result.plan),
            "paper_count": len(result.papers),
            "critic_score": result.critic_reviews[-1].score if result.critic_reviews else None,
            "report_preview": result.final_report_markdown[:1500],
        }
        agent.add_tool_result(f"research done. report at {result.report_file}")
        return agent.summarize_result(action, brief)

    return action.reply or "我暂时还不确定该怎么帮你，你可以再具体一点。"


if __name__ == "__main__":
    app()
