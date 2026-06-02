#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agents.conversational_agent import ConversationalAgent
from src.agents.manager import ResearchManager
from src.core.config import Settings
from src.core.llm import LLMClient
from src.orchestrator import ResearchOrchestrator


load_dotenv()

app = typer.Typer()
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
        console.print(f"\n[bold]找到 {len(papers)} 篇相关论文[/bold]")
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
def analyze(
    url: str = typer.Argument(..., help="论文 arXiv ID 或 URL"),
    focus: str = typer.Option(None, "--focus", "-f", help="关注点"),
):
    """深度分析一篇 arXiv 论文。"""
    orch = get_orchestrator()

    paper_id = url.rstrip("/").split("/")[-1].replace(".pdf", "")
    with console.status("分析论文中..."):
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            console.print("[red]未找到该论文[/red]")
            raise typer.Exit(1)
        result = orch.analyze_paper(papers[0], focus)

    console.print(Panel(f"[bold]{papers[0].title}[/bold]"))
    source = result.get("_source", "unknown")
    if source == "fulltext":
        console.print(
            f"[dim]分析来源: 全文 PDF ({result.get('_num_pages', '?')} 页, "
            f"{result.get('_num_chunks', '?')} 个章节块)[/dim]"
        )
    else:
        console.print("[yellow]分析来源: 仅摘要（全文下载失败，已降级）[/yellow]")

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


@app.command("read-paper")
def read_paper(
    pdf_path: str = typer.Argument(..., help="本地 PDF 论文路径"),
    title: str = typer.Option(None, "--title", "-t", help="可选：手动指定论文标题"),
    focus: str = typer.Option(None, "--focus", "-f", help="关注点"),
):
    """直接阅读本地 PDF 论文并生成结构化笔记。"""
    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.exists() or not pdf_file.is_file():
        console.print(f"[red]未找到 PDF 文件: {pdf_file}[/red]")
        raise typer.Exit(1)
    if pdf_file.suffix.lower() != ".pdf":
        console.print(f"[red]文件不是 PDF: {pdf_file}[/red]")
        raise typer.Exit(1)

    orch = get_orchestrator()
    display_title = title or orch.analyzer.fetcher.infer_title_from_pdf(pdf_file)
    console.print(Panel(f"[bold]本地论文阅读:[/bold] {display_title}", style="blue"))
    console.print(f"[dim]{pdf_file}[/dim]")

    with console.status("正在解析本地 PDF 并分析..."):
        result = orch.analyzer.analyze_local_pdf(pdf_file, title=title, focus=focus)

    console.print(Panel(f"[bold]{display_title}[/bold]"))
    source = result.get("_source", "unknown")
    if source == "local_pdf":
        console.print(
            f"[dim]分析来源: 本地 PDF ({result.get('_num_pages', '?')} 页, "
            f"{result.get('_num_chunks', '?')} 个章节块)[/dim]"
        )
    else:
        console.print("[yellow]本地 PDF 文本提取不足，已生成降级笔记[/yellow]")

    console.print(f"\n[bold]核心问题:[/bold] {result.get('problem', '')}")

    contributions = result.get("contributions", [])
    if contributions:
        console.print("\n[bold]主要贡献:[/bold]")
        for item in contributions:
            console.print(f"  • {item}")

    if result.get("method_summary"):
        console.print(f"\n[bold]方法概览:[/bold] {result.get('method_summary', '')}")

    datasets = result.get("datasets", [])
    if datasets:
        console.print("\n[bold]数据集:[/bold]")
        for dataset in datasets:
            console.print(f"  • {_format_dataset_brief(dataset)}")

    if result.get("results"):
        console.print(f"\n[bold]实验结果:[/bold] {result['results']}")

    weaknesses = result.get("weaknesses", [])
    if weaknesses:
        console.print("\n[bold]局限性:[/bold]")
        for item in weaknesses:
            console.print(f"  • {item}")

    note_path = result.get("_note_path")
    if note_path:
        console.print(f"\n[green]笔记已保存至: {note_path}[/green]")


@app.command()
def research(
    topic: str = typer.Argument(..., help="研究主题"),
    max_steps: int = typer.Option(30, "--max-steps", "-s", help="最大决策步数"),
    max_tokens: int = typer.Option(200000, "--max-tokens", "-t", help="总 token 上限"),
    legacy: bool = typer.Option(False, "--legacy", help="使用旧的编排式流程"),
):
    """对主题进行深度研究并生成报告。"""
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
        safe_topic = "".join(c if c.isalnum() or c in ' -_' else '' for c in topic)[:40].strip()
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


def _dispatch(orch, agent, action) -> str:
    """将路由动作翻译成具体后端调用。"""
    current_action = action.action

    if current_action in ("ask_user", "chitchat"):
        return action.reply or "我在听，请继续说。"

    if current_action == "memory_query":
        stats = orch.memory_stats()
        ctx = orch.memory.format_context_for_prompt(action.topic or "research")
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

    return action.reply or "我暂时还不确定该怎么帮你，能再说得具体一点吗？"


if __name__ == "__main__":
    app()
