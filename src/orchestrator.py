import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .core.config import Settings
from .core.llm import LLMClient
from .core.models import (
    PaperItem, ResearchResult, ReflectionResult,
    TaskPlanItem, TaskRunResult, Citation, SourceItem,
)
from .memory.memory_manager import MemoryManager
from .memory.store import NoteStore
from .agents.direction_evaluator import DirectionEvaluator
from .agents.paper_analyzer import PaperAnalyzer
from .agents.critic import CriticAgent
from .agents.reviser import ReviserAgent
from .services.paper_search import ArxivSearcher, SemanticScholarSearcher
from .services.paper_search import PaperDiscoveryService
from .core.prompts import (
    memory_augmented_planner_prompt, planner_prompt,
    summarizer_prompt, reflection_prompt, reporter_prompt,
)
from .core.utils import extract_json_object
from .learning.reflection import ReflectionEngine


class ResearchOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.memory = MemoryManager(settings)
        self.store = NoteStore(settings.workspace_dir / "notes")
        self.evaluator = DirectionEvaluator(self.llm, settings)
        self.analyzer = PaperAnalyzer(self.llm, settings, memory=self.memory)
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()
        self.discovery = PaperDiscoveryService(arxiv_max_results=settings.search_top_k)
        self.reflection_engine = ReflectionEngine(self.llm, self.memory)
        # 独立 LLMClient 实例，通过 system prompt 隔离角色
        self.critic = CriticAgent(
            LLMClient(settings),
            threshold=settings.critic_threshold,
            temperature=settings.critic_temperature,
        )
        self.reviser = ReviserAgent(
            LLMClient(settings),
            self.arxiv,
            self.s2,
            temperature=settings.reviser_temperature,
        )
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        (settings.workspace_dir / "reports").mkdir(exist_ok=True)

    def evaluate_direction(
        self, direction: str, queries: Optional[List[str]] = None, on_event=None,
    ) -> dict:
        result = self.evaluator.evaluate_direction(direction, queries=queries, on_event=on_event)
        self.memory.save_task_result(
            f"direction:{direction}", direction, result.get("analysis", "")
        )
        return result

    def analyze_paper(self, paper: PaperItem, focus: Optional[str] = None) -> dict:
        return self.analyzer.analyze(paper, focus)

    def analyze_paper_multimodal(self, paper: PaperItem, focus: Optional[str] = None) -> dict:
        return self.analyzer.analyze_multimodal(paper, focus)

    def search_papers_multi(self, queries: List[str], per_query: int = 4) -> List[PaperItem]:
        """用多个查询并行搜索，合并去重。"""
        all_papers: List[PaperItem] = []
        seen = set()
        for q in queries:
            for p in self.arxiv.search(q, max_results=per_query) + self.s2.search(q, max_results=per_query):
                if p.title and p.title not in seen:
                    seen.add(p.title)
                    all_papers.append(p)
        return all_papers

    def search_papers(self, query: str) -> List[PaperItem]:
        arxiv_papers = self.arxiv.search(query)
        s2_papers = self.s2.search(query, max_results=5)
        seen, papers = set(), []
        for p in arxiv_papers + s2_papers:
            if p.title not in seen:
                seen.add(p.title)
                papers.append(p)
        return papers

    def discover_papers(self, query: str, max_results: int = 10, on_event=None) -> List[PaperItem]:
        papers = self.discovery.search_topic(query, max_results=max_results, on_event=on_event)
        return self.discovery.enrich_with_code(papers, max_code_hits=3, on_event=on_event)

    def run_deep_research(self, topic: str, on_event=None) -> ResearchResult:
        """编排式深度研究。

        on_event 可选，传入时按阶段推送进度，并让最终报告逐字流式输出：
            {"type": "phase",  "label": str, "pct": int}
            {"type": "log",    "level": str, "text": str}
            {"type": "token",  "pane": "report", "text": str}
        """
        emit = on_event if on_event is not None else (lambda e: None)

        # 查询所有记忆层，为规划提供上下文
        emit({"type": "phase", "label": "回忆历史记忆", "pct": 5})
        memory_context = self.memory.format_context_for_prompt(topic)

        emit({"type": "phase", "label": "规划研究任务", "pct": 12})
        plan = self._plan(topic, memory_context)
        emit({"type": "log", "level": "info", "text": f"规划出 {len(plan)} 个子任务"})
        task_results, all_papers = [], []

        for task_idx, task in enumerate(plan):
            base_pct = 15 + int(45 * task_idx / max(1, len(plan)))
            emit({"type": "phase", "label": f"检索：{task.title}", "pct": base_pct})
            papers = self.search_papers(task.search_query)
            all_papers.extend(papers)
            emit({"type": "log", "level": "info",
                  "text": f"《{task.title}》检索到 {len(papers)} 篇"})
            sources = [
                SourceItem(title=p.title, url=p.url, snippet=p.abstract[:300], rank=i)
                for i, p in enumerate(papers[:self.settings.search_top_k])
            ]
            # 从向量存储检索语义相关记忆
            rag_hits = self.memory.vector.retrieve(task.goal, self.settings.memory_top_k)
            rag_context = "\n".join(h.content for h in rag_hits)

            emit({"type": "phase", "label": f"总结：{task.title}", "pct": base_pct + 4})
            result = self._summarize(topic, task, sources, rag_context)
            task_results.append(result)
            self.memory.save_task_result(
                f"task:{task.title}", task.goal, result.summary_markdown
            )
            self.store.save_note(
                f"task_{task.title[:30].replace(' ', '_')}",
                task.title, result.summary_markdown, {}
            )

        emit({"type": "phase", "label": "撰写综述报告", "pct": 65})
        report_md = self._write_report(topic, task_results, on_event=on_event)

        # 评审-修改循环：独立 Critic Agent 评审，低于阈值则 Reviser Agent 修改
        critic_reviews = []
        for round_i in range(self.settings.max_revision_rounds):
            emit({"type": "phase", "label": f"独立评审（第 {round_i + 1} 轮）", "pct": 80})
            review = self.critic.review(topic, report_md)
            critic_reviews.append(review)
            emit({"type": "log", "level": "info",
                  "text": f"Critic 评分 {review.score:.2f}"
                          + ("（需修改）" if review.needs_revision else "（通过）")})
            if not review.needs_revision:
                break
            emit({"type": "phase", "label": "根据评审修改报告", "pct": 88})
            report_md = self.reviser.revise(topic, report_md, review)

        report_file = self._save_report(topic, report_md)

        research_result = ResearchResult(
            topic=topic, plan=plan, task_results=task_results,
            final_report_markdown=report_md, report_file=report_file,
            papers=all_papers,
            critic_reviews=critic_reviews,
            revision_count=sum(1 for r in critic_reviews if r.needs_revision),
        )

        # 自我学习：反思本次研究，提炼洞见和技能存入记忆
        emit({"type": "phase", "label": "反思并沉淀记忆", "pct": 95})
        reflection_data = self.reflection_engine.reflect(research_result)
        research_result.reflection = ReflectionResult(
            episode_id=reflection_data["episode_id"],
            quality_score=reflection_data["quality_score"],
            insights_summary=reflection_data["insights_summary"],
            tags=reflection_data.get("tags", []),
            lessons_learned=reflection_data.get("lessons_learned", []),
            skills_learned=reflection_data["skills_learned"],
        )

        return research_result

    def memory_stats(self) -> dict:
        """返回三层记忆的统计信息。"""
        return self.memory.stats()

    def _plan(self, topic: str, memory_context: str = "") -> List[TaskPlanItem]:
        if memory_context:
            prompt = memory_augmented_planner_prompt(
                topic, self.settings.max_plan_items, memory_context
            )
        else:
            prompt = planner_prompt(topic, self.settings.max_plan_items)
        raw = self.llm.invoke(
            [{"role": "user", "content": prompt}], self.settings.planner_temperature
        )
        data = extract_json_object(raw)
        return [TaskPlanItem(**t) for t in data.get("tasks", [])]

    def _summarize(
        self, topic: str, task: TaskPlanItem,
        sources: List[SourceItem], rag_context: str
    ) -> TaskRunResult:
        prompt = summarizer_prompt(topic, task, sources, rag_context)
        raw = self.llm.invoke(
            [{"role": "user", "content": prompt}], self.settings.researcher_temperature
        )
        data = extract_json_object(raw)
        return TaskRunResult(
            task=task,
            summary_markdown=data.get("summary_markdown", raw),
            key_points=data.get("key_points", []),
            citations=[Citation(**c) for c in data.get("citations", [])],
            confidence=float(data.get("confidence", 0.5)),
            sources_used=sources,
        )

    def _write_report(self, topic: str, task_results: List[TaskRunResult], on_event=None) -> str:
        prompt = reporter_prompt(topic, task_results)
        messages = [{"role": "user", "content": prompt}]
        if on_event is None:
            return self.llm.invoke(messages, self.settings.writer_temperature)

        # 流式：逐字把报告推给前端打字机区，同时累计完整文本返回
        parts: List[str] = []
        for delta in self.llm.invoke_stream(messages, self.settings.writer_temperature):
            parts.append(delta)
            on_event({"type": "token", "pane": "report", "text": delta})
        return "".join(parts)

    def _save_report(self, topic: str, content: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{topic[:40].replace(' ', '_')}.md"
        path = self.settings.workspace_dir / "reports" / filename
        path.write_text(content, encoding="utf-8")
        return str(path)
