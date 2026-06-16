"""
ResearchManager — 自主研究经理 Agent

层级式 multi-agent 的主脑：
- 继承 BaseAgent，通过 function calling 自主决策每一步
- 工具箱包含搜索/阅读/记忆/委派评审/委派修改/保存
- 不硬编码研究流程，由 LLM 根据 system prompt 的策略建议自行规划
- Critic 和 Reviser 作为独立 worker 被委派调用（保证生成与评审分离）
"""

from ..core.config import Settings
from ..core.llm import LLMClient
from ..agents.base_agent import BaseAgent, AgentStep
from ..agents.paper_analyzer import PaperAnalyzer
from ..memory.memory_manager import MemoryManager
from ..learning.reflection import ReflectionEngine
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..services.github_search import GitHubCodeSearcher
from ..services.paper_fetcher import PaperFetcher
from ..services.web_searcher import WebSearcher
from ..tools.tool import ToolRegistry
from ..tools.search_tools import build_arxiv_search_tool, build_github_code_search_tool, build_s2_search_tool
from ..tools.paper_tools import build_fetch_fulltext_tool, build_analyze_paper_tool
from ..tools.memory_tools import (
    build_retrieve_memory_tool,
    build_save_note_tool,
    build_save_research_episode_tool,
)
from ..tools.delegation_tools import (
    build_delegate_to_critic_tool,
    build_delegate_to_reviser_tool,
)
from ..tools.web_tools import build_web_search_tool, build_web_fetch_tool


MANAGER_SYSTEM_PROMPT = """You are a Research Manager Agent — the central brain of a multi-agent research system.

## Your Role
You autonomously conduct deep academic research on a given topic. You have a toolbox of capabilities and must decide, step by step, what to do next based on what you've learned so far.

## Available Tools
- **search_arxiv** / **search_semantic_scholar**: Search for papers. Use short English keyword queries (3-8 words). Try multiple angles.
- **search_github_code**: Search GitHub repositories for public implementations. Use this to prioritize papers with code and collect reproducibility links.
- **fetch_paper_fulltext**: Download a paper's PDF and get section-level text. Use when abstract isn't enough.
- **analyze_paper**: Deep analysis of one paper (map-reduce over sections). Produces structured note with formulas, tables, critical view. Use on the 2-3 most important papers only.
- **retrieve_memory**: Check if you've researched this topic before. Call this FIRST to avoid repeating prior work.
- **save_note**: Save intermediate findings to memory (so future research can build on them).
- **delegate_to_critic**: Send your draft report to an independent Critic for peer review. The Critic scores coverage/evidence/coherence/actionability.
- **delegate_to_reviser**: Send report + critic review to a Reviser who will search for supplementary sources and rewrite weak sections.
- **save_research_episode**: Save the completed research session + trigger self-learning (extracts insights and skills). Call this ONCE at the very end.
- **finish**: End the research and return the final report.

## Research Strategy (suggested, not mandatory)

1. **Recall**: retrieve_memory first — see what you already know about this topic.
2. **Explore**: search with 3-5 different keyword angles. Don't just use one query.
3. **Code-first filtering**: for candidate papers, call search_github_code on paper titles or method names and prioritize papers with public code.
4. **Deep dive**: for the 2-3 most relevant papers, use analyze_paper to get full details.
5. **Synthesize**: save_note with your intermediate findings as you go.
6. **Draft**: when you have enough material, compose a comprehensive Markdown report.
7. **Review**: delegate_to_critic to get independent feedback.
8. **Revise**: if score < 0.7, delegate_to_reviser. You can also search more yourself.
9. **Finalize**: save_research_episode (triggers learning), then finish with the final report.

## Report Quality Standards

Your final report MUST include:
- Executive summary (TL;DR)
- Detailed findings organized by subtopic
- Specific numbers and citations (paper title + arXiv URL)
- Risks, limitations, and open questions
- Actionable recommendations
- References section with all cited papers
- Open-source code section listing GitHub repositories and which paper/method each repo supports
- Algorithm evolution section with dates ordered chronologically and no unsupported timeline claims

## Decision Rules

- If a search returns 0 results: try a different query formulation, don't give up.
- If you've searched 3+ times with no useful results: acknowledge the gap and move on.
- If the Critic scores < 0.5: you need major revision — consider searching more before revising.
- If the Critic scores 0.5-0.7: delegate_to_reviser should suffice.
- If the Critic scores > 0.7: report is acceptable, proceed to finalize.
- Maximum one revision cycle (critic → reviser). If still < 0.7 after revision, finalize anyway with a note about limitations.
- Always call save_research_episode before finish.

## HARD CONSTRAINTS (MUST FOLLOW)

1. **Total search calls ≤ 8**: Count ALL search tool calls (search_arxiv + search_semantic_scholar + web_search combined). After 8 searches, STOP searching and START writing the report immediately.
2. **By step 12, you MUST have called finish or be writing the report**: If you reach step 12 without having started drafting, immediately compose the report with whatever material you have and call finish.
3. **Never call the same tool with the same query twice**: If a search returned empty or failed, use a DIFFERENT tool or DIFFERENT query.
4. **If a tool returns an error mentioning "rate-limited"**: Do NOT retry that tool. Switch to an alternative immediately.

## Language

- Search queries: ALWAYS in English (arXiv/S2 don't support Chinese well)
- Report output: Chinese (unless user specifies otherwise)
- Tool arguments: English

## Token Budget Awareness

You have a STRICT token budget (~200K). Each step costs ~5-10K tokens. This means you have roughly 20-25 steps total. Plan accordingly:
- Steps 1-8: Search and gather material
- Steps 9-12: Write report + delegate_to_critic
- Steps 13-15: Revise if needed + save_research_episode + finish
- Do NOT spend more than 10 steps on searching. Material doesn't need to be perfect — write with what you have.
- If a search tool fails, don't retry — use a different source or move on.
"""


class ResearchManager(BaseAgent):
    """
    自主研究经理 Agent。

    使用方式：
        manager = ResearchManager(settings)
        result = manager.run("研究 DPO 在测试用例生成中的应用")
        print(result.final_output)   # 最终报告
        print(result.total_tokens)   # 总消耗
        print(len(result.steps))     # 决策步数
    """

    def __init__(
        self,
        settings: Settings,
        max_steps: int = 30,
        max_total_tokens: int = 200_000,
    ):
        self.settings = settings

        # 构造业务对象（生命周期由 Manager 管理）
        self.memory = MemoryManager(settings)
        self.reflection = ReflectionEngine(LLMClient(settings), self.memory)
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()
        self.github = GitHubCodeSearcher()
        self.fetcher = PaperFetcher(settings.workspace_dir / "pdf_cache")
        self.analyzer = PaperAnalyzer(LLMClient(settings), settings)
        self.web = WebSearcher()

        # 初始化 BaseAgent（会调用 build_tools + system_prompt）
        super().__init__(
            llm=LLMClient(settings),
            max_steps=max_steps,
            max_total_tokens=max_total_tokens,
            temperature=0.3,
        )

    def system_prompt(self) -> str:
        return MANAGER_SYSTEM_PROMPT

    def build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(build_arxiv_search_tool(self.arxiv))
        registry.register(build_s2_search_tool(self.s2))
        registry.register(build_github_code_search_tool(self.github))
        registry.register(build_fetch_fulltext_tool(self.fetcher))
        registry.register(build_analyze_paper_tool(self.analyzer, self.arxiv, self.memory))
        registry.register(build_retrieve_memory_tool(self.memory))
        registry.register(build_save_note_tool(self.memory))
        registry.register(build_save_research_episode_tool(self.memory, self.reflection))
        # Critic/Reviser 作为独立 worker（每次调用新建实例，角色隔离）
        registry.register(build_delegate_to_critic_tool(self.settings))
        registry.register(build_delegate_to_reviser_tool(self.settings))
        # Web 搜索（突破学术信息墙）
        registry.register(build_web_search_tool(self.web))
        registry.register(build_web_fetch_tool(self.web))
        return registry

    def on_step(self, step: AgentStep) -> None:
        """每步打印进度（可选，方便调试）。"""
        status = "✓" if (step.tool_result and step.tool_result.success) else "✗"
        if step.tool_name:
            print(f"  [{step.step_idx+1}] {status} {step.tool_name}  ({step.tokens_used} tokens, {step.elapsed_ms}ms)")
        elif step.thought:
            print(f"  [{step.step_idx+1}] 💭 {step.thought[:80]}")
