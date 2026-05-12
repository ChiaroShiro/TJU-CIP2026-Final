from typing import List

from .models import CriticReview, ResearchResult, SourceItem, TaskPlanItem, TaskRunResult


def critic_prompt(topic: str, report_md: str) -> str:
    """
    评审 Agent 提示词（独立 system prompt 隔离角色）

    评判维度：
    - coverage: 是否覆盖主题的核心方面（0-1）
    - evidence_quality: 引用是否充分、来源是否可信（0-1）
    - coherence: 逻辑是否连贯、结构是否清晰（0-1）
    - actionability: 建议是否具体可执行（0-1）
    """
    return f"""Topic under review: {topic}

Report to review:
{report_md}

Evaluate this research report strictly and objectively.
Return JSON only:
{{
  "dimension_scores": {{
    "coverage": 0.0,
    "evidence_quality": 0.0,
    "coherence": 0.0,
    "actionability": 0.0
  }},
  "score": 0.0,
  "strengths": ["strength 1", "strength 2"],
  "suggestions": ["specific improvement 1", "specific improvement 2"],
  "missing_topics": ["topic that needs more research 1"]
}}

Rules:
- score = average of dimension_scores
- suggestions must be specific and actionable, not vague
- missing_topics: list subtopics that lack sufficient evidence and need supplementary search
- be strict: a score above 0.85 means the report is publication-ready
""".strip()


def reviser_prompt(
    topic: str,
    report_md: str,
    review: CriticReview,
    extra_sources: List[SourceItem],
) -> str:
    """
    修改 Agent 提示词（独立 system prompt 隔离角色）

    基于评审意见和补充搜索结果，重写报告中的薄弱部分。
    """
    suggestions_text = "\n".join(f"- {s}" for s in review.suggestions)
    missing_text = "\n".join(f"- {t}" for t in review.missing_topics)
    dim_text = "\n".join(
        f"  {k}: {v:.2f}" for k, v in review.dimension_scores.items()
    )
    sources_text = ""
    if extra_sources:
        sources_text = "\nSupplementary sources found:\n" + "\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
            for i, s in enumerate(extra_sources)
        )

    return f"""Topic: {topic}

Critic score: {review.score:.2f}
Dimension scores:
{dim_text}

Suggestions from critic:
{suggestions_text}

Topics needing more coverage:
{missing_text}
{sources_text}

Original report:
{report_md}

Rewrite the report addressing all critic suggestions.
- Incorporate supplementary sources where relevant (only cite provided URLs)
- Strengthen weak sections identified by the critic
- Keep strong sections largely intact
- Return the complete revised report in Markdown format (no JSON wrapper)
""".strip()


def memory_augmented_planner_prompt(topic: str, max_items: int, memory_context: str) -> str:
    """
    记忆增强规划提示词

    在标准规划基础上注入历史研究情节和已学技能，
    让规划Agent避免重复已知内容，直接深入未知领域。
    """
    return f"""
You are a planner agent for deep research with access to accumulated research memory.
Generate a concise research plan for this topic: {topic}

Accumulated memory context (use to avoid repeating known findings and go deeper):
{memory_context if memory_context else "No prior memory available."}

Return JSON only with schema:
{{
  "tasks": [
    {{
      "title": "task short title",
      "goal": "what this task should answer",
      "search_query": "web search query"
    }}
  ]
}}

Constraints:
- 3 to {max_items} tasks
- no duplicated task intent
- tasks should progressively build toward a final report
- if memory context exists, build on it rather than repeating what is already known
""".strip()


def learning_reflection_prompt(result: ResearchResult, quality_score: float) -> str:
    """
    自我学习反思提示词

    研究完成后，提炼关键洞见、标签和经验教训，
    存入情节记忆供未来研究参考。
    """
    task_summaries = "\n\n".join(
        f"Task: {t.task.title}\nSummary: {t.summary_markdown[:500]}"
        for t in result.task_results
    )
    return f"""
You are a self-learning research agent reflecting on a completed research session.

Topic: {result.topic}
Quality Score: {quality_score:.2f} (0=poor, 1=excellent)

Task Summaries:
{task_summaries}

Extract the most important insights from this research session.
Return JSON only:
{{
  "insights_summary": "2-3 sentence summary of the most important findings worth remembering",
  "tags": ["tag1", "tag2", "tag3"],
  "lessons_learned": ["lesson about research approach 1", "lesson 2"]
}}
""".strip()


def skill_extraction_prompt(result: ResearchResult) -> str:
    """
    技能提取提示词（参考 Anthropic Skill-Creator 范式）

    核心设计原则（来自 Anthropic Agent Skills / Claude Code skill-creator）:

    1. **Discoverability via description**
       - name + description 必须能让一个完全不了解当前研究的 agent
         仅凭这两个字段判断 "是否该调用此技能"，不能泛泛而谈
       - description 长度 80-200 字符，前半段是"何时用"，后半段是"做什么"

    2. **Concrete over abstract**
       - 禁止 "做好研究" 这类空话，必须写具体操作
       - 最少包含一个 action verb 和一个 measurable outcome

    3. **Self-contained & progressive disclosure**
       - content 字段要包含完整的 step-by-step 指令，不依赖外部上下文
       - 结构：triggers / inputs / steps / outputs / anti-patterns

    4. **Anti-patterns are mandatory**
       - 写清楚 "什么时候不该用这个技能"，防止误触发

    5. **Verifiable trigger conditions**
       - trigger_conditions 必须是可由后续 router / planner 机械判断的条件
       - 格式：列出 3-5 个触发信号，每条不超过一行
    """
    task_summaries = "\n\n".join(
        f"Task: {t.task.title}\n"
        f"Goal: {t.task.goal}\n"
        f"Search Query: {t.task.search_query}\n"
        f"Confidence: {t.confidence:.2f}\n"
        f"Summary: {t.summary_markdown[:400]}"
        for t in result.task_results
    )
    return f"""You are a skill curator for a research agent, following the Anthropic skill-creator methodology.

Your job: examine a completed research session and extract reusable skills that would genuinely help future research on DIFFERENT topics. This is not a summary of what happened — it is abstracting patterns that generalize.

# Research Session to Analyze

Topic: {result.topic}

Task trace:
{task_summaries}

# Skill Quality Bar (Anthropic Skill-Creator Style)

A skill is worth extracting ONLY if all of the following hold:
- It describes a **reusable procedure**, not a one-off observation
- The procedure worked in this session (average confidence >= 0.5)
- The procedure is NOT trivially obvious (e.g. "use google to search" is not a skill)
- A model without prior context can execute the procedure purely from the skill's content field

If fewer than 1 pattern clears this bar, return an empty list. DO NOT pad.

# Output Format

Return JSON only:
{{
  "skills": [
    {{
      "name": "verb-phrase-kebab-case (<= 5 words)",
      "description": "One 80-200 char sentence: when to use + what it does. Must be discoverable by keyword match. Example: 'When evaluating a novel method combination with limited prior art, search each component separately then synthesize overlap gaps.'",
      "trigger_conditions": "3-5 bullet signals, one per line with '- ' prefix. Must be mechanically checkable, e.g. '- user query contains \\"novel combination\\" or \\"新组合\\"' or '- planning phase for a topic with <5 exact-match papers'",
      "content": "# Procedure\\n\\n## When to use\\n<concrete triggers>\\n\\n## Inputs needed\\n<list>\\n\\n## Steps\\n1. <action verb + measurable outcome>\\n2. ...\\n\\n## Outputs\\n<list>\\n\\n## Anti-patterns\\n- <when NOT to use>\\n\\n## Example\\n<concrete mini-example from this session>",
      "domain": "one of: research_methodology | search_strategy | evidence_synthesis | evaluation | writing | general"
    }}
  ]
}}

# Additional Rules

- "content" MUST be a full multi-section markdown block using the template above, not 1-3 sentences
- Use \\n for newlines inside JSON strings
- Do not reference the specific topic "{result.topic}" in name/description — skills must generalize
- If you find only 1 high-quality skill, return 1. Never inflate to 3.
- Maximum 3 skills per session.
""".strip()


def planner_prompt(topic: str, max_items: int) -> str:
    """
    规划Agent提示词

    功能：为研究主题生成结构化的任务计划
    输入：研究主题、最大任务数
    输出：JSON格式的任务列表，每个任务包含标题、目标、搜索查询

    约束：
    - 生成3到max_items个任务
    - 任务之间不重复
    - 任务应逐步构建最终报告
    """
    return f"""
You are a planner agent for deep research.
Generate a concise research plan for this topic: {topic}

Return JSON only with schema:
{{
  "tasks": [
    {{
      "title": "task short title",
      "goal": "what this task should answer",
      "search_query": "web search query"
    }}
  ]
}}

Constraints:
- 3 to {max_items} tasks
- no duplicated task intent
- tasks should progressively build toward a final report
""".strip()


def summarizer_prompt(
    topic: str,
    task: TaskPlanItem,
    sources: List[SourceItem],
    rag_context: str,
) -> str:
    """
    摘要Agent提示词

    功能：基于网络搜索结果和RAG记忆，为单个任务生成摘要
    输入：
    - topic: 研究主题
    - task: 当前任务信息（标题、目标）
    - sources: 网络搜索到的论文/资料列表
    - rag_context: 从记忆系统检索到的相关历史内容

    输出：JSON格式，包含：
    - summary_markdown: 任务摘要（Markdown格式）
    - key_points: 关键要点列表
    - citations: 引用列表（标题、URL、引用原因）
    - confidence: 置信度（0-1）

    规则：
    - 只能引用提供的网络证据中的URL
    - 置信度范围0到1
    - 如果证据不足，明确标记不确定性
    """
    sources_text = "\n".join(
        f"[{i+1}] {src.title}\nURL: {src.url}\nSnippet: {src.snippet}"
        for i, src in enumerate(sources)
    )
    return f"""
You are a research summarizer.
Topic: {topic}
Current task title: {task.title}
Current task goal: {task.goal}

Retrieved memory context (RAG):
{rag_context}

Web evidence:
{sources_text}

Return JSON only with schema:
{{
  "summary_markdown": "concise markdown summary for this task",
  "key_points": ["point 1", "point 2"],
  "citations": [
    {{
      "title": "source title",
      "url": "https://...",
      "reason": "why this source supports your point"
    }}
  ],
  "confidence": 0.0
}}

Rules:
- only cite URLs from provided web evidence
- confidence range is 0 to 1
- explicitly mark uncertainty if evidence is weak
""".strip()


def reflection_prompt(topic: str, result: TaskRunResult) -> str:
    """
    反思Agent提示词

    功能：检查研究质量，判断是否需要补充搜索
    输入：
    - topic: 研究主题
    - result: 当前任务的执行结果（包含摘要）

    输出：JSON格式，包含：
    - needs_more_research: 是否需要更多研究（布尔值）
    - follow_up_query: 后续搜索查询（如果需要）
    - reason: 判断原因

    规则：
    - 只有在存在重要事实空白时才设置needs_more_research=true
    """
    return f"""
You are a critic agent checking research quality.
Topic: {topic}
Task title: {result.task.title}
Task goal: {result.task.goal}
Current summary:
{result.summary_markdown}

Return JSON only:
{{
  "needs_more_research": true or false,
  "follow_up_query": "new query string if needed else empty",
  "reason": "short reason"
}}

Set needs_more_research=true only if there is an important factual gap.
""".strip()


def reporter_prompt(topic: str, task_results: List[TaskRunResult]) -> str:
    """
    报告写作Agent提示词

    功能：将所有任务结果整合为完整的研究报告
    输入：
    - topic: 研究主题
    - task_results: 所有任务的执行结果列表

    输出：Markdown格式的完整报告，必须包含：
    1. 执行摘要（Executive summary）
    2. 分章节的详细发现（Detailed findings）
    3. 风险和不确定性（Risks and uncertainty）
    4. 可行建议（Actionable recommendations）
    5. 带URL的参考文献（References）
    """
    blocks = []
    for i, item in enumerate(task_results):
        cits = "\n".join(f"- {c.title} ({c.url})" for c in item.citations)
        blocks.append(
            f"""
Task {i+1}: {item.task.title}
Goal: {item.task.goal}
Summary:
{item.summary_markdown}

Citations:
{cits or "- None"}
""".strip()
        )
    payload = "\n\n".join(blocks)
    return f"""
You are a report writer agent.
Produce a final markdown report for topic: {topic}

You must include:
1. Executive summary
2. Detailed findings by section
3. Risks and uncertainty
4. Actionable recommendations
5. References with URLs

Use these task materials:
{payload}
""".strip()
