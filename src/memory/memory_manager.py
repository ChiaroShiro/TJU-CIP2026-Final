"""
记忆管理器（Memory Manager）- Hermes风格三层记忆统一接口

三层架构：
  Layer 1 - Session Memory  : ContextManager（内存，当前会话对话历史）
  Layer 2 - Episodic Memory : SQLite+FTS5（跨会话，历史研究情节）
  Layer 3 - Skill Memory    : SQLite+FTS5（跨会话，学到的研究技能）
  + Vector Store            : Chroma（语义检索，跨所有内容）

检索流程：
  召回阶段（Recall） - 每路扩大召回，取较多候选
  精排阶段（Rerank） - Cross-Encoder 对所有候选统一重排
  输出阶段         - 按 source 回填到各个分类，或输出统一 top_k
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from ..core.context_manager import ContextManager
from ..core.models import MemoryHit
from .episodic_memory import Episode, EpisodicMemory
from .reranker import CrossEncoderReranker, RerankCandidate, build_candidates_from_memory
from .skill_memory import Skill, SkillMemory
from .vector_store import VectorMemory


class MemoryManager:
    """
    三层记忆管理器，统一管理 Session / Episodic / Skill 三层记忆。

    使用方式：
        ctx = memory.get_context_for_task("transformer attention mechanism")
        prompt_ctx = memory.format_context_for_prompt("transformer attention mechanism")
    """

    # 召回阶段的扩大倍数（召回更多再让 reranker 精排）
    RECALL_MULTIPLIER = 4

    def __init__(self, settings: Settings, enable_rerank: Optional[bool] = None):
        mem_dir = settings.workspace_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: 会话记忆（内存）
        self.session = ContextManager(settings.context_max_chars)

        # Layer 2: 情节记忆（SQLite + FTS5）
        self.episodic = EpisodicMemory(mem_dir / "episodic.db")

        # Layer 3: 技能记忆（SQLite + FTS5）
        self.skill = SkillMemory(mem_dir / "skills.db")

        # 语义向量存储（Chroma）
        self.vector = VectorMemory(settings.workspace_dir / "vector_db")

        # Cross-Encoder 精排器（懒加载）
        self.enable_rerank = (
            settings.enable_rerank if enable_rerank is None else enable_rerank
        )
        self.reranker: Optional[CrossEncoderReranker] = (
            CrossEncoderReranker(
                settings.rerank_model,
                score_threshold=settings.rerank_score_threshold,
            ) if self.enable_rerank else None
        )

        self._top_k = settings.memory_top_k
        self._ep_k = settings.memory_episode_k
        self._sk_k = settings.memory_skill_k

        # 技能使用追踪：最近一次 get_context_for_task 返回的技能 id 列表
        # 研究完成后 ReflectionEngine 会按质量反馈成功与否
        self._last_used_skill_ids: List[str] = []

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #

    def get_context_for_task(self, query: str) -> Dict[str, Any]:
        """
        同时查询所有记忆层，返回结构化上下文。

        流程：
          1. 每路以 RECALL_MULTIPLIER 扩大召回
          2. 统一成 RerankCandidate 列表
          3. Cross-Encoder 精排
          4. 按 source 回填，返回最终 top_k

        Returns:
            {
              "session":  session messages,
              "episodes": List[Episode],   # 精排后保留的
              "skills":   List[Skill],
              "vectors":  List[MemoryHit],
              "reranked": List[RerankCandidate],  # 统一精排 top_k（可直接给 LM）
            }
        """
        recall_k = self._top_k * self.RECALL_MULTIPLIER

        # 1. 扩大召回
        episodes = self.episodic.search(query, limit=recall_k)
        skills = self.skill.find_relevant(query, limit=recall_k)
        vectors = self.vector.retrieve(query, recall_k)

        # 1.5 向量命中回查 + 跨源去重
        # 向量库里的 doc_id 形如 "episode:xxx" / "skill:xxx" / "task:xxx" / "direction:xxx"
        # 对前两种做回查得到完整对象，合并进 episodes/skills 列表（去重）；
        # 其余（task/direction 等）保留为纯向量命中。
        episodes, skills, vectors = self._resolve_vector_hits(episodes, skills, vectors)

        # 2. 如果未启用 rerank，走原路径
        if not self.enable_rerank or self.reranker is None:
            sk_returned = skills[:self._sk_k]
            self._last_used_skill_ids = [s.id for s in sk_returned]
            return {
                "session": self.session.get_context(),
                "episodes": episodes[:self._ep_k],
                "skills": sk_returned,
                "vectors": vectors[:self._top_k],
                "reranked": [],
            }

        # 3. 构建统一候选并精排
        candidates = build_candidates_from_memory(episodes, skills, vectors)
        # 给下游三路分流留足空间：ep_k + sk_k + top_k
        rerank_output_k = self._ep_k + self._sk_k + self._top_k
        reranked = self.reranker.rerank(query, candidates, top_k=rerank_output_k)

        # 4. 按 source 回填
        ep_reranked = [c.raw for c in reranked if c.source == "episode"][:self._ep_k]
        sk_reranked = [c.raw for c in reranked if c.source == "skill"][:self._sk_k]
        vec_reranked = [c.raw for c in reranked if c.source == "vector"][:self._top_k]

        # 记录本轮精排后实际要注入的技能 id
        self._last_used_skill_ids = [s.id for s in sk_reranked]

        return {
            "session": self.session.get_context(),
            "episodes": ep_reranked,
            "skills": sk_reranked,
            "vectors": vec_reranked,
            "reranked": reranked[:self._top_k],
        }

    def format_context_for_prompt(self, query: str) -> str:
        """将所有记忆层的相关内容格式化为可注入提示词的字符串。"""
        ctx = self.get_context_for_task(query)
        parts: List[str] = []

        if ctx["episodes"]:
            parts.append("### 历史研究情节")
            for ep in ctx["episodes"]:
                parts.append(
                    f"**主题:** {ep.topic}  (质量: {ep.quality_score:.2f})\n"
                    f"**洞见:** {ep.insights}"
                )

        if ctx["skills"]:
            parts.append("### 已学研究技能（按 skill-creator 范式组织）")
            for sk in ctx["skills"]:
                # 先给 planner 看 description（最精炼） + trigger（判断是否匹配）
                # 再附 content 前 800 字符（完整 procedure 的头部，够用）
                head = (
                    f"**{sk.name}** [{sk.domain}]  "
                    f"(使用 {sk.usage_count} 次, 成功率 {sk.success_rate:.2f})\n"
                    f"_说明_: {sk.description}\n"
                    f"_触发_: {sk.trigger_conditions}"
                )
                body = sk.content[:800]
                if len(sk.content) > 800:
                    body += "\n...(truncated)"
                parts.append(f"{head}\n\n{body}")

        if ctx["vectors"]:
            parts.append("### 语义记忆片段")
            for hit in ctx["vectors"]:
                parts.append(hit.content[:400])

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------ #
    # 向量命中回查与跨源去重
    # ------------------------------------------------------------------ #

    def _resolve_vector_hits(self, episodes, skills, vectors):
        """
        处理向量库命中，让结果按"真实身份"归类：
          - 前缀 "episode:" → 回查 EpisodicMemory，合并进 episodes（已在情节列表中的跳过）
          - 前缀 "skill:"   → 回查 SkillMemory，合并进 skills
          - 其他前缀（task:/direction: 等）→ 保留为纯向量命中

        这样做的好处：
          1. 向量命中能拿到结构化完整字段（Episode.content/tags，Skill.trigger 等）
          2. Rerank 阶段按 doc_id 去重时，同一条 episode 只保留一份
        """
        ep_ids = {e.id for e in episodes}
        sk_ids = {s.id for s in skills}
        remaining_vectors = []

        for hit in vectors:
            doc_id = hit.doc_id or ""

            if doc_id.startswith("episode:"):
                ep_id = doc_id.split(":", 1)[1]
                if ep_id in ep_ids:
                    continue  # 情节库已召回，无需重复
                ep = self.episodic.get_by_id(ep_id)
                if ep is not None:
                    episodes.append(ep)
                    ep_ids.add(ep_id)
                # 回查失败时也不保留纯向量（避免只有缩略）
                continue

            if doc_id.startswith("skill:"):
                sk_id = doc_id.split(":", 1)[1]
                if sk_id in sk_ids:
                    continue
                sk = self.skill.get_by_id(sk_id)
                if sk is not None:
                    skills.append(sk)
                    sk_ids.add(sk_id)
                continue

            remaining_vectors.append(hit)

        return episodes, skills, remaining_vectors

    # ------------------------------------------------------------------ #
    # 写入接口
    # ------------------------------------------------------------------ #

    def save_task_result(self, doc_id: str, title: str, body: str):
        """将单个任务结果存入向量存储。"""
        self.vector.add(doc_id, body, {"title": title})

    def save_research_episode(
        self,
        topic: str,
        content: str,
        insights: str,
        tags: List[str],
        quality_score: float,
    ) -> str:
        """将完整研究会话存入情节记忆 + 向量存储。"""
        ep_id = self.episodic.add_episode(
            topic=topic,
            content=content,
            insights=insights,
            tags=tags,
            quality_score=quality_score,
        )
        self.vector.add(
            f"episode:{ep_id}",
            f"{topic}\n{insights}",
            {"topic": topic, "type": "episode", "ep_id": ep_id},
        )
        return ep_id

    def save_skill(
        self,
        name: str,
        description: str,
        trigger_conditions: str,
        content: str,
        domain: str = "general",
    ) -> str:
        """将学到的技能存入技能记忆 + 镜像到向量库（支持语义检索）。"""
        skill_id = self.skill.add_skill(
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            content=content,
            domain=domain,
        )
        # 镜像到向量库：description + trigger + content
        # description 首当其冲（它本身就是 discoverability 设计）
        # content 可能很长，截断到 1500 字符避免过度占用向量空间
        payload = f"{description}\n触发: {trigger_conditions}\n{content[:1500]}"
        self.vector.add(
            f"skill:{skill_id}",
            payload,
            {"type": "skill", "skill_id": skill_id, "domain": domain, "name": name},
        )
        return skill_id

    def feedback_skills_usage(self, success: bool) -> int:
        """
        对最近一次 get_context_for_task 返回的技能做使用反馈。

        通常由 ReflectionEngine 在研究完成后按质量阈值调用：
            memory.feedback_skills_usage(success=quality_score >= 0.7)

        返回：实际被反馈的技能数量。
        """
        count = 0
        for sk_id in self._last_used_skill_ids:
            try:
                self.skill.update_usage(sk_id, success=success)
                count += 1
            except Exception:
                continue
        self._last_used_skill_ids = []  # 反馈完清空，避免重复计数
        return count

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, int]:
        return {
            "episodes": self.episodic.count(),
            "skills": self.skill.count(),
            "vectors": self.vector.count(),
        }
