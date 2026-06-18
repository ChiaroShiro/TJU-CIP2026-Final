from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from ..core.context_manager import ContextManager
from ..core.models import MemoryHit
from .episodic_memory import EpisodicMemory
from .paper_graph import PaperGraphMemory
from .reranker import CrossEncoderReranker, RerankCandidate, build_candidates_from_memory
from .skill_memory import SkillMemory
from .vector_store import VectorMemory


class MemoryManager:
    """
    Unified memory manager for:
    - session memory
    - episodic memory
    - skill memory
    - vector memory
    - paper graph memory
    """

    RECALL_MULTIPLIER = 4
    MIN_RELATED_PAPER_CONFIDENCE = 0.45
    MIN_GRAPH_CONTEXT_CONFIDENCE = 0.58
    MIN_EDGE_CONFIDENCE = 0.60

    def __init__(self, settings: Settings, enable_rerank: Optional[bool] = None):
        mem_dir = settings.workspace_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.session = ContextManager(settings.context_max_chars)
        self.episodic = EpisodicMemory(mem_dir / "episodic.db")
        self.skill = SkillMemory(mem_dir / "skills.db")
        self.paper_graph = PaperGraphMemory(mem_dir / "paper_graph.db")
        self.vector = VectorMemory(settings.workspace_dir / "vector_db")

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
        self._last_used_skill_ids: List[str] = []

    def get_context_for_task(self, query: str) -> Dict[str, Any]:
        recall_k = self._top_k * self.RECALL_MULTIPLIER

        episodes = self.episodic.search(query, limit=recall_k)
        skills = self.skill.find_relevant(query, limit=recall_k)
        vectors = self.vector.retrieve(query, recall_k)
        episodes, skills, vectors = self._resolve_vector_hits(episodes, skills, vectors)

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

        candidates = build_candidates_from_memory(episodes, skills, vectors)
        rerank_output_k = self._ep_k + self._sk_k + self._top_k
        reranked = self.reranker.rerank(query, candidates, top_k=rerank_output_k)

        ep_reranked = [c.raw for c in reranked if c.source == "episode"][:self._ep_k]
        sk_reranked = [c.raw for c in reranked if c.source == "skill"][:self._sk_k]
        vec_reranked = [c.raw for c in reranked if c.source == "vector"][:self._top_k]
        self._last_used_skill_ids = [s.id for s in sk_reranked]

        return {
            "session": self.session.get_context(),
            "episodes": ep_reranked,
            "skills": sk_reranked,
            "vectors": vec_reranked,
            "reranked": reranked[:self._top_k],
        }

    def format_context_for_prompt(self, query: str) -> str:
        ctx = self.get_context_for_task(query)
        graph_ctx = self.get_paper_graph_context(query, top_k=4)
        parts: List[str] = []

        if ctx["episodes"]:
            parts.append("### 历史研究情节")
            for ep in ctx["episodes"]:
                parts.append(
                    f"**主题:** {ep.topic} (质量: {ep.quality_score:.2f})\n"
                    f"**洞见:** {ep.insights}"
                )

        if ctx["skills"]:
            parts.append("### 已学研究技能")
            for sk in ctx["skills"]:
                head = (
                    f"**{sk.name}** [{sk.domain}] "
                    f"(使用 {sk.usage_count} 次, 成功率 {sk.success_rate:.2f})\n"
                    f"_说明_: {sk.description}\n"
                    f"_触发_: {sk.trigger_conditions}"
                )
                body = sk.content[:800]
                if len(sk.content) > 800:
                    body += "\n...(truncated)"
                parts.append(f"{head}\n\n{body}")

        if graph_ctx["papers"]:
            parts.append("### 已读论文图记忆")
            for paper in graph_ctx["papers"]:
                parts.append(
                    f"**{paper.get('title', '')}**\n"
                    f"- 核心: {paper.get('tldr', '')}\n"
                    f"- 问题: {paper.get('problem', '')}"
                )

        if ctx["vectors"]:
            parts.append("### 语义记忆片段")
            for hit in ctx["vectors"]:
                parts.append(hit.content[:400])

        return "\n\n".join(parts) if parts else ""

    def get_related_papers(self, query: str, top_k: int = 5) -> List[MemoryHit]:
        hits = self.vector.retrieve(query, top_k * 4)
        paper_hits = [h for h in hits if (h.doc_id or "").startswith("paper:")]

        if not self.enable_rerank or not self.reranker or not paper_hits:
            return paper_hits[:top_k]

        candidates = [
            RerankCandidate(
                doc_id=h.doc_id,
                content=h.content,
                source="paper",
                raw=h,
            )
            for h in paper_hits
        ]
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return [c.raw for c in reranked]

    def get_recent_episodes(self, limit: int = 5) -> list:
        return self.episodic.get_recent(limit=limit)

    def get_relevant_skills(self, query: str, limit: int = 5) -> list:
        return self.skill.find_relevant(query, limit=limit)

    def find_paper_notes(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        notes_dir = self.vector.persist_dir.parent / "paper_notes"
        if not notes_dir.exists():
            return []

        query_lower = (query or "").lower().strip()
        matches: List[Dict[str, str]] = []
        for path in sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            if query_lower:
                title_match = query_lower in path.stem.lower()
                content_match = query_lower in content.lower()
                if not (title_match or content_match):
                    continue

            matches.append(
                {
                    "title": path.stem,
                    "path": str(path),
                    "preview": content[:1200],
                }
            )
            if len(matches) >= top_k:
                break

        if matches or query_lower:
            return matches

        for path in sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:top_k]:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            matches.append(
                {
                    "title": path.stem,
                    "path": str(path),
                    "preview": content[:1200],
                }
            )
        return matches

    def get_related_read_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        graph_hits = self.paper_graph.search_papers(query, limit=top_k)
        vector_hits = self.get_related_papers(query, top_k=top_k * 2)

        merged: List[Dict[str, Any]] = []
        seen = set()

        for node in graph_hits:
            seen.add(node.paper_id)
            confidence = self._compute_graph_match_confidence(node, query)
            merged.append(
                {
                    "paper_id": node.paper_id,
                    "title": node.title,
                    "method_name": node.method_name,
                    "problem": node.problem,
                    "tldr": node.tldr,
                    "note_path": node.note_path,
                    "source": "graph",
                    "confidence": round(confidence, 3),
                    "is_placeholder": bool((node.metadata or {}).get("placeholder")),
                }
            )

        for hit in vector_hits:
            metadata = hit.metadata or {}
            paper_id = metadata.get("paper_id") or metadata.get("arxiv_id") or hit.doc_id.replace("paper:", "", 1)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            confidence = self._compute_vector_match_confidence(hit, query)
            merged.append(
                {
                    "paper_id": paper_id,
                    "title": metadata.get("title") or metadata.get("method_name") or paper_id,
                    "method_name": metadata.get("method_name", ""),
                    "problem": metadata.get("problem", ""),
                    "tldr": hit.content[:300],
                    "note_path": metadata.get("path", ""),
                    "source": "vector",
                    "confidence": round(confidence, 3),
                    "is_placeholder": False,
                }
            )
            if len(merged) >= top_k:
                break

        merged.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return merged[:top_k]

    def get_paper_graph_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        papers = self.get_related_read_papers(query, top_k=top_k)
        qualified_papers = [
            item for item in papers
            if item.get("confidence", 0.0) >= self.MIN_RELATED_PAPER_CONFIDENCE
            and not item.get("is_placeholder", False)
        ]

        edges = []
        for item in qualified_papers[:3]:
            paper_id = item.get("paper_id", "")
            if not paper_id:
                continue
            neighbors = self.paper_graph.get_neighbors(paper_id, limit=6)
            for edge in neighbors:
                edge_confidence = self._compute_edge_confidence(edge.relation_strength, item.get("confidence", 0.0))
                edges.append(
                    {
                        "src_paper_id": edge.src_paper_id,
                        "dst_paper_id": edge.dst_paper_id,
                        "relation_type": edge.relation_type,
                        "relation_strength": edge.relation_strength,
                        "evidence": edge.evidence,
                        "source_kind": edge.source_kind,
                        "confidence": round(edge_confidence, 3),
                    }
                )

        qualified_edges = [
            edge for edge in edges
            if edge.get("confidence", 0.0) >= self.MIN_EDGE_CONFIDENCE
        ]

        top_confidences = [item["confidence"] for item in qualified_papers[:3]]
        if qualified_edges:
            top_confidences.extend(edge["confidence"] for edge in qualified_edges[:3])
        aggregate_confidence = (
            round(sum(top_confidences) / len(top_confidences), 3)
            if top_confidences else 0.0
        )
        has_confident_context = (
            bool(qualified_papers)
            and aggregate_confidence >= self.MIN_GRAPH_CONTEXT_CONFIDENCE
        )

        return {
            "papers": qualified_papers[:top_k] if has_confident_context else [],
            "edges": qualified_edges[:12] if has_confident_context else [],
            "node_count": self.paper_graph.count_nodes(),
            "edge_count": self.paper_graph.count_edges(),
            "aggregate_confidence": aggregate_confidence,
            "has_confident_context": has_confident_context,
        }

    def get_paper_graph_snapshot(
        self,
        query: str = "",
        node_limit: int = 40,
        edge_limit: int = 120,
        include_neighbors: bool = True,
    ) -> Dict[str, Any]:
        nodes = (
            self.paper_graph.search_papers(query, limit=node_limit)
            if (query or "").strip()
            else self.paper_graph.list_papers(limit=node_limit)
        )
        node_map: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            node_map[node.paper_id] = self._serialize_paper_node(node)

        edges = []
        seen_edges = set()

        def _append_edge(edge) -> None:
            key = (
                edge.src_paper_id,
                edge.dst_paper_id,
                edge.relation_type,
                edge.source_kind,
            )
            if key in seen_edges or len(edges) >= edge_limit:
                return
            seen_edges.add(key)
            edges.append(self._serialize_paper_edge(edge))

        if include_neighbors:
            for node in nodes:
                for edge in self.paper_graph.get_neighbors(node.paper_id, limit=12):
                    _append_edge(edge)
                    if edge.src_paper_id not in node_map:
                        src = self.paper_graph.get_paper(edge.src_paper_id)
                        if src:
                            node_map[src.paper_id] = self._serialize_paper_node(src)
                    if edge.dst_paper_id not in node_map:
                        dst = self.paper_graph.get_paper(edge.dst_paper_id)
                        if dst:
                            node_map[dst.paper_id] = self._serialize_paper_node(dst)
                    if len(edges) >= edge_limit:
                        break
                if len(edges) >= edge_limit:
                    break
        else:
            for edge in self.paper_graph.list_edges(limit=edge_limit):
                if edge.src_paper_id in node_map or edge.dst_paper_id in node_map:
                    _append_edge(edge)

        return {
            "query": query,
            "nodes": list(node_map.values()),
            "edges": edges,
            "stats": self.stats(),
        }

    def _compute_graph_match_confidence(self, node, query: str) -> float:
        query_lower = (query or "").strip().lower()
        if not query_lower:
            return 0.0

        title = (node.title or "").lower()
        method_name = (node.method_name or "").lower()
        problem = (node.problem or "").lower()
        tldr = (node.tldr or "").lower()
        tags = " ".join(node.tags or []).lower()
        metadata = node.metadata or {}

        confidence = 0.35
        if query_lower == title or query_lower == method_name:
            confidence = 0.95
        elif query_lower in title or query_lower in method_name:
            confidence = 0.82
        elif query_lower in problem:
            confidence = 0.68
        elif query_lower in tldr:
            confidence = 0.60
        elif query_lower in tags:
            confidence = 0.55

        if metadata.get("placeholder"):
            confidence -= 0.35
        if not node.note_path:
            confidence -= 0.10
        if not node.problem and not node.tldr:
            confidence -= 0.10
        return max(0.0, min(confidence, 1.0))

    def _compute_vector_match_confidence(self, hit: MemoryHit, query: str) -> float:
        query_lower = (query or "").strip().lower()
        metadata = hit.metadata or {}
        title = str(metadata.get("title", "")).lower()
        method_name = str(metadata.get("method_name", "")).lower()
        problem = str(metadata.get("problem", "")).lower()
        content = (hit.content or "").lower()

        confidence = float(hit.score or 0.0)
        if query_lower and (query_lower == title or query_lower == method_name):
            confidence = max(confidence, 0.92)
        elif query_lower and (query_lower in title or query_lower in method_name):
            confidence = max(confidence, 0.80)
        elif query_lower and query_lower in problem:
            confidence = max(confidence, 0.66)
        elif query_lower and query_lower in content:
            confidence = max(confidence, 0.52)

        if not metadata.get("path"):
            confidence -= 0.08
        return max(0.0, min(confidence, 1.0))

    @staticmethod
    def _compute_edge_confidence(relation_strength: float, paper_confidence: float) -> float:
        relation_strength = float(relation_strength or 0.0)
        paper_confidence = float(paper_confidence or 0.0)
        return max(0.0, min((relation_strength * 0.6) + (paper_confidence * 0.4), 1.0))

    def _resolve_vector_hits(self, episodes, skills, vectors):
        ep_ids = {e.id for e in episodes}
        sk_ids = {s.id for s in skills}
        remaining_vectors = []

        for hit in vectors:
            doc_id = hit.doc_id or ""

            if doc_id.startswith("episode:"):
                ep_id = doc_id.split(":", 1)[1]
                if ep_id in ep_ids:
                    continue
                ep = self.episodic.get_by_id(ep_id)
                if ep is not None:
                    episodes.append(ep)
                    ep_ids.add(ep_id)
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

    def save_task_result(self, doc_id: str, title: str, body: str):
        self.vector.add(doc_id, body, {"title": title})

    def save_paper_note(self, paper_id: str, title: str, analysis: Dict[str, Any]) -> None:
        tags = analysis.get("tags", []) or []
        contributions = analysis.get("contributions", []) or []
        datasets_raw = analysis.get("datasets", []) or []
        datasets = []
        for item in datasets_raw:
            if isinstance(item, str):
                datasets.append(item)
            elif isinstance(item, dict):
                datasets.append(item.get("name", "") or "Unknown")

        summary_text = (
            f"{title}\n"
            f"{analysis.get('tldr', '')}\n\n"
            f"Problem:\n{analysis.get('problem', '')}\n\n"
            f"Method:\n{analysis.get('method_summary', '')[:1200]}\n\n"
            "Contributions:\n"
            + "\n".join(f"- {c}" for c in contributions)
        )

        self.vector.add(
            doc_id=f"paper:{paper_id}",
            content=summary_text,
            metadata={
                "type": "paper_note",
                "paper_id": paper_id,
                "arxiv_id": paper_id,
                "title": title,
                "method_name": analysis.get("_method_name", ""),
                "path": analysis.get("_note_path", ""),
                "tags": ",".join(tags),
                "problem": analysis.get("problem", ""),
            },
        )

        self.paper_graph.upsert_paper(
            paper_id=paper_id,
            title=title,
            method_name=analysis.get("_method_name", ""),
            note_path=analysis.get("_note_path", ""),
            problem=analysis.get("problem", ""),
            method_summary=analysis.get("method_summary", ""),
            tldr=analysis.get("tldr", ""),
            tags=tags,
            datasets=datasets,
            related_work=analysis.get("related_work", []) or [],
            metadata={
                "analysis_mode": analysis.get("_analysis_mode", ""),
                "source": analysis.get("_source", ""),
                "focus": analysis.get("_focus", ""),
            },
        )

        for item in analysis.get("cited_similar_work", []) or []:
            if isinstance(item, str):
                title_text = item.strip()
                if not title_text:
                    continue
                target_id = title_text[:120]
                relation_type = "similar_to"
                evidence = ""
                source_kind = "inferred"
            elif isinstance(item, dict):
                title_text = str(item.get("title", "")).strip()
                if not title_text:
                    continue
                target_id = title_text[:120]
                category = (item.get("category", "") or "").lower()
                if "foundation" in category:
                    relation_type = "builds_on"
                elif "baseline" in category:
                    relation_type = "compares_with"
                else:
                    relation_type = "similar_to"
                evidence = (
                    str(item.get("why_related", "")).strip()
                    or str(item.get("difference_vs_this_paper", "")).strip()
                )
                source_kind = "explicit"
            else:
                continue

            existing = self.paper_graph.search_papers(title_text, limit=1)
            dst_paper_id = existing[0].paper_id if existing else target_id
            if not existing:
                self.paper_graph.upsert_paper(
                    paper_id=dst_paper_id,
                    title=title_text,
                    metadata={"placeholder": True},
                )

            self.paper_graph.add_edge(
                src_paper_id=paper_id,
                dst_paper_id=dst_paper_id,
                relation_type=relation_type,
                relation_strength=0.75,
                evidence=evidence[:500],
                source_kind=source_kind,
            )

    def merge_evolution_graph(self, papers, evo: Dict[str, Any], origin: str = "survey") -> Dict[str, int]:
        """把算法演进图的节点+关系并入论文图谱（作为"检索发现"层）。

        - 节点：以 `metadata.source="discovered"` 标记，作为独立层次着色；
          绝不覆盖已精读节点（有 note_path / problem / method_summary 的）。
        - 边：只持久化 builds_on / compares_with（语义边），用独立 source_kind="evolution"
          以免与精读抽取的引用边冲突；方向对齐既有约定（src=较新，dst=较旧）。
        返回 {"nodes_added", "edges_added"}。
        """
        if not evo or not isinstance(evo, dict):
            return {"nodes_added": 0, "edges_added": 0}

        nodes = evo.get("nodes") or []
        edges = evo.get("edges") or []
        by_id = {getattr(p, "paper_id", ""): p for p in (papers or []) if getattr(p, "paper_id", "")}

        row_to_pid: Dict[Any, str] = {}
        nodes_added = 0
        for n in nodes:
            pid = (n.get("paper_id") or "").strip()
            if not pid:
                continue
            row_to_pid[n.get("id")] = pid

            existing = self.paper_graph.get_paper(pid)
            # 不覆盖已精读节点
            if existing and (existing.note_path or existing.problem or existing.method_summary):
                continue

            paper = by_id.get(pid)
            title = n.get("title") or (getattr(paper, "title", "") if paper else "") or pid
            abstract = (getattr(paper, "abstract", "") if paper else "") or ""
            tldr = " ".join(abstract.split())[:200]
            cats = list(getattr(paper, "categories", []) or []) if paper else []
            self.paper_graph.upsert_paper(
                paper_id=pid,
                title=title,
                tldr=tldr,
                tags=cats[:6],
                metadata={
                    "source": "discovered",
                    "origin": origin,
                    "url": getattr(paper, "url", "") if paper else "",
                    "year": n.get("year", ""),
                    "has_code": bool(n.get("has_code")),
                },
            )
            nodes_added += 1

        edges_added = 0
        for e in edges:
            rel = e.get("type", "")
            if rel not in ("builds_on", "compares_with"):
                continue  # 启发式 similar_to 不入库，避免无语义连线污染知识图
            early_pid = row_to_pid.get(e.get("src"))   # src=较早
            late_pid = row_to_pid.get(e.get("dst"))    # dst=较晚
            if not early_pid or not late_pid or early_pid == late_pid:
                continue
            # 约定：src=较新的论文 builds_on/compares_with 较旧的 dst
            self.paper_graph.add_edge(
                src_paper_id=late_pid,
                dst_paper_id=early_pid,
                relation_type=rel,
                relation_strength=0.6,
                evidence=(e.get("reason", "") or "")[:500],
                source_kind="evolution",
            )
            edges_added += 1

        return {"nodes_added": nodes_added, "edges_added": edges_added}

    def save_research_episode(
        self,
        topic: str,
        content: str,
        insights: str,
        tags: List[str],
        quality_score: float,
    ) -> str:
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
        skill_id = self.skill.add_skill(
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            content=content,
            domain=domain,
        )
        payload = f"{description}\n触发: {trigger_conditions}\n{content[:1500]}"
        self.vector.add(
            f"skill:{skill_id}",
            payload,
            {"type": "skill", "skill_id": skill_id, "domain": domain, "name": name},
        )
        return skill_id

    def feedback_skills_usage(self, success: bool) -> int:
        count = 0
        for sk_id in self._last_used_skill_ids:
            try:
                self.skill.update_usage(sk_id, success=success)
                count += 1
            except Exception:
                continue
        self._last_used_skill_ids = []
        return count

    def stats(self) -> Dict[str, int]:
        return {
            "episodes": self.episodic.count(),
            "skills": self.skill.count(),
            "vectors": self.vector.count(),
            "paper_nodes": self.paper_graph.count_nodes(),
            "paper_edges": self.paper_graph.count_edges(),
        }

    @staticmethod
    def _node_layer(node) -> str:
        """判定节点所属层次：read（精读）/ discovered（检索发现）/ cited（引用占位）。

        优先级 read > discovered > cited：一篇论文若已精读，即便后续又被检索到，
        仍归为 read，不被降级。
        """
        md = node.metadata or {}
        if node.note_path or node.problem or node.method_summary:
            return "read"
        if md.get("source") == "discovered":
            return "discovered"
        if md.get("placeholder"):
            return "cited"
        return "read"

    @staticmethod
    def _serialize_paper_node(node) -> Dict[str, Any]:
        return {
            "paper_id": node.paper_id,
            "title": node.title,
            "method_name": node.method_name,
            "note_path": node.note_path,
            "problem": node.problem,
            "method_summary": node.method_summary,
            "tldr": node.tldr,
            "tags": node.tags or [],
            "datasets": node.datasets or [],
            "related_work": node.related_work or [],
            "metadata": node.metadata or {},
            "layer": MemoryManager._node_layer(node),
            "created_at": node.created_at,
            "updated_at": node.updated_at,
        }

    @staticmethod
    def _serialize_paper_edge(edge) -> Dict[str, Any]:
        return {
            "src_paper_id": edge.src_paper_id,
            "dst_paper_id": edge.dst_paper_id,
            "relation_type": edge.relation_type,
            "relation_strength": edge.relation_strength,
            "evidence": edge.evidence,
            "source_kind": edge.source_kind,
            "created_at": edge.created_at,
        }
