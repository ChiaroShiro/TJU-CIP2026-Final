import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PaperNode:
    paper_id: str
    title: str
    method_name: str = ""
    note_path: str = ""
    problem: str = ""
    method_summary: str = ""
    tldr: str = ""
    tags: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    related_work: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class PaperEdge:
    src_paper_id: str
    dst_paper_id: str
    relation_type: str
    relation_strength: float = 0.5
    evidence: str = ""
    source_kind: str = "inferred"
    created_at: str = ""


class PaperGraphMemory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_nodes (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    method_name TEXT DEFAULT '',
                    note_path TEXT DEFAULT '',
                    problem TEXT DEFAULT '',
                    method_summary TEXT DEFAULT '',
                    tldr TEXT DEFAULT '',
                    tags_json TEXT DEFAULT '[]',
                    datasets_json TEXT DEFAULT '[]',
                    related_work_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_edges (
                    src_paper_id TEXT NOT NULL,
                    dst_paper_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    relation_strength REAL DEFAULT 0.5,
                    evidence TEXT DEFAULT '',
                    source_kind TEXT DEFAULT 'inferred',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (src_paper_id, dst_paper_id, relation_type, source_kind)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_edges_src
                ON paper_edges(src_paper_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_edges_dst
                ON paper_edges(dst_paper_id)
                """
            )

    def upsert_paper(
        self,
        paper_id: str,
        title: str,
        method_name: str = "",
        note_path: str = "",
        problem: str = "",
        method_summary: str = "",
        tldr: str = "",
        tags: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        related_work: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT created_at FROM paper_nodes WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            created_at = row[0] if row else now
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_nodes (
                    paper_id, title, method_name, note_path, problem,
                    method_summary, tldr, tags_json, datasets_json,
                    related_work_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    title,
                    method_name,
                    note_path,
                    problem,
                    method_summary,
                    tldr,
                    json.dumps(tags or [], ensure_ascii=False),
                    json.dumps(datasets or [], ensure_ascii=False),
                    json.dumps(related_work or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )

    def add_edge(
        self,
        src_paper_id: str,
        dst_paper_id: str,
        relation_type: str,
        relation_strength: float = 0.5,
        evidence: str = "",
        source_kind: str = "inferred",
    ) -> None:
        if not src_paper_id or not dst_paper_id or src_paper_id == dst_paper_id:
            return
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_edges (
                    src_paper_id, dst_paper_id, relation_type,
                    relation_strength, evidence, source_kind, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    src_paper_id,
                    dst_paper_id,
                    relation_type,
                    relation_strength,
                    evidence,
                    source_kind,
                    now,
                ),
            )

    def get_paper(self, paper_id: str) -> Optional[PaperNode]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT paper_id, title, method_name, note_path, problem,
                       method_summary, tldr, tags_json, datasets_json,
                       related_work_json, metadata_json, created_at, updated_at
                FROM paper_nodes
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()
        return self._to_paper_node(row) if row else None

    def search_papers(self, query: str, limit: int = 5) -> List[PaperNode]:
        query = (query or "").strip()
        if not query:
            return self.get_recent_papers(limit)
        like = f"%{query}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, title, method_name, note_path, problem,
                       method_summary, tldr, tags_json, datasets_json,
                       related_work_json, metadata_json, created_at, updated_at
                FROM paper_nodes
                WHERE title LIKE ? OR problem LIKE ? OR method_summary LIKE ? OR tldr LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, like, like, like, limit),
            ).fetchall()
        return [self._to_paper_node(row) for row in rows]

    def get_recent_papers(self, limit: int = 5) -> List[PaperNode]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, title, method_name, note_path, problem,
                       method_summary, tldr, tags_json, datasets_json,
                       related_work_json, metadata_json, created_at, updated_at
                FROM paper_nodes
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_paper_node(row) for row in rows]

    def list_papers(self, limit: int = 100) -> List[PaperNode]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, title, method_name, note_path, problem,
                       method_summary, tldr, tags_json, datasets_json,
                       related_work_json, metadata_json, created_at, updated_at
                FROM paper_nodes
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_paper_node(row) for row in rows]

    def get_neighbors(self, paper_id: str, limit: int = 10) -> List[PaperEdge]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT src_paper_id, dst_paper_id, relation_type,
                       relation_strength, evidence, source_kind, created_at
                FROM paper_edges
                WHERE src_paper_id = ? OR dst_paper_id = ?
                ORDER BY relation_strength DESC, created_at DESC
                LIMIT ?
                """,
                (paper_id, paper_id, limit),
            ).fetchall()
        return [self._to_paper_edge(row) for row in rows]

    def list_edges(self, limit: int = 200) -> List[PaperEdge]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT src_paper_id, dst_paper_id, relation_type,
                       relation_strength, evidence, source_kind, created_at
                FROM paper_edges
                ORDER BY relation_strength DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_paper_edge(row) for row in rows]

    def count_nodes(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM paper_nodes").fetchone()[0]

    def count_edges(self) -> int:
        """边的总行数（同一对节点的不同 relation_type/source_kind 会各算一行）。"""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM paper_edges").fetchone()[0]

    def count_unique_edges(self) -> int:
        """按有向 (src, dst) 对去重后的关系数——与图谱视图“每对一条边”一致，更直观。"""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM "
                "(SELECT DISTINCT src_paper_id, dst_paper_id FROM paper_edges)"
            ).fetchone()[0]

    def _to_paper_node(self, row) -> PaperNode:
        return PaperNode(
            paper_id=row[0],
            title=row[1],
            method_name=row[2],
            note_path=row[3],
            problem=row[4],
            method_summary=row[5],
            tldr=row[6],
            tags=json.loads(row[7] or "[]"),
            datasets=json.loads(row[8] or "[]"),
            related_work=json.loads(row[9] or "[]"),
            metadata=json.loads(row[10] or "{}"),
            created_at=row[11],
            updated_at=row[12],
        )

    def _to_paper_edge(self, row) -> PaperEdge:
        return PaperEdge(
            src_paper_id=row[0],
            dst_paper_id=row[1],
            relation_type=row[2],
            relation_strength=row[3],
            evidence=row[4],
            source_kind=row[5],
            created_at=row[6],
        )
