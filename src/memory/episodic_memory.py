"""
情节记忆（Episodic Memory）- Hermes风格第二层记忆

基于 SQLite + FTS5 实现跨会话持久化：
- 存储每次研究会话的完整结果和提炼洞见
- FTS5全文检索，支持中英文关键词搜索
- 质量评分机制，记录每次研究的可信度
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    id: str
    topic: str
    content: str
    insights: str
    tags: List[str]
    quality_score: float
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """SQLite + FTS5 情节记忆，跨会话持久化研究历史。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id       TEXT PRIMARY KEY,
                    topic    TEXT NOT NULL,
                    content  TEXT NOT NULL,
                    insights TEXT DEFAULT '',
                    tags     TEXT DEFAULT '',
                    quality_score REAL DEFAULT 0.5,
                    created_at    TEXT NOT NULL,
                    metadata      TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(
                    id UNINDEXED, topic, content, insights, tags,
                    content='episodes', content_rowid='rowid'
                )
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS episodes_ai
                AFTER INSERT ON episodes BEGIN
                    INSERT INTO episodes_fts(rowid, id, topic, content, insights, tags)
                    VALUES (new.rowid, new.id, new.topic, new.content,
                            new.insights, new.tags);
                END
            """)

    def add_episode(
        self,
        topic: str,
        content: str,
        insights: str,
        tags: List[str],
        quality_score: float,
        metadata: Dict[str, Any] = None,
    ) -> str:
        ep_id = uuid.uuid4().hex[:8]
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
                (ep_id, topic, content, insights,
                 ",".join(tags), quality_score, now,
                 json.dumps(metadata or {})),
            )
        return ep_id

    def search(self, query: str, limit: int = 5) -> List[Episode]:
        """FTS5全文检索，返回最相关的情节。"""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT e.id, e.topic, e.content, e.insights, e.tags,
                           e.quality_score, e.created_at, e.metadata
                    FROM episodes_fts f
                    JOIN episodes e ON e.rowid = f.rowid
                    WHERE episodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS syntax error (e.g. special chars) — fall back to LIKE
            rows = self._fallback_search(query, limit)
        return [self._to_episode(r) for r in rows]

    def _fallback_search(self, query: str, limit: int) -> list:
        like = f"%{query}%"
        with self._conn() as conn:
            return conn.execute(
                "SELECT id, topic, content, insights, tags, quality_score, "
                "created_at, metadata FROM episodes "
                "WHERE topic LIKE ? OR insights LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()

    def get_recent(self, limit: int = 5) -> List[Episode]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, topic, content, insights, tags, quality_score, "
                "created_at, metadata FROM episodes "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_episode(r) for r in rows]

    def get_by_id(self, ep_id: str) -> Optional[Episode]:
        """按 ID 取完整情节，供向量命中后回查使用。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, topic, content, insights, tags, quality_score, "
                "created_at, metadata FROM episodes WHERE id = ?",
                (ep_id,),
            ).fetchone()
        return self._to_episode(row) if row else None

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def _to_episode(self, row) -> Episode:
        return Episode(
            id=row[0], topic=row[1], content=row[2], insights=row[3],
            tags=row[4].split(",") if row[4] else [],
            quality_score=row[5], created_at=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
        )
