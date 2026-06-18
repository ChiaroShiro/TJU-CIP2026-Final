"""
向量数据库RAG实现（基于Chroma）

相比TF-IDF的优势：
1. 语义检索 - 理解文本含义而非仅匹配关键词
2. 跨语言支持 - embedding模型支持中英文混合
3. 持久化 - 自动保存到磁盘，重启后可恢复
4. 可扩展 - 支持大规模文档库
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict
from pathlib import Path

from ..core.models import MemoryHit


class VectorMemory:
    """基于Chroma向量数据库的记忆系统"""

    def __init__(self, persist_dir: Path):
        """
        初始化向量数据库

        Args:
            persist_dir: 持久化目录路径
        """
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Chroma客户端（持久化模式）
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # 获取或创建collection
        # 使用默认的all-MiniLM-L6-v2 embedding模型（支持中英文）
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            metadata={"description": "Research task memory storage"}
        )

    def add(self, doc_id: str, content: str, metadata: Dict[str, str]) -> None:
        """
        添加/更新文档到向量数据库（幂等）。

        用 upsert 而非 add：同一 doc_id 重复写入时会更新内容而不是告警/抛错。
        这样重复分析同一篇论文（paper:{id}）能刷新摘要，且不会因“ID 已存在”
        中断后续的图谱写入（save_paper_note 中向量写在前、图谱写在后）。

        Args:
            doc_id: 文档唯一ID
            content: 文档内容（会自动生成embedding）
            metadata: 元数据（如标题、时间戳等）
        """
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata]
        )

    def retrieve(self, query: str, top_k: int) -> List[MemoryHit]:
        """
        语义检索相关文档

        Args:
            query: 查询文本
            top_k: 返回最相关的top_k个结果

        Returns:
            MemoryHit列表，按相关性降序排列
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append(MemoryHit(
                    doc_id=doc_id,
                    score=1.0 - results["distances"][0][i],  # 转换为相似度
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {}
                ))

        return hits

    def delete(self, doc_id: str) -> None:
        """删除指定文档"""
        self.collection.delete(ids=[doc_id])

    def clear(self) -> None:
        """清空所有文档"""
        self.client.delete_collection("research_memory")
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            metadata={"description": "Research task memory storage"}
        )

    def count(self) -> int:
        """返回文档总数"""
        return self.collection.count()

