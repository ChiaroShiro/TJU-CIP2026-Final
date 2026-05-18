"""
评估框架 - 记忆系统召回率 / 精确率 / Rerank 提升

评估方法：
1. 预先准备一组 (query, relevant_doc_ids) 的 ground truth
2. 对每个 query 执行检索，计算 Recall@K / Precision@K / NDCG@K
3. 对比 rerank 前后的指标变化

使用方式：
    python eval/eval_memory.py

前提：workspace/memory/ 下已有数据（跑过至少 1-2 次 research）
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.memory.memory_manager import MemoryManager
from src.memory.reranker import build_candidates_from_memory, RerankCandidate


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """计算 NDCG@K。"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / (i + 1)  # 简化版：相关=1，不相关=0
    # 理想 DCG
    ideal_dcg = sum(1.0 / (i + 1) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K = 检索到的相关文档数 / 总相关文档数。"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K = 检索到的相关文档数 / K。"""
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / k


def evaluate_memory_system(settings: Settings, test_cases: List[Dict]) -> Dict:
    """
    评估记忆系统。

    test_cases 格式：
    [
        {
            "query": "DPO test case generation",
            "relevant_doc_ids": ["task:xxx", "episode:yyy", "skill:zzz"],
            "description": "测试 DPO 相关记忆召回"
        },
        ...
    ]
    """
    memory = MemoryManager(settings, enable_rerank=True)
    memory_no_rerank = MemoryManager(settings, enable_rerank=False)

    results = {
        "per_query": [],
        "aggregate": {},
    }

    all_recall_rerank = []
    all_precision_rerank = []
    all_ndcg_rerank = []
    all_recall_no_rerank = []
    all_precision_no_rerank = []
    all_ndcg_no_rerank = []

    k = settings.memory_top_k

    for tc in test_cases:
        query = tc["query"]
        relevant = set(tc["relevant_doc_ids"])

        # 带 rerank
        ctx = memory.get_context_for_task(query)
        reranked_ids = []
        for ep in ctx.get("episodes", []):
            reranked_ids.append(f"ep:{ep.id}")
        for sk in ctx.get("skills", []):
            reranked_ids.append(f"sk:{sk.id}")
        for hit in ctx.get("vectors", []):
            reranked_ids.append(hit.doc_id)

        # 不带 rerank
        ctx_raw = memory_no_rerank.get_context_for_task(query)
        raw_ids = []
        for ep in ctx_raw.get("episodes", []):
            raw_ids.append(f"ep:{ep.id}")
        for sk in ctx_raw.get("skills", []):
            raw_ids.append(f"sk:{sk.id}")
        for hit in ctx_raw.get("vectors", []):
            raw_ids.append(hit.doc_id)

        # 计算指标
        r_rerank = recall_at_k(reranked_ids, relevant, k)
        p_rerank = precision_at_k(reranked_ids, relevant, k)
        n_rerank = ndcg_at_k(reranked_ids, relevant, k)

        r_raw = recall_at_k(raw_ids, relevant, k)
        p_raw = precision_at_k(raw_ids, relevant, k)
        n_raw = ndcg_at_k(raw_ids, relevant, k)

        all_recall_rerank.append(r_rerank)
        all_precision_rerank.append(p_rerank)
        all_ndcg_rerank.append(n_rerank)
        all_recall_no_rerank.append(r_raw)
        all_precision_no_rerank.append(p_raw)
        all_ndcg_no_rerank.append(n_raw)

        results["per_query"].append({
            "query": query,
            "description": tc.get("description", ""),
            "relevant_count": len(relevant),
            "retrieved_with_rerank": len(reranked_ids),
            "retrieved_without_rerank": len(raw_ids),
            "recall@k_rerank": round(r_rerank, 3),
            "precision@k_rerank": round(p_rerank, 3),
            "ndcg@k_rerank": round(n_rerank, 3),
            "recall@k_raw": round(r_raw, 3),
            "precision@k_raw": round(p_raw, 3),
            "ndcg@k_raw": round(n_raw, 3),
        })

    n = len(test_cases) or 1
    results["aggregate"] = {
        "k": k,
        "num_queries": len(test_cases),
        "with_rerank": {
            "avg_recall@k": round(sum(all_recall_rerank) / n, 3),
            "avg_precision@k": round(sum(all_precision_rerank) / n, 3),
            "avg_ndcg@k": round(sum(all_ndcg_rerank) / n, 3),
        },
        "without_rerank": {
            "avg_recall@k": round(sum(all_recall_no_rerank) / n, 3),
            "avg_precision@k": round(sum(all_precision_no_rerank) / n, 3),
            "avg_ndcg@k": round(sum(all_ndcg_no_rerank) / n, 3),
        },
        "rerank_improvement": {
            "recall_delta": round(
                (sum(all_recall_rerank) - sum(all_recall_no_rerank)) / n, 3
            ),
            "precision_delta": round(
                (sum(all_precision_rerank) - sum(all_precision_no_rerank)) / n, 3
            ),
            "ndcg_delta": round(
                (sum(all_ndcg_rerank) - sum(all_ndcg_no_rerank)) / n, 3
            ),
        },
    }

    return results


def get_memory_stats(settings: Settings) -> Dict:
    """获取记忆库基础统计。"""
    memory = MemoryManager(settings, enable_rerank=False)
    stats = memory.stats()
    return {
        "episodes": stats["episodes"],
        "skills": stats["skills"],
        "vectors": stats["vectors"],
        "total_documents": stats["episodes"] + stats["skills"] + stats["vectors"],
    }


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    settings = Settings.from_env(root)

    print("=" * 60)
    print("记忆系统评估")
    print("=" * 60)

    # 基础统计
    stats = get_memory_stats(settings)
    print(f"\n📊 记忆库统计:")
    print(f"  情节 (episodes): {stats['episodes']}")
    print(f"  技能 (skills):   {stats['skills']}")
    print(f"  向量 (vectors):  {stats['vectors']}")
    print(f"  总文档数:        {stats['total_documents']}")

    if stats["total_documents"] == 0:
        print("\n⚠️  记忆库为空！请先跑几次 research 积累数据后再评估。")
        print("  示例: python main.py research \"DPO 测试用例生成\"")
        sys.exit(0)

    # 自动生成 test cases（基于已有数据）
    # 从向量库取几条已有文档作为 ground truth
    memory = MemoryManager(settings, enable_rerank=False)
    sample_hits = memory.vector.retrieve("research", top_k=10)

    if len(sample_hits) < 3:
        print("\n⚠️  向量库文档太少（<3），无法有效评估。请多跑几次 research。")
        sys.exit(0)

    # 用已有文档的内容片段作为 query，对应 doc_id 作为 relevant
    test_cases = []
    for hit in sample_hits[:5]:
        # 用 content 前 50 字作为 query（模拟"用户搜相关内容"）
        query_text = hit.content[:50].strip()
        if not query_text:
            continue
        test_cases.append({
            "query": query_text,
            "relevant_doc_ids": [hit.doc_id],
            "description": f"检索 {hit.doc_id}",
        })

    if not test_cases:
        print("\n⚠️  无法生成测试用例。")
        sys.exit(0)

    print(f"\n🧪 自动生成 {len(test_cases)} 个测试用例（基于已有文档）")
    results = evaluate_memory_system(settings, test_cases)

    print(f"\n📈 聚合结果 (K={results['aggregate']['k']}):")
    print(f"\n  {'指标':<20} {'带 Rerank':<12} {'不带 Rerank':<12} {'提升':<10}")
    print(f"  {'─'*54}")
    agg = results["aggregate"]
    for metric in ["recall@k", "precision@k", "ndcg@k"]:
        with_r = agg["with_rerank"][f"avg_{metric}"]
        without_r = agg["without_rerank"][f"avg_{metric}"]
        delta = agg["rerank_improvement"][f"{metric.split('@')[0]}_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<20} {with_r:<12.3f} {without_r:<12.3f} {sign}{delta:.3f}")

    print(f"\n📋 逐 query 详情:")
    for pq in results["per_query"]:
        print(f"  [{pq['description']}]")
        print(f"    query: \"{pq['query'][:40]}...\"")
        print(f"    recall: {pq['recall@k_rerank']:.3f} (raw: {pq['recall@k_raw']:.3f})")
        print(f"    precision: {pq['precision@k_rerank']:.3f} (raw: {pq['precision@k_raw']:.3f})")
        print()

    # 保存结果
    output_path = root / "eval" / "memory_eval_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 详细结果已保存: {output_path}")
