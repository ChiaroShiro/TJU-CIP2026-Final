"""
评估框架 - Agent 执行效率 + 报告质量

评估维度：
1. 执行效率：步数 / token 消耗 / 完成率 / 工具调用分布
2. 报告质量：Critic 四维评分
3. 学习效果：技能积累速度 / 使用率

使用方式：
    python eval/eval_agent.py "研究主题"

会跑一次完整 research 并输出详细评估指标。
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.agents.manager import ResearchManager


def evaluate_agent_run(settings: Settings, topic: str, max_steps: int = 30) -> Dict:
    """跑一次完整 research 并收集评估指标。"""
    manager = ResearchManager(settings, max_steps=max_steps, max_total_tokens=200_000)

    start = time.time()
    result = manager.run(topic)
    wall_time = time.time() - start

    # 工具调用分布
    tool_counts: Dict[str, int] = {}
    tool_tokens: Dict[str, int] = {}
    tool_success: Dict[str, int] = {}
    tool_fail: Dict[str, int] = {}

    for step in result.steps:
        name = step.tool_name or "(thinking)"
        tool_counts[name] = tool_counts.get(name, 0) + 1
        tool_tokens[name] = tool_tokens.get(name, 0) + step.tokens_used
        if step.tool_result:
            if step.tool_result.success:
                tool_success[name] = tool_success.get(name, 0) + 1
            else:
                tool_fail[name] = tool_fail.get(name, 0) + 1

    # Critic 评分（如果 agent 调了 delegate_to_critic）
    critic_scores = []
    for step in result.steps:
        if step.tool_name == "delegate_to_critic" and step.tool_result and step.tool_result.success:
            content = step.tool_result.content
            if isinstance(content, dict) and "score" in content:
                critic_scores.append(content)

    # 报告长度
    report_text = ""
    if result.finished and isinstance(result.final_output, dict):
        report_text = result.final_output.get("output", "")
    elif result.finished:
        report_text = str(result.final_output or "")

    metrics = {
        "topic": topic,
        "execution": {
            "finished": result.finished,
            "finish_reason": result.finish_reason,
            "total_steps": len(result.steps),
            "total_tokens": result.total_tokens,
            "wall_time_seconds": round(wall_time, 1),
            "tokens_per_step": round(result.total_tokens / max(len(result.steps), 1)),
            "avg_step_latency_ms": round(result.total_elapsed_ms / max(len(result.steps), 1)),
        },
        "tool_usage": {
            "distribution": tool_counts,
            "tokens_by_tool": tool_tokens,
            "success_count": tool_success,
            "fail_count": tool_fail,
            "unique_tools_used": len(tool_counts),
            "total_tool_calls": sum(tool_counts.values()),
            "success_rate": round(
                sum(tool_success.values()) / max(sum(tool_counts.values()), 1), 3
            ),
        },
        "report_quality": {
            "report_length_chars": len(report_text),
            "report_length_words": len(report_text.split()),
            "critic_reviews": critic_scores,
            "final_critic_score": critic_scores[-1]["score"] if critic_scores else None,
            "dimension_scores": critic_scores[-1].get("dimension_scores") if critic_scores else None,
        },
        "learning": {
            "memory_stats_after": None,  # 填充在下面
        },
    }

    # 记忆统计
    try:
        stats = manager.memory.stats()
        metrics["learning"]["memory_stats_after"] = stats
    except Exception:
        pass

    return metrics


def print_metrics(metrics: Dict):
    """格式化打印评估结果。"""
    print("=" * 60)
    print(f"Agent 执行评估: {metrics['topic']}")
    print("=" * 60)

    ex = metrics["execution"]
    print(f"\n📊 执行效率:")
    print(f"  完成状态:     {'✅ 完成' if ex['finished'] else '❌ 未完成'} ({ex['finish_reason']})")
    print(f"  总步数:       {ex['total_steps']}")
    print(f"  总 tokens:    {ex['total_tokens']:,}")
    print(f"  墙钟时间:     {ex['wall_time_seconds']}s")
    print(f"  每步平均 tok: {ex['tokens_per_step']}")
    print(f"  每步平均延迟: {ex['avg_step_latency_ms']}ms")

    tu = metrics["tool_usage"]
    print(f"\n🔧 工具使用:")
    print(f"  唯一工具数:   {tu['unique_tools_used']}")
    print(f"  总调用次数:   {tu['total_tool_calls']}")
    print(f"  成功率:       {tu['success_rate']:.1%}")
    print(f"\n  调用分布:")
    for name, count in sorted(tu["distribution"].items(), key=lambda x: -x[1]):
        tokens = tu["tokens_by_tool"].get(name, 0)
        fails = tu["fail_count"].get(name, 0)
        status = f" ({fails} failed)" if fails else ""
        print(f"    {name:<30} {count:>3}x  ({tokens:>6} tok){status}")

    rq = metrics["report_quality"]
    print(f"\n📝 报告质量:")
    print(f"  报告长度:     {rq['report_length_chars']:,} 字符 / {rq['report_length_words']} 词")
    if rq["final_critic_score"] is not None:
        print(f"  Critic 评分:  {rq['final_critic_score']:.3f}")
        if rq["dimension_scores"]:
            for dim, score in rq["dimension_scores"].items():
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"    {dim:<20} {score:.2f}  {bar}")
    else:
        print(f"  Critic 评分:  未触发（Agent 未调用 delegate_to_critic）")

    learn = metrics["learning"]
    if learn["memory_stats_after"]:
        s = learn["memory_stats_after"]
        print(f"\n🧠 学习效果:")
        print(f"  情节数:       {s['episodes']}")
        print(f"  技能数:       {s['skills']}")
        print(f"  向量数:       {s['vectors']}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    settings = Settings.from_env(root)

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "DPO 在测试用例生成中的应用"

    print(f"🚀 开始评估 (主题: {topic})")
    print(f"   max_steps=30, max_tokens=200K\n")

    metrics = evaluate_agent_run(settings, topic)
    print_metrics(metrics)

    # 保存
    output_path = root / "eval" / "agent_eval_results.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n💾 详细结果已保存: {output_path}")
