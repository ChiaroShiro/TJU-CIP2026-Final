"""
对话式路由 Agent。

职责：
1. 判断用户是在闲聊、查记忆、查笔记，还是要调用研究能力
2. 给 chat 模式提供统一的动作结构
3. 把工具/记忆结果总结成自然语言回复
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.llm import LLMClient
from ..core.utils import extract_json_object
from ..memory.memory_manager import MemoryManager


ROUTER_SYSTEM_PROMPT = """你是一个研究助手的对话路由器。你的职责：
1. 理解用户的自然语言意图
2. 区分用户是要直接聊天、查询记忆、读取已有笔记，还是发起新的研究动作
3. 如果需要论文搜索，请优先抽取 3-5 个英文检索词
4. 输出严格 JSON，供后端执行

可用动作：
- "ask_user": 信息不足，需要追问
- "evaluate": 评估研究方向
- "search": 搜索论文
- "analyze": 深度分析一篇 arXiv 论文
- "research": 发起完整深度研究
- "memory_query": 查询历史研究记忆、技能记忆、最近研究
- "paper_note_query": 查询已有 paper_notes 笔记并基于内容回答
- "chitchat": 普通对话、解释、追问已有上下文

决策原则：
- 用户问“之前研究过什么 / 我们聊过什么 / 记忆里有什么” -> memory_query
- 用户问“之前那篇笔记 / paper_notes 里的某篇 / 帮我总结 VLA-JEPA 笔记” -> paper_note_query
- 用户给了 arXiv ID 或 arXiv URL -> analyze
- 用户说“查一下 XXX 的论文” -> search
- 用户说“评估 XXX 方向是否可行” -> evaluate
- 用户说“详细研究 XXX / 写一份报告” -> research
- 用户是在追问、解释、比较、总结已有内容 -> chitchat

检索词要求：
- 只在 search / evaluate / research 需要时生成
- 使用英文
- 每个查询 3-8 个词
- 不要整句
"""


@dataclass
class AgentAction:
    action: str
    reply: str = ""
    queries: List[str] = field(default_factory=list)
    topic: str = ""
    paper_id: str = ""
    focus: str = ""
    raw_intent: str = ""


class ConversationalAgent:
    def __init__(self, llm: LLMClient, memory: MemoryManager, max_history: int = 16):
        self.llm = llm
        self.memory = memory
        self.session = memory.session
        self.max_history = max_history

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self.session.history)

    def add_user_message(self, content: str):
        self.session.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.session.add_message("assistant", content)

    def add_tool_result(self, summary: str):
        self.session.add_message("assistant", f"[tool_observation]\n{summary}")

    def clear(self):
        self.session.clear()

    def decide(self) -> AgentAction:
        user_msgs = [m for m in self.history if m["role"] == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""

        long_term_context = ""
        if last_user:
            try:
                long_term_context = self.memory.format_context_for_prompt(last_user)
            except Exception:
                long_term_context = ""

        system_content = ROUTER_SYSTEM_PROMPT
        if long_term_context:
            system_content += (
                "\n\n# 与当前输入相关的长期记忆\n"
                f"{long_term_context[:2000]}"
            )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history[-self.max_history:])
        messages.append({"role": "user", "content": self._decision_prompt()})

        raw = self.llm.invoke(messages, temperature=0.2)
        data = extract_json_object(raw) or {}

        return AgentAction(
            action=data.get("action", "chitchat"),
            reply=data.get("reply", ""),
            queries=[q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:5],
            topic=data.get("topic", "").strip(),
            paper_id=data.get("paper_id", "").strip(),
            focus=data.get("focus", "").strip(),
            raw_intent=data.get("intent", "").strip(),
        )

    def _decision_prompt(self) -> str:
        return """基于以上对话历史和长期记忆，输出下一步动作。
严格返回 JSON：
{
  "intent": "你理解的用户当前意图（一句话）",
  "action": "ask_user | evaluate | search | analyze | research | memory_query | paper_note_query | chitchat",
  "reply": "仅 ask_user 或 chitchat 时填写；其他动作可留空",
  "topic": "主题原文，可用于 memory_query / paper_note_query / evaluate / research / search",
  "queries": ["english keyword query", "english keyword query"],
  "paper_id": "analyze 时的 arXiv ID 或 URL",
  "focus": "可选关注点"
}

只返回 JSON，不要输出其他文字。"""

    def _summarize_messages(self, action: AgentAction, tool_output: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位友好的研究助手。"
                    "请基于后端返回结果，用简洁、自然、有帮助的中文回答用户。"
                    "如果适合，可以顺带指出下一步建议。"
                ),
            }
        ]
        messages.extend(self.history[-self.max_history:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"动作 [{action.action}] 的执行结果如下：\n"
                    f"{json.dumps(tool_output, ensure_ascii=False, default=str)[:5000]}\n\n"
                    "请生成最终回复。"
                ),
            }
        )
        return messages

    def summarize_result(self, action: AgentAction, tool_output: Dict[str, Any]) -> str:
        return self.llm.invoke(self._summarize_messages(action, tool_output), temperature=0.4).strip()

    def summarize_result_stream(self, action: AgentAction, tool_output: Dict[str, Any]):
        """流式生成回复，逐块 yield 文本增量，供 chat 打字机使用。"""
        yield from self.llm.invoke_stream(self._summarize_messages(action, tool_output), temperature=0.4)
