"""
BaseAgent —— Autonomous ReAct loop 基类

所有自主 agent（Manager / CriticWorker / ReviserWorker 等）继承此基类。

核心循环：
    while not done:
        1. LLM 观察当前状态（system prompt + history + observations）
        2. LLM 通过 function calling 选择一个工具（或 finish）
        3. 执行工具，把 ToolResult 加入 history
        4. 检查终止条件（finish / 步数超限 / token 超限）

关键特性：
- Function calling 原生支持（不是 ReAct 文本解析）
- Token 预算硬上限，超出强制 finish
- 步数上限
- 所有历史保留在 agent 内部，支持 inspect trace
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.llm import LLMClient
from ..tools.tool import Tool, ToolRegistry, ToolResult


@dataclass
class AgentStep:
    """Agent 单步执行记录，便于事后审计。"""
    step_idx: int
    thought: str = ""                # LLM 的思考（如果模型输出了）
    tool_name: str = ""              # 调用的工具名
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[ToolResult] = None
    tokens_used: int = 0             # 此步 LLM 调用耗费的 tokens
    elapsed_ms: int = 0


@dataclass
class AgentRunResult:
    """Agent 运行结束后的汇总。"""
    final_output: Any                # finish() 返回值
    finished: bool                   # 是否正常 finish（False 表示被强制中止）
    finish_reason: str               # "finished" | "max_steps" | "token_budget" | "error"
    steps: List[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    total_elapsed_ms: int = 0


class BaseAgent(ABC):
    """
    Autonomous Agent 基类。

    子类必须实现：
      - system_prompt() : 返回给 LLM 的 system prompt
      - build_tools()   : 返回 ToolRegistry（包含 finish 工具）

    子类可选覆盖：
      - initial_user_message() : 第一条 user 消息（默认用 task 字段）
      - on_step(step)           : 钩子，每步结束后调用（日志/监控）
    """

    FINISH_TOOL_NAME = "finish"

    def __init__(
        self,
        llm: LLMClient,
        max_steps: int = 30,
        max_total_tokens: int = 200_000,
        temperature: float = 0.2,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.max_total_tokens = max_total_tokens
        self.temperature = temperature
        self.tools: ToolRegistry = self.build_tools()
        self._ensure_finish_tool()

    # ------------------------------------------------------------------ #
    # 子类需实现
    # ------------------------------------------------------------------ #

    @abstractmethod
    def system_prompt(self) -> str:
        """返回 agent 的 system prompt。应详细描述 agent 角色、工具使用策略、终止条件。"""
        raise NotImplementedError

    @abstractmethod
    def build_tools(self) -> ToolRegistry:
        """返回此 agent 可用的工具注册表。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 核心 loop
    # ------------------------------------------------------------------ #

    def run(self, task: str, on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> AgentRunResult:
        """
        执行 autonomous loop，直到 finish 或达到上限。

        task: 用户给 agent 的初始任务描述。
        on_event: 可选回调。传入时进入流式模式，逐步推送轨迹与 LLM 增量事件：
            {"type": "step", ...}        单步轨迹（工具/参数/状态/tokens/耗时/思考）
            {"type": "llm_delta", ...}   LLM 流式增量（content/reasoning/tool_name/tool_args）
            {"type": "budget_warning", ...}
          不传时走原非流式路径，行为与改动前完全一致。
        """
        emit = on_event if on_event is not None else (lambda e: None)
        streaming = on_event is not None

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.initial_user_message(task)},
        ]

        steps: List[AgentStep] = []
        total_tokens = 0
        total_start = time.time()

        for step_idx in range(self.max_steps):
            if total_tokens >= self.max_total_tokens:
                return AgentRunResult(
                    final_output=None,
                    finished=False,
                    finish_reason="token_budget",
                    steps=steps,
                    total_tokens=total_tokens,
                    total_elapsed_ms=int((time.time() - total_start) * 1000),
                )

            # 中途收敛提醒：步数过 60% 或 token 过 70% 时注入提示
            budget_pct = total_tokens / self.max_total_tokens if self.max_total_tokens > 0 else 0
            step_pct = step_idx / self.max_steps if self.max_steps > 0 else 0
            if (budget_pct > 0.7 or step_pct > 0.6) and step_idx > 0:
                # 只提醒一次（检查最后一条 user 消息是否已经是提醒）
                last_user_msgs = [m for m in messages if m["role"] == "user"]
                if last_user_msgs and "BUDGET WARNING" not in last_user_msgs[-1].get("content", ""):
                    messages.append({
                        "role": "user",
                        "content": (
                            "[BUDGET WARNING] You have used "
                            f"{budget_pct:.0%} of token budget and {step_pct:.0%} of max steps. "
                            "STOP searching and START writing your final report NOW. "
                            "Call finish with your report, or delegate_to_critic if you have a draft."
                        ),
                    })
                    emit({
                        "type": "budget_warning",
                        "budget_pct": round(budget_pct, 3),
                        "step_pct": round(step_pct, 3),
                    })

            step_start = time.time()
            step = AgentStep(step_idx=step_idx)

            # 1. LLM 决定下一步
            try:
                completion = self._call_llm(messages, on_event if streaming else None)
            except Exception as e:
                step.tool_result = ToolResult(
                    success=False,
                    error=f"LLM call failed: {type(e).__name__}: {e}",
                )
                steps.append(step)
                return AgentRunResult(
                    final_output=None,
                    finished=False,
                    finish_reason="error",
                    steps=steps,
                    total_tokens=total_tokens,
                    total_elapsed_ms=int((time.time() - total_start) * 1000),
                )

            msg = completion.choices[0].message
            usage = getattr(completion, "usage", None)
            step.tokens_used = int(usage.total_tokens) if usage else 0
            total_tokens += step.tokens_used

            # 2. 提取工具调用（function calling）
            tool_calls = getattr(msg, "tool_calls", None) or []
            # 兼容 thinking 模式（MiMo / DeepSeek-R1）：保留 reasoning_content
            reasoning = getattr(msg, "reasoning_content", None) or ""

            if not tool_calls:
                # LLM 没调工具但给了文本 —— 要求它继续决策
                content = (msg.content or "").strip()
                step.thought = content
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning
                messages.append(assistant_msg)
                messages.append({
                    "role": "user",
                    "content": "你必须通过 function calling 调用一个工具。没有其它方式推进。如需结束请调用 finish。",
                })
                step.elapsed_ms = int((time.time() - step_start) * 1000)
                steps.append(step)
                self.on_step(step)
                emit(self._step_event(step))
                continue

            # 3. 执行工具调用（只取第一个，multi-tool 先不支持）
            tool_call = tool_calls[0]
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            step.tool_name = fn_name
            step.tool_args = fn_args

            # 把 assistant 的 tool_call 加回 history（符合 OpenAI 协议）
            # 兼容 thinking 模式：带上 reasoning_content
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": tool_call.function.arguments or "{}",
                    },
                }],
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            # finish 工具特殊处理
            if fn_name == self.FINISH_TOOL_NAME:
                step.tool_result = ToolResult(success=True, content=fn_args)
                step.elapsed_ms = int((time.time() - step_start) * 1000)
                steps.append(step)
                self.on_step(step)
                emit(self._step_event(step))
                return AgentRunResult(
                    final_output=fn_args,
                    finished=True,
                    finish_reason="finished",
                    steps=steps,
                    total_tokens=total_tokens,
                    total_elapsed_ms=int((time.time() - total_start) * 1000),
                )

            result = self.tools.invoke(fn_name, fn_args)
            step.tool_result = result

            # 4. 把工具结果回灌给 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.to_llm_string(),
            })

            step.elapsed_ms = int((time.time() - step_start) * 1000)
            steps.append(step)
            self.on_step(step)
            emit(self._step_event(step))

        # 超出最大步数
        return AgentRunResult(
            final_output=None,
            finished=False,
            finish_reason="max_steps",
            steps=steps,
            total_tokens=total_tokens,
            total_elapsed_ms=int((time.time() - total_start) * 1000),
        )

    # ------------------------------------------------------------------ #
    # 可覆盖的钩子
    # ------------------------------------------------------------------ #

    def initial_user_message(self, task: str) -> str:
        """子类可覆盖以注入更结构化的初始消息。"""
        return task

    def on_step(self, step: AgentStep) -> None:
        """每步结束后调用，默认无操作。子类可用于日志/监控。"""
        pass

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _call_llm(self, messages: List[Dict[str, Any]], on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        通过 LLMClient 的 function calling 接口调用 LLM。
        返回 ChatCompletion（或同形的合成对象），便于 agent loop 访问 tool_calls / usage。

        on_event 不为空时走流式，把 LLM 增量（含 finish 报告的逐字 token）实时转发出去。
        """
        schemas = self.tools.openai_schemas()
        if on_event is None:
            return self.llm.invoke_with_tools(
                messages=messages,
                tools=schemas,
                temperature=self.temperature,
                tool_choice="auto",
            )

        def _on_delta(d: Dict[str, Any]) -> None:
            event = {"type": "llm_delta"}
            event.update(d)
            on_event(event)

        return self.llm.invoke_with_tools_stream(
            messages=messages,
            tools=schemas,
            temperature=self.temperature,
            tool_choice="auto",
            on_delta=_on_delta,
        )

    @staticmethod
    def _brief_args(args: Dict[str, Any]) -> str:
        """把工具参数压成一行简短摘要，便于前端轨迹展示。"""
        if not args:
            return ""
        parts: List[str] = []
        for key, value in args.items():
            text = str(value).replace("\n", " ")
            if len(text) > 60:
                text = text[:57] + "…"
            parts.append(f"{key}={text}")
        return ", ".join(parts)[:200]

    def _step_event(self, step: "AgentStep") -> Dict[str, Any]:
        """把一步执行记录转成可推送的 step 事件。"""
        res = step.tool_result
        if step.tool_name:
            status = "ok" if (res and res.success) else "err"
        else:
            status = "think"
        return {
            "type": "step",
            "idx": step.step_idx,
            "tool": step.tool_name,
            "args_brief": self._brief_args(step.tool_args),
            "status": status,
            "tokens": step.tokens_used,
            "ms": step.elapsed_ms,
            "thought": (step.thought or "")[:400],
            "error": (res.error if (res and not res.success and res.error) else ""),
        }

    def _ensure_finish_tool(self):
        """确保注册表中有 finish 工具。子类忘加时兜底。"""
        if self.FINISH_TOOL_NAME not in self.tools:
            self.tools.register(self._default_finish_tool())

    def _default_finish_tool(self) -> Tool:
        """默认的 finish 工具：接受任意 dict 作为 final output。"""
        return Tool(
            name=self.FINISH_TOOL_NAME,
            description="Call this when the task is complete. Pass the final answer as arguments.",
            parameters={
                "type": "object",
                "properties": {
                    "output": {
                        "type": "string",
                        "description": "Final answer or summary of what was accomplished.",
                    },
                },
                "required": ["output"],
            },
            run=lambda args: args,
        )
