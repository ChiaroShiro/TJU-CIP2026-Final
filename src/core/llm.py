from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from openai import OpenAI

from .config import Settings


# --------------------------------------------------------------------------- #
# 流式 function-calling 的合成对象
# --------------------------------------------------------------------------- #
# invoke_with_tools_stream 在流结束后，把增量拼装成与非流式 ChatCompletion
# 同形的对象，使 BaseAgent 的下游逻辑（读取 choices[0].message / tool_calls /
# usage）无需任何改动即可复用。
# --------------------------------------------------------------------------- #


@dataclass
class _SyntheticFunction:
    name: str = ""
    arguments: str = ""


@dataclass
class _SyntheticToolCall:
    id: str = ""
    type: str = "function"
    function: _SyntheticFunction = field(default_factory=_SyntheticFunction)


@dataclass
class _SyntheticUsage:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _SyntheticMessage:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: Optional[List[_SyntheticToolCall]] = None


@dataclass
class _SyntheticChoice:
    message: _SyntheticMessage


@dataclass
class _SyntheticCompletion:
    choices: List[_SyntheticChoice]
    usage: Optional[_SyntheticUsage] = None


class LLMClient:
    """
    LLM 调用的轻量封装。

    - invoke(): 纯文本输入输出，返回 str
    - invoke_stream(): 纯文本流式，逐块 yield content 增量
    - invoke_with_tools(): 保留原始 completion 对象，供 tool-calling 使用
    - invoke_with_tools_stream(): 流式 tool-calling，逐增量回调 on_delta，
      结束后返回与非流式同形的合成 completion 对象
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None
        if settings.llm_api_key:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )

    @property
    def available(self) -> bool:
        """是否已配置可用的 LLM API key。"""
        return self.client is not None

    def invoke(self, messages: List[Dict[str, str]], temperature: float) -> str:
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return self._extract_text_response(response)

    def invoke_stream(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> Iterator[str]:
        """纯文本流式调用，逐块 yield 文本增量。用于报告/回复的打字机效果。"""
        self._ensure_client()
        stream = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            text = self._chunk_content(chunk)
            if text:
                yield text

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        tool_choice: str = "auto",
    ):
        self._ensure_client()
        return self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

    def invoke_with_tools_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        tool_choice: str = "auto",
        on_delta: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> _SyntheticCompletion:
        """
        流式 tool-calling。

        on_delta 收到的事件（dict）：
          {"kind": "content",   "text": "..."}                 助手正文增量
          {"kind": "reasoning", "text": "..."}                 思考增量（thinking 模型）
          {"kind": "tool_name", "name": "...", "index": i}     某个工具调用开始
          {"kind": "tool_args", "text": "...", "name": "...", "index": i}  工具参数增量

        返回：与非流式 ChatCompletion 同形的合成对象（含 choices/message/usage）。
        即便底层网关不支持工具增量，结束时仍能拼出完整 message，保证正确性。
        """
        self._ensure_client()
        stream = self._create_tool_stream(messages, tools, temperature, tool_choice)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tc_acc: Dict[int, Dict[str, str]] = {}
        usage: Optional[_SyntheticUsage] = None

        def _emit(event: Dict[str, Any]) -> None:
            if on_delta is not None:
                try:
                    on_delta(event)
                except Exception:
                    pass

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _SyntheticUsage(
                    total_tokens=int(getattr(chunk_usage, "total_tokens", 0) or 0),
                    prompt_tokens=int(getattr(chunk_usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(chunk_usage, "completion_tokens", 0) or 0),
                )

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                _emit({"kind": "content", "text": text})

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                _emit({"kind": "reasoning", "text": reasoning})

            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", 0) or 0
                slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                fn_name = getattr(fn, "name", None)
                if fn_name:
                    slot["name"] = fn_name
                    _emit({"kind": "tool_name", "name": fn_name, "index": idx})
                args_delta = getattr(fn, "arguments", None)
                if args_delta:
                    slot["args"] += args_delta
                    _emit({"kind": "tool_args", "text": args_delta, "name": slot["name"], "index": idx})

        tool_calls: List[_SyntheticToolCall] = []
        for idx in sorted(tc_acc.keys()):
            slot = tc_acc[idx]
            if not slot["name"]:
                continue
            tool_calls.append(
                _SyntheticToolCall(
                    id=slot["id"] or f"call_{idx}",
                    function=_SyntheticFunction(name=slot["name"], arguments=slot["args"]),
                )
            )

        message = _SyntheticMessage(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=tool_calls or None,
        )
        return _SyntheticCompletion(choices=[_SyntheticChoice(message=message)], usage=usage)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _create_tool_stream(self, messages, tools, temperature, tool_choice):
        """创建流式 tool-calling 请求；优先带 usage 统计，网关不支持时降级。"""
        base_kwargs = dict(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        try:
            return self.client.chat.completions.create(
                **base_kwargs, stream_options={"include_usage": True}
            )
        except Exception:
            # 部分 OpenAI 兼容网关不接受 stream_options，降级为无 usage 的普通流
            return self.client.chat.completions.create(**base_kwargs)

    @staticmethod
    def _chunk_content(chunk: Any) -> str:
        """从一个流式 chunk 中提取文本增量，兼容对象/字典两种结构。"""
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", None) or ""

    def _ensure_client(self) -> None:
        if self.client is None:
            raise ValueError(
                "LLM_API_KEY is required for LLM-backed commands. "
                "Set it in .env before running research/analyze/evaluate."
            )

    def _extract_text_response(self, response: Any) -> str:
        """兼容不同 OpenAI 兼容网关返回的文本结构。"""
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str):
                return output_text

            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                return self._normalize_message_content(message.get("content"))

            return str(response)

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                return self._normalize_message_content(
                    getattr(message, "content", "")
                )

        return str(response)

    def _normalize_message_content(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                    continue

                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))

            return "\n".join(p for p in parts if p)

        return str(content)
