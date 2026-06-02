from typing import Any, Dict, List

from openai import OpenAI

from .config import Settings


class LLMClient:
    """
    LLM 调用的轻量封装。

    - invoke(): 纯文本输入输出，返回 str
    - invoke_with_tools(): 保留原始 completion 对象，供 tool-calling 使用
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required in .env.")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def invoke(self, messages: List[Dict[str, str]], temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return self._extract_text_response(response)

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        tool_choice: str = "auto",
    ):
        return self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
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
