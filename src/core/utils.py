import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    从可能带有冗余文本的字符串中抽取第一个完整 JSON 对象。

    LLM 经常在 JSON 外加自然语言说明（如 ```json ... ```），
    这个函数先尝试整体解析，失败后回退到正则抓取第一个 `{...}` 块。
    """
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
