"""src.tools — 工具抽象 + 注册表 + 各类具体工具 builder"""

from .tool import Tool, ToolRegistry, ToolResult
from .search_tools import build_arxiv_search_tool, build_github_code_search_tool, build_s2_search_tool
from .paper_tools import build_fetch_fulltext_tool, build_analyze_paper_tool
from .memory_tools import (
    build_retrieve_memory_tool,
    build_save_note_tool,
    build_save_research_episode_tool,
)
from .delegation_tools import build_delegate_to_critic_tool, build_delegate_to_reviser_tool
from .web_tools import build_web_search_tool, build_web_fetch_tool

__all__ = [
    "Tool", "ToolRegistry", "ToolResult",
    "build_arxiv_search_tool", "build_github_code_search_tool", "build_s2_search_tool",
    "build_fetch_fulltext_tool", "build_analyze_paper_tool",
    "build_retrieve_memory_tool", "build_save_note_tool",
    "build_save_research_episode_tool",
    "build_delegate_to_critic_tool", "build_delegate_to_reviser_tool",
    "build_web_search_tool", "build_web_fetch_tool",
]
