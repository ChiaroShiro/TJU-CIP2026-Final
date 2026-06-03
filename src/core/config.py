from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    vision_model: str
    vision_api_key: str
    vision_base_url: str
    planner_temperature: float = 0.2
    researcher_temperature: float = 0.3
    writer_temperature: float = 0.2
    critic_temperature: float = 0.1
    reviser_temperature: float = 0.3
    max_plan_items: int = 5
    search_backend: str = "duckduckgo"
    search_top_k: int = 6
    memory_top_k: int = 4
    memory_episode_k: int = 3
    memory_skill_k: int = 3
    context_max_chars: int = 7000
    max_reflection_rounds: int = 1
    max_revision_rounds: int = 2
    critic_threshold: float = 0.7
    enable_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_score_threshold: float = 0.3
    workspace_dir: Path = Path("workspace")

    @classmethod
    def from_env(cls, project_root: Path) -> "Settings":
        llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        llm_model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini").strip()
        vision_api_key = os.getenv("VISION_API_KEY", llm_api_key).strip()
        vision_base_url = os.getenv("VISION_BASE_URL", llm_base_url).strip()
        vision_model = os.getenv("VISION_MODEL_ID", llm_model).strip()
        workspace = project_root / os.getenv("RESEARCH_WORKSPACE_DIR", "workspace")

        return cls(
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            vision_model=vision_model,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
            planner_temperature=float(os.getenv("PLANNER_TEMPERATURE", "0.2")),
            researcher_temperature=float(os.getenv("RESEARCHER_TEMPERATURE", "0.3")),
            writer_temperature=float(os.getenv("WRITER_TEMPERATURE", "0.2")),
            critic_temperature=float(os.getenv("CRITIC_TEMPERATURE", "0.1")),
            reviser_temperature=float(os.getenv("REVISER_TEMPERATURE", "0.3")),
            max_plan_items=int(os.getenv("MAX_PLAN_ITEMS", "5")),
            search_backend=os.getenv("SEARCH_BACKEND", "duckduckgo"),
            search_top_k=int(os.getenv("SEARCH_TOP_K", "6")),
            memory_top_k=int(os.getenv("MEMORY_TOP_K", "4")),
            memory_episode_k=int(os.getenv("MEMORY_EPISODE_K", "3")),
            memory_skill_k=int(os.getenv("MEMORY_SKILL_K", "3")),
            context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "7000")),
            max_reflection_rounds=int(os.getenv("MAX_REFLECTION_ROUNDS", "1")),
            max_revision_rounds=int(os.getenv("MAX_REVISION_ROUNDS", "2")),
            critic_threshold=float(os.getenv("CRITIC_THRESHOLD", "0.7")),
            enable_rerank=os.getenv("ENABLE_RERANK", "true").lower() in ("1", "true", "yes"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
            rerank_score_threshold=float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3")),
            workspace_dir=workspace,
        )
