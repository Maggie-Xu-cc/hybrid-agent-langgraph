import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # AUP 等环境可能没有 python-dotenv：手动读 .env（只认 KEY=VALUE，不覆盖已有环境变量）
    from pathlib import Path

    _env_file = Path(__file__).resolve().parents[1] / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


@dataclass
class Settings:
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
    local_model: str = os.getenv("LOCAL_MODEL", "Qwen2.5-7B-AWQ")
    local_api_key: str = os.getenv("LOCAL_API_KEY", "EMPTY")

    cloud_base_url: str = os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com/v1")
    cloud_model: str = os.getenv("CLOUD_MODEL", "deepseek-chat")
    cloud_api_key: str = os.getenv("CLOUD_API_KEY", "")

    long_query_chars: int = int(os.getenv("LONG_QUERY_CHARS", "300"))
    local_max_tokens: int = int(os.getenv("LOCAL_MAX_TOKENS", "768"))
    cloud_max_tokens: int = int(os.getenv("CLOUD_MAX_TOKENS", "1024"))

    enable_router_model: bool = os.getenv("ENABLE_ROUTER_MODEL", "1") == "1"
    enable_escalation: bool = os.getenv("ENABLE_ESCALATION", "1") == "1"
    router_max_tokens: int = int(os.getenv("ROUTER_MAX_TOKENS", "48"))
    mock_mode: bool = os.getenv("MOCK_MODE", "0") == "1"

    # 语义缓存（app/cache.py）：相似问题直接复用答案，成本降为 0
    # threshold=0.45 是实测校准值：同义改写对 n-gram 余弦通常 ≥0.5，无关问题 ≤0.3
    enable_cache: bool = os.getenv("ENABLE_CACHE", "1") == "1"
    cache_threshold: float = float(os.getenv("CACHE_THRESHOLD", "0.45"))
    cache_ttl_days: int = int(os.getenv("CACHE_TTL_DAYS", "7"))
    cache_cache_hard: bool = os.getenv("CACHE_CACHE_HARD", "0") == "1"

    # 1 = 走 LangGraph 编排（app/graph.py）；0 = 走原来的显式流程（app/agent.py）
    # graph.py 导入失败（未装 langgraph）时 server.py 会自动回退
    enable_langgraph: bool = os.getenv("ENABLE_LANGGRAPH", "1") == "1"

    price_local_per_1k: float = float(os.getenv("PRICE_LOCAL_PER_1K", "0.0028"))
    price_cloud_in_per_1k: float = float(os.getenv("PRICE_CLOUD_IN_PER_1K", "0.005"))
    price_cloud_out_per_1k: float = float(os.getenv("PRICE_CLOUD_OUT_PER_1K", "0.015"))

    db_path: str = os.getenv("DB_PATH", "runs.db")


settings = Settings()
