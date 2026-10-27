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

    # 超时。openai SDK 不给 timeout 时默认 600 秒——本地端口"通但不响应"
    # （Ollama 半死、vLLM 加载中等）会让每条请求白等十分钟，这是纯本机跑最坑的地方。
    local_timeout: float = float(os.getenv("LOCAL_TIMEOUT", "15"))
    cloud_timeout: float = float(os.getenv("CLOUD_TIMEOUT", "60"))
    # SDK 默认失败重试 2 次：本地挂了会变成 3 倍等待，本地一律不重试
    cloud_max_retries: int = int(os.getenv("CLOUD_MAX_RETRIES", "1"))

    # 本地熔断：连续失败 N 次后短路，冷却期内直接跳过本地（0 延迟），
    # 冷却结束放一次试探请求，成功即恢复。纯云端模式下等于永久短路。
    local_breaker_threshold: int = int(os.getenv("LOCAL_BREAKER_THRESHOLD", "2"))
    local_breaker_cooldown: float = float(os.getenv("LOCAL_BREAKER_COOLDOWN", "60"))
    # 1 = 完全不调本地（纯云端跑基准/演示时最省时间）
    disable_local: bool = os.getenv("DISABLE_LOCAL", "0") == "1"

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

    # 公开演示护栏（app/guard.py）。0 = 不限制，本地开发保持全放开；
    # 部署到公网时必须收紧，否则链接被转发后会把 API 额度刷光。
    demo_quota_per_day: int = int(os.getenv("DEMO_QUOTA_PER_DAY", "0"))
    demo_rate_per_min: int = int(os.getenv("DEMO_RATE_PER_MIN", "0"))

    price_local_per_1k: float = float(os.getenv("PRICE_LOCAL_PER_1K", "0.0028"))
    price_cloud_in_per_1k: float = float(os.getenv("PRICE_CLOUD_IN_PER_1K", "0.005"))
    price_cloud_out_per_1k: float = float(os.getenv("PRICE_CLOUD_OUT_PER_1K", "0.015"))

    db_path: str = os.getenv("DB_PATH", "runs.db")


settings = Settings()
