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
    # threshold = "抄答案分数线"，含义完全取决于用哪把尺子，所以默认 0 = 按后端自动取：
    #   bge   → 0.60（实测 precision 100%、错误率 0、命中率 47%）
    #   ngram → 0.35（F0.5 最优；配合守卫 precision 96.7%）
    # 实测数据见 docs/evidence/threshold_sweep.md，脚本 tools/threshold_sweep.py。
    # 换后端必须重新标定——两个后端的分数分布完全不同，阈值不能互相搬。
    enable_cache: bool = os.getenv("ENABLE_CACHE", "1") == "1"
    cache_threshold: float = float(os.getenv("CACHE_THRESHOLD", "0"))
    cache_ttl_days: int = int(os.getenv("CACHE_TTL_DAYS", "7"))
    cache_cache_hard: bool = os.getenv("CACHE_CACHE_HARD", "0") == "1"
    # 载荷守卫（app/cacheguard.py）：向量粗筛之后再比一遍数字/单位/缩写/说法/褒贬。
    # 不带守卫实测最优阈值 precision 只有 66%，带上后 96%——见 docs/evidence/threshold_sweep.md
    cache_guard: bool = os.getenv("CACHE_GUARD", "1") == "1"

    # 向量后端（app/embedder.py）
    #   auto  模型文件在就用 bge，不在就 ngram（默认，部署包不带 91MB 模型也能跑）
    #   bge   强制本地小模型，加载失败降级
    #   ngram 强制字面 n-gram
    # 换后端会让旧缓存条目自动失效（维度不同不能比），这是有意的。
    embed_backend: str = os.getenv("EMBED_BACKEND", "auto")
    bge_model_dir: str = os.getenv("BGE_MODEL_DIR", "models/bge-small-zh")
    bge_max_len: int = int(os.getenv("BGE_MAX_LEN", "128"))
    bge_threads: int = int(os.getenv("BGE_THREADS", "2"))

    # RAG 检索（app/rag.py）：从本地文档召回相关片段注入 prompt
    # bm25_min 是实测校准的门槛：相关问题量产出的 BM25 最高分在 6~23，无关问题在 0~5
    # （"今天天气"例外地到 8.4，因为文档里恰好用它举例——这类误召回靠提示词
    #  "不相关就忽略"兜底，不靠调阈值，调高会把真召回一起误伤）
    enable_rag: bool = os.getenv("ENABLE_RAG", "1") == "1"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_bm25_min: float = float(os.getenv("RAG_BM25_MIN", "5.0"))
    # 相对分数截断：低于"第一名分数 × 这个系数"的候选丢掉。
    # 固定 top_k 的问题是它不管你检索得好不好，一律塞满 4 条——
    # 第一名 0.87、第三名 0.43 的时候，那两条弱相关只会稀释模型注意力。
    rag_rel_cutoff: float = float(os.getenv("RAG_REL_CUTOFF", "0.55"))
    rag_budget: int = int(os.getenv("RAG_BUDGET", "500"))  # 注入 prompt 的 token 配额
    kb_dirs: str = os.getenv("KB_DIRS", "docs")

    # 1 = 走 LangGraph 编排（app/graph.py）；0 = 走原来的显式流程（app/agent.py）
    # graph.py 导入失败（未装 langgraph）时 server.py 会自动回退
    enable_langgraph: bool = os.getenv("ENABLE_LANGGRAPH", "1") == "1"

    # ---------------- 工具调用 / ReAct（app/tools/, app/react.py）----------------
    # 为什么要有这一层：路由+缓存+RAG 解决的是"答得省、答得对"，
    # 但它们都只读预先准备好的资料。真正要"算一下/查库/跑代码/抓网页"时，
    # 模型必须能动手——这就是 tool calling，也是 Agent 与问答系统的分界线。
    enable_tools: bool = os.getenv("ENABLE_TOOLS", "1") == "1"
    # 循环上限。没有它，模型会一直"再确认一下"，token 烧光还可能死循环
    tool_max_steps: int = int(os.getenv("TOOL_MAX_STEPS", "5"))
    # 观察结果截断长度：工具返回 200KB HTML 直接塞回 prompt 会一次性吃掉上下文
    tool_result_chars: int = int(os.getenv("TOOL_RESULT_CHARS", "1600"))
    tool_timeout: float = float(os.getenv("TOOL_TIMEOUT", "8"))
    # 单次请求的工具循环预算（人民币）。超了强制收口，避免一条问题烧掉几块钱
    tool_budget_cny: float = float(os.getenv("TOOL_BUDGET_CNY", "0.05"))
    # 是否允许执行 run_python 这类"危险级"工具。P3 会换成人工确认（HITL）
    tool_allow_danger: bool = os.getenv("TOOL_ALLOW_DANGER", "1") == "1"
    # 抓网页的域名白名单（逗号分隔）。开放抓取等于给 SSRF 开后门
    tool_fetch_domains: str = os.getenv(
        "TOOL_FETCH_DOMAINS",
        "docs.python.org,github.com,raw.githubusercontent.com,pypi.org,example.com")
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "50"))

    # 公开演示护栏（app/guard.py）。0 = 不限制，本地开发保持全放开；
    # 部署到公网时必须收紧，否则链接被转发后会把 API 额度刷光。
    demo_quota_per_day: int = int(os.getenv("DEMO_QUOTA_PER_DAY", "0"))
    demo_rate_per_min: int = int(os.getenv("DEMO_RATE_PER_MIN", "0"))

    price_local_per_1k: float = float(os.getenv("PRICE_LOCAL_PER_1K", "0.0028"))
    price_cloud_in_per_1k: float = float(os.getenv("PRICE_CLOUD_IN_PER_1K", "0.005"))
    price_cloud_out_per_1k: float = float(os.getenv("PRICE_CLOUD_OUT_PER_1K", "0.015"))

    db_path: str = os.getenv("DB_PATH", "runs.db")


settings = Settings()
