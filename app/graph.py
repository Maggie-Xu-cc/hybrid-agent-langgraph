"""LangGraph 版主流程（阶段一重构）。

与 agent.py 的显式三步流程完全等价，但改成 StateGraph 编排：

    START
      │
    route ──easy──▶ local_answer ──▶ self_check ──需升级──▶ escalate ──┐
      │                                        │                      │
      └──hard──▶ cloud_answer                  └──通过──▶ bookkeep ◀──┘
                                     (cloud_answer ──────────▶ bookkeep)
                                                    │
                                                   END

迁移原则（为什么不推倒重来）：
- 路由逻辑 router.decide() 原样保留，从"函数调用"变成"条件边"；
- 记账逻辑 store.record() 原样保留，集中在 bookkeep 节点统一落库；
- agent.py 保留为基线实现，eval 可对比两条链路（ENABLE_LANGGRAPH=0 回退）。

相比 agent.py 新增的能力：
- checkpointer 按 session_id 持久化会话状态，天然支持多轮对话；
- 上下文工程（阶段二）：按 token 预算组装 prompt，长期记忆注入、历史摘要压缩。

图的结构（阶段二后）：

    START ─▶ load_memory ─▶ cache_lookup ─命中─▶ bookkeep ─▶ save_memory ─▶ END
                                 │
                                 └未命中─▶ route ──easy──▶ local_answer ─▶ self_check ─需升级─▶ escalate ─┐
                                              │                                │                         │
                                              └──hard──▶ cloud_answer          └─通过─▶ bookkeep ◀───────┘
                                                         (cloud_answer ─────────────────▶ bookkeep)

语义缓存（阶段三）在路由之前：相似问题直接复用答案，成本降为 0。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import cache, context, memory, rag, store
from .dbutil import with_retry
from .clients import ProviderError, call_cloud, call_local
from .config import settings
from .router import Route, decide

SYSTEM = (
    "你是一个严谨的助手。用中文回答，直接给结论，不要复述问题。"
    "如果不确定，明确说\"我不确定\"，不要编造。"
)

ESCALATE_MARKERS = [
    "我不确定",
    "不确定",
    "无法回答",
    "我不知道",
    "作为AI",
    "没有足够信息",
]

# 会话历史：超过 20 条时把最老的部分压缩成摘要（软上限，不是硬丢弃）
CONTEXT_WINDOW_MESSAGES = 8
MAX_HISTORY_MESSAGES = 20
SUMMARY_KEEP_MESSAGES = 12        # 压缩后保留的最近消息条数

SUMMARIZE_PROMPT = (
    "请把以下多轮对话压缩成一段不超过120字的中文摘要，"
    "保留用户的关键事实（身份、偏好、在做什么）和重要结论，不要评论。\n\n对话：\n"
)

# checkpointer 的落盘位置。会话列表也读这个库，所以统一成一个常量，避免两处路径漂移。
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.db")


class AgentState(TypedDict, total=False):
    """图的全局状态。节点只返回自己更新的字段（partial update）。"""

    query: str
    force_route: str | None
    user_id: str  # 长期记忆的身份键；空 = 不启用记忆（评测/匿名）
    history: list[dict]  # 多轮历史（checkpointer 持久化）
    facts: dict  # 长期记忆召回的用户事实（load_memory 注入）
    summary: str  # 早前对话的压缩摘要（bookkeep 生成，跨轮持久化）
    messages: list[dict]  # 本轮实际发给模型的消息
    ctx_stats: dict  # 上下文组装统计（各部分 token、裁剪条数）
    route: dict  # {"difficulty","reason","by","latency_ms","tokens_in","tokens_out"}
    answer: str
    model_used: str
    local_in: int
    local_out: int
    cloud_in: int
    cloud_out: int
    escalated: bool
    degraded: bool
    needs_escalation: bool
    cache_hit: bool  # 语义缓存是否命中（cache_lookup 节点判定）
    cache_score: float  # 命中时的相似度
    cache_cost: float  # 命中条目当初生成时花的钱（用来算"省了多少"）
    use_cache: bool  # 本轮是否启用缓存（评测时关掉，避免污染对照实验）
    rag_chunks: list  # 检索到的相关片段（retrieve 节点，注入 prompt 用）
    rag_hit: bool  # 本轮是否检索到资料
    rag_stats: dict  # 检索统计（条数、耗时、来源、最高分）
    use_rag: bool  # 本轮是否启用检索（评测/对照实验可关）
    started: float
    result: dict


def _needs_escalation(text: str) -> bool:
    if len(text.strip()) < 12:
        return True
    return any(marker in text for marker in ESCALATE_MARKERS)


def _build_messages(state: AgentState, query: str) -> tuple[list[dict], dict]:
    """按 token 预算组装 prompt：system + 记忆 + 检索资料 + 摘要 + 历史 + 提问。"""
    return context.build_messages(
        system=SYSTEM,
        facts=state.get("facts") or {},
        summary=state.get("summary") or "",
        history=state.get("history") or [],
        query=query,
        rag_chunks=state.get("rag_chunks") or [],
    )


# ---------------- 节点 ----------------

def load_memory_node(state: AgentState) -> dict:
    """长期记忆召回：按 user_id 取用户事实，供 route 组 prompt 时注入。

    user_id 为空（评测/匿名请求）时返回空 facts——保证与基线链路行为一致。
    """
    user_id = state.get("user_id") or ""
    facts = memory.recall(user_id) if user_id else {}
    return {"user_id": user_id, "facts": facts}


def cache_lookup_node(state: AgentState) -> dict:
    """语义缓存查找：路由之前拦一道，命中则整个推理链路都不用跑。

    命中时直接构造 answer/route 等字段，条件边把流程引向 bookkeep——
    bookkeep 仍走原路径构造 result，保证缓存命中的返回结构与普通请求完全一致。
    查找成本 ≈ 一次 256 维向量线性扫描，毫秒级，0 token。

    传 kb_version 是为了让"答案依赖文档"这个前提可校验：知识库没变就照样复用，
    重建过索引就自动失效。原来的做法是一律不缓存检索过的答案，结果缓存几乎不工作。
    """
    if not state.get("use_cache", settings.enable_cache):
        return {"cache_hit": False}
    hit = cache.lookup(state["query"], kb_version=rag.version())
    if not hit:
        return {"cache_hit": False}
    return {
        "cache_hit": True,
        "cache_score": hit["score"],
        "cache_cost": hit.get("cost", 0.0),
        "started": time.perf_counter(),  # 命中路径不经过 route，这里补上计时起点
        "answer": hit["answer"],
        "model_used": f"{hit['model']} (cache)",
        "route": {
            "difficulty": hit["difficulty"],
            "reason": f"语义缓存命中(相似度{hit['score']:.2f})",
            "by": "cache",
            "latency_ms": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        },
        "messages": [],
        "ctx_stats": {},
        "local_in": 0,
        "local_out": 0,
        "cloud_in": 0,
        "cloud_out": 0,
        "escalated": False,
        "degraded": False,
    }


def retrieve_node(state: AgentState) -> dict:
    """RAG 检索：从本地知识库召回相关片段，供 route 组 prompt 时注入。

    为什么放在"缓存未命中之后、路由之前"：
    - 缓存命中时已经是答案了，检索纯属浪费；
    - route 节点要组装 messages，检索结果必须在那之前就位。

    失败不阻塞：知识库没建、表不存在、内容为空，一律退化为"没有资料的普通问答"，
    和没上 RAG 时行为完全一致。检索是增强，不是链路的前提。
    """
    started = time.perf_counter()
    if not state.get("use_rag", settings.enable_rag):
        return {"rag_chunks": [], "rag_hit": False, "rag_stats": {"n": 0, "enabled": False}}
    try:
        chunks = rag.search(state["query"])
    except Exception as exc:   # noqa: BLE001 — 检索挂了也不能让整条请求挂
        chunks = []
        err = f"{type(exc).__name__}: {exc}"
    else:
        err = ""
    return {
        "rag_chunks": chunks,
        "rag_hit": bool(chunks),
        "rag_stats": {
            "enabled": True,
            "n": len(chunks),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "top_score": chunks[0]["score"] if chunks else 0.0,
            "sources": [f"{c['doc']} § {c['heading']}" for c in chunks],
            "error": err,
        },
    }


def route_node(state: AgentState) -> dict:
    """路由节点：原来的 agent.run() 第一步。路由 token 记在本地账上。"""
    started = time.perf_counter()
    query = state["query"].strip()

    route: Route = decide(query)
    force = state.get("force_route")
    if force in ("easy", "hard"):
        route = Route(force, f"强制{force}", route.by)

    messages, ctx_stats = _build_messages({**state, "query": query}, query)

    return {
        "query": query,
        "started": started,
        "route": {
            "difficulty": route.difficulty,
            "reason": route.reason,
            "by": route.by,
            "latency_ms": route.latency_ms,
            "tokens_in": route.tokens_in,
            "tokens_out": route.tokens_out,
        },
        "messages": messages,
        "ctx_stats": ctx_stats,
        "local_in": route.tokens_in,
        "local_out": route.tokens_out,
        "cloud_in": 0,
        "cloud_out": 0,
        "escalated": False,
        "degraded": False,
    }


def local_answer_node(state: AgentState) -> dict:
    """本地小模型作答；不可用时降级到云端（与 agent.py 行为一致）。"""
    messages = state["messages"]
    route = state["route"]
    local_in = state.get("local_in", 0)
    local_out = state.get("local_out", 0)
    update: AgentState = {}

    try:
        answer, tin, tout = call_local(messages)
        update.update(answer=answer, model_used=settings.local_model,
                      local_in=local_in + tin, local_out=local_out + tout,
                      degraded=False)
    except ProviderError:
        answer, tin, tout = call_cloud(messages)
        update.update(answer=answer, model_used=settings.cloud_model,
                      cloud_in=tin, cloud_out=tout, degraded=True,
                      route={**route, "reason": f"{route['reason']}(本地不可用)"})
    return update


def cloud_answer_node(state: AgentState) -> dict:
    """云端大模型作答；不可用时降级到本地。"""
    messages = state["messages"]
    route = state["route"]
    update: AgentState = {}

    try:
        answer, tin, tout = call_cloud(messages)
        update.update(answer=answer, model_used=settings.cloud_model,
                      cloud_in=tin, cloud_out=tout, degraded=False)
    except ProviderError:
        answer, tin, tout = call_local(messages)
        update.update(answer=answer, model_used=settings.local_model,
                      local_in=state.get("local_in", 0) + tin,
                      local_out=state.get("local_out", 0) + tout,
                      degraded=True,
                      route={**route, "reason": f"{route['reason']}(云端不可用)"})
    return update


def self_check_node(state: AgentState) -> dict:
    """质量自检：本地答案过短或含不确定表述 → 标记需升级。"""
    needs = (
        settings.enable_escalation
        and not state.get("degraded")
        and _needs_escalation(state.get("answer", ""))
    )
    return {"needs_escalation": needs}


def escalate_node(state: AgentState) -> dict:
    """升级重答：云端重新生成一次；失败则保留本地答案（不阻塞主流程）。"""
    update: AgentState = {"escalated": True}
    try:
        answer, tin, tout = call_cloud(state["messages"])
        update.update(
            answer=answer,
            cloud_in=state.get("cloud_in", 0) + tin,
            cloud_out=state.get("cloud_out", 0) + tout,
            model_used=f"{settings.local_model} -> {settings.cloud_model}",
        )
    except ProviderError:
        pass  # 保留本地答案
    return update


def _summarize(old_msgs: list[dict]) -> tuple[str, int, int]:
    """把老对话压缩成摘要。返回 (摘要, tin, tout)。

    优先本地小模型；不可用（含 MOCK）时退化为"逐条截取前 30 字"的机械摘要——
    效果差一些，但保证链路不阻塞，且成本为 0。
    """
    if not old_msgs:
        return "", 0, 0
    transcript = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in old_msgs)
    try:
        text, tin, tout = call_local(
            [{"role": "user", "content": SUMMARIZE_PROMPT + transcript[:3000]}],
            max_tokens=160,
        )
        return text.strip()[:context.SUMMARY_MAX_CHARS], tin, tout
    except ProviderError:
        parts = [f"{m.get('role','user')}:{m.get('content','')[:30]}"
                 for m in old_msgs[-8:]]
        return ("；".join(parts))[:SUMMARY_MAX_CHARS], 0, 0


def bookkeep_node(state: AgentState) -> dict:
    """记账节点：成本计算 + 历史摘要压缩 + 落库 + 更新 history。"""
    route = state["route"]
    local_in = state.get("local_in", 0)
    local_out = state.get("local_out", 0)
    cloud_in = state.get("cloud_in", 0)
    cloud_out = state.get("cloud_out", 0)
    summary_in = summary_out = 0

    # ---- 语义缓存写入：把这份答案存起来，下次相似问题就能 0 成本复用 ----
    # 只存"首轮独立问题"：多轮问题依赖会话上下文，换会话直接复用会答错。
    # hard 默认不存（质量敏感+重复率低），CACHE_CACHE_HARD=1 可打开。
    # 注意不排除 degraded：降级只是换了provider（本地挂了走云端），答案本身是好的；
    # 若排除，本地模型一挂缓存就彻底不工作——恰是最需要省钱的时候失效，不合理。
    #
    # 关于"检索过的答案能不能缓存"：能，但要绑定知识库版本。
    # 早先的做法是 not rag_hit —— 只要检索命中就不写，理由"文档改了答案就过期"。
    # 这个顾虑成立，但做法把粒度搞错了：线上有知识库时几乎每条查询都会检索命中，
    # 于是缓存条目永远是 0，第一道省钱关卡形同关闭（实测线上 entries=0 就是这个原因）。
    # 现在写进 kb_version，命中时比对版本：文档没改照样复用，重建过索引自动失效。
    pre_history = state.get("history") or []
    rag_hit = bool(state.get("rag_hit"))
    cache_writable = (
        state.get("use_cache", settings.enable_cache)
        and not state.get("cache_hit")
        and not pre_history
        and bool(state.get("answer"))
        and (route["difficulty"] == "easy" or settings.cache_cache_hard)
    )

    # ---- 先追加 history，再决定是否要压缩（摘要在落库前做，成本才能进账）----
    history = list(pre_history)
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": state.get("answer", "")})

    summary = state.get("summary") or ""
    summarized = False
    if len(history) > MAX_HISTORY_MESSAGES:
        old, keep = history[:-SUMMARY_KEEP_MESSAGES], history[-SUMMARY_KEEP_MESSAGES:]
        piece, tin, tout = _summarize(old)
        summary_in, summary_out = tin, tout
        local_in += tin
        local_out += tout
        # 新旧摘要合并：旧的浓缩成一句话前缀，避免摘要无限膨胀
        summary = (summary[:120] + "\n" if summary else "") + piece
        history = keep
        summarized = True

    tokens_in = local_in + cloud_in
    tokens_out = local_out + cloud_out
    latency_ms = int((time.perf_counter() - state["started"]) * 1000)

    cost = (
        (local_in + local_out) / 1000 * settings.price_local_per_1k
        + cloud_in / 1000 * settings.price_cloud_in_per_1k
        + cloud_out / 1000 * settings.price_cloud_out_per_1k
    )
    baseline = (
        tokens_in / 1000 * settings.price_cloud_in_per_1k
        + tokens_out / 1000 * settings.price_cloud_out_per_1k
    )
    # 缓存命中的基线要单独处理：这一轮 0 token、0 成本，按 token 算出来的
    # baseline 也是 0，于是 (baseline - cost) / baseline 永远是 0/0，
    # 页面上的"总节省比例"就成了恒定的 0%——省了钱但账上看不出来。
    # 正确口径是"如果这次没命中缓存、真的调了一次模型要花多少"，
    # 也就是这条答案当初生成时的成本（存在缓存条目里）。
    if state.get("cache_hit"):
        baseline = float(state.get("cache_cost") or 0.0)

    store.record(
        query=state["query"],
        route=route["difficulty"],
        reason=route["reason"],
        decided_by=route["by"],
        model=state.get("model_used", ""),
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        baseline_cost=baseline,
        escalated=state.get("escalated", False),
    )

    if cache_writable:
        cache.put(
            query=state["query"],
            answer=state["answer"],
            model=state.get("model_used", ""),
            difficulty=route["difficulty"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            # 用过检索就记下当前知识库版本，没用过留空（不受文档更新影响）
            kb_version=rag.version() if rag_hit else "",
        )

    facts = state.get("facts") or {}
    result = {
        "answer": state.get("answer", ""),
        "route": route["difficulty"],
        "reason": route["reason"],
        "decided_by": route["by"],
        "model": state.get("model_used", ""),
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_cny": round(cost, 6),
        "baseline_cost_cny": round(baseline, 6),
        "escalated": state.get("escalated", False),
        "degraded": state.get("degraded", False),
        "cache_hit": bool(state.get("cache_hit")),
        "cache_score": round(state.get("cache_score") or 0, 3),
        # 上下文工程的可观测性：注入了哪些记忆、裁了多少历史
        "memory": {
            "enabled": bool(state.get("user_id")),
            "facts_injected": list(facts.keys()),
            "summary_used": bool(state.get("summary")),
            "summarized_now": summarized,
            "ctx_tokens": state.get("ctx_stats") or {},
        },
        # 检索的可观测性：命了几段、来自哪个文档、最高分多少、耗时
        "rag": {
            **(state.get("rag_stats") or {"enabled": False, "n": 0}),
            "hit": bool(state.get("rag_hit")),
            "chunks": [
                {"doc": c["doc"], "heading": c["heading"],
                 "score": c["score"], "bm25": c["bm25"], "vec": c["vec"]}
                for c in (state.get("rag_chunks") or [])
            ],
        },
    }
    return {"history": history, "summary": summary, "result": result}


def save_memory_node(state: AgentState) -> dict:
    """长期记忆写入：从本轮提问抽取事实落库。

    只抽 query 不抽 answer：模型回复里可能出现"用户说他是学生"这类转述，
    直接抽会把转述错当陈述。宁可少记，不可记错。
    """
    user_id = state.get("user_id") or ""
    saved = memory.remember(user_id, state["query"]) if user_id else {}
    if saved:
        result = dict(state.get("result") or {})
        if "memory" in result:
            result["memory"]["facts_saved"] = list(saved.keys())
            return {"result": result}
    return {}


# ---------------- 条件边 ----------------

def after_cache(state: AgentState) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def after_route(state: AgentState) -> str:
    return state["route"]["difficulty"]


def after_local(state: AgentState) -> str:
    return "done" if state.get("degraded") else "check"


def after_check(state: AgentState) -> str:
    return "escalate" if state.get("needs_escalation") else "done"


# ---------------- 组装 ----------------

def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("cache_lookup", cache_lookup_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("route", route_node)
    g.add_node("local_answer", local_answer_node)
    g.add_node("cloud_answer", cloud_answer_node)
    g.add_node("self_check", self_check_node)
    g.add_node("escalate", escalate_node)
    g.add_node("bookkeep", bookkeep_node)
    g.add_node("save_memory", save_memory_node)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "cache_lookup")
    g.add_conditional_edges("cache_lookup", after_cache, {
        "hit": "bookkeep", "miss": "retrieve",
    })
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", after_route, {
        "easy": "local_answer", "hard": "cloud_answer",
    })
    g.add_conditional_edges("local_answer", after_local, {
        "check": "self_check", "done": "bookkeep",
    })
    g.add_conditional_edges("self_check", after_check, {
        "escalate": "escalate", "done": "bookkeep",
    })
    g.add_edge("escalate", "bookkeep")
    g.add_edge("cloud_answer", "bookkeep")
    g.add_edge("bookkeep", "save_memory")
    g.add_edge("save_memory", END)
    return g.compile(checkpointer=checkpointer)


# 全局一把可重入锁：既保护建表 DDL，也作为所有 SqliteSaver 实例共用的写锁
_ckpt_lock = threading.RLock()

# 编译好的图按"检查点库路径"缓存：主库一个，并发基准的每线程库各一个。
# 用锁保护初始化，避免多个请求同时发现"还没建"而各建一份（那样就又变成多连接了）。
_apps: dict[str, object] = {}
_app_lock = threading.Lock()


def _with_db_retry(fn, tries: int = 6, delay: float = 0.15):
    """checkpointer 的读写重试（统一实现见 app/dbutil.py）。

    checkpoint 落盘比业务记录更频繁，这里把重试窗口放大到约 2 秒：
    Windows 上杀毒软件扫描 -wal 文件会把锁住几百毫秒，重试一两次不够。
    """
    return with_retry(fn, tries=tries, delay=delay)


# 已经初始化过（建表 + 设 WAL）的库
_ready_dbs: set[str] = set()


def _make_checkpointer(db_path: str | None = None):
    """优先 SQLite 持久化（重启不丢会话），装不上退回内存版。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = db_path or CHECKPOINT_DB
        # timeout 就是 sqlite 的 busy_timeout：并发写时排队等，
        # 而不是立刻抛 database is locked。
        conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        saver = SqliteSaver(conn)
        saver.lock = _ckpt_lock

        # 建表 + PRAGMA journal_mode=WAL 只做一次！
        # setup() 里有 PRAGMA journal_mode=WAL，而 SQLite 切换日志模式需要独占访问：
        # 多个连接（尤其在别的连接有活动事务时）各自执行，会切不动、退回 rollback
        # journal，之后混着写就会抛 "attempt to write a readonly database"。
        # 实测：这个是并发读写偶发失败的真凶（每线程独立库时不出现，因为每库只有
        # 一个连接）。所以首个连接负责建库，后续连接直接标记已就绪、跳过 DDL。
        with _ckpt_lock:
            if path not in _ready_dbs or not os.path.exists(path):
                saver.setup()
                _ready_dbs.add(path)
            else:
                saver.is_setup = True

        # 兜底：即便上面处理了，Windows 上仍可能偶发瞬时锁（杀软扫文件等），
        # 只对数据库操作重试，不重跑模型——重跑整图会再花一次模型钱
        for name in ("put", "put_writes", "get_tuple", "list"):
            if hasattr(saver, name):
                setattr(saver, name, _with_db_retry(getattr(saver, name)))
        return saver
    except Exception:
        return MemorySaver()


def get_app(db_path: str | None = None):
    """取编译好的图（按库路径缓存，全局共享）。

    为什么是全局单例而不是每线程一个：
    SQLite 的 WAL 自动 checkpoint 需要独占锁，多个连接同时写同一个库时，
    checkpoint 长期拿不到锁 → WAL 只涨不缩（实测涨到 4MB）→ 最终所有写都失败，
    表现为满屏 "attempt to write a readonly database"，且**不可自愈**。
    改用"一条连接 + 全局锁串行"后，SQLite 内部没有连接间争用，checkpoint 正常。
    代价是并发请求在落盘时排队——但那是毫秒级，真正的耗时在网络调用上。

    并发基准仍然可以每线程传不同 db_path，不同文件互不影响。
    """
    key = db_path or CHECKPOINT_DB
    with _app_lock:
        app = _apps.get(key)
        if app is None:
            app = build_graph(_make_checkpointer(db_path))
            _apps[key] = app
        return app


def run_graph(query: str, session_id: str = "default",
              force_route: str | None = None,
              user_id: str = "",
              use_cache: bool | None = None,
              use_rag: bool | None = None,
              checkpoint_db: str | None = None) -> dict:
    """对外入口：与 agent.run() 返回字段完全一致。

    session_id  → 短期记忆（本段对话的上下文，checkpointer 持久化）
    user_id     → 长期记忆（跨会话的用户事实，SQLite user_facts 表）
    use_cache   → 语义缓存开关；None = 跟随全局配置。评测时显式传 False，
                  否则同义用例第二次出现会命中缓存，污染对照实验。
    use_rag     → 检索开关；None = 跟随全局配置。做"有无 RAG"对照实验时显式传值，
                  否则两组结果不具有可比性。
    checkpoint_db → 会话状态落到哪个库；默认用主库。并发基准传独立文件，
                  避免多线程争抢同一个 sqlite 的 WAL 文件。
    两个记忆键独立：新会话不会忘掉用户，新身份不会读到别人的会话。
    """
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}

    app = get_app(checkpoint_db)
    config = {"configurable": {"thread_id": session_id}}
    out = app.invoke(
        {"query": query, "force_route": force_route, "user_id": user_id,
         "use_cache": settings.enable_cache if use_cache is None else use_cache,
         "use_rag": settings.enable_rag if use_rag is None else use_rag},
        config,
    )
    return out["result"]


# ---------------------------------------------------------------------------
# 会话浏览：checkpointer 只提供"按 thread_id 存取"，不提供"列出有哪些会话"。
# 但用户需要能找回之前的对话，所以这里反过来读 checkpointer 自己的表。
# ---------------------------------------------------------------------------

# 评测/冒烟脚本造的线程不该出现在用户界面上
_HIDDEN_PREFIXES = ("eval-", "smoke-", "dbg-", "other-")


def _uuid6_time(checkpoint_id: str) -> float | None:
    """从 checkpoint_id 解出时间戳。

    LangGraph 的 checkpoint_id 是 UUIDv6（时间有序），前 60 bit 就是时间。
    解不出来就返回 None，调用方退回"按插入顺序"，不让 UI 因为解析失败而崩。
    """
    try:
        p = checkpoint_id.split("-")
        ts60 = (int(p[0], 16) << 28) | (int(p[1], 16) << 12) | int(p[2][1:], 16)
        return ts60 / 1e7 - 12219292800          # 1582-10-15 起的 100ns 数 → Unix 秒
    except Exception:
        return None


def _fmt_time(ts: float | None, now: datetime | None = None) -> str:
    if not ts:
        return ""
    dt = datetime.fromtimestamp(ts)
    now = now or datetime.now()
    delta = (now - dt).total_seconds()
    if delta < 60:
        return "刚刚"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    return dt.strftime("%m-%d %H:%M")


def _load_history(session_id: str) -> list[dict]:
    try:
        state = get_app().get_state({"configurable": {"thread_id": session_id}})
    except Exception:
        return []
    if not state or not state.values:
        return []
    return state.values.get("history", []) or []


def list_sessions(limit: int = 20, include_hidden: bool = False) -> list[dict]:
    """列出历史会话，按最近活动排序。title 取第一轮用户提问。"""
    if not os.path.exists(CHECKPOINT_DB):
        return []
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        rows = conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) cid FROM checkpoints"
            " GROUP BY thread_id ORDER BY MAX(rowid) DESC LIMIT ?",
            (limit * 4,),
        ).fetchall()
        conn.close()
    except Exception:
        return []

    out: list[dict] = []
    now = datetime.now()
    for tid, cid in rows:
        if not include_hidden and tid.startswith(_HIDDEN_PREFIXES):
            continue
        history = _load_history(tid)
        if not history:
            continue
        first_user = next(
            (m.get("content", "") for m in history if m.get("role") == "user"), "")
        title = first_user.strip().replace("\n", " ")
        if len(title) > 26:
            title = title[:26] + "…"
        ts = _uuid6_time(cid)
        out.append({
            "session_id": tid,
            "title": title or "(空会话)",
            "turns": sum(1 for m in history if m.get("role") == "user"),
            "updated_at": _fmt_time(ts, now),
            "updated_ts": ts or 0,
        })
        if len(out) >= limit:
            break
    return out


def get_session_messages(session_id: str, limit: int = 40) -> list[dict]:
    """取某个会话的消息记录，用于前端切换会话时回显。"""
    history = _load_history(session_id)
    return [{"role": m.get("role", ""), "content": m.get("content", "")}
            for m in history[-limit:]]


def _count_checkpoints(conn: sqlite3.Connection, session_id: str) -> int:
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (session_id,)
        ).fetchone()[0]
    except Exception:
        return 0


def delete_session(session_id: str) -> int:
    """删除一个会话的全部记忆，返回删掉的 checkpoint 数（0 = 本来就没有）。

    用 checkpointer 自带的 delete_thread（官方接口，会同时清 checkpoints 和 writes）；
    版本不兼容时退回直接删表——两条路都保证不会留下"半截会话"。
    """
    if not os.path.exists(CHECKPOINT_DB):
        return 0
    conn = sqlite3.connect(CHECKPOINT_DB)
    n = _count_checkpoints(conn, session_id)
    conn.close()
    if n == 0:
        return 0

    try:
        get_app().checkpointer.delete_thread(session_id)
    except Exception:
        conn = sqlite3.connect(CHECKPOINT_DB)
        with conn:
            conn.execute("DELETE FROM writes WHERE thread_id=?", (session_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (session_id,))
        conn.close()
    return n


def delete_all_sessions() -> int:
    """清空全部会话记忆，返回删掉的会话数。"""
    if not os.path.exists(CHECKPOINT_DB):
        return 0
    conn = sqlite3.connect(CHECKPOINT_DB)
    n = conn.execute(
        "SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
    with conn:
        conn.execute("DELETE FROM writes")
        conn.execute("DELETE FROM checkpoints")
    conn.close()
    return n
