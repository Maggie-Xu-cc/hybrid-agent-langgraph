"""FastAPI 服务：一个页面 + 一组 REST 接口（聊天/确认/统计/能力/缓存/知识库/会话/记忆等）。"""

from pathlib import Path

import sqlite3
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import cache, embedder, guard, memory, rag, store, tools as toolreg
from .mcp import bridge as mcp_bridge
from .agent import run
from .clients import local_breaker
from .config import settings

# LangGraph 编排（阶段一重构）：装了 langgraph 且未显式关闭就走图编排，
# 否则自动回退到 agent.py 的显式流程，两条链路可用 ENABLE_LANGGRAPH 切换对比。
try:
    from .graph import (delete_all_sessions, delete_session, get_session_messages,
                        list_sessions, pending_of, resume_graph, run_graph)
except ImportError:
    run_graph = None
    resume_graph = None
    pending_of = None
    list_sessions = None
    get_session_messages = None
    delete_session = None
    delete_all_sessions = None

USE_GRAPH = settings.enable_langgraph and run_graph is not None

app = FastAPI(title="Hybrid Routing Agent Demo", version="1.1.0")
WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"


class ChatIn(BaseModel):
    query: str
    force_route: str | None = None
    session_id: str = "default"  # 短期记忆：同 session_id 共享对话上下文
    user_id: str = ""            # 长期记忆：跨会话记住用户事实；空 = 不启用


class ConfirmIn(BaseModel):
    session_id: str
    approve: bool
    note: str = ""               # 拒绝时可以写一句原因，会转告给模型


# session_id → 弹出确认的时刻。用来算"你犹豫了多久"，记账时从延迟里扣掉。
# 为什么放内存而不是落库：确认卡本身存在 checkpointer 里（重启也在），
# 但"什么时候弹的"只影响一条延迟统计，丢了顶多算 0，不值得为它加一张表。
_asked_at: dict[str, float] = {}


@app.on_event("startup")
def _startup() -> None:
    store.init()
    # MCP：连上配置里的外部工具服务，把它们注册进本地工具表。
    # 失败只打警告不阻塞——一个外部服务没起来，不该让整个 Agent 挂掉。
    try:
        st = mcp_bridge.bootstrap()
        if st.get("enabled"):
            print(f"[mcp] {st['count']} 个服务 / {st['tools']} 个外部工具已就绪", flush=True)
    except Exception as exc:   # noqa: BLE001
        print(f"[mcp] 初始化失败（服务继续，只是没有外部工具）：{exc}", flush=True)
    memory.init()
    guard.init()          # 演示额度表（不设额度时仅建表，无实际限制）
    if settings.enable_cache:
        cache.init()
        cache.purge_expired()   # 启动时清一次过期条目，之后命中时也会顺带跳过
    if settings.enable_rag:
        rag.init()
        # 知识库为空时只警告不阻塞：检索层设计成"没有资料就静默退化"，
        # 服务照常可用。但必须打出来，否则用户会以为 RAG 在工作而实际没有。
        st = rag.stats()
        if st["chunks"] == 0:
            print("[warn] 知识库为空，RAG 检索不会生效。"
                  "先跑：python tools/build_kb.py", flush=True)
        else:
            print(f"[rag] 知识库就绪：{st['docs']} 篇 / {st['chunks']} 块", flush=True)


@app.middleware("http")
async def demo_guard_middleware(request: Request, call_next):
    """只拦花钱的 /api/chat：读统计、看会话都免费，没必要限。

    两个开关都为 0 时整段直接跳过——本地开发完全不受影响。
    """
    if (request.url.path in ("/api/chat", "/api/confirm")
            and (settings.demo_quota_per_day > 0 or settings.demo_rate_per_min > 0)):
        ip = request.client.host if request.client else ""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:                      # 反向代理后面拿真实来源
            ip = forwarded.split(",")[0].strip()
        ok, reason = guard.check(ip or "unknown")
        if not ok:
            return JSONResponse(status_code=429, content={"error": reason})
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB)


def _layer_of(result: dict) -> str:
    """这一轮答案实际是哪一层产出的：cache / local / cloud。

    为什么不复用 route（easy/hard）：route 是"打算走哪一层"，
    layer 是"实际走了哪一层"——本地模型挂了降级到云端时，route 仍是 easy，
    但答案明明是云端生成的。页面上要显示后者，否则等于撒谎。
    """
    if result.get("decided_by") == "cache" or result.get("cache_hit"):
        return "cache"
    model = str(result.get("model", ""))
    if settings.local_model and model.startswith(settings.local_model):
        return "local"
    return "cloud"


@app.post("/api/chat")
def api_chat(body: ChatIn) -> dict:
    if USE_GRAPH:
        try:
            result = run_graph(body.query, session_id=body.session_id,
                               force_route=body.force_route, user_id=body.user_id)
        except sqlite3.OperationalError as exc:
            # 高并发下 SQLite 偶发写锁争用（checkpoint 落盘时）。
            # 不要把堆栈甩给前端：告诉用户"重试一次"，并保留诊断信息。
            msg = str(exc)
            if "locked" in msg or "readonly" in msg:
                raise HTTPException(
                    status_code=503,
                    detail="会话存储正忙（SQLite 写锁争用），请重试一次；"
                           "批量压测请用 tools/cache_bench.py --workers",
                ) from exc
            raise
        engine = "langgraph"
    else:
        result = run(body.query, force_route=body.force_route)
        engine = "plain"
    if result.get("status") == "awaiting_confirm":
        # 挂起：这一轮还没有答案，也没有记账（bookkeep 还没跑到）。
        # 记下弹出时刻，用户点确认时用它算等待时长。
        _asked_at[body.session_id] = time.time()
        result["engine"] = engine
        return result
    _asked_at.pop(body.session_id, None)
    if "error" not in result:
        result["engine"] = engine
        result["layer"] = _layer_of(result)
    return result


@app.post("/api/confirm")
def api_confirm(body: ConfirmIn) -> dict:
    """确认卡的回执：同意 / 拒绝，然后从中断的地方继续跑。

    为什么是单独的接口而不是复用 /api/chat：原来的请求早就返回了（返回的是
    "等你确认"），这一轮的状态停在 checkpointer 里。要继续只能带着 resume 语义
    调一次图——这不是一个新问题，加个 query 字段只会让人误会。
    """
    if not USE_GRAPH or resume_graph is None:
        raise HTTPException(status_code=400,
                            detail="人工确认需要 LangGraph 链路（ENABLE_LANGGRAPH=1）")
    asked = _asked_at.pop(body.session_id, None)
    waited_ms = int((time.time() - asked) * 1000) if asked else 0
    try:
        result = resume_graph(body.session_id, approve=body.approve,
                              note=body.note, waited_ms=waited_ms)
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "locked" in msg or "readonly" in msg:
            raise HTTPException(status_code=503,
                                detail="会话存储正忙，请再点一次") from exc
        raise
    if result.get("status") == "awaiting_confirm":
        # 一步里有两批写操作时会再问一次，所以这个分支是正常的，不是错误
        _asked_at[body.session_id] = time.time()
        return result
    if "error" not in result:
        result["engine"] = "langgraph"
        result["layer"] = _layer_of(result)
    return result


@app.get("/api/stats")
def api_stats(session_id: str | None = None, recent_limit: int = 30) -> dict:
    """统计 + 最近请求。

    summary 是全站累计（含别人的请求），页面标注为"全站"；
    recent 传 session_id 就只给这个会话的记录——混在一起显示既看不懂，
    也会让人误以为别人的对话是自己的。
    """
    return {"summary": store.stats(),
            "recent": store.recent(recent_limit, session_id=session_id),
            "cache": cache.stats() if settings.enable_cache else None,
            # 知识库状态：搜了几篇、多少块。为空说明没建索引，检索层在空转
            "kb": ({"enabled": True, **rag.stats()} if settings.enable_rag
                   else {"enabled": False}),
            # 本地熔断状态：本地模型挂了时页面能看到"已短路"，不至于以为卡死
            "local_breaker": local_breaker.state(),
            # 本地模型层是否启用。云端部署时一定是 False——页面据此说明
            # "线上版走纯云端"，避免访客以为自己看到的是混合路由的结果。
            "local_enabled": not settings.disable_local,
            # 本地模型名：页面要据此判断"这一轮是不是本地小模型答的"
            # （历史记录里只有 model 字段，layer 是当场算出来的，得给个判定依据）
            "local_model": settings.local_model,
            # 演示额度：公网版看"还剩多少次"，本地版是 None（不限制）
            "quota": ({"limit": settings.demo_quota_per_day,
                       "used": guard.today_used()}
                      if settings.demo_quota_per_day > 0 else None)}


@app.on_event("shutdown")
def _shutdown() -> None:
    mcp_bridge.shutdown()


@app.get("/api/tools")
def api_tools() -> dict:
    """工具清单：Agent 现在会哪些动作、危险级别、循环上限。

    单独一个接口而不是塞进 /api/stats：这份清单是**静态配置**（代码里注册的），
    和"跑了多少次"不是一类东西；页面用它渲染"能力面板"，不该跟着统计一起变。
    """
    return {"enabled": settings.enable_tools,
            "max_steps": settings.tool_max_steps,
            "budget_cny": settings.tool_budget_cny,
            "result_chars": settings.tool_result_chars,
            "allow_danger": settings.tool_allow_danger,
            "fetch_domains": [d.strip() for d in settings.tool_fetch_domains.split(",") if d.strip()],
            # 这一轮实际放行的级别：公网只放 read，本机可以开到 danger
            "levels": list(settings.allowed_levels()),
            # 人在环路：哪些级别动手前要人点头、这个能力开没开
            "confirm_enabled": bool(settings.tool_graph and settings.enable_tools),
            "confirm_levels": list(settings.confirm_levels()),
            "tools": toolreg.describe(settings.allowed_levels()),
            # 多智能体审查：开没开、审哪些轮次、最多返工几次。
            # 页面要能说清"这次多花的钱买了什么"，所以这几项得能看见。
            "critic": {"enabled": settings.enable_critic,
                       "when": settings.critic_when,
                       "max_rounds": settings.critic_max_rounds,
                       "min_score": settings.critic_min_score},
            # 外部（MCP）工具：连了谁、各自有几个工具、协议版本
            "mcp": mcp_bridge.status()}


@app.get("/api/cache")
def api_cache_stats() -> dict:
    """语义缓存概况：条目数、累计命中、省下的 token 和成本。"""
    return {"enabled": settings.enable_cache,
            "threshold": cache.effective_threshold(),
            "embedder": embedder.name(),
            "guard": settings.cache_guard,
            "ttl_days": settings.cache_ttl_days,
            # 缓存与知识库版本绑定：重建索引后，基于旧文档的条目自动失效
            "kb_version": rag.version(),
            **cache.stats()}


@app.delete("/api/cache")
def api_cache_clear() -> dict:
    """清空语义缓存（不影响记账和会话记忆）。"""
    return {"deleted": cache.clear(), "summary": cache.stats()}


@app.get("/api/kb")
def api_kb_stats() -> dict:
    """知识库概况：多少篇、多少块、最后更新时间。"""
    return {"enabled": settings.enable_rag,
            "top_k": settings.rag_top_k,
            "bm25_min": settings.rag_bm25_min,
            "dirs": settings.kb_dirs,
            **rag.stats()}


@app.post("/api/kb/rebuild")
def api_kb_rebuild() -> dict:
    """重建索引。改了 docs/ 里的文档之后调它，否则检索用的还是旧快照。"""
    if not settings.enable_rag:
        raise HTTPException(400, "RAG 未启用（ENABLE_RAG=0）")
    dirs = [d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
    r = rag.index_dirs(dirs, Path.cwd())
    return {"rebuilt": r, "summary": rag.stats()}


@app.get("/api/sessions")
def api_sessions(limit: int = 20) -> dict:
    """历史会话列表（按最近活动排序）。数据来自 checkpointer，不是另建的索引表。"""
    if not USE_GRAPH or list_sessions is None:
        return {"sessions": [], "note": "会话列表需要 LangGraph 链路（ENABLE_LANGGRAPH=1）"}
    return {"sessions": list_sessions(limit), "engine": "langgraph"}


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str) -> dict:
    """某个会话的消息记录 + 每一轮的决策元信息。

    消息本身来自 checkpointer（原文），元信息来自记账表（走了哪层、花了多少、
    有没有被门卫拦下）。分开存是刻意的：一个记"说了什么"，一个记"花了多少"，
    合并归档反而会互相拖累。这里按顺序对起来返回，页面就能一句话带出处。
    """
    if not USE_GRAPH or get_session_messages is None:
        return {"session_id": session_id, "messages": [], "turns": []}
    # 挂起中的确认请求也要一起返回：刷新页面后确认卡要能重新出现，
    # 否则图还在等人点，页面上却什么都没有——用户只能开新会话，那一轮就白跑了。
    pending = pending_of(session_id) if pending_of else None
    return {"session_id": session_id,
            "messages": get_session_messages(session_id),
            # 记账表按 id 倒序取，再反转成时间正序，和消息顺序一致
            "turns": list(reversed(store.recent(200, session_id=session_id))),
            "pending": pending}


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict:
    """删除单个会话的全部记忆（不可恢复）。"""
    if not USE_GRAPH or delete_session is None:
        return {"deleted": 0, "note": "删除会话需要 LangGraph 链路"}
    return {"deleted": delete_session(session_id), "session_id": session_id}


@app.delete("/api/sessions")
def api_delete_all_sessions() -> dict:
    """清空全部会话记忆（不可恢复）。"""
    if not USE_GRAPH or delete_all_sessions is None:
        return {"deleted": 0, "note": "删除会话需要 LangGraph 链路"}
    return {"deleted": delete_all_sessions()}


@app.delete("/api/memory/{user_id}")
def api_forget_user(user_id: str) -> dict:
    """清空某用户的长期记忆（短期会话不受影响）。"""
    return {"deleted": memory.forget(user_id), "user_id": user_id}


@app.get("/api/memory/{user_id}")
def api_get_memory(user_id: str) -> dict:
    """查看某用户已被记住的事实（调试/演示用）。"""
    return {"user_id": user_id, "facts": memory.recall(user_id)}


@app.post("/api/clear")
def api_clear() -> dict:
    deleted = store.clear()
    return {"deleted": deleted, "summary": store.stats()}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
