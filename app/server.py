"""FastAPI 服务：一个页面 + 两个接口。"""

from pathlib import Path

import sqlite3

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import cache, guard, memory, store
from .agent import run
from .clients import local_breaker
from .config import settings

# LangGraph 编排（阶段一重构）：装了 langgraph 且未显式关闭就走图编排，
# 否则自动回退到 agent.py 的显式流程，两条链路可用 ENABLE_LANGGRAPH 切换对比。
try:
    from .graph import (delete_all_sessions, delete_session, get_session_messages,
                        list_sessions, run_graph)
except ImportError:
    run_graph = None
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


@app.on_event("startup")
def _startup() -> None:
    store.init()
    memory.init()
    guard.init()          # 演示额度表（不设额度时仅建表，无实际限制）
    if settings.enable_cache:
        cache.init()
        cache.purge_expired()   # 启动时清一次过期条目，之后命中时也会顺带跳过


@app.middleware("http")
async def demo_guard_middleware(request: Request, call_next):
    """只拦花钱的 /api/chat：读统计、看会话都免费，没必要限。

    两个开关都为 0 时整段直接跳过——本地开发完全不受影响。
    """
    if (request.url.path == "/api/chat"
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
    if "error" not in result:
        result["engine"] = engine
        result["layer"] = _layer_of(result)
    return result


@app.get("/api/stats")
def api_stats() -> dict:
    return {"summary": store.stats(), "recent": store.recent(8),
            "cache": cache.stats() if settings.enable_cache else None,
            # 本地熔断状态：本地模型挂了时页面能看到"已短路"，不至于以为卡死
            "local_breaker": local_breaker.state(),
            # 本地模型层是否启用。云端部署时一定是 False——页面据此说明
            # "线上版走纯云端"，避免访客以为自己看到的是混合路由的结果。
            "local_enabled": not settings.disable_local,
            # 演示额度：公网版看"还剩多少次"，本地版是 None（不限制）
            "quota": ({"limit": settings.demo_quota_per_day,
                       "used": guard.today_used()}
                      if settings.demo_quota_per_day > 0 else None)}


@app.get("/api/cache")
def api_cache_stats() -> dict:
    """语义缓存概况：条目数、累计命中、省下的 token 和成本。"""
    return {"enabled": settings.enable_cache,
            "threshold": settings.cache_threshold,
            "ttl_days": settings.cache_ttl_days,
            **cache.stats()}


@app.delete("/api/cache")
def api_cache_clear() -> dict:
    """清空语义缓存（不影响记账和会话记忆）。"""
    return {"deleted": cache.clear(), "summary": cache.stats()}


@app.get("/api/sessions")
def api_sessions(limit: int = 20) -> dict:
    """历史会话列表（按最近活动排序）。数据来自 checkpointer，不是另建的索引表。"""
    if not USE_GRAPH or list_sessions is None:
        return {"sessions": [], "note": "会话列表需要 LangGraph 链路（ENABLE_LANGGRAPH=1）"}
    return {"sessions": list_sessions(limit), "engine": "langgraph"}


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str) -> dict:
    """某个会话的消息记录，用于切换会话时回显。"""
    if not USE_GRAPH or get_session_messages is None:
        return {"session_id": session_id, "messages": []}
    return {"session_id": session_id,
            "messages": get_session_messages(session_id)}


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
