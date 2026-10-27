"""FastAPI 服务：一个页面 + 两个接口。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import memory, store
from .agent import run
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB)


@app.post("/api/chat")
def api_chat(body: ChatIn) -> dict:
    if USE_GRAPH:
        result = run_graph(body.query, session_id=body.session_id,
                           force_route=body.force_route, user_id=body.user_id)
        engine = "langgraph"
    else:
        result = run(body.query, force_route=body.force_route)
        engine = "plain"
    if "error" not in result:
        result["engine"] = engine
    return result


@app.get("/api/stats")
def api_stats() -> dict:
    return {"summary": store.stats(), "recent": store.recent(8)}


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
