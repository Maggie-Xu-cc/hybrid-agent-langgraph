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
- history 带滑动窗口裁剪（上下文工程的第一步，阶段二会加摘要压缩）。
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import store
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

# 会话历史滑动窗口：发给模型的最近 N 条 + 持久化的最多 M 条
CONTEXT_WINDOW_MESSAGES = 8
MAX_HISTORY_MESSAGES = 20

# checkpointer 的落盘位置。会话列表也读这个库，所以统一成一个常量，避免两处路径漂移。
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.db")


class AgentState(TypedDict, total=False):
    """图的全局状态。节点只返回自己更新的字段（partial update）。"""

    query: str
    force_route: str | None
    history: list[dict]  # 多轮历史（checkpointer 持久化）
    messages: list[dict]  # 本轮实际发给模型的消息
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
    started: float
    result: dict


def _needs_escalation(text: str) -> bool:
    if len(text.strip()) < 12:
        return True
    return any(marker in text for marker in ESCALATE_MARKERS)


def _build_messages(state: AgentState) -> list[dict]:
    """system + 滑动窗口内的历史 + 本轮提问。"""
    history = state.get("history") or []
    trimmed = history[-CONTEXT_WINDOW_MESSAGES:]
    return [
        {"role": "system", "content": SYSTEM},
        *trimmed,
        {"role": "user", "content": state["query"]},
    ]


# ---------------- 节点 ----------------

def route_node(state: AgentState) -> dict:
    """路由节点：原来的 agent.run() 第一步。路由 token 记在本地账上。"""
    started = time.perf_counter()
    query = state["query"].strip()

    route: Route = decide(query)
    force = state.get("force_route")
    if force in ("easy", "hard"):
        route = Route(force, f"强制{force}", route.by)

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
        "messages": _build_messages({**state, "query": query}),
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


def bookkeep_node(state: AgentState) -> dict:
    """记账节点：成本计算 + 落库 + 把本轮对话追加进 history。"""
    route = state["route"]
    local_in = state.get("local_in", 0)
    local_out = state.get("local_out", 0)
    cloud_in = state.get("cloud_in", 0)
    cloud_out = state.get("cloud_out", 0)
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

    history = list(state.get("history") or [])
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": state.get("answer", "")})

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
    }
    return {"history": history[-MAX_HISTORY_MESSAGES:], "result": result}


# ---------------- 条件边 ----------------

def after_route(state: AgentState) -> str:
    return state["route"]["difficulty"]


def after_local(state: AgentState) -> str:
    return "done" if state.get("degraded") else "check"


def after_check(state: AgentState) -> str:
    return "escalate" if state.get("needs_escalation") else "done"


# ---------------- 组装 ----------------

def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("local_answer", local_answer_node)
    g.add_node("cloud_answer", cloud_answer_node)
    g.add_node("self_check", self_check_node)
    g.add_node("escalate", escalate_node)
    g.add_node("bookkeep", bookkeep_node)

    g.add_edge(START, "route")
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
    g.add_edge("bookkeep", END)
    return g.compile(checkpointer=checkpointer)


def _make_checkpointer():
    """优先 SQLite 持久化（重启不丢会话），装不上退回内存版。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        return MemorySaver()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph(_make_checkpointer())
    return _app


def run_graph(query: str, session_id: str = "default",
              force_route: str | None = None) -> dict:
    """对外入口：与 agent.run() 返回字段完全一致（多轮时需带 session_id）。"""
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}

    app = get_app()
    config = {"configurable": {"thread_id": session_id}}
    out = app.invoke({"query": query, "force_route": force_route}, config)
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
