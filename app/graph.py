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

图的结构（v4.0 现状，13 个节点）：

    START ─▶ load_memory ─▶ cache_lookup ─命中─▶ bookkeep ─▶ save_memory ─▶ END
                                 │
                                 └未命中─▶ retrieve ─▶ route ─┬─easy──▶ local_answer ─▶ self_check ─需升级─▶ escalate ─┐
                                                              │                            └─通过────────────────────▶ review
                                                              ├─hard──▶ cloud_answer ─────────────────────────────▶ review
                                                              └─tools─▶ tool_agent ──────────────────────────────▶ review
    review ─未过且未到返工上限─▶ revise ─▶ review ─通过/到上限─▶ bookkeep

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
from langgraph.types import Command

from . import cache, context, critic, memory, rag, react, store
from .toolgraph import build_tool_graph
from .dbutil import with_retry
from .clients import ProviderError, call_cloud, call_local
from .config import settings
from .router import Route, decide, needs_tools

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
    session_id: str  # 会话键（记账时带上，页面按会话分开看）
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
    cache_blocked: str  # 门卫拒绝原因（空 = 没被拦）。未命中也要带出去，这是演示重点
    cache_blocked_text: str  # 被拦下的那条缓存原文
    cache_top_score: float  # 全场最高相似度（含被拦的候选）
    use_cache: bool  # 本轮是否启用缓存（评测时关掉，避免污染对照实验）
    rag_chunks: list  # 检索到的相关片段（retrieve 节点，注入 prompt 用）
    rag_hit: bool  # 本轮是否检索到资料
    rag_stats: dict  # 检索统计（条数、耗时、来源、最高分）
    use_rag: bool  # 本轮是否启用检索（评测/对照实验可关）
    use_tools: bool  # 本轮是否走工具循环（route 节点判定的）
    tools_reason: str  # 为什么判定要用工具（页面上要能解释）
    tool_steps: list  # 工具调用轨迹（给页面和记账用）
    tool_calls: int  # 实际执行了几次工具（不含重复调用）
    tool_stop: str  # 循环怎么结束的：answer/max_steps/budget/no_progress/error/cancelled/timeout
    tool_waited_ms: int  # 等用户点"同意"的时间：要从延迟里扣掉，那不是系统慢
    # ---- 多智能体审查（app/critic.py）----
    critic_rounds: int  # 审了几轮（含返工后再审）
    critic_pass: bool  # 最后一轮是否通过
    critic_score: int  # 审查员给的分数（0-100）
    critic_issues: list  # 审查员提的问题（打回的依据）
    critic_hint: str  # 修改方向
    critic_fixes: int  # 返工了几次
    critic_by: str  # 谁判的：rule（数字核对）/llm/error（审查员自己挂了，放行）/skip
    critic_skipped: bool  # 这一轮没审（开关关着，或缓存命中）
    started: float  # 旧的单调钟起点：端到端延迟已改用 started_wall，这个只为兼容旧 checkpoint 留着
    # 墙钟起点：端到端延迟用它算。为什么不能只用上面那个——状态会在 interrupt
    # （等用户确认）处落盘，恢复时可能已经是另一个进程甚至另一台机器，
    # 两边的 perf_counter 基准没有任何关系，相减会得到天文数字（线上出过 60 天）。
    started_wall: float
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
    查找成本 ≈ 一次向量线性扫描（维度随后端：ngram 256 / bge 512），毫秒级，0 token。

    传 kb_version 是为了让"答案依赖文档"这个前提可校验：知识库没变就照样复用，
    重建过索引就自动失效。原来的做法是一律不缓存检索过的答案，结果缓存几乎不工作。
    """
    if not state.get("use_cache", settings.enable_cache):
        # 三个缓存字段一律显式清零。**不能只返回 cache_hit**：
        # 同一个 session_id 的第二轮会拿到上一轮攒下的 state（checkpointer 回放），
        # 上一轮命中过、分数 0.969，这轮没查缓存却照样把 0.969 带出去——
        # 页面就会显示"命中 0.969"而实际走的是云端，是最难查的那种假数据。
        return {"cache_hit": False, "cache_score": 0.0, "cache_top_score": 0.0,
                "cache_blocked": "", "cache_blocked_text": ""}
    look = cache.lookup_ex(state["query"], kb_version=rag.version())
    hit = look["hit"]
    if not hit:
        # 没命中也要把"本来差点命中、被门卫拦下"带出去：
        # 图上要能看见"相似度 0.906 但数字不同，所以拒绝复用"，
        # 否则页面只会显示"走了云端"，看不出系统刚避免了一次错答。
        return {"cache_hit": False, "cache_score": 0.0,
                "cache_blocked": look.get("blocked", ""),
                "cache_blocked_text": look.get("blocked_text", ""),
                "cache_top_score": look.get("score", 0.0)}
    return {
        "cache_hit": True,
        "cache_score": hit["score"],
        "cache_cost": hit.get("cost", 0.0),
        # 命中时不报"被拦下的别条候选"：页面要回答的是"这一轮为什么这样走"，
        # 混进另一条被拒的候选只会让人看不懂。
        "cache_blocked": "",
        "cache_blocked_text": "",
        "cache_top_score": hit["score"],
        "started": time.perf_counter(),  # 命中路径不经过 route，这里补上计时起点
        "started_wall": time.time(),
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
    # 要动手的请求不预先检索：工具里有 search_docs，它按需检索。
    # 两边都注入等于同一批文档进两次上下文——实测多花约 1000 token，
    # 而且模型会分不清"这两段资料哪个是这一轮的"，答案反而更容易串。
    if needs_tools(state.get("query") or "")[0] and settings.enable_tools:
        return {"rag_chunks": [], "rag_hit": False,
                "rag_stats": {"n": 0, "enabled": True, "skipped": "工具路径自带检索"}}
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
    started_wall = time.time()
    query = state["query"].strip()

    route: Route = decide(query)
    force = state.get("force_route")
    if force in ("easy", "hard"):
        route = Route(force, f"强制{force}", route.by)

    messages, ctx_stats = _build_messages({**state, "query": query}, query)

    want_tools, why = needs_tools(query)
    want_tools = want_tools and settings.enable_tools

    return {
        "query": query,
        "started": started,
        "started_wall": started_wall,
        "route": {
            "difficulty": route.difficulty,
            "reason": route.reason,
            "by": route.by,
            "latency_ms": route.latency_ms,
            "tokens_in": route.tokens_in,
            "tokens_out": route.tokens_out,
        },
        # 路由和"要不要动手"是两件事：一个决定用多贵的模型，一个决定要不要调工具。
        # 合成一个字段会互相污染（"难"不代表要查库，"简单"也可能要算一下）。
        "use_tools": want_tools,
        "tools_reason": why,
        "tool_steps": [],
        "tool_calls": 0,
        "tool_stop": "",
        "messages": messages,
        "ctx_stats": ctx_stats,
        "local_in": route.tokens_in,
        "local_out": route.tokens_out,
        "cloud_in": 0,
        "cloud_out": 0,
        "escalated": False,
        "degraded": False,
    }


def tool_agent_node(state: AgentState) -> dict:
    """工具循环节点：ReAct（想→动手→看结果→再想）。

    为什么跳过后面的 self_check/escalate：
    那两步是为"模型没把握"设计的——让它换更强的模型重答一次。而这里的答案
    建立在**工具实际返回的事实**上，重答一次只会把刚查到的东西丢掉。
    工具链路的质量问题应该在循环内部解决（收口那次调用），不是靠外部重试。
    """
    messages = state["messages"]
    res = react.run(messages)

    cloud_in = res.tokens_in
    cloud_out = res.tokens_out
    update: AgentState = {
        "tool_steps": res.brief_steps(),
        "tool_calls": sum(1 for s in res.steps if not s.repeat and s.name != "(模型)"),
        "tool_stop": res.stop,
    }

    if not res.answer:
        # 工具链路失败（模型挂了/没有 openai 包）：退回普通云端作答。
        # 用户要的是答案，不是"工具链路出错了"——降级要自己吃掉。
        try:
            answer, tin, tout = call_cloud(messages)
            update.update(answer=answer, model_used=settings.cloud_model,
                          cloud_in=cloud_in + tin, cloud_out=cloud_out + tout,
                          degraded=True, tool_stop=res.stop or "error")
            return update
        except ProviderError:
            return {**update, "answer": "工具调用没能完成，模型也不可用。请稍后再试。",
                    "model_used": "tool-error", "cloud_in": cloud_in,
                    "cloud_out": cloud_out, "degraded": True,
                    "tool_stop": res.stop or "error"}

    update.update(answer=res.answer, model_used=f"{settings.cloud_model} +tools",
                  cloud_in=cloud_in, cloud_out=cloud_out, degraded=False)
    return update


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


def _confirm_stats(steps: list) -> dict:
    """从轨迹里数出人在环路的账：问了几次、批了几次、拒了几次。

    为什么要单独统计：一趟跑下来"调了 2 次工具"看不出有没有人把关。
    页面上真正有说服力的是"其中 1 次是你点的同意"，以及"有 1 次被你拒了"。
    """
    asked = [s for s in steps if s.get("confirm") in ("approved", "rejected")]
    return {
        "asked": len(asked),
        "approved": sum(1 for s in asked if s.get("confirm") == "approved"),
        "rejected": sum(1 for s in asked if s.get("confirm") == "rejected"),
    }


def _evidence_of(state: AgentState) -> tuple[str, bool]:
    """把这一轮"手上有的证据"摊成文本，交给审查员比对。

    证据只有三种来源：检索到的资料、工具的实际返回、长期记忆里的事实。
    返回 (文本, 是否被截断)——工具结果超长会被截成头尾，这个事实必须往上传：
    否则审查员会把"被截掉的数字"判成"编的数字"（实测误伤过一次）。
    """
    parts: list[str] = []
    truncated = False
    for c in state.get("rag_chunks") or []:
        parts.append(f"[资料 {c.get('doc', '')} § {c.get('heading', '')}] {c.get('text', '')}")
    for s in state.get("tool_steps") or []:
        name = str(s.get("name") or "")
        if name.startswith("("):          # 收口步骤不是证据
            continue
        # 优先用真实输出：summary 只是"3 行输出"这类摘要，拿它当证据等于没证据
        body = s.get("out") or s.get("summary", "")
        is_cut = bool(s.get("truncated") or s.get("out_cut"))
        cut = "（此结果被截断，只保留头尾）" if is_cut else ""
        truncated = truncated or is_cut
        parts.append(f"[工具 {name} 参数 {s.get('args', '')}]{cut} {body}")
    facts = state.get("facts") or {}
    if facts:
        parts.append("[记忆] " + "；".join(f"{k}={v}" for k, v in facts.items()))
    return "\n".join(parts), truncated


def review_node(state: AgentState) -> dict:
    """审查员：判断这一轮的答案能不能放行，不能就打回返工。

    为什么单独一个节点：它是"第二个人"，要有自己的输入输出、自己能停能走，
    才能在图上被看见（页面上能指着说"这一步是审查员，它发现了两条问题"）。
    藏在一个函数内部，就只是一次额外的模型调用，讲不出"多智能体"。
    """
    rounds = int(state.get("critic_rounds") or 0)
    used_tools = bool(state.get("tool_steps"))
    if not settings.should_review(used_tools, bool(state.get("cache_hit"))):
        # 关着开关、或缓存命中：原样放行。节点照样留在图里，
        # 这样"审/不审"只差一个配置，而不是两张图——图一换，前面所有埋点都要重测。
        return {"critic_skipped": True, "critic_pass": True, "critic_by": "skip"}

    evidence, cut = _evidence_of(state)
    rev = critic.review(state.get("query") or "", state.get("answer") or "",
                        evidence, min_score=settings.critic_min_score, truncated=cut)
    return {
        "critic_rounds": rounds + 1,
        "critic_pass": rev.passed,
        "critic_score": rev.score,
        "critic_issues": rev.issues,
        "critic_hint": rev.hint,
        "critic_by": rev.by,
        "critic_skipped": False,
        # 审查本身也花钱：token 记进云端账，总成本才是真的。
        # 把审查成本藏起来，A/B 对比就成了"白送一轮质检"，结论会偏。
        "cloud_in": state.get("cloud_in", 0) + rev.tin,
        "cloud_out": state.get("cloud_out", 0) + rev.tout,
    }


def after_review(state: AgentState) -> str:
    if state.get("critic_skipped") or state.get("critic_pass"):
        return "bookkeep"
    # 返工次数用完就收工：每打回一次都是真金白银，而且第二轮基本只在措辞上打转
    if int(state.get("critic_fixes") or 0) >= settings.critic_max_rounds:
        return "bookkeep"
    return "revise"


def revise_node(state: AgentState) -> dict:
    """返工：带着审查意见和证据重写答案。

    失败了不抛错，保留原答案——审查意见已经记在 state 里，页面照样能看到
    "它被打了回票但没改成功"。这比整轮失败好：用户至少拿到了原来那份答案。
    """
    rev = critic.Review(passed=False, score=int(state.get("critic_score") or 0),
                        issues=list(state.get("critic_issues") or []),
                        hint=state.get("critic_hint") or "")
    msgs = critic.build_revise_messages(state.get("query") or "",
                                        state.get("answer") or "", rev,
                                        _evidence_of(state)[0])
    fixes = int(state.get("critic_fixes") or 0) + 1
    try:
        answer, tin, tout = call_cloud(msgs)
    except ProviderError:
        return {"critic_fixes": fixes}
    return {
        "answer": answer,
        "model_used": f"{state.get('model_used', '')} -> 返工",
        "cloud_in": state.get("cloud_in", 0) + tin,
        "cloud_out": state.get("cloud_out", 0) + tout,
        "critic_fixes": fixes,
    }


def _elapsed_ms(state: AgentState, waited_ms: int | None = None) -> int:
    """端到端耗时（毫秒）：墙钟差 − 等人确认的时间，越界归零。

    为什么不用 perf_counter：这张图的中间状态会在 interrupt（等用户点确认）处
    落盘，恢复时可能已经是另一个进程、甚至平台换了一台机器重建的容器。
    两个进程的单调钟基准毫无关系，相减能得出"这条请求跑了 60 天"——
    线上真的记出过 51 亿毫秒，一条就把平均延迟拉成了天文数字。
    墙钟（time.time）跨进程、跨机器都对得上，代价是受 NTP 校时影响（毫秒级，记账够用）。

    为什么不直接 max(0, ...)：等待时间扣完可能变负（起点在未来 / 时钟被往回拨），
    这时说明数据本身不可信，记 0 而不是记一个"看起来很小"的假数。

    等人点"同意"的那几十秒不是系统慢：不扣掉的话，页面上会出现"这一轮跑了 42 秒"，
    而真实原因是用户自己在犹豫——这种数字会误导所有性能结论。
    """
    started_wall = float(state.get("started_wall") or 0)
    if not started_wall:
        # 旧 checkpoint 里没有这个字段（升级前挂起的会话）：不猜，记 0。
        return 0
    # 不传就自己从状态里取。默认 0 的写法看着方便，实际是个坑：
    # 调用方漏传一次，等待时间就白扣了，而账面上完全看不出来。
    if waited_ms is None:
        waited_ms = int(state.get("tool_waited_ms") or 0)
    elapsed = int((time.time() - started_wall) * 1000) - int(waited_ms or 0)
    if elapsed > store.MAX_PLAUSIBLE_LATENCY_MS:
        print(f"[bookkeep] 延迟异常已归零：elapsed={elapsed}ms waited={waited_ms}ms "
              f"started_wall={started_wall:.0f}", flush=True)
        return 0
    return max(0, elapsed)


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
        # 调过工具的答案不写缓存：它的正确性依赖"当时那一次"外部状态
        # （数据库有几行、网页写了什么），这些变化没有版本号可比。
        # kb_version 能管"文档改过"，管不了"数据库多了一条记录"。
        and not state.get("tool_steps")
        # 没过审查的答案不写缓存：缓存是"这份答案值得复用"的承诺，
        # 审查员刚说它有问题，还存起来就是把这个错误复制到后面所有相似问题上。
        # 注意"审了但没改成功"（有 issues、critic_pass=False）也不能存。
        and (state.get("critic_skipped") or state.get("critic_pass"))
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
    waited_ms = int(state.get("tool_waited_ms") or 0)
    latency_ms = _elapsed_ms(state, waited_ms)

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
        # 带上 session_id：页面要按会话分开显示"这个对话里发生过什么"，
        # 混在一张全局列表里既看不懂也分不清是谁的。
        session_id=state.get("session_id", ""),
        guard_blocked=state.get("cache_blocked") or "",
        # 本轮看到过的最高相似度：命中时是被复用的那条，被拦时是差点被复用的那条。
        # 页面靠它讲清楚"为什么这样走"，所以必须落库，否则历史记录只剩一个结论。
        cache_score=state.get("cache_top_score") or 0,
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
        # 工具轨迹落库：页面刷新后要能重放"这一轮调了什么、结果是什么"
        tool_steps=state.get("tool_steps") or [],
        # 审查记录落库：不打进库就只剩"这一轮花了多少钱"，
        # 没法回答"多花的钱买到了什么"——而这正是要不要留审查员的唯一依据。
        critic_rounds=int(state.get("critic_rounds") or 0),
        critic_pass=bool(state.get("critic_pass", True)),
        critic_score=int(state.get("critic_score") or 0),
        critic_fixes=int(state.get("critic_fixes") or 0),
        critic_issues="；".join(state.get("critic_issues") or [])[:400],
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
        # 门卫拦截：这一轮差点抄到一条"字面很像但参数不同"的答案，被字符串规则拒绝了。
        # 它不是失败，是拦住了一次错答——页面上要单独标出来，不能混在"走了云端"里。
        "cache_guard": {
            "blocked": bool(state.get("cache_blocked")),
            "reason": state.get("cache_blocked") or "",
            "matched": state.get("cache_blocked_text") or "",
            "score": round(state.get("cache_top_score") or 0, 3),
        },
        # 上下文工程的可观测性：注入了哪些记忆、裁了多少历史
        "memory": {
            "enabled": bool(state.get("user_id")),
            "facts_injected": list(facts.keys()),
            "summary_used": bool(state.get("summary")),
            "summarized_now": summarized,
            "ctx_tokens": state.get("ctx_stats") or {},
        },
        # 工具调用的可观测性：这一轮动了几次手、每步的参数/结果/耗时、为什么停
        "tools": {
            "enabled": settings.enable_tools,
            "used": bool(state.get("tool_steps")),
            "reason": state.get("tools_reason") or "",
            "calls": state.get("tool_calls") or 0,
            "stop": state.get("tool_stop") or "",
            "steps": state.get("tool_steps") or [],
            # 人在环路的痕迹：问了几次、批了几次、拒了几次、你犹豫了多久。
            # 页面要能说清"这一步是人放的，不是它自作主张"。
            "confirm": _confirm_stats(state.get("tool_steps") or []),
            "waited_ms": waited_ms,
        },
        # 多智能体的可观测性：审了几轮、谁判的、发现什么问题、改了几次
        "critic": {
            "enabled": settings.enable_critic,
            "skipped": bool(state.get("critic_skipped", True)),
            "rounds": int(state.get("critic_rounds") or 0),
            "pass": bool(state.get("critic_pass", True)),
            "score": int(state.get("critic_score") or 0),
            "fixes": int(state.get("critic_fixes") or 0),
            "issues": list(state.get("critic_issues") or []),
            "by": state.get("critic_by") or "",
            "when": settings.critic_when,
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
    if state.get("use_tools"):
        return "tools"
    return state["route"]["difficulty"]


def after_local(state: AgentState) -> str:
    return "done" if state.get("degraded") else "check"


def after_check(state: AgentState) -> str:
    return "escalate" if state.get("needs_escalation") else "done"


# ---------------- 组装 ----------------

def tool_node():
    """工具节点：默认走 think/act 子图（写操作要人点头），关掉开关则退回纯函数版。

    为什么两条都留着：这是一个能写进报告的对照实验——同一个问题，
    有确认 vs 没确认，成本/延迟差多少、被拒绝时回答质量掉多少。
    只有基线在，这些数字才有意义。
    """
    return build_tool_graph() if settings.tool_graph else tool_agent_node


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("cache_lookup", cache_lookup_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("route", route_node)
    g.add_node("local_answer", local_answer_node)
    g.add_node("cloud_answer", cloud_answer_node)
    g.add_node("tool_agent", tool_node())
    g.add_node("self_check", self_check_node)
    g.add_node("escalate", escalate_node)
    g.add_node("review", review_node)
    g.add_node("revise", revise_node)
    g.add_node("bookkeep", bookkeep_node)
    g.add_node("save_memory", save_memory_node)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "cache_lookup")
    g.add_conditional_edges("cache_lookup", after_cache, {
        "hit": "bookkeep", "miss": "retrieve",
    })
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", after_route, {
        "easy": "local_answer", "hard": "cloud_answer", "tools": "tool_agent",
    })
    # 所有"出答案"的口子统一收进 review：审查员是最后一道闸，
    # 它可能直接放行（`critic_skipped`/通过），也可能打回 revise 重写一轮。
    # 缓存命中不经过它——见 settings.should_review 里的理由（审一下就白省了）。
    g.add_edge("tool_agent", "review")
    g.add_conditional_edges("local_answer", after_local, {
        "check": "self_check", "done": "review",
    })
    g.add_conditional_edges("self_check", after_check, {
        "escalate": "escalate", "done": "review",
    })
    g.add_edge("escalate", "review")
    g.add_edge("cloud_answer", "review")
    g.add_conditional_edges("review", after_review, {
        "bookkeep": "bookkeep", "revise": "revise",
    })
    # 返工完再审一次：不能因为"改过了"就默认它改对了
    g.add_edge("revise", "review")
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


def _pending_of(app, config: dict) -> dict | None:
    """这一会话有没有"卡在等人确认"的操作。有就返回确认单，没有返回 None。

    为什么读 get_state 而不是自己维护一张表：中断状态就存在 checkpointer 里，
    自己再记一份就是两个真相源，重启进程后必然对不上。LangGraph 已经把
    待处理的 interrupt 挂在 state 上，直接问它就是最准的。
    """
    try:
        st = app.get_state(config)
    except Exception:          # noqa: BLE001 — 空会话/旧版本可能抛，一律当作"没有待确认"
        return None
    ints = getattr(st, "interrupts", None) or ()
    if not ints:
        return None
    first = ints[0]
    return getattr(first, "value", None) or {"kind": "tool_confirm", "question": "有一步在等你确认"}


def _shape(out: dict, session_id: str) -> dict:
    """把图的输出整成接口返回：要么是答案，要么是"等你确认"。

    两种结果结构不同，前端必须能分辨——把确认单当成答案渲染，用户会看到一段 JSON。
    """
    ints = out.get("__interrupt__") or []
    if ints:
        first = ints[0]
        pending = getattr(first, "value", None) or {}
        return {"status": "awaiting_confirm", "session_id": session_id,
                "pending": pending}
    result = out.get("result") or {}
    return {"status": "ok", **result}


def run_graph(query: str, session_id: str = "default",
              force_route: str | None = None,
              user_id: str = "",
              use_cache: bool | None = None,
              use_rag: bool | None = None,
              checkpoint_db: str | None = None) -> dict:
    """对外入口：与 agent.run() 返回字段一致，外加一个 status（ok / awaiting_confirm）。

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

    # 同一会话里还挂着一步没确认，而用户直接问了新问题：
    # 先把那一步按"取消"收尾（不花钱：act 直接返回，不再调模型），再跑新问题。
    # 不这么做会有两种坏结果——新问题被挂起的运行挡住，或者用旧的待办答新问题。
    if _pending_of(app, config):
        app.invoke(Command(resume={"approve": False, "cancel": True,
                                   "note": "用户没有确认，直接问了新的问题"}), config)

    out = app.invoke(
        {"query": query, "force_route": force_route, "user_id": user_id,
         "session_id": session_id,
         "use_cache": settings.enable_cache if use_cache is None else use_cache,
         "use_rag": settings.enable_rag if use_rag is None else use_rag},
        config,
    )
    return _shape(out, session_id)


def pending_of(session_id: str, checkpoint_db: str | None = None) -> dict | None:
    """这个会话有没有卡在等人确认的操作。

    页面刷新之后要用它把确认卡重新画出来——否则用户一刷就卡死在这里：
    图还在等他点，而页面上什么都没了，他也想不起自己在等什么。
    """
    app = get_app(checkpoint_db)
    return _pending_of(app, {"configurable": {"thread_id": session_id}})


def resume_graph(session_id: str, approve: bool, note: str = "",
                 waited_ms: int = 0, checkpoint_db: str | None = None) -> dict:
    """用户在确认卡上点了同意/拒绝：从中断处继续跑。

    waited_ms 由调用方（服务端）算好传进来——只有它知道"弹出确认"到"用户点击"
    之间隔了多久。act 节点把它累加进状态，记账时从延迟里扣掉。
    """
    app = get_app(checkpoint_db)
    config = {"configurable": {"thread_id": session_id}}
    if not _pending_of(app, config):
        return {"error": "这个会话没有待确认的操作（可能已经处理过，或者被新问题取消了）"}
    out = app.invoke(Command(resume={"approve": approve, "note": note,
                                     "waited_ms": int(waited_ms)}), config)
    return _shape(out, session_id)


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
