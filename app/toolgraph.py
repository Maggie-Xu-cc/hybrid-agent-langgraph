"""人在环路（HITL）：把 ReAct 循环拆成 think / act 两个节点的子图。

为什么必须拆成两个节点
----------------------
`interrupt()` 只能在**节点边界**暂停。react.py 里那个 `for step in range(...)` 是一个
函数闷头跑到底，中途没有任何地方能停下来等人——这正是 v2.0 文档里写的
"到 P3 必须拆成节点"的硬约束。

拆完之后，图长这样：

    START ─▶ think ──有工具调用──▶ act ──继续──▶ think
               │                    │
               │有答案               │到上限/打转
               ▼                    ▼
             finish ◀──────────── finalize
               │
               ▼
              END

`act` 里在真正动手**之前**调用 `interrupt()`，图就停在那一刻，状态落进 checkpointer；
用户点"同意"后从断点继续，工具才真的执行。

三条硬约束（都是踩出来的，不是看文档看来的）
------------------------------------------
1. **恢复时节点会从头重跑**。LangGraph 恢复中断的节点不是"从 interrupt 那行接着跑"，
   而是**重新执行整个节点函数**，`interrupt()` 第二次调用时直接返回 resume 值。
   所以：模型调用绝对不能和 interrupt 放在同一个节点里——重跑一次就是再花一次钱。
   这也是 think / act 必须分开的真正原因，不只是"看起来清楚"。
2. **确认必须先于任何副作用**。act 里先算出"哪些要问"，一次性问完，再去执行。
   如果边执行边问，恢复时前半截副作用会做两次（查一次库、写一条便签，都变成两次）。
3. **一个节点只问一次**。多个 interrupt 要靠 resume 值按顺序配对，一错位就串了。
   所以**一步里的所有写操作合成一张确认单**，一次问清、一次批完。

为什么还要保留 react.py
----------------------
`ENABLE_TOOL_GRAPH=0` 时走纯函数版。这不是怀旧：它给了你一个干净的对照——
同一个问题，加人工确认之后成本/延迟差多少、被拒绝时回答质量掉多少。
没有基线，"上了 HITL 更安全"只是一句口号。
"""

from __future__ import annotations

import json
import time
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import react, tools as toolreg
from .clients import ProviderError, call_cloud
from .config import settings

# 确认单里给用户看的一句话解释。级别是代码里的词，用户看不懂"write 级"是什么意思。
LEVEL_WHY = {
    "write": "会改动数据（写进去就回不去了）",
    "danger": "可能执行任意代码，风险最高",
    "read": "只是查询，不改任何东西",
}


class ToolState(TypedDict, total=False):
    """子图自己的状态。

    只放"循环内部要记住的东西"：对话上下文、待执行调用、步数、重复计数、花了多少。
    跨轮该留下的（history/summary/result）全在父图，这里一律不碰。
    """

    query: str                # 本轮问题原文（确认单要带上：刷新后卡上得知道问的是什么）
    messages: list[dict]      # 父图传进来的这一轮 prompt
    tool_ctx: list[dict]      # 循环内的完整对话（含 assistant 的 tool_calls 与 tool 结果）
    tool_pending: list[dict]  # 本步模型要求执行的调用（含解析好的参数）
    tool_step: int
    tool_seen: dict           # "工具名|参数" → 第几步调过（查重复）
    tool_repeats: int
    tool_started: float
    tool_steps: list          # 给页面/记账的轨迹
    tool_calls: int
    tool_stop: str
    tool_in: int
    tool_out: int
    tool_cost: float
    tool_waited_ms: int       # 等人点确认的时间（要从延迟里扣掉，否则延迟里混了人的反应时间）
    answer: str
    model_used: str
    cloud_in: int
    cloud_out: int
    degraded: bool


def _confirm_payload(step: int, calls: list[dict], query: str = "") -> dict:
    """给前端渲染确认卡的载荷。**会被序列化进 checkpoint**，所以只能是纯数据。"""
    names = "、".join(c["name"] for c in calls)
    return {
        "kind": "tool_confirm",
        "step": step,
        # 问题原文：这一轮还没跑完，历史里没有这条记录。刷新页面后如果只画一张卡，
        # 用户会看到"要你确认 notes_add"却不知道自己在问什么。
        "query": query,
        "question": f"模型想执行 {names}，要允许吗？",
        "calls": [{"name": c["name"], "args": c.get("args") or {},
                   "level": c.get("level", ""),
                   "why": LEVEL_WHY.get(c.get("level", ""), "")} for c in calls],
        # 拒绝不是错误：说清楚拒绝之后会发生什么，用户才敢点
        "reject_hint": "拒绝不会中断对话，模型会换一种做法，或者直接告诉你它缺这个权限。",
    }


def build_tool_graph(model_fn: react.ModelFn | None = None,
                     max_steps: int | None = None,
                     budget_cny: float | None = None,
                     levels: tuple[str, ...] | None = None,
                     confirm_levels: tuple[str, ...] | None = None):
    """编译 think/act 子图（不带 checkpointer——中断由父图负责，这是硬要求）。"""
    max_steps = max_steps or settings.tool_max_steps
    budget = settings.tool_budget_cny if budget_cny is None else budget_cny
    levels = settings.allowed_levels() if levels is None else levels
    confirm_levels = settings.confirm_levels() if confirm_levels is None else confirm_levels
    model_fn = model_fn or react.default_model

    specs = toolreg.openai_tools(levels)

    # ------------------------------------------------------------------ think

    def think(state: ToolState) -> dict:
        """问模型：要调工具，还是能直接答。

        这个节点**绝不能 interrupt**：恢复时它会被重跑，模型调用就白花一次钱。
        """
        convo = state.get("tool_ctx")
        if not convo:
            convo = [dict(m) for m in state.get("messages") or []]
            idx = 1 if convo and convo[0].get("role") == "system" else 0
            convo.insert(idx, {"role": "system",
                               "content": react.TOOL_SYSTEM.format(max_steps=max_steps)})

        step = (state.get("tool_step") or 0) + 1
        tin = state.get("tool_in") or 0
        tout = state.get("tool_out") or 0
        cost = state.get("tool_cost") or 0.0
        started = state.get("tool_started") or time.perf_counter()

        # 三道闸在上限面前一律"收口而不是硬跑"：再给 3 步只会烧钱，用户还是拿不到答案
        over = ("max_steps" if step > max_steps
                else "budget" if cost > budget
                else "timeout" if time.perf_counter() - started > settings.tool_timeout * max_steps
                else "")
        if over:
            answer, ai, ao, err = react.finalize(convo, model_fn)
            return {"tool_ctx": convo, "tool_stop": over, "answer": answer,
                    "tool_in": tin + ai, "tool_out": tout + ao,
                    "tool_cost": cost + react.cost_of(ai, ao),
                    "tool_steps": list(state.get("tool_steps") or []) + (
                        [react.Step(0, "(收口)", {}, not err, "到上限，收口",
                                    0, error=err).brief()])}

        try:
            content, calls, ai, ao = model_fn(convo, specs)
        except ProviderError as exc:
            return {"tool_ctx": convo, "tool_stop": "error", "answer": "",
                    "tool_steps": list(state.get("tool_steps") or []) + [
                        react.Step(step, "(模型)", {}, False, "模型调用失败", 0,
                                   error=str(exc)).brief()]}

        tin += ai
        tout += ao
        cost += react.cost_of(ai, ao)

        if not calls:
            # 模型不再要工具 = 自然结束。这是最常见也最便宜的收尾。
            return {"tool_ctx": convo, "tool_step": step, "tool_stop": "answer",
                    "answer": content or "", "tool_in": tin, "tool_out": tout,
                    "tool_cost": cost}

        convo = convo + [{"role": "assistant", "content": content or None,
                          "tool_calls": [
                              {"id": c["id"], "type": "function",
                               "function": {"name": c["name"], "arguments": c["args_raw"]}}
                              for c in calls]}]

        pending = []
        for c in calls:
            args, err = react.parse_args(c["args_raw"])
            spec = toolreg.get(c["name"])
            pending.append({"id": c["id"], "name": c["name"], "args": args or {},
                            "args_raw": c["args_raw"], "err": err,
                            "level": spec.level if spec else ""})
        return {"tool_ctx": convo, "tool_step": step, "tool_pending": pending,
                "tool_in": tin, "tool_out": tout, "tool_cost": cost}

    # -------------------------------------------------------------------- act

    def act(state: ToolState) -> dict:
        """执行本步的调用；写/危险级先问人。

        恢复时这个节点会从头重跑，所以：**interrupt 之前不许有任何副作用**。
        这里的"重跑"是安全的——算"哪些要问"是纯计算，问的那一次会直接拿到上次的答案。
        """
        calls = state.get("tool_pending") or []
        convo = list(state.get("tool_ctx") or [])
        steps = list(state.get("tool_steps") or [])
        step = state.get("tool_step") or 1
        seen = dict(state.get("tool_seen") or {})
        repeats = state.get("tool_repeats") or 0
        waited = state.get("tool_waited_ms") or 0

        need = [c for c in calls if c.get("level") in confirm_levels]
        decision: dict = {}
        if need:
            decision = interrupt(_confirm_payload(step, need,
                                                  state.get("query") or "")) or {}
            waited += int(decision.get("waited_ms") or 0)

        # 用户没理这一步、直接问了别的问题：直接收尾，不再调模型（一分钱都不该花）。
        # 和"拒绝"不同——拒绝是明确表态，要让它换个方法继续；取消是"这条不算了"。
        if decision.get("cancel"):
            steps.append(react.Step(step, "(取消)", {}, True, "用户放弃了这次操作", 0,
                                    confirm="rejected").brief())
            return {"tool_pending": [], "answer": "（上一轮待确认的操作已取消，未执行。）",
                    "tool_stop": "cancelled", "tool_steps": steps,
                    "tool_waited_ms": waited}

        approve = bool(decision.get("approve"))
        note = str(decision.get("note") or "")

        progressed = False
        for c in calls:
            # 参数都不合法的调用不用问人：它本来也执行不了，直接把错误还给模型
            if c.get("err"):
                steps.append(react.Step(step, c["name"], {}, False, "参数解析失败", 0,
                                        error=c["err"], confirm="auto").brief())
                convo.append({"role": "tool", "tool_call_id": c["id"],
                              "content": f"参数错误：{c['err']}。请重新给出**合法 JSON**，"
                                         f"字段参考工具定义。"})
                progressed = True
                continue

            if c.get("level") in confirm_levels and not approve:
                why = note or "用户没有同意"
                steps.append(react.Step(step, c["name"], c["args"], False,
                                        f"你拒绝了这次执行（{why}）", 0,
                                        confirm="rejected").brief())
                convo.append({"role": "tool", "tool_call_id": c["id"],
                              "content": f"用户拒绝执行 {c['name']}（原因：{why}）。"
                                         f"不要重试同一个调用。基于已有信息回答，"
                                         f"或者直接说明你需要这项权限。"})
                continue

            key = f"{c['name']}|{json.dumps(c['args'], sort_keys=True, ensure_ascii=False)}"
            if key in seen:
                repeats += 1
                steps.append(react.Step(step, c["name"], c["args"], True,
                                        f"重复调用（同第 {seen[key]} 步）", 0,
                                        repeat=True, confirm="auto").brief())
                convo.append({"role": "tool", "tool_call_id": c["id"],
                              "content": f"（这一步和第 {seen[key]} 步完全相同，结果也相同，"
                                         f"不要重复调用。请换方法或直接回答。）"})
                continue

            out = toolreg.call(c["name"], c["args"], levels=levels)
            seen[key] = step
            repeats = 0
            progressed = True
            steps.append(react.Step(step, c["name"], c["args"], out.ok, out.summary,
                                    out.ms, truncated=out.truncated, error=out.error,
                                    confirm="approved" if c.get("level") in confirm_levels
                                    else "auto").brief())
            convo.append({"role": "tool", "tool_call_id": c["id"],
                          "content": out.for_model()})

        stop = ""
        # 连续两次原地打转（全是重复调用）→ 收口。留着它只会烧钱
        if not progressed and repeats >= 2:
            stop = "no_progress"
        return {"tool_ctx": convo, "tool_pending": [], "tool_steps": steps,
                # 括号开头的都是内部收尾步骤（(模型) / (收口)），不算"动了几次手"
                "tool_calls": sum(1 for s in steps if not s.get("repeat")
                                  and not str(s.get("name", "")).startswith("(")),
                "tool_seen": seen, "tool_repeats": repeats,
                "tool_waited_ms": waited, "tool_stop": stop}

    # ---------------------------------------------------------------- finalize

    def finalize(state: ToolState) -> dict:
        """到上限/打转时的收口：让它把已经查到的东西讲完。"""
        convo = list(state.get("tool_ctx") or [])
        answer, tin, tout, err = react.finalize(convo, model_fn)
        steps = list(state.get("tool_steps") or [])
        if err:
            steps.append(react.Step(0, "(收口)", {}, False, "收口失败", 0, error=err).brief())
        return {"answer": answer,
                "tool_in": (state.get("tool_in") or 0) + tin,
                "tool_out": (state.get("tool_out") or 0) + tout,
                "tool_cost": (state.get("tool_cost") or 0.0) + react.cost_of(tin, tout),
                "tool_steps": steps,
                "tool_stop": state.get("tool_stop") or "max_steps"}

    # ------------------------------------------------------------------ finish

    def finish(state: ToolState) -> dict:
        """统一出口：算清这一轮花了多少、答案是谁给的。

        工具链路整个失败（模型不可用）时退回普通云端作答——用户要的是答案，
        不是"工具循环出错了"这个内部状态。降级要自己吃掉。
        """
        tin = state.get("tool_in") or 0
        tout = state.get("tool_out") or 0
        answer = state.get("answer") or ""
        update: dict = {"cloud_in": tin, "cloud_out": tout,
                        "model_used": f"{settings.cloud_model} +tools",
                        "degraded": False}
        if not answer:
            try:
                text, ai, ao = call_cloud(state.get("messages") or [])
                update.update(answer=text, cloud_in=tin + ai, cloud_out=tout + ao,
                              degraded=True,
                              model_used=settings.cloud_model,
                              tool_stop=state.get("tool_stop") or "error")
            except ProviderError:
                update.update(answer="工具调用没能完成，模型也不可用。请稍后再试。",
                              degraded=True, model_used="tool-error",
                              tool_stop=state.get("tool_stop") or "error")
        else:
            update["answer"] = answer
        update["tool_steps"] = state.get("tool_steps") or []
        return update

    # ------------------------------------------------------------------- 组装

    def after_think(state: ToolState) -> str:
        return "act" if state.get("tool_pending") else "finish"

    def after_act(state: ToolState) -> str:
        stop = state.get("tool_stop")
        if stop == "cancelled":
            return "finish"          # 取消：一句话收尾，不再花模型钱
        if stop in ("no_progress", "timeout"):
            return "finalize"
        return "think"

    g = StateGraph(ToolState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("finalize", finalize)
    g.add_node("finish", finish)

    g.add_edge(START, "think")
    g.add_conditional_edges("think", after_think, {"act": "act", "finish": "finish"})
    g.add_conditional_edges("act", after_act,
                            {"think": "think", "finalize": "finalize", "finish": "finish"})
    g.add_edge("finalize", "finish")
    g.add_edge("finish", END)
    # 不带 checkpointer：中断要靠父图那个带持久化的 checkpointer 才能落盘。
    # 子图自己再挂一个，恢复时状态对不上，表现为"resume 之后又从头问一遍"。
    return g.compile()
