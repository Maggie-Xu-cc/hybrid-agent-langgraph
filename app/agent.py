"""混合推理主流程：路由 -> 本地作答 -> 质量自检 -> 必要时升级到云端 -> 记账。

刻意不用 LangGraph：第一版流程是线性的，显式三步比图编排更好调试，
而且能讲清"每一步为什么存在"比"我用了框架"更有说服力。后续加多轮记忆与
状态分支后迁到 app/graph.py，本文件保留为对照基线（ENABLE_LANGGRAPH=0）。
"""

import time

from .clients import ProviderError, call_cloud, call_local
from .config import settings
from .router import decide
from . import store

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


def _needs_escalation(text: str) -> bool:
    if len(text.strip()) < 12:
        return True
    return any(marker in text for marker in ESCALATE_MARKERS)


def run(query: str, force_route: str | None = None) -> dict:
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}

    started = time.perf_counter()

    route = decide(query)
    if force_route in ("easy", "hard"):
        route.difficulty = force_route
        route.reason = f"强制{force_route}"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    # 分开计账：本地 token 与云端 token 单价不同
    local_in, local_out = route.tokens_in, route.tokens_out  # 路由本身的推理开销
    cloud_in = cloud_out = 0

    escalated = False
    degraded = False
    answer = ""
    model_used = ""

    if route.difficulty == "easy":
        try:
            answer, tin, tout = call_local(messages)
            local_in += tin
            local_out += tout
            model_used = settings.local_model
        except ProviderError as exc:
            degraded = True
            answer, tin, tout = call_cloud(messages)
            cloud_in += tin
            cloud_out += tout
            model_used = settings.cloud_model
            route.reason = f"{route.reason}(本地不可用)"

        if not degraded and settings.enable_escalation and _needs_escalation(answer):
            escalated = True
            try:
                answer2, tin2, tout2 = call_cloud(messages)
                cloud_in += tin2
                cloud_out += tout2
                answer = answer2
                model_used = f"{settings.local_model} -> {settings.cloud_model}"
            except ProviderError:
                pass  # 升级失败保留本地答案，不阻塞主流程
    else:
        try:
            answer, tin, tout = call_cloud(messages)
            cloud_in += tin
            cloud_out += tout
            model_used = settings.cloud_model
        except ProviderError:
            degraded = True
            answer, tin, tout = call_local(messages)
            local_in += tin
            local_out += tout
            model_used = settings.local_model
            route.reason = f"{route.reason}(云端不可用)"

    latency_ms = int((time.perf_counter() - started) * 1000)
    tokens_in = local_in + cloud_in
    tokens_out = local_out + cloud_out

    cost = (
        (local_in + local_out) / 1000 * settings.price_local_per_1k
        + cloud_in / 1000 * settings.price_cloud_in_per_1k
        + cloud_out / 1000 * settings.price_cloud_out_per_1k
    )
    # 基线：同样的 token 量全部走云端
    baseline = (
        tokens_in / 1000 * settings.price_cloud_in_per_1k
        + tokens_out / 1000 * settings.price_cloud_out_per_1k
    )

    store.record(
        query=query,
        route=route.difficulty,
        reason=route.reason,
        decided_by=route.by,
        model=model_used,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        baseline_cost=baseline,
        escalated=escalated,
    )

    return {
        "answer": answer,
        "route": route.difficulty,
        "reason": route.reason,
        "decided_by": route.by,
        "model": model_used,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_cny": round(cost, 6),
        "baseline_cost_cny": round(baseline, 6),
        "escalated": escalated,
        "degraded": degraded,
    }
