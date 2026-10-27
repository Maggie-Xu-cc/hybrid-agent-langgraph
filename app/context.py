"""上下文工程：把有限的 token 预算分配给人设、长期记忆、摘要和最近对话。

为什么需要这一层：
- 不加预算控制，长对话会把 prompt 撑爆，成本和延迟一起失控；
- 但"一刀切砍掉"会丢信息。所以按重要性排序分配，而不是简单截断。

分配顺序（也是优先级）：
1. system 人设（必须有，很小）
2. 长期记忆事实（跨会话最有价值，且很短）
3. 早前对话摘要（压缩过的历史精华）
4. 最近对话原文（细节，预算内能装多少装多少）
5. 本轮问题（永远保留）

token 估算故意用"字数折算"而不是真 tokenizer：
- 中文约 0.75 token/字，英文约 0.3 token/字符，误差在 ±15% 内；
- 为了估算引入一个 tokenizer 依赖不值当，预算本身留了余量。
"""

CJK_RANGE = ("\u4e00", "\u9fff")

# 默认预算（token）。按"经济用量"设，不是模型上限。
DEFAULT_BUDGET = {
    "system": 200,    # 人设 + 记忆/摘要注入
    "memory": 200,    # 长期记忆事实块
    "summary": 300,   # 早前对话摘要
    "history": 800,   # 最近对话原文
}

MEMORY_MAX_ITEMS = 6      # 最多注入几条事实
SUMMARY_MAX_CHARS = 600   # 摘要硬上限（字符）


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 0.75/字，其它 0.3/字符。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if CJK_RANGE[0] <= ch <= CJK_RANGE[1])
    other = len(text) - cjk
    return int(cjk * 0.75 + other * 0.3) + 1


def _facts_block(facts: dict[str, str]) -> str:
    lines = [f"- {k}: {v}" for k, v in list(facts.items())[:MEMORY_MAX_ITEMS]]
    return "\n\n已知用户信息（来自长期记忆，可直接使用）：\n" + "\n".join(lines)


def build_messages(
    system: str,
    facts: dict[str, str],
    summary: str,
    history: list[dict],
    query: str,
    budget: dict | None = None,
) -> tuple[list[dict], dict]:
    """拼出发给模型的消息列表，同时返回各部分 token 统计（可观测性）。

    返回 (messages, stats)。stats 里 history_dropped 表示因预算被裁掉的消息条数——
    这个数字是后续优化（换更大窗口/更好摘要）的依据。
    """
    b = {**DEFAULT_BUDGET, **(budget or {})}
    stats = {"system": estimate_tokens(system), "memory": 0, "summary": 0,
             "history": 0, "query": estimate_tokens(query),
             "history_dropped": 0, "memory_dropped": 0}
    system_content = system

    # ① 长期记忆注入 system。整块装不下就整块丢弃，不塞半个——
    #    半截的"我叫小"比没有更糟。
    if facts:
        block = _facts_block(facts)
        cost = estimate_tokens(block)
        if stats["system"] + cost <= b["system"] + b["memory"]:
            system_content += block
            stats["memory"] = cost
        else:
            stats["memory_dropped"] = 1

    # ② 早前对话摘要：紧跟在记忆后面注入 system。
    if summary:
        trimmed = summary[:SUMMARY_MAX_CHARS]
        system_content += "\n\n早前对话摘要：\n" + trimmed
        stats["summary"] = estimate_tokens(trimmed)

    messages = [{"role": "system", "content": system_content}]

    # ③ 最近对话原文：从最新往回装，预算内能装多少装多少。
    used = 0
    kept: list[dict] = []
    for msg in reversed(history[-40:]):   # 先粗剪到 40 条，避免极端长历史全量计算
        t = estimate_tokens(msg.get("content", "")) + 4  # 每条含 role/格式开销
        if used + t > b["history"] and kept:
            break
        kept.append(msg)
        used += t
    kept.reverse()
    stats["history"] = used
    stats["history_dropped"] = max(0, len(history) - len(kept))
    messages.extend(kept)

    # ④ 本轮问题永远保留
    messages.append({"role": "user", "content": query})
    stats["total"] = sum(v for k, v in stats.items()
                         if k in ("system", "memory", "summary", "history", "query"))
    return messages, stats
