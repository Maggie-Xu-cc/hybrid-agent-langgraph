"""上下文工程：把有限的 token 预算分配给人设、记忆、检索资料、摘要和最近对话。

为什么需要这一层：
- 不加预算控制，长对话会把 prompt 撑爆，成本和延迟一起失控；
- 但"一刀切砍掉"会丢信息。所以按重要性排序分配，而不是简单截断。

分配顺序（也是优先级）：
1. system 人设（必须有，很小）
2. 长期记忆事实（跨会话最有价值，且很短）
3. 检索到的资料（RAG 召回，决定答案准不准）
4. 早前对话摘要（压缩过的历史精华）
5. 最近对话原文（细节，预算内能装多少装多少）
6. 本轮问题（永远保留）

token 估算故意用"字数折算"而不是真 tokenizer：
- 中文约 0.75 token/字，英文约 0.3 token/字符，误差在 ±15% 内；
- 为了估算引入一个 tokenizer 依赖不值当，预算本身留了余量。
"""

CJK_RANGE = ("\u4e00", "\u9fff")

# 默认预算（token）。按"经济用量"设，不是模型上限。
DEFAULT_BUDGET = {
    "system": 200,    # 人设 + 记忆/摘要注入
    "memory": 200,    # 长期记忆事实块
    "rag": 500,       # 检索资料块
    "summary": 300,   # 早前对话摘要
    "history": 800,   # 最近对话原文
}

MEMORY_MAX_ITEMS = 6      # 最多注入几条事实
SUMMARY_MAX_CHARS = 600   # 摘要硬上限（字符）
RAG_MAX_CHUNKS = 4        # 最多注入几段检索结果


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


def _rag_block(chunks: list[dict]) -> str:
    """把检索结果拼成"参考资料"块，每段带上出处。

    带出处不是为了好看：模型据此可以说"根据 docs/xxx.md 的说法"，
    用户能核验；没有出处时模型会把自己编的内容说得同样自信，
    出了问题也追不回来。
    """
    lines = ["\n\n可参考的资料（摘自本地文档。与问题相关时优先据此回答，"
             "并在回答中注明出处；不相关就忽略，不要硬套）："]
    for i, c in enumerate(chunks[:RAG_MAX_CHUNKS], 1):
        src = c["doc"] + (f" § {c['heading']}" if c.get("heading") else "")
        lines.append(f"\n[{i}] {src}\n{c['text']}")
    return "\n".join(lines)


def build_messages(
    system: str,
    facts: dict[str, str],
    summary: str,
    history: list[dict],
    query: str,
    budget: dict | None = None,
    rag_chunks: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """拼出发给模型的消息列表，同时返回各部分 token 统计（可观测性）。

    返回 (messages, stats)。stats 里 history_dropped 表示因预算被裁掉的消息条数——
    这个数字是后续优化（换更大窗口/更好摘要）的依据。
    """
    b = {**DEFAULT_BUDGET, **(budget or {})}
    stats = {"system": estimate_tokens(system), "memory": 0, "rag": 0, "summary": 0,
             "history": 0, "query": estimate_tokens(query),
             "history_dropped": 0, "memory_dropped": 0, "rag_dropped": 0}
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

    # ② 检索资料：和记忆相反，这里**逐条装、装不下就停**。
    #    因为每段资料是独立事实，少给一段只是少一条参考，不会变成"半句话"。
    #    但至少保证第 1 段能进——否则整块预算白算，检索等于没做。
    if rag_chunks:
        header = ("\n\n可参考的资料（摘自本地文档。与问题相关时优先据此回答，"
                  "并在回答中注明出处；不相关就忽略，不要硬套）：")
        used = estimate_tokens(header)
        kept: list[dict] = []
        for c in rag_chunks[:RAG_MAX_CHUNKS]:
            t = estimate_tokens(c.get("text", "")) + 12   # 出处行 + 序号开销
            if kept and used + t > b["rag"]:
                break
            kept.append(c)
            used += t
        if kept:
            system_content += _rag_block(kept)
            stats["rag"] = used
            stats["rag_used"] = len(kept)
            stats["rag_dropped"] = max(0, len(rag_chunks) - len(kept))

    # ③ 早前对话摘要：紧跟在记忆后面注入 system。
    if summary:
        trimmed = summary[:SUMMARY_MAX_CHARS]
        system_content += "\n\n早前对话摘要：\n" + trimmed
        stats["summary"] = estimate_tokens(trimmed)

    messages = [{"role": "system", "content": system_content}]

    # ④ 最近对话原文：从最新往回装，预算内能装多少装多少。
    used_hist = 0
    kept_msgs: list[dict] = []
    for msg in reversed(history[-40:]):   # 先粗剪到 40 条，避免极端长历史全量计算
        t = estimate_tokens(msg.get("content", "")) + 4  # 每条含 role/格式开销
        if used_hist + t > b["history"] and kept_msgs:
            break
        kept_msgs.append(msg)
        used_hist += t
    kept_msgs.reverse()
    stats["history"] = used_hist
    stats["history_dropped"] = max(0, len(history) - len(kept_msgs))
    messages.extend(kept_msgs)

    # ⑤ 本轮问题永远保留
    messages.append({"role": "user", "content": query})
    stats["total"] = sum(v for k, v in stats.items()
                         if k in ("system", "memory", "rag", "summary", "history", "query"))
    return messages, stats
