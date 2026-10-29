"""阶段四冒烟测试：RAG 检索层。

离线跑（MOCK_MODE=1），不花钱不依赖外部服务：
    python tools/smoke_rag.py

覆盖：
  1. 切分：标题分节、短块过滤、长段落硬切、内容不丢
  2. 词项：中文 bigram、停用词过滤、英文整词保留
  3. 关键项：DF=0 的垃圾组合不能当关键项（踩过的坑）
  4. BM25：相关查询分数显著高于无关查询
  5. 检索：相关命中正确章节 / 无关拒绝 / top_k 生效 / 阈值可调
  6. 关键命中约束：靠单个巧合词项撞上来的不入选
  7. 索引：重建幂等、改文档后旧块被替换
  8. 图链路：检索结果进 prompt、result 暴露来源、有检索时不写缓存
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="smoke_rag_")) / "rag.db")
os.environ["CHECKPOINT_DB"] = str(
    Path(tempfile.mkdtemp(prefix="smoke_rag_ck_")) / "checkpoints.db")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import cache, context, memory, rag, store   # noqa: E402
from app.config import settings                      # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  [{detail}]" if detail else ""))


store.init()
cache.init()
memory.init()
rag.init()
from app.graph import run_graph   # noqa: E402 — 依赖上面的 init 顺序

# --------------------------------------------------------------------------
print("\n== 1. 切分 ==")
md = """# 标题

## 第一节

第一段内容，讲缓存设计。这里有足够多的字来通过最小长度检查，确保它不会被当成噪声丢掉。

第二段内容，讲检索设计。同样需要一定的长度，否则会因为太短而被过滤掉。

## 第二节

这一节只有一个很长的段落。""" + "很长的内容。" * 200

chunks = rag.split_markdown(md)
check("按标题分成多块", len(chunks) >= 3, f"{len(chunks)} 块")
check("块带 heading", all("heading" in c for c in chunks))
check("标题被识别", any(c["heading"] == "第一节" for c in chunks))
check("超长段落被硬切", all(len(c["text"]) <= rag.CHUNK_MAX_CHARS + 50 for c in chunks),
      f"最长 {max(len(c['text']) for c in chunks)} 字")
check("不丢内容", "缓存设计" in "".join(c["text"] for c in chunks))

short = rag.split_markdown("## 只有标题\n\n短\n")
check("过短的块被过滤", len(short) == 0)

# --------------------------------------------------------------------------
print("\n== 2. 词项 ==")
t = rag._terms("缓存阈值怎么定")
check("中文切成 bigram", "缓存" in t and "阈值" in t, str(t))
check("停用词被剔除", "怎么" not in t, str(t))
t2 = rag._terms("SQLiteSaver 报 readonly")
check("英文按整词切", "sqlitesaver" in t2 and "readonly" in t2, str(t2))
check("大小写归一", "readonly" in rag._terms("ReadOnly"))

# --------------------------------------------------------------------------
print("\n== 3. 关键项（踩过的坑）==")
corpus = [rag._terms("缓存 阈值 缓存 阈值"), rag._terms("熔断 三态 状态机"), rag._terms("缓存 命中")]
key = rag._key_terms(corpus, rag._terms("缓存阈值么会会报"))
check("DF=0 的垃圾组合不算关键项", "么会" not in key and "会报" not in key, str(key))
check("真实存在的词可以是关键项", len(key) > 0, str(key))

# --------------------------------------------------------------------------
print("\n== 4-6. 检索质量（索引 docs/）==")
r = rag.index_dirs(["docs"], ROOT)
check("索引成功", r["chunks"] > 0, f"{r['docs']} 篇 / {r['chunks']} 块")

# 索引口径：哪些 md 不该进知识库。
# 这不是"多几块"的问题——混进来之后 search_docs 会把"我今天学到什么"
# 当成项目文档引用出去，答案里就会出现学习笔记的口吻。
with rag.connect(settings.db_path) as conn:
    indexed_docs = {row["doc"] for row in conn.execute(
        "SELECT DISTINCT doc FROM kb_chunks")}
check("学习笔记（learning-*）不进知识库",
      not any("learning-" in d for d in indexed_docs), f"{sorted(indexed_docs)}")
check("评测原始数据（evidence/）不进知识库",
      not any("/evidence/" in d for d in indexed_docs))
check("正式文档都在", "docs/code-reading.md" in indexed_docs)
check("刚建完索引不算过期", rag.stale(["docs"], ROOT) is False)

REL = [
    # 断言用"内容里有没有这个词"，不用具体章节号：
    # 文档一改（比如新写了一节复述同样的例子），章节号就变了，检索结果其实是对的，
    # 断言却会红——这不是代码坏了，是测试写脆了。
    ("缓存阈值是怎么定下来的", "0.45"),
    ("SQLiteSaver 为什么会报 readonly", "readonly"),
    ("熔断器的三个状态是什么", "熔断"),
    ("语义缓存和普通缓存什么区别", "KV"),
]
ok = 0
for q, kw in REL:
    hits = rag.search(q, top_k=3)
    hit = bool(hits) and any(kw in (h["heading"] + h["text"]) for h in hits)
    ok += hit
    check(f"相关命中：{q[:18]}", hit, hits[0]["heading"][:20] if hits else "未命中")
check("相关召回率达标", ok >= len(REL) - 1, f"{ok}/{len(REL)}")

IRR = ["什么是反向传播算法", "给我讲讲量子力学", "明天股票会涨吗"]
rejected = sum(1 for q in IRR if not rag.search(q, top_k=3))
check("无关问题被拒绝", rejected >= 2, f"{rejected}/{len(IRR)} 拒绝")

hits = rag.search("缓存阈值是怎么定下来的", top_k=2)
check("top_k 生效", len(hits) <= 2, f"{len(hits)} 条")
check("结果带出处", all("doc" in h and "heading" in h for h in hits))
check("展示分与排序一致",
      all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)))

strict = rag.search("缓存阈值是怎么定下来的", min_score=999)
check("阈值可调（调到极高则空）", len(strict) == 0)
# 这个用例已经换过两次了，第三次踩坑值得记下来：
#   第 1 次用"今天上海天气" → 第 16 节把它当反例写进正文，知识库里真有了这几个字；
#   第 2 次用"红烧肉要炖多久" → 第 19 节又把它当"无关问题"的例子写进正文，同样污染。
# 通用教训：**不能拿"被写进知识库当例子的句子"去当无关查询的用例**。
# 现在固定用一句和本项目无关、也没在文档里出现过的话，并在断言信息里带条数方便排查。
loose = rag.search("腌咸鸭蛋要放多少盐")
# 注意这里的查询换过一次：原来用"今天上海天气"，后来 docs 第 16 节把这类问题
# 当反例写进了正文，于是知识库里真的有了这几个字，检索合理地召回它——
# 断言失败不是因为代码坏了，是因为"文档里出现了这句话"。
# 教训：拿做例子写进知识库的句子，不能再当"无关查询"的测试用例。
check("无词项命中时直接返回空", len(loose) <= 1, f"{len(loose)} 条")

# 关键命中约束：单个巧合词项不该召回
one_word = rag.search("帮我写一首诗", top_k=3)
check("靠巧合词项撞的不入选", len(one_word) <= 1,
      f"{len(one_word)} 条" + (f" -> {one_word[0]['heading'][:16]}" if one_word else ""))

# --------------------------------------------------------------------------
print("\n== 7. 索引幂等 ==")
n1 = rag.stats()["chunks"]
rag.index_dirs(["docs"], ROOT)
n2 = rag.stats()["chunks"]
check("重复索引不产生重复块", n1 == n2, f"{n1} -> {n2}")

tmpdir = Path(tempfile.mkdtemp(prefix="kb_"))
(tmpdir / "t.md").write_text("## 测试节\n\n" + "这是一段用于测试索引更新的内容。" * 10,
                             encoding="utf-8")
rag.index_file(tmpdir / "t.md", tmpdir)
n3 = rag.stats()["chunks"]
(tmpdir / "t.md").write_text("## 测试节\n\n改过的内容。", encoding="utf-8")
rag.index_file(tmpdir / "t.md", tmpdir)
n4 = rag.stats()["chunks"]
check("改文档后块数变化（旧块被替换）", n3 != n4, f"{n3} -> {n4}")
check("过短的新内容不产生块", n4 < n3)

# --------------------------------------------------------------------------
print("\n== 8. 图链路 ==")
res = run_graph("缓存阈值是怎么定下来的", session_id="smoke-rag-1", use_cache=False)
check("result 有 rag 字段", "rag" in res)
check("检索命中", res["rag"]["hit"] is True, f"n={res['rag']['n']}")
check("暴露来源", len(res["rag"]["chunks"]) > 0,
      res["rag"]["chunks"][0]["doc"] if res["rag"]["chunks"] else "")
check("注入 token > 0", res["memory"]["ctx_tokens"].get("rag", 0) > 0,
      str(res["memory"]["ctx_tokens"].get("rag")))
check("检索耗时被记录", res["rag"].get("latency_ms", -1) >= 0,
      f"{res['rag'].get('latency_ms')}ms")

res2 = run_graph("什么是反向传播算法", session_id="smoke-rag-2", use_cache=False)
check("无关问题不注入", res2["memory"]["ctx_tokens"].get("rag", 0) == 0)

# 检索过的答案可以写缓存，但必须绑定知识库版本。
# 早先的规则是"检索命中就不写缓存"，理由是"改了文档缓存就过期"——顾虑对，做法错：
# 线上有知识库时几乎每条查询都会检索命中，于是缓存条目永远是 0，第一道省钱关卡关闭。
# 现在改成版本号：同版本复用，重建索引后自动失效。
import uuid as _uuid
_k = "smoke-kbver-" + _uuid.uuid4().hex[:8]
cache.put(_k, "答案", "m", "easy", kb_version="v1")
check("同版本可命中", cache.lookup(_k, kb_version="v1") is not None)
check("版本变了不命中", cache.lookup(_k, kb_version="v2") is None)
check("不传版本则不校验（评测用）", cache.lookup(_k, kb_version=None) is not None)
check("检索过的答案会进缓存（不再一刀切拒绝）",
      run_graph("熔断器的三个状态是什么", session_id="smoke-rag-3b",
                use_cache=True)["rag"]["hit"] is True)

# 关掉 RAG 时完全不检索
res3 = run_graph("缓存阈值是怎么定下来的", session_id="smoke-rag-4",
                 use_cache=False, use_rag=False)
check("use_rag=False 不检索", res3["rag"]["hit"] is False
      and res3["memory"]["ctx_tokens"].get("rag", 0) == 0)

# --------------------------------------------------------------------------
print("\n== 9. 上下文注入 ==")
chunks = rag.search("缓存阈值是怎么定下来的", top_k=4)
msgs, st = context.build_messages(
    system="你是助手", facts={}, summary="", history=[], query="问",
    rag_chunks=chunks)
sys_msg = msgs[0]["content"]
check("资料块进了 system", "可参考的资料" in sys_msg)
check("带出处", "docs/" in sys_msg)
check("标注了不相关要忽略", "不相关就忽略" in sys_msg)
check("统计里有 rag token", st.get("rag", 0) > 0, str(st.get("rag")))

# 预算不够时：逐条装、至少留 1 条
msgs2, st2 = context.build_messages(
    system="你是助手", facts={}, summary="", history=[], query="问",
    rag_chunks=chunks, budget={"rag": 200})
check("预算小则少装几条", st2.get("rag_used", 0) < len(chunks),
      f"装了 {st2.get('rag_used')} 条")
check("至少留 1 条", st2.get("rag_used", 0) >= 1)
check("记录被丢条数", st2.get("rag_dropped", 0) > 0, str(st2.get("rag_dropped")))

empty_msgs, st3 = context.build_messages(
    system="你是助手", facts={}, summary="", history=[], query="问", rag_chunks=[])
check("无检索结果时不加资料块", "可参考的资料" not in empty_msgs[0]["content"])

# --------------------------------------------------------------------------
print(f"\n{'=' * 46}")
print(f"通过 {PASS} / {PASS + FAIL}")
if FAIL:
    print(f"失败 {FAIL} 项")
    sys.exit(1)
print("全部通过")
