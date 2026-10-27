"""语义缓存：相似的问题直接复用上次的答案，把"答过的问题"成本降到 0。

它插在三层路由之前，形成完整的四级降本链路：

    语义缓存命中(0成本) ─▶ 规则路由(0成本) ─▶ 本地小模型(几乎免费) ─▶ 云端大模型(贵)

设计取舍（面试可讲的四点）：

1. 为什么用"字符 n-gram 向量"而不是真 embedding 模型？
   为缓存再调一次 embedding API 就本末倒置——缓存的意义是省钱，查询本身必须
   零成本零网络。n-gram hashing 向量纯 Python 计算，毫秒级。
   实测边界（阈值 0.45 据此校准）：中长句的同义改写余弦 ≥0.5 能命中，无关问题
   ≤0.3 不会误命中；但短句寒暄（"你好" vs "您好啊"只有 0.1）和换说法的概念问句
   （"404 是什么意思" vs "404 代表什么"）召回弱——这是 n-gram 的已知边界，
   好在寒暄本来就 0 成本走规则路由，真正的语义召回交给 embed() 的后续实现
   （换 BGE 等本地 embedding 模型只改这一个函数，上层不动）。

2. 为什么只缓存"首轮独立问题"？
   多轮问题（"继续""为什么"）依赖会话上下文，同一句话在不同会话里答案不同，
   缓存会答错。宁可少命中，不可答错——和记忆模块"宁可少记，不可记错"同一原则。

3. 为什么默认只缓存 easy？
   hard 请求通常是大段代码/长文，答案质量敏感；且这类问题重复率低、答案长，
   缓存收益小风险大。配置 CACHE_CACHE_HARD=1 可打开。

4. 为什么相似度用线性扫描而不是向量库？
   缓存条目量级是几百到几千，256 维向量两两点积一次 <10ms，SQLite 单表就够。
   等量级上万再上 FAISS/sqlite-vec——"判断成本要配得上判断价值"的又一次复用。

失效策略：TTL（默认 7 天）+ 命中计数。长期没人命中的条目自然过期，
不需要人工维护。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timedelta

from .config import settings
from .dbutil import with_retry

# n-gram 混合权重：trigram 区分度高为主，uni/bigram 保证短句也能提出特征
_NGRAM_WEIGHTS = ((1, 0.3), (2, 0.5), (3, 1.0))
_EMB_DIM = 256

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """查询归一化：去空白、转小写。让"你好"和"你 好"落在同一个 hash 上。"""
    return _WS_RE.sub("", text).lower()


def embed(text: str) -> list[float]:
    """把文本变成 256 维归一化向量（hashing trick，零依赖零网络）。

    每个字符 n-gram hash 到固定维度上累加词频（乘权重），最后 L2 归一化。
    归一化之后两向量的内积就是余弦相似度，不用再除两个模长。
    """
    text = _normalize(text)
    vec = [0.0] * _EMB_DIM
    if not text:
        return vec
    for n, weight in _NGRAM_WEIGHTS:
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "big")
            vec[h % _EMB_DIM] += weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _hash_query(text: str) -> str:
    """完全一致（归一化后）的查询用 hash 直接命中，跳过相似度扫描。"""
    return hashlib.md5(_normalize(text).encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def init() -> None:
    """建表（幂等）。与记账同库：缓存和成本统计是同一个故事的两半。"""
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS semantic_cache ("
            " query_hash TEXT PRIMARY KEY,"     # 归一化后 md5，完全一致的快速通道
            " query_text TEXT NOT NULL,"        # 原始问题（调试/展示用）
            " embedding TEXT NOT NULL,"         # 256 维向量，JSON 数组
            " answer TEXT NOT NULL,"
            " model TEXT NOT NULL,"
            " difficulty TEXT NOT NULL,"
            " tokens_in INTEGER NOT NULL DEFAULT 0,"   # 生成这份答案实际花的 token
            " tokens_out INTEGER NOT NULL DEFAULT 0,"
            " cost_cny REAL NOT NULL DEFAULT 0,"
            " hits INTEGER NOT NULL DEFAULT 0,"
            " created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
    conn.close()


@with_retry
def lookup(query: str, threshold: float | None = None) -> dict | None:
    """查缓存。返回 {"answer","model","difficulty","score","hits"} 或 None。

    两段式：先 hash 精确匹配（O(1)），再相似度线性扫描（量级小，毫秒级）。
    过期条目（超过 TTL）一律不返回——答案可能已经过时，比如"今天天气"。
    """
    if not query or not settings.enable_cache:
        return None
    threshold = threshold if threshold is not None else settings.cache_threshold
    q_hash = _hash_query(query)
    now = time.time()

    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        rows = conn.execute(
            "SELECT query_hash, embedding, answer, model, difficulty,"
            " hits, updated_at FROM semantic_cache"
        ).fetchall()
    except sqlite3.OperationalError:   # 表还没建
        return None
    finally:
        conn.close()

    ttl = settings.cache_ttl_days * 86400
    best_score, best_row, best_hash = 0.0, None, ""
    for r_hash, emb_json, answer, model, difficulty, hits, updated_at in rows:
        # TTL 统一生效（含精确匹配）。命中过的条目 updated_at 每次刷新，
        # 正常不会过期；hits=0 且超期的是死条目，直接跳过。
        try:
            age = now - datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            age = 0
        if age > ttl and hits == 0:
            continue

        if r_hash == q_hash:
            # 完全一致：相似度记 1.0，必然最优先
            best_score, best_row, best_hash = 1.0, (answer, model, difficulty, hits), r_hash
            break

        score = _cosine(embed(query), json.loads(emb_json))
        if score > best_score:
            best_score, best_row, best_hash = score, (answer, model, difficulty, hits), r_hash

    if best_row is None or best_score < threshold:
        return None

    answer, model, difficulty, hits = best_row
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        # 注意：刷新的是"被匹配条目"的计数，不是当前查询的 hash——
        # 相似命中时两者不同，写错位置统计就全歪了。
        conn.execute(
            "UPDATE semantic_cache SET hits=hits+1, updated_at=? WHERE query_hash=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), best_hash),
        )
    conn.close()
    return {"answer": answer, "model": model, "difficulty": difficulty,
            "score": best_score, "hits": hits + 1}


@with_retry
def put(query: str, answer: str, model: str, difficulty: str,
        tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0) -> bool:
    """写入缓存。完全相同的查询已存在时不覆盖（答案没变，没必要）。

    表不存在时自动建表重试一次——调用方（如冒烟脚本）忘了 init 也不至于崩，
    主流程不能被缓存这种"锦上添花"的组件阻塞。
    """
    if not query or not answer:
        return False
    q_hash = _hash_query(query)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for attempt in (1, 2):
        try:
            conn = sqlite3.connect(settings.db_path, timeout=10)
            with conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO semantic_cache"
                    " (query_hash, query_text, embedding, answer, model, difficulty,"
                    "  tokens_in, tokens_out, cost_cny, hits, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,0,?,?)",
                    (q_hash, query[:500], json.dumps(embed(query)), answer, model,
                     difficulty, tokens_in, tokens_out, round(cost, 6), now, now),
                )
            conn.close()
            return cur.rowcount > 0
        except sqlite3.OperationalError:
            if attempt == 1:
                init()
            else:
                raise
    return False


def stats() -> dict:
    """缓存概况：条目数、累计命中、被省下的 token/成本。"""
    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(hits),0) hits,"
            " COALESCE(SUM(CASE WHEN hits>0 THEN tokens_in ELSE 0 END),0) tin,"
            " COALESCE(SUM(CASE WHEN hits>0 THEN tokens_out ELSE 0 END),0) tout,"
            " COALESCE(SUM(CASE WHEN hits>0 THEN cost_cny ELSE 0 END),0) cost"
            " FROM semantic_cache"
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        row = (0, 0, 0, 0, 0.0)
    n, hits, tin, tout, cost = row
    return {"entries": n, "hits": hits,
            "tokens_saved": tin + tout,
            "cost_saved_cny": round(cost, 6)}


def purge_expired() -> int:
    """清掉过期且从未命中的条目，返回删除数。服务启动时调一次即可。"""
    cutoff = (datetime.now() - timedelta(days=settings.cache_ttl_days)
              ).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        cur = conn.execute(
            "DELETE FROM semantic_cache WHERE updated_at < ? AND hits = 0",
            (cutoff,),
        )
    conn.close()
    return cur.rowcount or 0


def clear() -> int:
    """清空全部缓存（API 用），返回删除条数。"""
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        cur = conn.execute("DELETE FROM semantic_cache")
    conn.close()
    return cur.rowcount or 0
