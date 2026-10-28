"""语义缓存：相似的问题直接复用上次的答案，把"答过的问题"成本降到 0。

它插在三层路由之前，形成完整的四级降本链路：

    语义缓存命中(0成本) ─▶ 规则路由(0成本) ─▶ 本地小模型(几乎免费) ─▶ 云端大模型(贵)

设计取舍（面试可讲的四点）：

1. 为什么现在用本地小模型，而不是当初的字符 n-gram？
   一开始的理由是"为缓存再调一次 embedding API 就本末倒置"——这半句是对的，
   但结论错了：**零的是 API 成本，不是零模型**。本地跑 91MB 的 BGE-small-zh
   （ONNX + CPU），一样一分钱不花，单条十几毫秒。

   而 n-gram 的代价是硬的，89 条标注集实测（docs/evidence/threshold_sweep.md，脚本
   tools/threshold_sweep.py 可复现）：
       该命中的同义改问法  中位相似度 0.437   「您好呀」vs「你好」只有 0.11
       必须拒绝的近似问题  中位相似度 0.746   「3点45」vs「3点15」高达 0.906
   分数与"该不该复用"是反着的。扫遍 0.20~0.95 全区间，precision 最高 57%；
   0.45 时命中 41 次里 18 次答错，整体答对率 51.7%，跟抛硬币差不多。
   根因：n-gram 量的是字面重合，而判断该不该复用看的是语义——一个字的变化
   （3→4、大写→小写、404→403）能翻转答案，整句换个说法却是同一件事。

   所以阈值不是"调得不准"，是尺子不对。换刻度救不了，只能换尺子。
   这也是 GPTCache / MeanCache / vCache 这些工作无一例外都建在真 embedding 上的原因。

   保留 n-gram 作为降级路径：部署包不携带 91MB 模型时自动退回，
   主流程不因为缓存组件装不上而挂掉。切换逻辑在 app/embedder.py。

2. 为什么只缓存"首轮独立问题"？
   多轮问题（"继续""为什么"）依赖会话上下文，同一句话在不同会话里答案不同，
   缓存会答错。宁可少命中，不可答错——和记忆模块"宁可少记，不可记错"同一原则。

3. 为什么默认只缓存 easy？
   hard 请求通常是大段代码/长文，答案质量敏感；且这类问题重复率低、答案长，
   缓存收益小风险大。配置 CACHE_CACHE_HARD=1 可打开。

4. 为什么相似度用线性扫描而不是向量库？
   缓存条目量级是几百到几千，256 维向量两两点积一次 <10ms，SQLite 单表就够。
   等量级上万再上 FAISS/sqlite-vec——"判断成本要配得上判断价值"的又一次复用。

失效策略（三条，各管一类"为什么不能再复用"）：
    1. TTL（默认 7 天）：时间会让答案过时（"今天天气"）
    2. 向量后端切换：维度不同不能比（见 embed_model 列）
    3. 知识库版本变化：答案依赖检索内容，文档改了就不成立（见 kb_version 列）
第 3 条原来是用"检索命中过就一律不写缓存"实现的，代价是只要有知识库缓存就永不工作。
改成版本号以后，粒度和收益才对齐。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timedelta

from . import cacheguard, embedder
from .config import settings
from .dbutil import with_retry

_WS_RE = re.compile(r"\s+")

# 向量化统一走 app/embedder.py：缓存和评测必须用同一把尺子，否则阈值不可复用。
# 保留 embed 这个名字是为了既有调用方（tools/threshold_sweep.py 等）不用改。
embed = embedder.embed


def _normalize(text: str) -> str:
    """查询归一化：去空白、转小写。让"你好"和"你 好"落在同一个 hash 上。"""
    return _WS_RE.sub("", text).lower()


def _hash_query(text: str) -> str:
    """完全一致（归一化后）的查询用 hash 直接命中，跳过相似度扫描。"""
    return hashlib.md5(_normalize(text).encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# 阈值与后端强绑定：两个后端的分数分布完全不同，数值不可互相搬。
# bge 0.60 是实测拐点（precision 100%，错误率 0），ngram 0.35 是 F0.5 最优点。
_AUTO_THRESHOLD = {"bge": 0.60, "ngram": 0.35}


def effective_threshold() -> float:
    """实际生效的分数线：CACHE_THRESHOLD>0 用手动值，否则按当前后端取标定值。"""
    if settings.cache_threshold > 0:
        return settings.cache_threshold
    return _AUTO_THRESHOLD.get(embedder.active(), 0.45)


def init() -> None:
    """建表（幂等）。与记账同库：缓存和成本统计是同一个故事的两半。"""
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS semantic_cache ("
            " query_hash TEXT PRIMARY KEY,"     # 归一化后 md5，完全一致的快速通道
            " query_text TEXT NOT NULL,"        # 原始问题（调试/展示用）
            " embedding TEXT NOT NULL,"         # 向量，JSON 数组（维度随后端变）
            " embed_model TEXT NOT NULL DEFAULT 'ngram-256',"  # 哪把尺子量的
            " answer TEXT NOT NULL,"
            " model TEXT NOT NULL,"
            " difficulty TEXT NOT NULL,"
            " tokens_in INTEGER NOT NULL DEFAULT 0,"   # 生成这份答案实际花的 token
            " tokens_out INTEGER NOT NULL DEFAULT 0,"
            " cost_cny REAL NOT NULL DEFAULT 0,"
            " hits INTEGER NOT NULL DEFAULT 0,"
            " kb_version TEXT NOT NULL DEFAULT '',"   # 生成答案时的知识库版本
            " created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
        # 老库补列：换后端才有意义，缺了它就只能整表重建
        cols = {r[1] for r in conn.execute("PRAGMA table_info(semantic_cache)")}
        if "embed_model" not in cols:
            conn.execute("ALTER TABLE semantic_cache ADD COLUMN"
                         " embed_model TEXT NOT NULL DEFAULT 'ngram-256'")
        if "kb_version" not in cols:
            conn.execute("ALTER TABLE semantic_cache ADD COLUMN"
                         " kb_version TEXT NOT NULL DEFAULT ''")
    conn.close()


@with_retry
def lookup_ex(query: str, threshold: float | None = None,
              kb_version: str | None = None) -> dict:
    """查缓存，并把"为什么没命中"也一起返回。

    只返回 None 的话，"被门卫拦下"和"压根没有相似条目"在外部看来一模一样——
    而这两件事的含义完全相反：前者说明系统刚刚防止了一次错误回答，后者只是没省钱。
    这个区别是页面演示里最有价值的一处，所以必须带出来。

    返回 {"hit": dict|None, "score": 最高相似度, "blocked": 拒绝原因,
          "blocked_text": 被拦下的那条缓存原文}
    """
    miss = {"hit": None, "score": 0.0, "blocked": "", "blocked_text": ""}
    if not query or not settings.enable_cache:
        return miss
    threshold = threshold if threshold is not None else effective_threshold()
    q_hash = _hash_query(query)
    now = time.time()

    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        rows = conn.execute(
            "SELECT query_hash, embedding, query_text, answer, model, difficulty,"
            " hits, updated_at, embed_model, kb_version, cost_cny FROM semantic_cache"
        ).fetchall()
    except sqlite3.OperationalError:   # 表还没建
        return miss
    finally:
        conn.close()

    cur_model = embedder.name()
    ttl = settings.cache_ttl_days * 86400
    best_score, best_row, best_hash = 0.0, None, ""
    guarded_out, guarded_text, guarded_score = "", "", 0.0
    for (r_hash, emb_json, query_text, answer, model, difficulty,
         hits, updated_at, emb_model, kb_ver, row_cost) in rows:
        # 换过向量后端：256 维和 512 维算出来的余弦毫无意义，整条跳过。
        # 代价是切换后端后缓存相当于清空——这是正确的，宁可重算不可答错。
        if emb_model != cur_model:
            continue

        # 知识库变过：这条答案是按旧文档生成的，可能已经不成立，整条跳过。
        # 空版本表示"这份答案没用到检索"，不受文档更新影响，照常可用。
        if kb_version is not None and kb_ver and kb_ver != kb_version:
            continue

        # TTL 统一生效（含精确匹配）。命中过的条目 updated_at 每次刷新，
        # 正常不会过期；hits=0 且超期的是死条目，直接跳过。
        try:
            age = now - datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            age = 0
        if age > ttl and hits == 0:
            continue

        if r_hash == q_hash:
            # 完全一致：字面都一模一样，不需要再过守卫
            best_score, best_hash = 1.0, r_hash
            best_row = (answer, model, difficulty, hits, row_cost)
            break

        score = _cosine(embed(query), json.loads(emb_json))
        # 先过分数线，再过守卫。顺序不能反：
        # 反过来的话，一个相似度只有 0.12 的无关条目也会被守卫"拒绝"，
        # 于是页面把"压根不像"误报成"门卫拦下了一次危险复用"——这属于编故事。
        # 只有"本来会命中、但参数不同"才配叫拦截。
        if score < threshold or score <= best_score:
            continue
        if settings.cache_guard:
            ok, why = cacheguard.check(query, query_text)
            if not ok:
                # 记下最高分的那条被拦候选，用于展示"0.906 却没复用"
                if score > guarded_score:
                    guarded_score = score
                guarded_out, guarded_text = why, query_text
                continue
        best_score, best_hash = score, r_hash
        best_row = (answer, model, difficulty, hits, row_cost)

    if best_row is None:
        return {"hit": None, "score": max(best_score, guarded_score),
                "blocked": guarded_out, "blocked_text": guarded_text}

    answer, model, difficulty, hits, saved_cost = best_row
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        # 注意：刷新的是"被匹配条目"的计数，不是当前查询的 hash——
        # 相似命中时两者不同，写错位置统计就全歪了。
        conn.execute(
            "UPDATE semantic_cache SET hits=hits+1, updated_at=? WHERE query_hash=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), best_hash),
        )
    conn.close()
    return {"hit": {"answer": answer, "model": model, "difficulty": difficulty,
                    "score": best_score, "hits": hits + 1,
                    # 这条答案当初生成的成本：命中时成本记 0，基线要按它算，否则"省了多少"看不出来
                    "cost": float(saved_cost or 0.0),
                    "guard": "pass" if guarded_out else "",
                    "guard_blocked": guarded_out},
            "score": best_score, "blocked": guarded_out, "blocked_text": guarded_text}


def lookup(query: str, threshold: float | None = None,
           kb_version: str | None = None) -> dict | None:
    """查缓存，只要结果。返回 {"answer","model","difficulty","score","hits"} 或 None。

    两段式：先 hash 精确匹配（O(1)），再相似度线性扫描（量级小，毫秒级）。
    从未命中过的过期条目（hits=0 且超 TTL）不返回——答案可能已经过时，比如"今天天气"；
    命中过的条目 updated_at 每次命中都会刷新，正常不会到期。

    kb_version：当前知识库版本（rag.version()）。传了就只认同一版本下生成的条目，
    传 None 表示不校验（评测/实验用——它们要的是纯粹的相似度行为）。
    """
    return lookup_ex(query, threshold, kb_version)["hit"]


@with_retry
def put(query: str, answer: str, model: str, difficulty: str,
        tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0,
        kb_version: str = "") -> bool:
    """写入缓存。完全相同的查询已存在时不覆盖（答案没变，没必要）。

    表不存在时自动建表重试一次——调用方（如冒烟脚本）忘了 init 也不至于崩，
    主流程不能被缓存这种"锦上添花"的组件阻塞。

    kb_version：生成这份答案时的知识库版本。答案没用到检索时传空串。
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
                    " (query_hash, query_text, embedding, embed_model, answer, model,"
                    "  difficulty, tokens_in, tokens_out, cost_cny, hits,"
                    "  kb_version, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?)",
                    (q_hash, query[:500], json.dumps(embed(query)),
                     embedder.name(), answer, model,
                     difficulty, tokens_in, tokens_out, round(cost, 6),
                     kb_version, now, now),
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
