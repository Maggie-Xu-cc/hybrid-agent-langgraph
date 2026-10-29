"""RAG 检索层：把本地文档切块建索引，按查询召回最相关的几段塞进 prompt。

它解决的问题：
    上下文工程（context.py）解决的是"预算怎么分配"，但预算里只有历史对话和记忆。
    模型不知道你的项目细节、不知道你的规范文档——问它"这个项目的缓存阈值怎么定的"，
    它只能编。检索就是把这个缺口补上：先找到原文，再让模型基于原文回答。

在链路里的位置：

    缓存未命中 ─▶ retrieve（本模块）─▶ route ─▶ 本地/云端作答
                    │
                    └ 命中则把片段注入 prompt 的"参考资料"块

三个关键设计（面试常问）：

1. 为什么切块，不整篇塞？
   一篇 3000 字的文档塞进去就是 2250 token，占掉整个预算。切块后只取最相关的
   3~4 块（约 400~600 token），召回靠检索而不是靠"全给模型自己找"。

2. 为什么向量 + BM25 两路，而不是只用向量？
   两者擅长的事不一样：
   - 向量（字符 n-gram）擅长"换个说法"：问"阈值怎么定的"能召回"阈值是实测出来的"
   - BM25 擅长"专有名词精确命中"：问"SQLiteSaver 为什么会 readonly"，
     向量对这种英文标识符几乎无能为力（n-gram 哈希撞不上），BM25 一击即中
   只上一路必然在某类问题上失手，所以两路都跑。

3. 为什么融合权重是 85% BM25 + 15% 向量，而不是标准 RRF？
   初版用的就是 RRF（工业界默认做法），实测发现它是错的——详见 search() 的
   长注释。一句话：向量这一路在"短查询 vs 长文档块"上没有判别力，
   RRF 会给噪声也发排名分，把 BM25 正确排第一的结果顶掉。
   换成显式加权后，向量只负责"BM25 打平时看谁字面更像"。

为什么不用真正的向量库（sqlite-vec / Chroma）：
   和缓存模块同一个判断——几千个 chunk 的线性扫描是毫秒级，不值得引入依赖。
   embed() 是独立函数：embedding 后端（ngram / bge）已由 app/embedder.py 统一支持，
   换后端只改它一处，上层不动。
   这个决定的边界也很清楚：文档量到十万级、或要毫秒级响应，就该换向量库了。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .cache import embed
from .config import settings
from .dbutil import connect, with_retry

# 切块参数
CHUNK_CHARS = 450        # 单块目标长度（字）。约 340 token，2~4 块正好填满 rag 预算
CHUNK_MIN_CHARS = 80     # 太短的块（如只有一行标题）不入库，避免噪声
CHUNK_MAX_CHARS = 900    # 超长段落硬切，防止单块吃满预算

# BM25 参数（取文献常用默认值，没调过——文档说了这是通用设置）
BM25_K1 = 1.5
BM25_B = 0.75

# 融合权重：BM25 主导。这个 0.85 不是拍的，见 search() 里的实测说明——
# 向量的判别力在"短查询 vs 长块"场景下接近于零，只保留它做 tiebreaker。
W_BM25 = 0.85
W_VEC = 0.15

# 中文停用 bigram：疑问词/功能词。不加这张表会出什么问题（实测）：
#   问"今天天气怎么样"能命中"缓存设计"章节，BM25 拿到 9.61 ——
#   因为"怎么""么样"这类 bigram 遍布全文，IDF 压不住，噪声分数盖过真信号。
#   这类词不承载语义，只会稀释排序，直接剔掉。
_STOP_BIGRAMS = {
    "怎么", "什么", "如何", "为什", "是什", "么是", "哪些", "哪个", "是否",
    "可以", "能否", "怎样", "么样", "为么", "这个", "那个", "一个", "些什",
    "我们", "你们", "他们", "它们", "因为", "所以", "但是", "如果", "这样",
    "那样", "就是", "不是", "没有", "现在", "或者", "而且", "并且", "以及",
    "问题", "情况", "时候", "东西", "一个", "一下", "可能", "应该", "需要",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc        TEXT NOT NULL,          -- 相对路径，如 docs/code-reading.md
    heading    TEXT NOT NULL DEFAULT '', -- 所在小节标题，回答时可注明出处
    chunk_no   INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    n_chars    INTEGER NOT NULL,
    embedding  TEXT NOT NULL,          -- JSON 数组，维度随后端（ngram 256 / bge 512）
    updated_at TEXT NOT NULL,
    UNIQUE(doc, chunk_no)
);
CREATE INDEX IF NOT EXISTS idx_kb_doc ON kb_chunks(doc);
"""


def init() -> None:
    with connect(settings.db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------- 切分 ----------------

def split_markdown(text: str) -> list[dict]:
    """按 markdown 标题分节，节内按段落聚合到 CHUNK_CHARS。

    为什么按标题切而不是定长滑窗：
    标题是作者自己划的语义边界，比按字数硬切准得多——一块里混着"缓存设计"和
    "部署步骤"两件事，召回时分数会被稀释，模型也容易答串。
    节内再聚合是因为有些小节只有两三行，单独成块信息量太少，
    但把整篇按标题切完就完事又会让长小节超标，所以要二级处理。
    """
    chunks: list[dict] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        """把当前缓冲区按空行段落聚合成块。"""
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        # 段落聚合：短段落合并，长了就断开
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 <= CHUNK_CHARS:
                cur = f"{cur}\n\n{p}" if cur else p
                continue
            if cur:
                chunks.append({"heading": heading, "text": cur})
            # 单段就超长：硬切
            while len(p) > CHUNK_MAX_CHARS:
                chunks.append({"heading": heading, "text": p[:CHUNK_MAX_CHARS]})
                p = p[CHUNK_MAX_CHARS:]
            cur = p
        if cur:
            chunks.append({"heading": heading, "text": cur})

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    flush()
    return [c for c in chunks if len(c["text"]) >= CHUNK_MIN_CHARS]


def index_file(path: Path, root: Path) -> int:
    """索引单个文件（先删旧块再写，保证改文档后索引跟着变）。返回入库块数。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    doc = path.relative_to(root).as_posix()
    chunks = split_markdown(text)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM kb_chunks WHERE doc = ?", (doc,))
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO kb_chunks(doc, heading, chunk_no, text, n_chars,"
                " embedding, updated_at) VALUES (?,?,?,?,?,?,?)",
                (doc, c["heading"], i, c["text"], len(c["text"]),
                 json.dumps(embed(c["text"])), now),
            )
    return len(chunks)


# 哪些 markdown 不该进知识库：
#   evidence/   —— 评测脚本自动生成的原始数据，检索命中它只会把数字当结论背出来；
#   figures/    —— 图目录，里面就算有说明文件也不该被当成文档；
#   learning-*  —— 学习笔记。那是"人对项目说的话"，不是项目文档；混进来之后
#                  search_docs 会把"我今天学到什么"当成项目交付物引用出去。
KB_SKIP_DIRS = ("evidence", "figures")
KB_SKIP_NAME_PREFIX = ("learning-",)


def should_index(p: Path, root: Path | None = None) -> bool:
    """这个 md 该不该进知识库。

    索引和"启动时判断索引完不完整"必须用同一个口径——两边算法不一样，
    篇数就永远对不上，服务每次启动都会白重建一遍（本地踩过）。

    只拿**相对路径**的目录部分做判断：项目本身要是放在 /data/evidence/ 这类
    路径下，用绝对路径判断会把整个知识库误排除。
    """
    rel = p.relative_to(root) if root else p
    return (not any(part in KB_SKIP_DIRS for part in rel.parts[:-1])
            and not p.name.startswith(KB_SKIP_NAME_PREFIX))


def index_dirs(dirs: list[str] | None = None, root: Path | None = None) -> dict:
    """索引若干目录下的 .md 文件。返回 {docs, chunks, skipped}。"""
    root = root or Path.cwd()
    dirs = dirs if dirs is not None else [
        d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
    total_docs = total_chunks = 0
    skipped: list[str] = []

    for d in dirs:
        folder = root / d
        if not folder.exists():
            skipped.append(f"{d}(目录不存在)")
            continue
        for p in sorted(folder.rglob("*.md")):
            if not should_index(p, root):  # 评测原始数据、学习笔记不进知识库
                continue
            n = index_file(p, root)
            if n:
                total_docs += 1
                total_chunks += n
            else:
                skipped.append(f"{p.relative_to(root).as_posix()}(无有效内容)")
    invalidate_version()
    return {"docs": total_docs, "chunks": total_chunks, "skipped": skipped}


# ---------------- 检索 ----------------

def _cosine(a: list[float], b: list[float]) -> float:
    # embed() 已做 L2 归一化，内积即余弦
    return sum(x * y for x, y in zip(a, b))


def _terms(text: str) -> list[str]:
    """BM25 词项：英文/数字按整词，中文按 bigram（去掉功能词）。

    中文不引入分词器（jieba 要额外依赖，且领域词还得加词典）。
    bigram 的精度够用：查"缓存阈值"切成 缓存/存阈/阈值，
    只要原文有"阈值"就能命中，代价是会产生无意义组合，靠 IDF + 停用词表压下去。
    """
    text = text.lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        if token[0].isascii():
            terms.append(token)
        elif len(token) == 1:
            terms.append(token)
        else:
            terms.extend(t for t in (token[i:i + 2] for i in range(len(token) - 1))
                         if t not in _STOP_BIGRAMS)
    return terms


def _key_terms(corpus: list[list[str]], query: list[str]) -> set[str]:
    """挑出查询里"有区分度且真实存在于语料"的词项：DF 低于查询内中位数的那些。

    这里踩过一个坑，值得记下来：最初按 IDF 取"最高的一半"当关键项，
    结果反了——IDF 最高的恰恰是**从没在语料里出现过的垃圾组合**。
    比如问"SQLiteSaver 为什么会报 readonly"，中文部分切出"么会""会报"两个
    跨词边界的碎片，它们在语料里一次都没有，IDF 冲到最高，于是被当成关键项；
    而真正该命中这条文档的 "readonly" 反而被排除，导致这条查询直接漏召。

    所以判断标准改成 DF（出现次数）而不是 IDF，并且排除 DF=0：
    - DF=0 不是"区分度高"，是"不存在"，拿它当门槛会永远不命中
    - 关键项 = 存在且相对稀有的词 = 真正能把这几个块区分开的词

    对"今天上海天气"这类：上海、海天在语料里没有，会被排除；剩下"今天""天气"
    有区分度，于是仍可能召回含"今天天气"的缓存章节——这类误召回不靠调阈值解决，
    留给提示词里的"不相关就忽略"兜底。理由：漏给资料比多给一段更糟。
    """
    if not query:
        return set()
    df = Counter()
    for d in corpus:
        for t in set(d):
            df[t] += 1
    dfs = sorted(df[t] for t in query if df[t] > 0)
    if not dfs:
        return set()
    med = dfs[len(dfs) // 2]
    return {t for t in query if 0 < df[t] <= med}


def _bm25(corpus: list[list[str]], query: list[str],
          key_terms: set[str] | None = None) -> tuple[list[float], list[int], list[int]]:
    """返回 (每个块的 BM25 分数, 命中词项数, 命中关键项数)。

    为什么要多返回两个计数（都是实测逼出来的）：
    中文 bigram 会切出跨词边界的垃圾——"推荐几本机器学习书"切成
    推荐/荐几/几本/本机/机器/器学/学习/习书，其中"本机"在项目文档里高频出现
    （"本机怎么跑最快"），于是一个问书的查询硬是召回了部署章节。分数再高也是噪声。

    所以 search() 会要求：
      - 命中词项数 >= 2：拦住靠单个巧合词项撞上来的
      - 命中关键项（高 IDF 的词）>= 1：拦住"只命中了'怎么''写一'这类泛词"的
    后者是关键：问"今天上海天气"，命中的是"天气"（文档里恰有此例），
    而"上海"这个真正有区分度的词一次都没出现 —— 这种情况就不该召回。
    """
    n = len(corpus)
    if not n:
        return [], [], []
    avgdl = sum(len(d) for d in corpus) / n or 1.0
    df = Counter()
    for d in corpus:
        for t in set(d):
            df[t] += 1

    scores: list[float] = []
    hit_counts: list[int] = []
    key_hits: list[int] = []
    for d in corpus:
        tf = Counter(d)
        s = 0.0
        hits = 0
        khits = 0
        for t in query:
            if t not in tf:
                continue
            hits += 1
            if key_terms and t in key_terms:
                khits += 1
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (tf[t] * (BM25_K1 + 1)) / (
                tf[t] + BM25_K1 * (1 - BM25_B + BM25_B * len(d) / avgdl))
        scores.append(s)
        hit_counts.append(hits)
        key_hits.append(khits)
    return scores, hit_counts, key_hits


def _load_all() -> list[dict]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT id, doc, heading, chunk_no, text, embedding FROM kb_chunks"
        ).fetchall()
    return [{"id": r["id"], "doc": r["doc"], "heading": r["heading"],
             "chunk_no": r["chunk_no"], "text": r["text"],
             "embedding": json.loads(r["embedding"])} for r in rows]


@with_retry
def search(query: str, top_k: int | None = None,
           min_score: float | None = None) -> list[dict]:
    """混合检索：BM25 主导排序，向量做小幅加权补召回。

    【这里有一处实测驱动的设计修正，值得记下来】
    初版用标准 RRF 融合（两路排名倒数相加），跑出来是错的：
    问"缓存阈值是怎么定下来的"，BM25 正确地把 §13.3 排第一（13.79 分），
    但融合后第一名变成了一块 BM25=0、只有微弱向量相似度的无关块——
    因为 RRF 假设两路都有判别力，而实际上一路是噪声时，它给噪声也发了"排名分"。

    根因：n-gram 向量对"短查询 vs 长文档块"这种长度悬殊的匹配天然吃亏。
    查询只有 8 个 bigram，文档块有几百个，归一化后点积被长块的主导维度稀释，
    分数集中在 0.25~0.5 的窄带里——相关与无关完全重叠（实测：相关 0.30~0.51，
    无关 0.27~0.40）。它没有判别力，就不该参与决定排序。

    所以改成身份明确的加权：BM25 归一化后占 85%，向量只占 15%。
    向量的作用退化为"BM25 打出平手时，谁字面更像就选谁"。
    —— 这不是最优解，是当前特征下的诚实解。bge 这类真 embedding 后端
    已在 embedder.py 落地（缓存侧已用上），等 RAG 侧标定出可靠的相关性
    区分度后，向量才有资格拿到更高权重，那时再调这个比例。

    过滤用 BM25 绝对值（min_score 参数在这里实际指 BM25 门槛），
    因为它是本模块唯一有跨查询可比性的量纲。向量低但 BM25 强 → 照样入选，
    这正是"SQLiteSaver 这类专有名词"要靠 BM25 救回来的场景。
    """
    if not query or len(query.strip()) < 2:
        return []
    top_k = top_k or settings.rag_top_k
    bm25_min = settings.rag_bm25_min if min_score is None else min_score

    rows = _load_all()
    if not rows:
        return []

    # ---- 第一路：BM25（主）----
    q_terms = _terms(query)
    corpus = [_terms(r["text"]) for r in rows]

    # 关键项 = 本次查询里 IDF 高于中位数的词项（即真正有区分度的那些）。
    # 用"本次查询内相对高低"而不是绝对阈值：查询长短差异大，绝对值不可比。
    key_terms = _key_terms(corpus, q_terms)

    bm, hits, key_hits = _bm25(corpus, q_terms, key_terms)
    bm_max = max(bm) if bm else 0.0
    if bm_max <= 0:
        # 一个词项都没命中，说明问的东西知识库里没有。此时不硬凑结果——
        # 塞几段无关资料比不塞更糟：模型会拿它们编答案。
        return []

    # 至少命中几个词项才算"真的相关"。查询本身很短（一两个词项）时没得挑，
    # 放宽到 1；否则要求 2 —— 单个巧合词项撞上来的（"几本机器"里的"本机"）拦掉。
    min_hits = 1 if len(q_terms) <= 2 else 2
    need_key = 1 if key_terms else 0

    # ---- 第二路：向量余弦（辅）----
    qv = embed(query)
    vec = [_cosine(qv, r["embedding"]) for r in rows]

    # ---- 排序：BM25 归一化 85% + 向量 15% ----
    def final_score(i: int) -> float:
        b = bm[i] / bm_max                       # 0~1，本查询内相对强度
        v = max(0.0, (vec[i] - 0.2) / 0.8)       # 0.2 以下视为无信号，压到 0
        return W_BM25 * b + W_VEC * v

    order = sorted(range(len(rows)), key=final_score, reverse=True)

    out: list[dict] = []
    top_score = 0.0
    for idx in order:
        b, v, h = bm[idx], vec[idx], hits[idx]
        if h < min_hits or key_hits[idx] < need_key:
            continue        # 靠单个巧合词项、或只命中泛词撞上来的，跳过
        # 门槛：BM25 达标的直接进；差一点的，如果向量也不弱就放行（补召回）
        if not (b >= bm25_min or (b >= bm25_min * 0.6 and v >= 0.5)):
            break                                 # 已按分数排序，后面只会更低
        fs = final_score(idx)
        # 相对截断：和第一名差太远的不要。固定 top_k 不管检索质量一律塞满，
        # 第一名 0.87、第三名 0.43 时后面几条只是稀释模型注意力。
        if out and fs < top_score * settings.rag_rel_cutoff:
            break
        top_score = top_score or fs
        r = rows[idx]
        out.append({
            "doc": r["doc"],
            "heading": r["heading"],
            "chunk_no": r["chunk_no"],
            "text": r["text"],
            "score": round(fs, 3),                 # 展示分：和排序一致，第一名为 1.0
            "bm25": round(b, 2),
            "vec": round(v, 3),
            "hits": h,
        })
        if len(out) >= top_k:
            break
    return out


def stats() -> dict:
    try:
        with connect(settings.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) n, COUNT(DISTINCT doc) d,"
                " COALESCE(SUM(n_chars),0) chars, MAX(updated_at) ts FROM kb_chunks"
            ).fetchone()
    except sqlite3.OperationalError:
        return {"docs": 0, "chunks": 0, "chars": 0, "updated_at": ""}
    return {"docs": row["d"], "chunks": row["n"],
            "chars": row["chars"], "updated_at": row["ts"] or ""}


def stale(dirs: list[str] | None = None, root: Path | None = None) -> bool:
    """磁盘上的文档比索引新吗（按修改时间比）。

    为什么需要这个判断：启动时只比"篇数"测不出内容变化——文档改一段、篇数还是 4，
    索引就一直是旧的。线上实际发生过：10-27 建的索引里是 58k 字，文档后来写到 78k 字，
    检索永远命中旧快照，答案引用的段落可能早就删了。篇数对、块数对、只有内容变了，
    这是最难发现的一种"索引假健康"。
    """
    root = root or Path.cwd()
    dirs = dirs if dirs is not None else [
        d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
    newest = 0.0
    for d in dirs:
        folder = root / d
        if not folder.exists():
            continue
        for p in folder.rglob("*.md"):
            if should_index(p, root):
                try:
                    newest = max(newest, p.stat().st_mtime)
                except OSError:
                    continue
    if not newest:
        return False
    ts = stats().get("updated_at") or ""
    if not ts:
        return True
    try:
        indexed = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return True
    # 1 秒容差：索引写入和文件落盘可能在同一秒内，不该判成过期
    return newest > indexed + 1


# ---------------- 知识库版本（缓存复用要用） ----------------

_ver_lock = threading.Lock()
_ver_cache: tuple[str, float] | None = None
_VER_TTL = 5.0        # 秒。版本号查库要一次 SQL，5 秒内复用结果


def version() -> str:
    """知识库版本：篇数 / 块数 / 最后写入时间。

    为什么语义缓存需要它：
        原来的规则是"检索命中过的问题一律不写缓存"，理由是"答案依赖文档内容，
        改了文档缓存就过期"。这个顾虑是对的，但**做法太粗**——结果是只要有知识库，
        缓存就永远不写（线上 entries 一直是 0）。正确做法是记下生成答案时的知识库
        版本，命中时比对：一样就复用，不一样就自动失效。粒度从"有没有用检索"变成
        "文档有没有变过"，缓存才真正工作起来。

    5 秒 TTL 的理由：这个函数在每次缓存查找/写入时都会调用，不能每次都查库；
    而重建索引通常持续数秒，5 秒的缓存窗口既够快，也不会让新版本迟迟不生效。
    """
    global _ver_cache
    now = time.monotonic()
    if _ver_cache is not None and _ver_cache[1] > now:
        return _ver_cache[0]
    st = stats()
    ver = f"{st['docs']}:{st['chunks']}:{st['updated_at']}"
    with _ver_lock:
        _ver_cache = (ver, now + _VER_TTL)
    return ver


def invalidate_version() -> None:
    """重建索引后立刻作废版本缓存，让新版本马上生效。"""
    global _ver_cache
    with _ver_lock:
        _ver_cache = None
