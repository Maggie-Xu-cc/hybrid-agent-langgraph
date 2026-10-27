"""文本向量化：语义缓存和阈值实验共用的那把「尺子」。

两个后端，一条接口
------------------
  ngram（默认，零依赖）  字符 n-gram hashing，256 维，纯 Python，毫秒级
  bge（本地小模型）      BGE-small-zh-v1.5，512 维，ONNX + CPU，零 API 成本

为什么要换后者
--------------
以前用 n-gram，理由是"缓存本身是为了省钱，不该再调一次 embedding API"。
这个理由只对了一半：**零的是 API 成本，不是零模型**。本地跑一个 100MB 的小模型，
一样一分钱不花，10~30ms 一条。

而 n-gram 的代价是硬的。用 89 条标注集实测（docs/evidence/threshold_sweep.md）：

  该命中的同义改问法  中位相似度 0.437    ← "您好呀" vs "你好" 只有 0.11
  必须拒绝的近似问题  中位相似度 0.746    ← "3点45" vs "3点15" 高达 0.906

分数和"该不该复用答案"是反着的。因为 n-gram 量的是字面重合度，而判断该不该复用
看的是语义；`3点15`→`3点45` 只差一个字符却差了一小时，`您好呀`→`你好` 几乎不重合
却是同一句话。**换刻度救不了，只能换尺子**——这也是 GPTCache / MeanCache / vCache
这些工作无一例外都建在真 embedding 上的原因。

降级策略
--------
想用 bge 但装不上（没 onnxruntime、模型文件不在、内存不够）时，自动退回 ngram
并打印一次警告。缓存是锦上添花的组件，不能因为它拖垮主流程。

切换后端有个副作用必须知道：向量维度变了（256 → 512），旧条目不能拿来比。
所以缓存表里存了 embed_model 字段，查的时候只比同后端的条目——切一次后端，
旧缓存自动失效，不会算出毫无意义的余弦值。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from pathlib import Path

from .config import settings

_ROOT = Path(__file__).resolve().parents[1]

# n-gram 混合权重：trigram 区分度高为主，uni/bigram 保证短句也能提出特征
_NGRAM_WEIGHTS = ((1, 0.3), (2, 0.5), (3, 1.0))
_NGRAM_DIM = 256
_WS_RE = re.compile(r"\s+")

_lock = threading.Lock()
_active: str | None = None
_bge: tuple | None = None
_bge_error = ""


def _normalize(text: str) -> str:
    """去空白、转小写。让「你好」和「你 好」落在同一个 hash 上。"""
    return _WS_RE.sub("", text).lower()


# ---------------------------------------------------------------- n-gram 后端

def _ngram_embed(text: str) -> list[float]:
    """字符 n-gram hashing 到固定维度，L2 归一化后内积即余弦。"""
    text = _normalize(text)
    vec = [0.0] * _NGRAM_DIM
    if not text:
        return vec
    for n, weight in _NGRAM_WEIGHTS:
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "big")
            vec[h % _NGRAM_DIM] += weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


# ---------------------------------------------------------------- BGE 后端

def _model_dir() -> Path:
    p = Path(settings.bge_model_dir)
    return p if p.is_absolute() else _ROOT / p


def _load_bge() -> tuple | None:
    """懒加载 ONNX 会话 + 分词器。失败返回 None，由调用方降级。"""
    global _bge, _bge_error
    if _bge is not None:
        return _bge

    with _lock:
        if _bge is not None:
            return _bge
        d = _model_dir()
        model_file, tok_file = d / "model.onnx", d / "tokenizer.json"
        if not model_file.exists() or not tok_file.exists():
            _bge_error = f"模型文件缺失：{d}（需要 model.onnx 与 tokenizer.json）"
            return None
        try:
            import numpy as np                       # onnxruntime 自带依赖
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            # 不抢满 CPU：服务器还要同时跑 Web 和 SQLite，
            # 而且缓存判断是"顺路做的事"，不该把核吃光。
            so.intra_op_num_threads = max(1, settings.bge_threads)
            so.inter_op_num_threads = 1
            so.log_severity_level = 3                # 只报错，别刷警告
            sess = ort.InferenceSession(str(model_file), so,
                                        providers=["CPUExecutionProvider"])

            tok = Tokenizer.from_file(str(tok_file))
            tok.enable_truncation(max_length=settings.bge_max_len)

            _bge = (sess, tok, np, [i.name for i in sess.get_inputs()])
            return _bge
        except Exception as exc:                      # 装不上、版本不匹配、内存不足
            _bge_error = f"{type(exc).__name__}: {exc}"
            return None


def _bge_embed(texts: list[str]) -> list[list[float]]:
    sess, tok, np, input_names = _bge
    # 必须先转小写。BGE-small-zh 的 BertNormalizer 是 lowercase=False，而词表里
    # 只收了小写英文（`http` 在表里，`HTTP` 不在）。不转的话所有大写技术词——
    # TCP / UDP / HTTP / REST / SOAP——全部落到 [UNK]，于是
    # 「TCP 和 HTTP 的区别」与「TCP 和 UDP 的区别」编码出完全相同的向量（余弦 1.000）。
    # 实测：不转小写时文本无关但同模板的样本分不开；转了之后能分开。
    # 中文不受影响（lower 对汉字是恒等变换）。
    texts = [t.lower() for t in texts]
    encs = tok.encode_batch(texts)
    maxlen = max(len(e.ids) for e in encs) or 1

    ids = np.zeros((len(encs), maxlen), dtype=np.int64)
    mask = np.zeros((len(encs), maxlen), dtype=np.int64)
    for i, e in enumerate(encs):
        n = len(e.ids)
        ids[i, :n] = e.ids
        mask[i, :n] = e.attention_mask

    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(ids)

    hidden = sess.run(None, feeds)[0]                 # (batch, seq, 512)
    # BGE 系列用 CLS pooling（取第 0 个 token），不是对全部 token 取平均
    vecs = hidden[:, 0, :]
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norm, 1e-12)
    return vecs.tolist()


# ---------------------------------------------------------------- 对外接口

def _requested() -> str:
    return (settings.embed_backend or "auto").strip().lower()


def _model_present() -> bool:
    d = _model_dir()
    return (d / "model.onnx").exists() and (d / "tokenizer.json").exists()


def active() -> str:
    """实际生效的后端名：'bge' 或 'ngram'。

    三档语义：auto 看模型文件在不在；bge 强制（失败降级）；ngram 强制。
    结果缓存，避免每次 embed 都去查文件。
    """
    global _active
    if _active is None:
        mode = _requested()
        if mode == "ngram":
            _active = "ngram"
        elif mode in ("bge", "bge-small-zh", "onnx"):
            _active = "bge" if _load_bge() is not None else "ngram"
            if _active == "ngram":
                print(f"[embedder] BGE 加载失败，降级为 n-gram：{_bge_error}")
        else:                                        # auto
            if not _model_present():
                _active = "ngram"
            elif _load_bge() is not None:
                _active = "bge"
            else:
                print(f"[embedder] BGE 加载失败，降级为 n-gram：{_bge_error}")
                _active = "ngram"
    return _active


def name() -> str:
    """写进缓存条目的标识。维度不同的向量永远不该互相比较。"""
    return "bge-small-zh-v1.5" if active() == "bge" else "ngram-256"


def force(backend: str) -> str:
    """实验用：在同一次进程里切换后端，绕开配置（配置在 import 时就固化了）。

    用途是 tools/threshold_sweep.py 拿同一批样本跑两条曲线做对照——
    换尺子前后必须用同一套题，否则数字没法比。返回实际生效的后端名。
    """
    global _active
    if backend == "ngram":
        _active = "ngram"
    else:
        _active = "bge" if _load_bge() is not None else "ngram"
    return _active


def dim() -> int:
    return 512 if active() == "bge" else _NGRAM_DIM


def embed(text: str) -> list[float]:
    if active() == "bge":
        return _bge_embed([text])[0]
    return _ngram_embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量编码。建索引/跑评测时比分批单条快一个数量级。"""
    if not texts:
        return []
    if active() == "bge":
        return _bge_embed(texts)
    return [_ngram_embed(t) for t in texts]
