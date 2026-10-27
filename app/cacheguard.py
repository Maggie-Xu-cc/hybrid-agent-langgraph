"""缓存守卫：句向量负责「像不像」，这里负责「能不能换」。

为什么必须有这一层
------------------
换成 BGE 之后，同义改写的召回从 44% 提到 86%（「您好呀」终于能认成「你好」），
但误命中没解决：最优阈值 0.70 下 precision 只有 66%。剩下的错误**全都长一个样**：

    把 3 点 45 分转成 24 小时制   ←→   把 3 点 15 分转成 24 小时制   0.975
    HTTP 403 是什么意思          ←→   HTTP 404 是什么意思           0.944
    「今天天气不好」翻成英文       ←→   「今天天气不错」翻成英文        0.946
    改成全小写：HELLO WORLD      ←→   改成全大写：hello world        0.966
    TCP 和 HTTP 的主要区别        ←→   TCP 和 UDP 的主要区别          0.911

这些句子的**语义完全一样**，只有一个词不同——而那个词恰好是整句话唯一的有效信息。
任何纯句向量都会给它们 0.9 以上的分，因为句向量编码的是"这句话在讲什么"，
不是"这句话的参数是什么"。这不是模型不够大，是表示方式的边界：
把 0.975 和 0.944 分开，需要的是逐字比对，不是语义理解。

论文里也是这么处理（vCache 的结论是静态阈值无正确性保证；工业界的银行案例要
叠到第六层才把误命中压到 5% 以下）。所以这一层不是补丁，是第二级过滤器：

    第一级  句向量粗筛（BGE，召回高，允许宽）
    第二级  载荷一致性校验（本模块，精确，只放行"参数也一致"的候选）

校验四个维度，任一不符就拒绝复用
--------------------------------
1. 数字     3点45 ≠ 3点15、404 ≠ 403。中文数字先转阿拉伯再比，所以
            「三点十五分」和「3 点 15 分」算同一组。
2. 大写缩写 TCP / HTTP / REST / SOAP 这类 2~6 位大写词，集合必须相同。
3. 拉丁词   hello world ≠ hello python。单位、寒暄等非载荷词在白名单里跳过，
            避免「1km 等于多少 m」被误判。
4. 变体组   同一维度的互斥说法（大写/小写、英文/日文/韩文、正式/口语、
            区别/相同点、今天/明天/后天）必须落在同一档；同义词先归一，
            所以「书面化」和「正式」算一档，不会误伤。
5. 极性     不好 ≠ 不错。

设计原则与缓存本身一致：**宁可少命中，不可答错**。守卫误拒的代价是退回去调一次
模型（多花一份钱）；守卫漏放的代价是用户拿着错的答案去办事。
"""

from __future__ import annotations

import re

_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_CN_NUM_RE = re.compile(r"[零〇一二三四五六七八九十百千万两]+")
_ASCII_NUM_RE = re.compile(r"\d+")
_LATIN_RE = re.compile(r"[a-zA-Z]{2,}")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

# 非载荷的拉丁词：单位、寒暄、常见虚词。比对它们只会造成误拒。
_LATIN_STOP = {
    "hi", "ok",            # 寒暄
    "km", "cm", "mm", "kg", "mg", "mb", "gb", "tb", "kb",
    "am", "pm", "vs", "and", "the", "or", "to", "in", "of", "is", "it",
}
# hello / world 故意不放进白名单：它们在"改成全大写：hello world"这类任务里就是载荷，
# 放行会让 hello world 和 hello python 分不开。

# 同一维度的互斥说法，同义词先归一到 canonical
_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "case": {"upper": ("大写",), "lower": ("小写",)},
    "lang": {"en": ("英文", "英语"), "ja": ("日文", "日语"),
             "ko": ("韩文", "韩语"), "zh": ("中文", "汉语")},
    "tone": {"formal": ("正式", "书面"), "casual": ("口语", "随意", "通俗")},
    "cmp": {"diff": ("区别", "差异", "不同", "差别", "差在哪", "不一样"),
            "same": ("相同点", "共同点", "一样", "相同")},
    "day": {"today": ("今天",), "tomorrow": ("明天",),
            "dayafter": ("后天",), "yesterday": ("昨天",), "before": ("前天",)},
}

_POS = ("不错", "很好", "优秀", "正常", "成功")
_NEG = ("不好", "糟糕", "很差", "有问题", "报错")

# 单位要单独比：1 公里 和 1 英里 数字一样、答案不同
_UNIT_ALIAS = {
    "km": ("公里", "千米"), "m": ("米",), "mile": ("英里",), "chi": ("尺",),
    "cm": ("厘米",), "mm": ("毫米",), "kg": ("公斤", "千克"), "g": ("克",),
    "hour": ("小时", "点钟"), "minute": ("分钟", "分"),
}

# 「一下」「一句」里的「一」是量词不是数量，不清理会把寒暄全判成含数字 1
_EMPTY_YI = re.compile(r"(一下|一句|一些|一样|一起|一点|一般|一定|一直|一共|一切|一边|一番)")


def _cn_to_int(s: str) -> int:
    """中文数字转整数。「十五」→15，「一百零五」→105，「三」→3。"""
    total = section = number = 0
    for ch in s:
        if ch in _DIGITS:
            number = _DIGITS[ch]
        elif ch in _UNITS:
            unit = _UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
    return total + section + number


def _numbers(text: str) -> frozenset[str]:
    """抽出全部数字（中文先转阿拉伯，两边口径一致）。"""
    text = _EMPTY_YI.sub("", text)
    text = _CN_NUM_RE.sub(lambda m: str(_cn_to_int(m.group())), text)
    return frozenset(_ASCII_NUM_RE.findall(text))


def _acronyms(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _ACRONYM_RE.findall(text)
                     if w.lower() not in _LATIN_STOP)


def _latin(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _LATIN_RE.findall(text)
                     if w.lower() not in _LATIN_STOP)


def _units(text: str) -> frozenset[str]:
    return frozenset(canon for canon, words in _UNIT_ALIAS.items()
                     if any(w in text for w in words))


def _variant(text: str, dim: str) -> frozenset[str]:
    return frozenset(k for k, words in _GROUPS[dim].items()
                     if any(w in text for w in words))


def _polarity(text: str) -> str:
    """先判负面（"不好"必须优先于"好"），再判正面，都没有返回空。"""
    if any(w in text for w in _NEG):
        return "neg"
    if any(w in text for w in _POS):
        return "pos"
    return ""


def check(query: str, cached: str) -> tuple[bool, str]:
    """返回 (能否复用, 原因)。原因用于页面展示与调试——拒绝也要讲得清。"""
    nq, nc = _numbers(query), _numbers(cached)
    if nq != nc:
        # 「公里和米怎么换算」对「1 公里等于多少米」：一边不写数量、另一边写 1，
        # 是同一个问题的两种说法，不该判成冲突。
        loose = (not nq or not nc) and (nq <= {"1"} and nc <= {"1"})
        if not loose:
            return False, f"数字不同：{sorted(nq) or '无'} vs {sorted(nc) or '无'}"

    uq, uc = _units(query), _units(cached)
    # 查询可以少说单位（「3:15 PM 换 24 小时制」没说分钟），但不能说了别的单位
    if uq and uc and not (uq <= uc or uc <= uq):
        return False, f"单位不同：{sorted(uq)} vs {sorted(uc)}"

    # 缩写与拉丁词用「两边都有才比」：一边没写不代表主题不同
    # （问「404 状态码」的人不会说 HTTP，但问「SOAP API」的人一定写了 SOAP）
    aq, ac = _acronyms(query), _acronyms(cached)
    if aq and ac and aq != ac:
        return False, f"缩写不同：{sorted(aq)} vs {sorted(ac)}"

    lq, lc = _latin(query), _latin(cached)
    if lq and lc and lq != lc:
        return False, f"关键词不同：{sorted(lq)} vs {sorted(lc)}"

    for dim in _GROUPS:
        vq, vc = _variant(query, dim), _variant(cached, dim)
        # 一句话里同时出现两个档位（"这句太随意了，改成正式一点"）说明在描述转换关系，
        # 这个维度就失去了判别力，跳过而不是误拒。
        if len(vq) > 1 or len(vc) > 1:
            continue
        if vq != vc:
            return False, f"说法冲突：{'/'.join(sorted(vq)) or '无'} vs {'/'.join(sorted(vc)) or '无'}"

    pq, pc = _polarity(query), _polarity(cached)
    if pq != pc:
        return False, f"褒贬相反：{pq or '中性'} vs {pc or '中性'}"

    # 短缓存的答案撑不起长问题："谢谢！" 不该接住 "谢谢，但是我要申请退货"，
    # "你是谁？" 不该接住 "你支持多长的上下文"
    if len(cached) <= 6 and len(query) > max(8, 2.2 * len(cached)):
        return False, "问题比缓存条目长太多，不像同一件事"

    return True, "载荷一致"
