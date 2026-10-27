"""审查员（critic）：让另一个角色专门挑错，答错了能被打回返工。

为什么要拆成两个角色：同一个模型"自己检查自己"效果很差——它刚写完答案，
再问它"对不对"，它倾向于说"对"（自洽偏误）。换一个**只负责挑错、不负责答**的角色，
提示词完全不同（不给它"要给出好答案"的压力），才挑得出来。

两层审查，成本差一个数量级：

1. **确定性数字核对（零成本）**：答案里的数字必须能在证据里找到。
   这一层不需要模型，纯字符串比对，抓的是"凭空冒出来的数字"——最常见的幻觉形态。
2. **LLM 审查（一次调用）**：结论有没有依据、有没有和资料矛盾、是否答非所问。

第一层先跑：数字对不上就直接打回，省掉那次 LLM 调用。第二层只处理"数字没问题但说法可疑"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .clients import call_cloud

CRITIC_SYSTEM = """你是审查员。你不重写答案，只做判断。
检查四件事：
1. 答案里的关键结论是否被【资料】支持；没有依据的断言要指出来。
2. 有没有和【资料】矛盾的地方（数字、版本、结论）。
3. 是否答非所问（问 A 答 B，或者回避了问题的核心）。
4. 【资料】为空时，只检查答案是否有明显事实错误，不要因为"没资料"就一律打回。
5. 【资料】标注"被截断"时，它是不完整的：**不能**因为"资料里查不到这个数字"就
   断言答案编造。只有和资料**直接矛盾**时才算错。

规则里最重要的一条：**资料里没有的，就属于"没有依据"，不要替它脑补成对的。**

只输出一个 JSON 对象，不要任何解释文字：
{"pass": true 或 false, "score": 0-100 的整数, "issues": ["问题1", "问题2"], "fix_hint": "一句话说怎么改"}
评分口径：有事实错误或凭空数字 → 40 以下；有未被支持的断言 → 60 上下；完全有据且答到点上 → 80 以上。
"""

REVISE_SYSTEM = """你是修订者。根据审查意见修改答案。
硬性要求：
1. **只输出答案正文**。不要复述问题、不要复述原答案、不要写"审查意见"或"修改说明"，
   不要加【】小标题，不要解释你改了什么。
2. 只改有问题的地方，正确的部分保持原样，篇幅与原文相当。
3. 没有依据的断言，要么删掉，要么明确说"资料里没有相关信息"。
4. 不要为了通过审查而编造依据——审查员会再查一次。
"""


@dataclass
class Review:
    """一次审查的结论。"""

    passed: bool
    score: int
    issues: list[str] = field(default_factory=list)
    hint: str = ""
    tin: int = 0
    tout: int = 0
    by: str = "llm"          # 谁判的：rule（数字核对）/llm/error（审查员自己挂了，放行）
    skipped: bool = False

    def brief(self) -> dict:
        return {"pass": self.passed, "score": self.score, "issues": self.issues[:5],
                "hint": self.hint[:120], "by": self.by, "skipped": self.skipped}


# ---------------- 第一层：确定性数字核对 ----------------

# 抓数字：允许千分位和百分号，避免把 "1,048,576" 切成两截
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def _norm(raw: str) -> str:
    """归一化：去千分位、去百分号、去掉尾随的 .0。"""
    s = raw.replace(",", "").rstrip("%")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_of(text: str) -> set[str]:
    return {_norm(m) for m in _NUM.findall(text or "")}


def grounding_issues(answer: str, evidence: str, query: str = "",
                     truncated: bool = False) -> list[str]:
    """答案里的数字，必须能和证据（资料 + 工具结果 + 记忆）或问题对上。

    为什么不能做严格的字符串比对：答案里大量数字是**派生**的——
    证据给 `0.009561`，答案写 `0.01`（四舍五入），证据给 `16`，答案写 `16 条`（带单位）。
    严格比对会把这类正确结论一律判成"编的"。实测中就因此把一条正确的 SQL 聚合答案打回了。

    所以判据是"能不能对上"，按宽松程度依次是：
      1. 直接出现（归一化后相等，千分位/百分号不影响）
      2. 是证据里某个数的四舍五入（按答案自身的小数位比）
      3. 与证据里某个数相差 2% 以内（换行、近似值、百分比换算）
    这三个都不满足，才算"找不到出处"。

    再漏掉三类不该管的：单个数位（"第 3 步"）、年份、问题里本来就有的数字。
    """
    if truncated:
        return []                      # 证据不完整，没资格判"找不到出处"
    if not (evidence or "").strip():
        return []                      # 没有资料可比，交给 LLM 那层

    allowed = numbers_of(evidence) | numbers_of(query)
    ev_floats: list[float] = []
    for raw in _NUM.findall(evidence or "") + _NUM.findall(query or ""):
        try:
            ev_floats.append(float(_norm(raw)))
        except ValueError:
            continue

    out: list[str] = []
    for raw in _NUM.findall(answer or ""):
        v = _norm(raw)
        if v in allowed:
            continue
        digits = v.replace(".", "")
        if len(digits) < 2:
            continue                   # 序数、单个数位：不足以判定
        try:
            f = float(v)
        except ValueError:
            continue
        if "." not in v and 1900 <= f <= 2100:
            continue                   # 年份

        dec = len(v.split(".")[1]) if "." in v else 0
        ok = False
        for e in ev_floats:
            if round(e, dec) == f:                     # 四舍五入能对上
                ok = True
                break
            if abs(f - e) <= max(abs(e) * 0.02, 1e-9):  # 近似值（含百分比换算）
                ok = True
                break
        if not ok:
            out.append(f"数字 {raw} 在资料里找不到出处")
        if len(out) >= 3:
            break
    return out


# ---------------- 第二层：LLM 审查 ----------------


def _parse_json_obj(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):         # 有的模型会把 JSON 包在代码块里
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            val = json.loads(text[start:end + 1])
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def build_prompt(query: str, answer: str, evidence: str) -> list[dict]:
    ev = (evidence or "").strip() or "（本轮没有检索资料，也没有调用工具：只检查明显的事实错误）"
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"【问题】\n{query}\n\n【资料】\n{ev[:4000]}\n\n【待审答案】\n{answer}"},
    ]


def review(query: str, answer: str, evidence: str = "", *, min_score: int = 70,
           truncated: bool = False, model_fn=None) -> Review:
    """审查一份答案。**不抛异常**：审查员自己出错时放行，不能因为它挂掉而卡住主流程。"""
    rule_issues = grounding_issues(answer, evidence, query, truncated)
    if rule_issues:
        # 数字都对不上，没必要再花一次模型调用
        return Review(passed=False, score=35, issues=rule_issues,
                      hint="删掉没有出处的数字，或说明这是估算", by="rule")

    if not (answer or "").strip():
        return Review(passed=False, score=0, issues=["空答案"], hint="重新作答",
                      by="rule")

    fn = model_fn or call_cloud
    ev_for_prompt = evidence
    if truncated:
        ev_for_prompt = "（注意：以下资料被截断，只保留了头尾，中间内容缺失）\n" + evidence
    try:
        content, tin, tout = fn(build_prompt(query, answer, ev_for_prompt), 300)
    except Exception as exc:                       # noqa: BLE001 审查员不能拖垮主链路
        return Review(passed=True, score=0, issues=[], hint="", by="error")

    obj = _parse_json_obj(content)
    if obj is None:
        # 解析不出来就是审查员自己没干好：放行并标 by=error，别把好答案毙掉
        return Review(passed=True, score=0, issues=[], hint="", tin=tin, tout=tout, by="error")

    try:
        score = int(float(obj.get("score", 0)))
    except (TypeError, ValueError):
        score = 0
    issues = [str(x)[:160] for x in (obj.get("issues") or []) if str(x).strip()]
    passed = bool(obj.get("pass")) and score >= min_score and not issues
    return Review(passed=passed, score=score, issues=issues,
                  hint=str(obj.get("fix_hint") or "")[:200], tin=tin, tout=tout)


def build_revise_messages(query: str, answer: str, rev: Review,
                          evidence: str = "") -> list[dict]:
    """返工的输入：原问题 + 原答案 + 审查意见 + 证据。

    为什么要连证据一起给：只说"这句没依据"，模型会换个说法继续编；
    把资料摊在它面前，它才有东西可依。
    """
    issues = "\n".join(f"- {x}" for x in rev.issues) or "- （未说明具体问题）"
    ev = (evidence or "").strip()
    # 用"任务式"叙述而不是【小标题】堆叠：小标题会让模型顺着抄格式，
    # 实测返工答案直接以"【原问题】…【原答案】…"开头，等于把提示词泄漏给用户。
    body = (f"用户的问题是：{query}\n\n"
            f"现在的答案：\n{answer}\n\n"
            f"审查员指出的问题：\n{issues}\n"
            f"修改方向：{rev.hint or '按审查意见修正'}\n\n"
            f"请直接输出修改后的答案正文（不要复述以上任何内容）。")
    if ev:
        body += f"\n\n可供引用的资料：\n{ev[:3000]}"
    return [{"role": "system", "content": REVISE_SYSTEM},
            {"role": "user", "content": body}]
