"""难度路由：决定一个请求走本地小模型还是云端大模型。

三层设计（同一份代码里依次生效）：
1. 规则层：0 成本，处理明显 case（代码块、数学符号、超长文本、纯寒暄）。
2. 模型层：本地小模型做一次极短分类（max_tokens=48），只在规则不确定时触发。
3. 兜底层：分类失败或未开启模型路由时，按长度阈值决定。

为什么要有规则层：路由本身也是一次推理，规则能拦掉约一半请求的路由开销。
"""

import json
import re
import time
from dataclasses import dataclass

from .clients import ProviderError, call_local
from .config import settings

HARD_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"import\s+\w+",
    r"证明|推导|反证|复杂度|时间复杂度|递推",
    r"调试|报错|异常栈|堆栈|bug|debug",
    r"重构|优化到|性能瓶颈|死锁|并发安全",
    r"设计一个|实现一个|写一个|写段代码|写个",
    r"分析.{0,8}(原因|原理|机制)",
    r"对比.{0,10}与.{0,10}的",
    r"[∑∫√≤≥±∞]|O\(n",
]

HARD_RE = re.compile("|".join(HARD_PATTERNS), re.IGNORECASE)
GREETING_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|在吗|谢谢|多谢|早上好|晚上好|哈喽)[!！。~～\s]*$",
    re.IGNORECASE,
)

ROUTER_PROMPT = """你是一个请求路由器。判断下面这个请求需要多强的模型。

判定标准：
- hard：需要多步推理、写代码、数学证明、长文分析、专业深度知识
- easy：闲聊、单点事实、简单改写、短句翻译、格式转换

只输出一行 JSON，不要解释：
{"difficulty":"easy","reason":"不超过15字"}
或
{"difficulty":"hard","reason":"不超过15字"}

请求："""




@dataclass
class Route:
    difficulty: str  # "easy" | "hard"
    reason: str
    by: str  # "rule" | "model" | "default"
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


def _rule_route(query: str) -> Route | None:
    text = query.strip()
    if not text:
        return Route("easy", "空输入", "rule")

    if GREETING_RE.match(text):
        return Route("easy", "纯寒暄", "rule")

    if len(text) > settings.long_query_chars:
        return Route("hard", f"文本超长({len(text)}字)", "rule")

    hit = HARD_RE.search(text)
    if hit:
        return Route("hard", f"命中强特征:{hit.group(0)[:12]}", "rule")

    if len(text) <= 20:
        return Route("easy", "短句无强特征", "rule")

    return None


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _model_route(query: str) -> Route | None:
    started = time.perf_counter()
    try:
        text, tin, tout = call_local(
            [{"role": "user", "content": ROUTER_PROMPT + query[:800]}],
            max_tokens=settings.router_max_tokens,
        )
    except ProviderError:
        return None

    elapsed = int((time.perf_counter() - started) * 1000)
    data = _parse_json(text)
    if not data:
        return None

    difficulty = str(data.get("difficulty", "")).lower()
    if difficulty not in ("easy", "hard"):
        return None

    reason = str(data.get("reason", ""))[:20] or "模型判定"
    return Route(difficulty, reason, "model", elapsed, tin, tout)


def decide(query: str) -> Route:
    route = _rule_route(query)
    if route is not None:
        return route

    if settings.enable_router_model:
        route = _model_route(query)
        if route is not None:
            return route

    # 兜底：长度阈值
    if len(query) < 120:
        return Route("easy", "兜底:短", "default")
    return Route("hard", "兜底:长", "default")


# ---------------------------------------------------------------- 要不要动手

# 关键词判据：这些说法意味着"光靠资料答不了，得去算/查/跑/抓"。
# 为什么不交给模型判：这是一次额外推理，而工具判断错了代价很高
# （该调的时候不调＝瞎编；不该调的时候调＝白烧钱）。规则层 0 成本、可解释、
# 出错能一眼改；等这套跑稳了再考虑用模型兜住长尾。
TOOL_HINTS = [
    (re.compile(r"https?://", re.I), "含链接"),
    (re.compile(r"算一下|算算|计算|求值|换算"), "要计算"),
    (re.compile(r"统计|多少条|几条|一共多少|合计|平均"), "要统计"),
    (re.compile(r"数据库|sql|查一下表|记账表", re.I), "要查库"),
    (re.compile(r"跑一下|执行|运行一下|写段代码|写个脚本|验证一下"), "要跑代码"),
    (re.compile(r"查一下|查查|搜一下|检索一下|翻一下文档"), "要查资料"),
    (re.compile(r"抓一下|抓取|爬一下|打开这个?链接|网页内容"), "要抓网页"),
]


def needs_tools(query: str) -> tuple[bool, str]:
    """返回 (是否需要工具, 理由)。理由会显示在页面上，方便解释"这轮为什么动了手"。"""
    text = (query or "").strip()
    if not text:
        return False, ""
    for pattern, reason in TOOL_HINTS:
        m = pattern.search(text)
        if m:
            return True, f"{reason}:{m.group(0)[:12]}"
    return False, ""
