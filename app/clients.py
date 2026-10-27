"""统一的大模型客户端。

关键设计：本地 vLLM 与云端 API 都是 OpenAI 兼容协议，所以只是 base_url 不同。
这样"路由"退化成一次 endpoint 切换，不需要两套调用逻辑。
"""

from typing import List, Tuple

import threading
import time

try:
    from openai import OpenAI
except ImportError:  # AUP 等环境不一定装了 openai 包；标准库工具链(*_stdlib.py)不经过这里
    OpenAI = None

from .config import settings

if OpenAI is None:
    local_client = cloud_client = None
else:
    local_client = OpenAI(
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
        # 不给 timeout 时 SDK 默认 600 秒：本地端口通但不响应（Ollama 半死、
        # vLLM 还在加载权重）就会每条请求白等十分钟
        timeout=settings.local_timeout,
        max_retries=0,   # 本地挂了重试没意义，只会把失败代价放大 3 倍
    )
    cloud_client = OpenAI(
        base_url=settings.cloud_base_url,
        api_key=settings.cloud_api_key or "EMPTY",
        timeout=settings.cloud_timeout,
        max_retries=settings.cloud_max_retries,   # SDK 默认 2，偶发 429 才值得重试
    )


class ProviderError(RuntimeError):
    pass


class LocalBreaker:
    """本地模型熔断器（三态：闭合 → 打开 → 半开）。

    为什么需要：本地模型不可用时，每条请求仍会白试一次本地（路由一次 + 生成一次）。
    连接被拒是毫秒级的无所谓，但"端口通、迟迟不回"（半死/OOM/加载中）会让
    每条请求各白等一个 timeout——10 条请求就是几分钟，看着就像卡死。

    连续失败 threshold 次 → 打开，冷却 cooldown 秒内**不发任何网络请求**，
    call_local 立刻抛 ProviderError 让上层走降级；
    冷却到期放一次试探（半开），成功复位、失败继续打开。
    """

    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold = max(1, threshold)
        self.cooldown = max(0.0, cooldown)
        self._fails = 0
        self._open_until = 0.0
        self._probing = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """当前是否允许发起本地调用（冷却到期时放一条试探）。"""
        if settings.disable_local:
            return False
        with self._lock:
            if self._open_until == 0.0:
                return True
            if time.monotonic() >= self._open_until:
                self._probing = True   # 半开：只放行，结果由 ok()/fail() 决定
                return True
            return False

    def ok(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0
            self._probing = False

    def fail(self) -> None:
        with self._lock:
            self._fails += 1
            if self._probing or self._fails >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown
                self._fails = 0
                self._probing = False

    def state(self) -> dict:
        with self._lock:
            left = max(0.0, self._open_until - time.monotonic())
            return {
                "open": left > 0,
                "cooldown_left_s": round(left, 1),
                "fails": self._fails,
                "disabled": settings.disable_local,
            }


local_breaker = LocalBreaker(
    settings.local_breaker_threshold, settings.local_breaker_cooldown
)


def _mock_chat(messages: List[dict], model: str, max_tokens: int) -> Tuple[str, int, int]:
    """离线假模型：没有 GPU、没有 API Key 时也能跑通全流程与页面联调。"""
    prompt = messages[-1]["content"] if messages else ""

    if "请求路由器" in prompt:
        query = prompt.split("请求：")[-1].strip()
        markers = ("写", "代码", "证明", "对比", "分析", "优化", "设计", "复杂度", "显存", "SQL", "sql", "推导")
        difficulty = "hard" if (len(query) > 60 or any(m in query for m in markers)) else "easy"
        return (
            '{"difficulty":"%s","reason":"mock判定"}' % difficulty,
            20,
            12,
        )

    if model == settings.cloud_model:
        text = "（mock/云端）这是来自大模型分支的回答。问题长度 %d 字。" % len(prompt)
    else:
        # 3% 概率触发"我不确定"，用来演示升级链路
        if len(prompt) % 33 == 0:
            text = "我不确定这个问题的答案。"
        else:
            text = "（mock/本地）这是来自本地小模型的回答。"

    return text, max(8, len(prompt) // 2), max(8, len(text) // 2)


def chat(
    client: OpenAI,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float = 0.3,
) -> Tuple[str, int, int]:
    """返回 (文本, 输入 token, 输出 token)。"""
    if settings.mock_mode:                  # 离线假模型
        return _mock_chat(messages, model, max_tokens)

    if client is None:
        raise ProviderError(
            "本机未安装 openai 包：pip install openai，或改用 *_stdlib.py 工具链"
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # 网络/额度/模型名错误统一抛出，由上层决定降级
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or 0
    return text, tin, tout


def call_local(messages: List[dict], max_tokens: int | None = None) -> Tuple[str, int, int]:
    # 熔断打开时立刻失败：不发网络请求、不等待，上层直接走降级
    if not local_breaker.allow():
        raise ProviderError("local circuit open: 上次本地调用失败，冷却期内跳过")
    try:
        out = chat(
            local_client,
            settings.local_model,
            messages,
            max_tokens or settings.local_max_tokens,
        )
    except Exception:
        local_breaker.fail()
        raise
    local_breaker.ok()
    return out


def call_cloud(messages: List[dict], max_tokens: int | None = None) -> Tuple[str, int, int]:
    return chat(
        cloud_client,
        settings.cloud_model,
        messages,
        max_tokens or settings.cloud_max_tokens,
    )


# ---------------------------------------------------------------- 工具调用（function calling）

def chat_tools(
    messages: List[dict],
    tools: List[dict],
    max_tokens: int | None = None,
    tool_choice: str = "auto",
) -> Tuple[str, List[dict], int, int]:
    """带工具的一次对话。返回 (文本, 工具调用列表, 输入token, 输出token)。

    为什么工具调用只能走云端：本地 7B 级模型对 JSON Schema 的遵守度很差，
    经常吐出 `{"query": "..."}` 之外的自由文本，或者把工具名写成近义词。
    工具调用的正确性是"要么对要么错"的，不像闲聊可以容错，所以这里不降级到本地——
    宁可多花一点钱，也不要在演示时出现"调了一个不存在的工具"。

    工具调用列表里每项是 {"id","name","args_raw"}：args_raw 保持原始字符串，
    解析失败要原样回给模型看（"你给的 JSON 不合法"比"参数错误"有用得多）。
    """
    if settings.mock_mode:
        return ("（mock）工具模式需要真实模型，离线演示不做工具调用。", [], 10, 10)
    if cloud_client is None:
        raise ProviderError("未安装 openai 包，无法使用工具调用")

    try:
        resp = cloud_client.chat.completions.create(
            model=settings.cloud_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or settings.cloud_max_tokens,
            temperature=0.2,
        )
    except Exception as exc:  # 网络/额度/模型不支持 tools，统一交给上层处理
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

    msg = resp.choices[0].message
    calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        calls.append({"id": tc.id,
                      "name": tc.function.name,
                      "args_raw": tc.function.arguments or "{}"})
    usage = getattr(resp, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or 0
    return (msg.content or "").strip(), calls, tin, tout
