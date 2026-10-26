"""统一的大模型客户端。

关键设计：本地 vLLM 与云端 API 都是 OpenAI 兼容协议，所以只是 base_url 不同。
这样"路由"退化成一次 endpoint 切换，不需要两套调用逻辑。
"""

from typing import List, Tuple

try:
    from openai import OpenAI
except ImportError:  # AUP 等环境不一定装了 openai 包；标准库工具链(*_stdlib.py)不经过这里
    OpenAI = None

from .config import settings

if OpenAI is None:
    local_client = cloud_client = None
else:
    local_client = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key)
    cloud_client = OpenAI(
        base_url=settings.cloud_base_url,
        api_key=settings.cloud_api_key or "EMPTY",
        timeout=60.0,
    )


class ProviderError(RuntimeError):
    pass


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
    return chat(
        local_client,
        settings.local_model,
        messages,
        max_tokens or settings.local_max_tokens,
    )


def call_cloud(messages: List[dict], max_tokens: int | None = None) -> Tuple[str, int, int]:
    return chat(
        cloud_client,
        settings.cloud_model,
        messages,
        max_tokens or settings.cloud_max_tokens,
    )
