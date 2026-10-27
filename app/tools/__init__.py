"""工具层：注册表 + 参数校验 + 统一调用口。

这一层解决四个"不做就一定会出事"的问题
--------------------------------------
1. **参数不能信模型**。模型会漏字段、把数字写成字符串、凭空多塞一个键。
   不校验就直接执行，等于把 JSON 当命令行用。
2. **工具必须有权限分级**。查资料（read）可以自动跑，
   执行代码/写文件（danger）必须先过人工确认——这是 P3 要接的 HITL。
3. **结果必须有上限**。网页 200KB、SQL 一万行，塞回 prompt 会一次性吃光上下文，
   而且模型根本读不完。统一截断，头尾都留，中间标注被砍了多少。
4. **失败要能被模型看见**。工具报错不要抛到顶层炸掉整轮请求——
   把错误当"观察结果"还给它，它才有机会换个参数重试。

和 MCP 的关系：这里的 ToolSpec 是内部表示，后面 P2 会写一个 MCP server
把它导出成 MCP 工具（`tools/list` + `tools/call`），这样同一批工具
能被 Claude Desktop 这类外部客户端装配，而不用改 Agent 代码。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import settings

# 工具级别：read 只读、write 会改数据、danger 能执行任意代码/影响宿主机
LEVELS = ("read", "write", "danger")


class ToolError(Exception):
    """可预期的失败（参数不合法、被规则拦下、目标不存在）。

    **不要**用它包装程序 bug：这类错误会把原文返回给模型让它自我修正，
    堆栈和内部路径不该出现在 prompt 里。
    """


class ToolArgError(ToolError):
    """参数不符合 schema。单独一类，测试和统计要能区分"模型不会用工具"和"工具坏了"。"""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON Schema（只用 object/string/integer/number/boolean/array）
    handler: Callable[[dict], dict]
    level: str = "read"
    timeout: float | None = None

    def openai_spec(self) -> dict:
        """转成 OpenAI tools 格式。description 写给模型看，要说清"什么时候该用"。"""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str              # 一行摘要，给页面显示（模型看不到）
    content: str              # 完整结果，还给模型（已截断）
    ms: int = 0
    truncated: bool = False
    error: str = ""
    meta: dict = field(default_factory=dict)

    def for_model(self) -> str:
        # 有 content 就用 content：失败也要给模型**能看懂的信息**。
        # 「工具执行失败：未注册的工具 web_search」不如
        # 「没有 web_search，可用的是 search_docs、sql_query…」有用——
        # 后者能直接导致下一步的正确动作，前者只会让它再猜一次。
        return self.content or f"工具执行失败：{self.error}"


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.level not in LEVELS:
        raise ValueError(f"未知工具级别：{spec.level}")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def all_specs(levels: tuple[str, ...] = LEVELS) -> list[ToolSpec]:
    return [s for s in _REGISTRY.values() if s.level in levels]


def openai_tools(levels: tuple[str, ...] = LEVELS) -> list[dict]:
    return [s.openai_spec() for s in all_specs(levels)]


def describe() -> list[dict]:
    """给页面/调试用的工具清单（不含 handler）。"""
    return [{"name": s.name, "description": s.description, "level": s.level,
             "parameters": s.parameters} for s in all_specs()]


# ---------------------------------------------------------------- 参数校验

_TYPE_MAP = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}


def validate_args(spec: ToolSpec, args: dict) -> dict:
    """按 schema 校验并返回清洗后的参数。

    只支持我们真正用到的子集：type / required / properties / enum / items。
    不引入 jsonschema 依赖——工具参数都很简单，多一个依赖就多一次部署风险。
    """
    if not isinstance(args, dict):
        raise ToolArgError(f"参数必须是对象，收到的是 {type(args).__name__}")

    props: dict = spec.parameters.get("properties") or {}
    required = spec.parameters.get("required") or []

    missing = [k for k in required if k not in args or args[k] in (None, "")]
    if missing:
        raise ToolArgError(f"缺少必填参数：{'、'.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolArgError(
            f"不存在的参数：{'、'.join(unknown)}（可用参数：{'、'.join(props)}）")

    clean: dict = {}
    for key, value in args.items():
        schema = props[key]
        want = schema.get("type")
        # 模型很爱把 3 写成 "3"：能救就救，不要为了一个引号让整轮失败
        if want == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if want == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        expected = _TYPE_MAP.get(want)
        if expected and not isinstance(value, expected):
            raise ToolArgError(f"参数 {key} 应为 {want}，收到 {type(value).__name__}")
        if want == "boolean" and isinstance(value, bool) is False:   # int 也是 int
            raise ToolArgError(f"参数 {key} 应为 boolean")
        if schema.get("enum") and value not in schema["enum"]:
            raise ToolArgError(f"参数 {key} 只能取 {schema['enum']}，收到 {value!r}")
        clean[key] = value

    for key, schema in props.items():
        if key not in clean and "default" in schema:
            clean[key] = schema["default"]
    return clean


def clip(text: str, limit: int | None = None) -> tuple[str, bool]:
    """超长观察结果截断：留头留尾，中间写清砍了多少。

    为什么不留头或留尾之一：网页/日志的结论常在后面，前言常在前面，
    只留一边会让模型"看过却漏掉关键结论"，然后一本正经地猜。
    """
    limit = limit or settings.tool_result_chars
    if len(text) <= limit:
        return text, False
    head = max(1, int(limit * 0.7))
    tail = max(1, limit - head)
    cut = len(text) - head - tail
    return (text[:head] + f"\n…（中间省略 {cut} 字）…\n" + text[-tail:]), True


def call(name: str, args: dict, levels: tuple[str, ...] = LEVELS) -> ToolResult:
    """统一调用口。**任何异常都不往外抛**——工具失败是这一轮的观察结果，不是系统故障。"""
    spec = _REGISTRY.get(name)
    if spec is None:
        avail = [s.name for s in all_specs(levels)]
        return ToolResult(name, False, "没有这个工具",
                          f"没有名为 {name} 的工具。可用工具：{'、'.join(avail)}。"
                          f"请从这些里面选，不要自己发明工具名。",
                          error=f"未注册的工具：{name}",
                          meta={"available": avail})
    if spec.level not in levels:
        return ToolResult(name, False, "被权限拦下",
                          f"{name} 是 {spec.level} 级工具，本轮权限不允许调用。"
                          f"请改用只读工具（search_docs / sql_query）获取信息，"
                          f"或直接说明做不到。",
                          error=f"{name} 属于 {spec.level} 级工具，超出本轮权限")

    started = time.perf_counter()
    try:
        clean = validate_args(spec, args if isinstance(args, dict) else {})
        out = spec.handler(clean)
    except ToolArgError as exc:
        return ToolResult(name, False, f"参数不对：{exc}", "",
                          ms=_ms(started), error=str(exc))
    except ToolError as exc:
        return ToolResult(name, False, f"被拦下：{exc}", "",
                          ms=_ms(started), error=str(exc))
    except Exception as exc:  # noqa: BLE001 — 工具内部 bug 也不能拖垮主链路
        return ToolResult(name, False, f"执行出错：{type(exc).__name__}",
                          "", ms=_ms(started),
                          error=f"{type(exc).__name__}: {exc}")

    content, truncated = clip(str(out.get("content", "")))
    return ToolResult(name=name, ok=True,
                      summary=str(out.get("summary", ""))[:80],
                      content=content, ms=_ms(started), truncated=truncated,
                      meta=out.get("meta") or {})


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# 注册内置工具（放在最后导入，避免 handler 反过来 import 本模块时循环依赖）
from . import builtin  # noqa: E402,F401
