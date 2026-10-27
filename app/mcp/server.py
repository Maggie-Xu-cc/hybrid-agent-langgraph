"""MCP server：把一批函数暴露成"任何 MCP 客户端都能用"的工具。

设计上刻意与 App 解耦
--------------------
这一层只认 `ToolDef`（名字/说明/inputSchema/只读标记/handler），不 import 项目里的
任何业务模块。于是同一份代码可以：
  1. 被 `mcp_servers/project_tools.py` 用来把本项目的四个工具导出去；
  2. 被 `mcp_servers/notes.py` 用来起一个完全独立的小服务（自己的库、自己的状态）。
工具提供方变了，server 一行都不用改——这就是"协议"和"实现"分开的收益。

实现的方法（MCP 的最小可用集合）
-------------------------------
  initialize                    握手：协商版本，声明我们支持 tools
  notifications/initialized     客户端说"我准备好了"（通知，不回复）
  tools/list                    列出工具（含 inputSchema）
  tools/call                    调一个工具，拿回 content 数组
  ping                          保活

**通知没有 id，绝对不能回复**——回了客户端会当成一条无法匹配的响应，
轻则报错重则卡死。这是手写 MCP 最容易踩的坑。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable, IO

from .protocol import (FALLBACK_VERSION, INTERNAL_ERROR, INVALID_PARAMS,
                       METHOD_NOT_FOUND, MCPError, error, read_message, result,
                       text_result, write_message)


def ensure_utf8() -> None:
    """把 stdio 钉成 UTF-8。

    Windows 下子进程的 stdio 默认跟本地代码页（cp936），而客户端（官方 SDK /
    Claude Desktop）一律按 UTF-8 解析——中文一出场就是乱码，报错却是
    "JSON 解析失败"这种完全不相干的话。Linux 上不会出现，
    所以特别容易只在某台机器上炸。

    **要在任何 print 之前调用**：启动日志如果用 cp936 写出去，
    在客户端看来同样是乱码（这条踩过）。
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", newline="\n")   # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]
    read_only: bool = True
    # MCP 的 annotations 是"给客户端的提示"，不是强约束：
    # 客户端据此决定要不要弹确认框（只读的通常直接放行）
    destructive: bool = False


@dataclass
class MCPServer:
    name: str
    version: str = "1.0.0"
    tools: list[ToolDef] = field(default_factory=list)
    instructions: str = ""

    def add(self, tool: ToolDef) -> None:
        self.tools.append(tool)

    # ---------------------------------------------------------------- 方法分发

    def handle(self, msg: dict) -> dict | None:
        """处理一条消息。返回 None 表示"这条不需要回复"（通知）。"""
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method is None:
            return None                      # 客户端发来的响应（我们不主动发请求）

        if method == "initialize":
            # 沿用客户端版本：客户端带的是 2026-07-28 而我们回 2025-06-18，
            # 严格的客户端会直接断开。能对上就对上，这是最省事的兼容方式。
            want = str(params.get("protocolVersion") or FALLBACK_VERSION)
            return result(req_id, {
                "protocolVersion": want or FALLBACK_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
                "instructions": self.instructions or "把工具当函数调用即可。",
            })

        if method in ("notifications/initialized", "notifications/cancelled"):
            return None                      # 通知：不回

        if method == "ping":
            return result(req_id, {})

        if method == "tools/list":
            return result(req_id, {"tools": [
                {"name": t.name,
                 "description": t.description,
                 "inputSchema": t.input_schema,
                 "annotations": {"readOnlyHint": t.read_only,
                                 "destructiveHint": t.destructive}}
                for t in self.tools]})

        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            tool = next((t for t in self.tools if t.name == name), None)
            if tool is None:
                # 工具不存在属于"参数错误"而不是崩溃：让客户端/模型看到可用清单
                return result(req_id, text_result(
                    f"没有名为 {name} 的工具。可用：{'、'.join(t.name for t in self.tools)}",
                    is_error=True))
            if not isinstance(args, dict):
                raise MCPError(INVALID_PARAMS, "arguments 必须是对象")
            try:
                out = tool.handler(args)
            except Exception as exc:          # noqa: BLE001 — 工具失败要回给客户端，不是杀进程
                return result(req_id, text_result(
                    f"工具执行失败：{type(exc).__name__}: {exc}", is_error=True))
            return result(req_id, text_result(str(out)))

        return error(req_id, METHOD_NOT_FOUND, f"未知方法：{method}")

    # ---------------------------------------------------------------- stdio 主循环

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """在 stdin/stdout 上跑协议循环，直到对端关闭管道。

        注意：**除了协议消息，不要在 stdout 写任何东西**。调试信息一律
        `print(..., file=sys.stderr)`，否则客户端解析到一半就崩。
        """
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        ensure_utf8()
        while True:
            try:
                msg = read_message(stdin)
            except MCPError as exc:
                write_message(stdout, error(None, exc.code, exc.message))
                continue
            if msg is None:
                return
            try:
                reply = self.handle(msg)
            except MCPError as exc:
                write_message(stdout, error(msg.get("id"), exc.code, exc.message))
                continue
            except Exception as exc:          # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                write_message(stdout, error(msg.get("id"), INTERNAL_ERROR, str(exc)))
                continue
            if reply is not None:
                write_message(stdout, reply)
