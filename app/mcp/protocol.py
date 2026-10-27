"""MCP（Model Context Protocol）的最小实现：协议层。

为什么要自己写 ~150 行，而不是装官方 SDK 直接用
------------------------------------------------
1. **运行时零依赖**。官方 `mcp` 包会带进来 anyio / httpx / pydantic 一堆东西，
   而云端部署包越小越不容易失败。协议本身很朴素：**按行分隔的 JSON-RPC 2.0**。
2. **看得见每一行**。MCP 被讲得神神秘秘，实际握手就四个方法：
   `initialize` → `notifications/initialized` → `tools/list` → `tools/call`。
   自己实现一遍，才知道"动态发现工具"到底发生了什么。
3. 代价是可能踩协议细节的坑，所以 `tools/smoke_mcp.py` 会用**官方 SDK 当客户端**
   连我们自己写的 server——能连上、能列工具、能调通，才算实现对了。
   （这条验证比自己说自己兼容可信得多。）

传输层约定（MCP 规范）：
- stdio：每条消息一行 JSON，`\\n` 分隔；**stdout 只能放协议消息**，
  任何调试 print 都必须走 stderr，否则客户端的解析直接崩。
- 字段是 camelCase（`protocolVersion` / `inputSchema` / `isError`）。
"""

from __future__ import annotations

import json
from typing import IO, Any

# 我们说"支持"的协议版本。客户端会带自己的版本过来，能对上就沿用它的，
# 对不上就回一个我们自己实现的版本——MCP 允许两端协商，客户端可以选择断开。
FALLBACK_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def read_message(stream: IO[str]) -> dict | None:
    """读一条消息。返回 None 表示对端关了管道（正常退出）。"""
    line = stream.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return read_message(stream)
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MCPError(PARSE_ERROR, f"不是合法 JSON：{exc}") from exc
    if not isinstance(msg, dict):
        raise MCPError(INVALID_REQUEST, "消息必须是 JSON 对象")
    return msg


def write_message(stream: IO[str], payload: dict) -> None:
    # ensure_ascii=False：中文原样出去。ASCII 转义也能用，但日志里没法读。
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def result(req_id: Any, value: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": value}


def error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def text_result(text: str, is_error: bool = False) -> dict:
    """tools/call 的返回结构。content 是数组，可以多段（文本/图片/资源）。"""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}
