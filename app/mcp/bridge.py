"""把 MCP server 暴露的工具，接进本项目的工具注册表。

这一步是 MCP 的全部意义所在
--------------------------
Agent 侧只认"注册表里有什么工具"。MCP server 在启动时被连上、`tools/list` 拿到
schema、**包装成本地 ToolSpec 注册进去**——之后模型看到的工具表和本地工具没有区别，
循环、校验、截断、权限分级、记账全部复用。

于是"加一个工具"这件事从**改 Agent 代码**变成**加一行配置**（`MCP_SERVERS`）。
这就是面试里问"你为什么用 MCP"的标准答案：不是因为它新，而是因为它把
"谁会什么"从代码里解耦成了运行时契约。

设计取舍
--------
- **命名带前缀**（`notes_add`）：远端工具重名会互相覆盖，前缀是唯一的防撞手段。
- **权限按服务端的 annotations 映射**：readOnlyHint → read，destructiveHint → danger。
  但不盲信：服务端说只读也可以撒谎，所以远端工具**默认按最低权限**处理——
  只读标记是"允许自动执行"的必要条件，不是"可以乱来"的许可证。
- **启动失败不阻塞服务**：连不上就记下错误、跳过这些工具，主链路照跑。
  一个 MCP server 挂了不该让整个 Agent 起不来。
"""

from __future__ import annotations

from .. import tools as toolreg
from ..config import settings
from .client import MCPClientError, StdioMCPClient, parse_servers

# name → {"client":..., "server_info":..., "protocol":..., "tools":[...], "error":...}
_servers: dict[str, dict] = {}
_registered: dict[str, list[str]] = {}


def _level_of(tool: dict) -> str:
    ann = tool.get("annotations") or {}
    if ann.get("destructiveHint"):
        return "danger"
    return "read" if ann.get("readOnlyHint") else "write"


def _make_handler(server: str, tool_name: str):
    def handler(args: dict) -> dict:
        entry = _servers.get(server) or {}
        client: StdioMCPClient | None = entry.get("client")
        if client is None:
            raise toolreg.ToolError(f"MCP 服务 {server} 未连接")
        try:
            text, is_error = client.call_tool(tool_name, args)
        except MCPClientError as exc:
            # 崩了先尝试重启一次：MCP server 是独立进程，重启比整个服务重启便宜得多
            try:
                client.close()
                client.start()
                text, is_error = client.call_tool(tool_name, args)
                entry["error"] = ""
            except MCPClientError as exc2:
                entry["error"] = str(exc2)
                raise toolreg.ToolError(f"MCP 服务 {server} 不可用：{exc2}") from exc2
        if is_error:
            raise toolreg.ToolError(text)
        return {"summary": text.splitlines()[0][:60] if text else "完成",
                "content": text,
                "meta": {"mcp": server, "tool": tool_name}}
    return handler


def bootstrap(force: bool = False) -> dict:
    """连接所有配置的 MCP server 并注册它们的工具。返回一份状态摘要。"""
    if _servers and not force:
        return status()
    spec = settings.mcp_servers
    if not spec.strip():
        return {"enabled": False, "servers": []}

    for name, argv in parse_servers(spec):
        if name in _servers and not force:
            continue
        client = StdioMCPClient(argv, name=name,
                               timeout=settings.mcp_timeout)
        entry = {"client": client, "server_info": {}, "protocol": "",
                 "tools": [], "error": ""}
        try:
            client.start()
            listed = client.list_tools()
        except MCPClientError as exc:
            entry["error"] = str(exc)
            client.close()
            _servers[name] = entry
            print(f"[mcp] {name} 连接失败（跳过它的工具）：{exc}", flush=True)
            continue

        entry["server_info"] = client.server_info
        entry["protocol"] = client.protocol_version
        names = []
        for t in listed:
            remote = t.get("name") or ""
            if not remote:
                continue
            local = f"{name}_{remote}"
            desc = (t.get("description") or "").strip()
            toolreg.register(toolreg.ToolSpec(
                name=local,
                description=f"[MCP:{name}] {desc}",
                parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                handler=_make_handler(name, remote),
                level=_level_of(t),
                # timeout 由 MCP client 控制，这里留 None 表示交给它
            ))
            names.append(local)
            entry["tools"].append({"name": local, "remote": remote,
                                   "level": _level_of(t),
                                   "description": desc[:120]})
        _registered[name] = names
        _servers[name] = entry
        print(f"[mcp] {name} 已连接：{len(names)} 个工具"
              f"（协议 {client.protocol_version}）", flush=True)
    return status()


def status() -> dict:
    servers = []
    for name, e in _servers.items():
        servers.append({
            "name": name,
            "connected": bool(e.get("client") and e["client"].proc),
            "protocol": e.get("protocol") or "",
            "server_info": e.get("server_info") or {},
            "tools": e.get("tools") or [],
            "error": e.get("error") or "",
        })
    return {"enabled": bool(servers),
            "count": len(servers),
            "tools": sum(len(s["tools"]) for s in servers),
            "servers": servers}


def shutdown() -> None:
    for e in _servers.values():
        try:
            e["client"].close()
        except Exception:      # noqa: BLE001
            pass
    _servers.clear()
