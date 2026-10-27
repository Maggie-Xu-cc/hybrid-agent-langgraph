"""MCP 支持：自实现的协议层 + server + client + 注册表桥接。

约定：`app.mcp` 只管协议与连接，不认识项目业务；
`mcp_servers/` 下的每个文件是一个可独立运行的 MCP server（自己的状态、自己的库）。
"""
