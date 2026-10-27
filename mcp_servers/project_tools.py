"""把本项目自己的工具，通过 MCP 暴露给任何客户端。

用途：Claude Desktop / 别的 Agent 框架想用"查本项目知识库、查记账库、跑 Python"，
不需要装这个项目、不需要读它的源码，配一行命令就行：

    {
      "mcpServers": {
        "hybrid-agent": {
          "command": "python",
          "args": ["-m", "mcp_servers.project_tools"],
          "cwd": "/path/to/hybrid-agent-langgraph"
        }
      }
    }

注意一个容易绕晕的地方：**不要把 project_tools 自己配进 MCP_SERVERS**。
那等于把 app 的工具通过 MCP 再导回 app，工具表里会出现
`project_tools_sql_query` 和 `sql_query` 两份同名不同名的工具，模型会随机挑一个。
这不是 bug，是"不要把回路接成环"的设计常识。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tools as toolreg          # noqa: E402
from app.mcp.server import MCPServer, ToolDef, ensure_utf8  # noqa: E402


def build() -> MCPServer:
    srv = MCPServer(
        name="hybrid-agent-tools",
        version="2.0.0",
        instructions="本项目的能力：检索知识库、只读查记账库、抓网页、跑 Python。",
    )
    for spec in toolreg.all_specs():
        def make(spec=spec):
            def handler(args: dict) -> str:
                # 复用同一套校验/截断/权限：MCP 只换了"怎么送进来"，没换"怎么执行"
                res = toolreg.call(spec.name, args)
                if not res.ok:
                    raise ValueError(res.error or "执行失败")
                return res.content
            return handler

        srv.add(ToolDef(
            name=spec.name,
            description=spec.description,
            input_schema=spec.parameters,
            handler=make(),
            read_only=spec.level == "read",
            destructive=spec.level == "danger",
        ))
    return srv


if __name__ == "__main__":
    ensure_utf8()   # 必须在任何 print 之前
    print(f"[project_tools] 暴露 {len(toolreg.all_specs())} 个工具", file=sys.stderr, flush=True)
    build().serve()
