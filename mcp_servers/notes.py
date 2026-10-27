"""一个完全独立的 MCP server：便签（notes）。

为什么单独做一个"便签"服务，而不是只把项目已有工具导出去
------------------------------------------------------
因为它能证明 MCP 真正的好处：**这个服务跟 Agent 毫无关系**——
自己的库、自己的状态、自己的一套工具，Agent 一行代码都不知道它存在，
只靠配置就把它接上了。面试演示时这条比"我把自己的函数包装了一遍"有说服力得多。

它还能演示一件工具层做不到的事：**跨进程的状态**。
你让它"记一条便签"，数据落在 notes.db；第二天重启 Agent 再问，便签还在。
状态在服务里，不在 Agent 的上下文里。

跑法（也能被 Claude Desktop 这类客户端直接装配）：
    python -m mcp_servers.notes
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 允许 `python -m mcp_servers.notes` 在任意目录下运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mcp.server import MCPServer, ToolDef, ensure_utf8  # noqa: E402

DB = Path(os.getenv("NOTES_DB", "notes.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, text TEXT NOT NULL, tags TEXT DEFAULT '')""")
    return conn


def add(args: dict) -> str:
    text = str(args["text"]).strip()
    tags = str(args.get("tags") or "").strip()
    if not text:
        raise ValueError("text 不能为空")
    with _conn() as conn:
        cur = conn.execute("INSERT INTO notes(ts, text, tags) VALUES (?,?,?)",
                           (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, tags))
        n = cur.lastrowid
    return f"已记下便签 #{n}：{text}" + (f"（标签：{tags}）" if tags else "")


def search(args: dict) -> str:
    kw = str(args["query"]).strip()
    limit = int(args.get("limit") or 5)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, ts, text, tags FROM notes"
            " WHERE text LIKE ? OR tags LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{kw}%", f"%{kw}%", limit)).fetchall()
    if not rows:
        return f"没有找到包含「{kw}」的便签。"
    return "\n".join(f"#{r['id']} [{r['ts']}] {r['text']}"
                     + (f"（{r['tags']}）" if r["tags"] else "") for r in rows)


def listing(args: dict) -> str:
    limit = int(args.get("limit") or 10)
    with _conn() as conn:
        rows = conn.execute("SELECT id, ts, text FROM notes"
                            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    if not rows:
        return "便签是空的。"
    return "\n".join(f"#{r['id']} [{r['ts']}] {r['text']}" for r in rows)


def build() -> MCPServer:
    srv = MCPServer(name="notes", version="1.0.0",
                    instructions="个人便签：可以新增、按关键词搜索、列出最近的。")
    srv.add(ToolDef(
        name="add",
        description="新增一条便签。用户说“记一下/帮我记住/存起来”时用它。",
        input_schema={"type": "object",
                      "properties": {"text": {"type": "string", "description": "便签内容"},
                                     "tags": {"type": "string", "description": "可选标签，逗号分隔"}},
                      "required": ["text"]},
        handler=add, read_only=False))
    srv.add(ToolDef(
        name="search",
        description="按关键词搜索便签内容或标签。",
        input_schema={"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "limit": {"type": "integer", "default": 5}},
                      "required": ["query"]},
        handler=search, read_only=True))
    srv.add(ToolDef(
        name="list",
        description="列出最近的便签（不知道关键词时先看这个）。",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 10}}},
        handler=listing, read_only=True))
    return srv


def _seed_if_empty() -> None:
    """首次启动灌几条示例便签（NOTES_SEED=1 时）。

    为什么需要：公网演示把工具级别收到 `read`（不给 write），`add` 会被权限闸门
    拦掉。如果不预置数据，`search`/`list` 只能返回"便签是空的"——
    MCP 接得再好也演示不出"跨进程状态真的存在"这件事。

    默认关闭（NOTES_SEED 不设就是不灌），否则会污染本地开发和冒烟测试里的编号假设。
    """
    if os.getenv("NOTES_SEED", "0") != "1":
        return
    with _conn() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
        if n:
            return
        conn.executemany("INSERT INTO notes(ts, text, tags) VALUES (?,?,?)", [
            ("2026-10-24 09:12:03", "周五交开题报告，先把相关工作整理成一页", "学业"),
            ("2026-10-25 21:40:11", "这周把 MCP 那节看完，重点看 tools/list 的 schema", "学习"),
            ("2026-10-26 08:05:47", "线上演示只开只读工具，run_python 一律关掉", "运维"),
        ])
    print(f"[notes] 已灌入 3 条示例便签（NOTES_SEED=1）", file=sys.stderr, flush=True)


if __name__ == "__main__":
    ensure_utf8()   # 必须在任何 print 之前
    # 绝对不能往 stdout 打日志：那是协议通道。调试信息走 stderr。
    print(f"[notes] 便签服务启动，库：{DB.resolve()}", file=sys.stderr, flush=True)
    _seed_if_empty()
    build().serve()
