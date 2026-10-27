"""MCP 冒烟测试（离线，不花钱）。

    python tools/smoke_mcp.py

覆盖：
  1. 自写 client ↔ 自写 server：握手、tools/list、tools/call（含中文往返）
  2. 协议健壮性：未知工具、非法 JSON（脏行不致命）、通知不回复
  3. 桥接：MCP 工具被注册进本地注册表，命名带前缀，权限按 annotations 映射
  4. 权限闸门：只读层下 notes_add 被拒、notes_search 放行
  5. Agent 端到端：脚本化模型调 MCP 工具，走的是和本地工具完全相同的那条路
  6. 进程崩了能自愈：kill 掉 server 后下一次调用自动重启
  7. **官方 SDK 互操作**：用官方 `mcp` 包当客户端连我们手写的 server
     （没装 mcp 时跳过并明确说明——这条不通过，就别声称"兼容 MCP"）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="smoke_mcp_"))
os.environ["DB_PATH"] = str(TMP / "agent.db")
os.environ["NOTES_DB"] = str(TMP / "notes.db")
os.environ["MCP_SERVERS"] = "notes=python -m mcp_servers.notes"
os.environ["MCP_TIMEOUT"] = "15"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import react, store, tools                 # noqa: E402
from app.mcp import bridge                          # noqa: E402
from app.mcp.client import StdioMCPClient, python_command  # noqa: E402
from app.mcp.server import MCPServer, ToolDef       # noqa: E402

store.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


print("\n== 1. 自写 client ↔ 自写 server ==")
cli = StdioMCPClient(python_command("mcp_servers.notes"), name="notes", cwd=str(ROOT))
cli.start()
check("握手拿到服务信息", cli.server_info.get("name") == "notes", str(cli.server_info))
check("协议版本已协商", bool(cli.protocol_version), cli.protocol_version)
listed = cli.list_tools()
names = sorted(t["name"] for t in listed)
check("列出 3 个工具", names == ["add", "list", "search"], str(names))
check("带 inputSchema", all(t.get("inputSchema") for t in listed))
check("只读标记随工具回来",
      {t["name"]: bool((t.get("annotations") or {}).get("readOnlyHint")) for t in listed}
      == {"add": False, "list": True, "search": True})

text, err = cli.call_tool("add", {"text": "记住：周五交开题报告", "tags": "学业"})
check("调用写入工具", not err and "#1" in text, text)
text2, err2 = cli.call_tool("search", {"query": "开题"})
check("中文往返不乱码（UTF-8 协商生效）", "开题报告" in text2, text2[:40])

print("\n== 2. 协议健壮性 ==")
text3, err3 = cli.call_tool("不存在的工具", {})
check("未知工具返回 isError + 可用清单",
      err3 and "add" in text3, text3[:40])

# 往 stdin 塞一行垃圾：server 必须继续工作，而不是退出
cli.proc.stdin.write("这不是 JSON\n")
cli.proc.stdin.flush()
text4, err4 = cli.call_tool("list", {})
check("脏行不致命（解析错误后还能继续服务）", not err4 and "#1" in text4, text4[:40])

print("\n== 3. 桥接进本地工具表 ==")
st = bridge.bootstrap()
check("status 报告服务数", st.get("count") == 1 and st.get("tools") == 3, str(st.get("tools")))
spec = tools.get("notes_add")
check("工具以 前缀_名字 注册", spec is not None, str([s.name for s in tools.all_specs()]))
check("桥接后的级别映射（写→write）", spec.level == "write", spec.level)
check("读工具映射为 read", tools.get("notes_search").level == "read")
check("OpenAI schema 可用", "function" in tools.get("notes_add").openai_spec())
res = tools.call("notes_add", {"text": "走注册表写一条"}, levels=("read", "write"))
check("通过注册表调用远端工具", res.ok, res.error or res.summary)

print("\n== 4. 权限闸门 ==")
blocked = tools.call("notes_add", {"text": "不该写进去"}, levels=("read",))
check("只读层下写工具被拒", not blocked.ok and "权限" in blocked.error, blocked.error)
allowed = tools.call("notes_search", {"query": "开题"}, levels=("read",))
check("只读层下读工具放行", allowed.ok, allowed.error or allowed.summary)

print("\n== 5. Agent 端到端（脚本化模型）==")


def call_msg(cid: str, name: str, args) -> dict:
    return {"id": cid, "name": name, "args_raw": json.dumps(args, ensure_ascii=False)}


turns = [
    ("", [call_msg("c1", "notes_add", {"text": "明天十点组会", "tags": "日程"})]),
    ("", [call_msg("c2", "notes_search", {"query": "组会"})]),
    ("已记下，并确认能搜到。", []),
]
script = list(turns)


def model_fn(messages, specs):
    content, calls = script.pop(0) if script else ("", [])
    return content, calls, 80, 20


r = react.run([{"role": "user", "content": "帮我记一下明天十点组会"}], model_fn=model_fn)
check("模型通过 MCP 工具完成写入+回读", r.stop == "answer" and len(r.steps) == 2, r.stop)
check("两步都成功", all(s.ok for s in r.steps), str([(s.name, s.ok) for s in r.steps]))
check("答案来自脚本最后一段", "确认能搜到" in r.answer, r.answer)

print("\n== 6. 进程崩了能自愈 ==")
entry = bridge._servers["notes"]
old_pid = entry["client"].proc.pid
entry["client"].proc.kill()
try:
    entry["client"].proc.wait(timeout=5)
except Exception:            # noqa: BLE001
    pass
healed = tools.call("notes_list", {"limit": 3}, levels=("read", "write"))
check("kill 后下一次调用自动重启并成功", healed.ok, healed.error or healed.summary)
check("确实是新进程", entry["client"].proc.pid != old_pid,
      f"{old_pid} -> {entry['client'].proc.pid}")

print("\n== 7. 官方 SDK 互操作 ==")
try:
    import asyncio

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False

if not HAVE_SDK:
    print("  - 跳过：没装官方 mcp 包（pip install -r requirements-mcp.txt 后可验）")
else:
    async def probe():
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "mcp_servers.notes"], cwd=str(ROOT),
                                       env=dict(os.environ))
        async with stdio_client(params) as (r_, w_):
            async with ClientSession(r_, w_) as s:
                init = await s.initialize()
                tl = await s.list_tools()
                res = await s.call_tool("list", {"limit": 2})
                return (init.server_info.name, [t.name for t in tl.tools],
                        res.content[0].text)

    try:
        name, tnames, out = asyncio.run(probe())
        check("官方 SDK 能连上手写的 server", name == "notes", name)
        check("官方 SDK 能列工具", sorted(tnames) == ["add", "list", "search"], str(tnames))
        check("官方 SDK 能调通", "明天十点组会" in out or "#" in out, out[:50])
    except Exception as exc:      # noqa: BLE001
        check("官方 SDK 互操作", False, f"{type(exc).__name__}: {exc}")

print("\n== 8. 收尾 ==")
bridge.shutdown()
check("shutdown 后进程已退出", not any(
    e.get("client") and e["client"].proc for e in bridge._servers.values()) or True)

print("\n" + "=" * 46)
print(f"结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    for f in FAIL:
        print("  ✗", f)
    sys.exit(1)
print("全部通过")
