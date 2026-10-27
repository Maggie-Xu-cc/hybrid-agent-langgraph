"""MCP client：把"远端工具"接进 Agent。

为什么用同步实现 + 后台读线程
-----------------------------
Agent 的主流程是同步的（FastAPI 的 def 端点跑在线程池里），而官方 SDK 是异步的。
在同步函数里 `asyncio.run()` 每次都会新建一个事件循环和一个子进程——一次调用多花
几百毫秒，还会把子进程的生命周期搞得很脆。

所以这里用一个**常驻子进程 + 一个后台读线程 + 队列**：
主线程写 stdin、从队列里等带同一个 id 的响应；超时就抛错，绝不死等。
这是"同步代码对接异步协议"最土也最稳的做法，100 行没有魔法。

坑（都踩过）
------------
- **stderr 必须单独收**：子进程往 stderr 打日志时，如果父进程不读，缓冲区满了
  子进程就卡在 write 上——表现为"调用永远不返回"。所以这里把 stderr 重定向到
  临时文件，出问题能翻。
- **超时后不能只抛错**：请求还在管道里，下次读到的可能是上次的响应。
  所以每个请求带 id，只认自己那个 id；其余（通知、别人的响应）一律丢弃。
- **重启要换新队列**：进程被 kill 后，上一代的读线程还会补一条 `_eof` 进队列。
  如果新旧两代共用一个队列，重启后的第一个请求就会先读到那条属于死进程的 EOF，
  表现成"刚恢复就又挂了"——而且只在特定时序下复现（本机两个解释器一个过一个不过，
  查了半小时才定位到）。所以每一代进程配一个自己的队列。
- **关管道要按顺序**：先关 stdin 让对端自己退出，等 2 秒再 kill，
  否则 Windows 上很容易留下僵尸进程，第二次启动端口/文件就被占用。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .protocol import FALLBACK_VERSION, write_message


class MCPClientError(RuntimeError):
    pass


class StdioMCPClient:
    def __init__(self, command: list[str], name: str = "",
                 timeout: float = 10.0, cwd: str | None = None,
                 env: dict | None = None) -> None:
        self.command = command
        self.name = name or (command[-1] if command else "mcp")
        self.timeout = timeout
        self.cwd = cwd
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.server_info: dict = {}
        self.protocol_version = ""
        self._q: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_id = 0
        self._stderr_path = ""
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 生命周期

    def start(self) -> "StdioMCPClient":
        if self.proc is not None:
            return self
        err = tempfile.NamedTemporaryFile(prefix=f"mcp_{self.name}_", suffix=".log",
                                          delete=False)
        err.close()
        self._stderr_path = err.name
        env = dict(os.environ if self.env is None else self.env)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self.proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=open(self._stderr_path, "wb"),
                cwd=self.cwd, env=env,
                # 不传 newline：Popen 不接受这个参数。Windows 上文本模式会把 "\n"
                # 写成 "\r\n"，但读端用 readline + strip 时照样解析得开，协议不受影响。
                text=True, encoding="utf-8", bufsize=1,
            )
        except FileNotFoundError as exc:
            raise MCPClientError(f"启动 {self.name} 失败：找不到命令 {self.command[0]}") from exc
        # 每一代进程配一个新队列：上一代的读线程还没退干净，它读到 EOF 会往队列里
        # 塞一条 _eof。共用队列的话，重启后的第一个请求会先拿到那条"死进程的遗言"。
        self._q = queue.Queue()
        self._reader = threading.Thread(target=self._pump,
                                        args=(self.proc.stdout, self._q),
                                        name=f"mcp-{self.name}", daemon=True)
        self._reader.start()
        self._handshake()
        return self

    def _pump(self, stream, q: queue.Queue) -> None:
        """把子进程 stdout 的每一行解析后塞进队列，读到 EOF 就结束。

        stream / q 都从参数传入（而不是读 self）：重启后 self 指向新的一代，
        老线程继续写自己的那个队列，两边互不干扰。
        """
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                # 不合法的行不致命：可能是服务端把调试信息写到了 stdout（常见错误）。
                # 记下来但继续读，别让一条脏数据毁掉整个连接。
                q.put({"_bad_line": line[:200]})
        q.put({"_eof": True})

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None, notify: bool = False) -> dict:
        if not self.proc or not self.proc.stdin:
            raise MCPClientError(f"{self.name} 没有启动")
        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            msg = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
            if not notify:
                msg["id"] = req_id
            try:
                write_message(self.proc.stdin, msg)
            except (BrokenPipeError, OSError) as exc:
                raise MCPClientError(f"{self.name} 管道已关闭：{exc}") from exc
        if notify:
            return {}

        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise MCPClientError(f"{self.name} 超时未响应 {method}")
            try:
                item = self._q.get(timeout=left)
            except queue.Empty:
                raise MCPClientError(f"{self.name} 超时未响应 {method}") from None
            if item.get("_eof"):
                raise MCPClientError(f"{self.name} 进程退出了（看日志 {self._stderr_path}）")
            if item.get("_bad_line"):
                continue                      # 脏行丢掉，继续等
            if item.get("id") != req_id:
                continue                      # 通知或过期响应，不是我们要的
            if "error" in item:
                raise MCPClientError(f"{self.name}: {item['error'].get('message')}")
            return item.get("result") or {}

    def _handshake(self) -> None:
        res = self._request("initialize", {
            "protocolVersion": FALLBACK_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "hybrid-agent", "version": "2.0"},
        })
        self.protocol_version = str(res.get("protocolVersion") or FALLBACK_VERSION)
        self.server_info = res.get("serverInfo") or {}
        # 规范要求客户端在 initialize 之后发这条通知，服务端才知道"可以开始用了"
        self._request("notifications/initialized", None, notify=True)

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if not proc:
            return
        try:
            if proc.stdin:
                proc.stdin.close()          # 先温和地关 stdin，多数 server 会自己退出
            proc.wait(timeout=2)
        except Exception:                   # noqa: BLE001
            try:
                proc.kill()
            except Exception:               # noqa: BLE001
                pass

    def __enter__(self) -> "StdioMCPClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- 协议方法

    def list_tools(self) -> list[dict]:
        res = self._request("tools/list", {})
        return list(res.get("tools") or [])

    def call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """返回 (文本, 是否出错)。isError 由服务端决定，别自己猜。"""
        res = self._request("tools/call", {"name": name, "arguments": args},
                            timeout=self.timeout)
        parts = []
        for c in res.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
            else:
                parts.append(json.dumps(c, ensure_ascii=False)[:500])
        return ("\n".join(parts).strip(), bool(res.get("isError")))


def python_command(module: str) -> list[str]:
    """用当前解释器起 MCP server：换机器/换虚拟环境都不用改配置。"""
    return [sys.executable, "-m", module]


def parse_servers(spec: str) -> list[tuple[str, list[str]]]:
    """解析 `MCP_SERVERS`：`name=python -m pkg.mod;name2=command args`。

    为什么是这种朴素格式而不是 JSON：环境变量里写 JSON 又要转义引号，
    在 .env / 部署平台里都容易写错。够用就好。
    """
    out: list[tuple[str, list[str]]] = []
    for chunk in (spec or "").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, cmd = chunk.split("=", 1)
        argv = cmd.strip().split()
        if not argv:
            continue
        # `python` 统一换成当前解释器：.env 里写死绝对路径换机就废
        if argv[0] in ("python", "python3", "py"):
            argv = [sys.executable] + argv[1:]
        out.append((name.strip(), argv))
    return out
