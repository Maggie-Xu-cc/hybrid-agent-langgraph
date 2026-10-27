"""部署入口：适配云端的 PORT 约定，并自动判断要不要启用本地模型层。

为什么单独一个文件：
    本地开发和云端部署的启动方式不一样——本地 `uvicorn app.server:app` 默认
    8000 端口、只听 127.0.0.1 就够了；云端平台只给一个 PORT 环境变量，
    并且要求监听 0.0.0.0。把这些差异收进入口文件，比让每个环境各记一套命令可靠。

两件在这里做的事（都必须在 import app.server 之前完成）：
    1. 探测本地模型端口，不通就 DISABLE_LOCAL=1
       不关会怎样：云端一定连不上本地，每个 easy 请求先白等 15 秒超时再降级；
       熔断器两次之后会短路，但前两条请求的体验是"点了没反应"。
    2. 收紧演示额度
       公网链接背后是真实 API Key，默认按 300 次/天、单 IP 10 次/分钟止损。

本地想跑这个入口也可以：python demo_server.py，默认 8000 端口。
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse


def _local_model_reachable() -> bool:
    """只探端口不够严谨（服务可能"端口通但推理起不来"），但足够做初筛：
    真正半死的情况交给熔断器兜底，这里只负责把"根本没有本地模型"这种
    最常见的情况识别出来。"""
    if os.getenv("DISABLE_LOCAL") == "1":
        return False
    url = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
    parsed = urlparse(url)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 8000), timeout=0.5
        ):
            return True
    except OSError:
        return False


if not _local_model_reachable():
    # settings 在 import 时固化，所以这一步必须早于 app.server 的导入
    os.environ["DISABLE_LOCAL"] = "1"

os.environ.setdefault("DEMO_QUOTA_PER_DAY", "300")
os.environ.setdefault("DEMO_RATE_PER_MIN", "10")


def _ensure_kb() -> None:
    """启动时确保 RAG 知识库有索引。

    为什么需要这一步：云端每次部署都是全新的库，`kb_chunks` 表是空的。
    代码在、数据没在，检索会静默返回空——**RAG 看起来像开着，实际什么都没做**。
    这种"功能在但不出效果"的问题最难发现，所以放在启动时兜底。

    只在为空时建（163 块约 1~2 秒），已有索引就跳过，不重复计算。
    """
    try:
        from pathlib import Path

        from app import rag, store
        from app.config import settings

        if not settings.enable_rag:
            print("[kb] RAG 未启用（ENABLE_RAG=0），跳过", flush=True)
            return
        store.init()
        rag.init()
        st = rag.stats()
        if st["chunks"] > 0:
            print(f"[kb] 已有索引：{st['docs']} 篇 / {st['chunks']} 块", flush=True)
            return
        dirs = [d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
        r = rag.index_dirs(dirs, Path.cwd())
        print(f"[kb] 已建索引：{r['docs']} 篇 / {r['chunks']} 块"
              f"（目录 {dirs}）", flush=True)
        if r["chunks"] == 0:
            print("[kb] ⚠ 索引为空：docs/ 没上传或路径不对，"
                  "RAG 会退化（不报错，但检索不到东西）", flush=True)
    except Exception as exc:   # noqa: BLE001 — 建索引失败不该让服务起不来
        print(f"[kb] 建索引失败（服务继续，RAG 退化）：{exc}", flush=True)


import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    _ensure_kb()
    # 用字符串形式而不是 app 对象：确保 uvicorn 在环境变量设好之后才导入应用
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, log_level="info")
