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

    只在为空时建（184 块约 1~2 秒），已有索引就跳过，不重复计算。

    但"不为空"不等于"建完了"：建到一半被重启（平台部署会先杀旧进程），
    库里就留着半套索引，而 chunks>0 会让人以为已经就绪。所以判据不是
    "有没有块"，而是"块数对应的篇数是否等于磁盘上的篇数"——对不上就重建
    （index_dirs 按文件幂等，重复执行只是覆盖，不会重复堆块）。
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
        dirs = [d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
        expect = 0
        for d in dirs:
            folder = Path.cwd() / d
            if folder.exists():
                expect += sum(1 for p in folder.rglob("*.md") if "evidence" not in p.parts)

        if st["chunks"] > 0 and st["docs"] == expect:
            print(f"[kb] 已有索引：{st['docs']} 篇 / {st['chunks']} 块", flush=True)
            return
        if st["chunks"] > 0:
            print(f"[kb] 索引不完整（{st['docs']}/{expect} 篇），重建", flush=True)
        r = rag.index_dirs(dirs, Path.cwd())
        print(f"[kb] 已建索引：{r['docs']} 篇 / {r['chunks']} 块"
              f"（目录 {dirs}）", flush=True)
        if r["chunks"] == 0:
            print("[kb] ⚠ 索引为空：docs/ 没上传或路径不对，"
                  "RAG 会退化（不报错，但检索不到东西）", flush=True)
    except Exception as exc:   # noqa: BLE001 — 建索引失败不该让服务起不来
        print(f"[kb] 建索引失败（服务继续，RAG 退化）：{exc}", flush=True)


import uvicorn  # noqa: E402


def _warm_embedder() -> None:
    """启动时把向量后端确定下来并打一行日志。

    为什么要预热：BGE 是懒加载的，第一次用到才建 ONNX 会话（本机实测 0.35 秒）。
    不预热的话这 0.35 秒会算在第一个用户的第一条请求上——而那条请求本来就慢，
    容易让人误判成"服务卡了"。更麻烦的是降级：加载失败只在用到的那一刻才发现，
    日志里只剩一句轻飘飘的警告。放在启动时打，一眼能看出线上到底用的哪把尺子。

    失败不影响服务启动：缓存是锦上添花，退化成 n-gram 照样能用（守卫兜正确性）。
    """
    try:
        from app import embedder

        backend = embedder.active()
        if backend == "bge":
            embedder.embed("预热")     # 真正跑一次，把 ONNX 的解释器开销吃掉
            print(f"[embedder] 后端 = {embedder.name()}（{embedder.dim()} 维，"
                  f"本地模型，无 API 成本）", flush=True)
        else:
            print("[embedder] 后端 = n-gram（未检测到本地模型，或模型加载失败；"
                  "命中率会低一些，但正确性由载荷守卫保证）", flush=True)
    except Exception as exc:   # noqa: BLE001
        print(f"[embedder] 预热失败（服务继续）：{exc}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    _ensure_kb()
    _warm_embedder()
    # 用字符串形式而不是 app 对象：确保 uvicorn 在环境变量设好之后才导入应用
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, log_level="info")
