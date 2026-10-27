"""语义缓存基准：同一批请求跑多遍，输出命中率/延迟/成本的真实对比。

    python tools/cache_bench.py                          # 真实模型（按 .env 配置）
    python tools/cache_bench.py --mock                   # 离线跑通流程（数字不可用于报告）
    python tools/cache_bench.py --limit 6                # 只跑前 6 条，先确认链路通
    python tools/cache_bench.py --easy-only --workers 6  # 本机最快跑法（见下）
    python tools/cache_bench.py --workers 8 --quiet      # 并发跑全量

三轮实验设计：
  第 1 遍  原始问题（缓存空）    → 基线，全部 miss
  第 2 遍  相同问题再来一遍      → 精确命中（hash O(1) 通道）
  第 3 遍  换个说法再问一遍      → 语义命中（n-gram 相似度通道，暴露真实召回率）

产物：终端 markdown 表格，可直接贴进 README / 报告。
诚实原则：第 3 遍命不中就是命不中，n-gram 的召回边界如实呈现——
这份数据本身就是"为什么下一步该换真 embedding 模型"的论据。

本机想跑快，三个开关叠加（实测可把 3~5 分钟压到 20 秒级）：
  --easy-only  只跑 easy 用例。hard 按设计不进缓存，跑它们对命中率毫无贡献，
               却在云端生成上最慢（长回答），纯属陪跑。
  --workers N  并发。瓶颈是网络等待不是 CPU，8 路并发基本就是 8 倍。
  DISABLE_LOCAL=1（.env）  完全不调本地。本机没 GPU / 只有 CPU 时，
               本地那一路既慢又必然超时，直接跳过最省时间。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --mock 必须在 import app 之前生效：settings 在 import 时固化，
# 事后设环境变量不改写已创建的配置对象（smoke 脚本同理）。
if "--mock" in sys.argv:
    os.environ.setdefault("MOCK_MODE", "1")

# 独立临时库：不污染开发数据，重复跑结果可复现
os_db = Path(tempfile.mkdtemp(prefix="bench_cache_"))
os.environ["DB_PATH"] = str(os_db / "bench.db")
os.environ["CHECKPOINT_DB"] = str(os_db / "checkpoints.db")

from app import cache, store          # noqa: E402
from app.config import settings       # noqa: E402
from app.graph import run_graph       # noqa: E402

# 第 3 遍的改写问法（人工写的自然改写，不是程序生成的）
PARAPHRASES = {
    "你好": "您好呀",
    "什么是 REST API？": "给我讲讲 REST API 这个概念",
    "1 公里等于多少米": "一公里是多少米？",
    "什么是 CAP 定理": "解释一下 CAP 定理说的是什么",
    "把 3 点 15 分转换成 24 小时制的数字写法": "下午 3 点 15 分用 24 小时制怎么表示",
    "用一句话说明 TCP 和 UDP 的主要区别": "简单说说 TCP 跟 UDP 差在哪",
    "HTTP 状态码 404 是什么意思": "404 状态码代表什么含义",
    "把「今天天气不错」翻译成英文": "帮我把「今天天气不错」翻成英语",
}


def load_cases() -> list[dict]:
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def preflight() -> bool:
    """开跑前检查模型可达性，早失败早报错——不要让人对着黑屏等几分钟。"""
    if settings.mock_mode:
        print("模式：MOCK（离线假模型，不发起真实调用）")
        return True

    u = urlparse(settings.local_base_url)
    host, port = u.hostname or "127.0.0.1", u.port or 8000
    local_ok = local_ready = False
    if settings.disable_local:
        local_note = "已禁用（DISABLE_LOCAL=1）"
    else:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                local_ok = True
        except OSError:
            pass
        if local_ok:
            # 端口开着不等于服务能用（模型没下完、显存不够、进程半死都会这样），
            # 这种情况请求会一直等到超时——提前探一下 /models 能省掉一轮白等
            local_ready = _http_ok(settings.local_base_url.rstrip("/") + "/models")
        local_note = ("就绪" if local_ready else
                      ("端口可达但接口未就绪" if local_ok else "不可达"))

    cloud_ok = bool(settings.cloud_api_key)
    print(f"模式：真实调用  |  本地 {host}:{port} {local_note}  |  "
          f"云端 Key {'已配置' if cloud_ok else '未配置'}")

    if not local_ready and not cloud_ok:
        print("\n两个模型都不可用，跑下去只会全是 ProviderError。先做下面任一件事：")
        print("  1) 加 --mock 离线跑通流程")
        print("  2) 启动本地模型：python deploy/local_server.py（或 ollama serve）")
        print("  3) 在 .env 填 CLOUD_API_KEY")
        return False

    if not local_ready:
        print("提示：本地这一路不可用，easy 请求会全部降级到云端——"
              "链路仍然正确，只是 latency/成本偏高；建议加 DISABLE_LOCAL=1 直接跳过本地。")
    return True


def run_pass(name: str, queries: list[str], verbose: bool,
             workers: int = 1) -> dict:
    """跑一遍。workers>1 时并发发起请求——瓶颈几乎全在网络等待，并发是最大杠杆。

    并发安全性：每条用独立 session_id（独立 thread），互不共享状态；
    SQLite 写入由各自短连接 + WAL/busy timeout 兜住（毫秒级，抢锁概率极低）。
    """
    n = len(queries)
    if verbose:
        print(f"\n-- {name} --", flush=True)

    def _one(q: str) -> tuple[str, dict]:
        # 每线程一个 checkpoint 库：多线程写同一个 sqlite 的 WAL 文件在 Windows
        # 上会偶发 "attempt to write a readonly database"，隔离后彻底消失
        ckpt = os_db / f"ckpt-{threading.get_ident()}.db"
        return q, run_graph(q, session_id=f"bench-{uuid.uuid4().hex}",
                            use_cache=True, checkpoint_db=str(ckpt))

    rows: list[dict] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, q) for q in queries]
            for done, fut in enumerate(as_completed(futs), 1):
                q = queries[futs.index(fut)]
                try:
                    q, r = fut.result()
                except Exception as exc:          # 单条失败不拖垮整轮
                    print(f"  [{done}/{n}] ERROR {type(exc).__name__}: {exc}", flush=True)
                    if verbose:
                        traceback.print_exc()
                    continue
                rows.append(r)
                if verbose:
                    tag = "HIT " if r.get("cache_hit") else "miss"
                    print(f"  [{done}/{n}] {tag} {r.get('route', '?'):<4} "
                          f"{r.get('latency_ms', 0):>5}ms  {q[:26]}", flush=True)
    else:
        for i, q in enumerate(queries, 1):
            _, r = _one(q)
            rows.append(r)
            if verbose:
                tag = "HIT " if r.get("cache_hit") else "miss"
                print(f"  [{i}/{n}] {tag} {r.get('route', '?'):<4} "
                      f"{r.get('latency_ms', 0):>5}ms  {q[:26]}", flush=True)

    hits = sum(1 for r in rows if r.get("cache_hit"))
    cost = sum(r.get("cost_cny", 0) for r in rows)
    lat = sum(r.get("latency_ms", 0) for r in rows)
    return {"pass": name, "n": len(rows), "hits": hits,
            "hit_ratio": hits / len(rows) if rows else 0,
            "cost": cost, "avg_lat": lat / len(rows) if rows else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="离线 mock 模式")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（快速验证链路）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐条进度")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发数，默认 1；网络等待为主，开 4~8 通常快数倍")
    parser.add_argument("--easy-only", action="store_true",
                        help="只跑标注为 easy 的用例（hard 本就不进缓存，纯陪跑）")
    args = parser.parse_args()

    store.init()
    cache.init()
    cases = load_cases()
    if args.easy_only:
        cases = [c for c in cases if c.get("difficulty") == "easy"]
    if args.limit > 0:
        cases = cases[: args.limit]
    queries = [c["q"] for c in cases]

    if not preflight():
        sys.exit(2)
    print(f"测试集 {len(queries)} 条  |  阈值 {settings.cache_threshold}  |  "
          f"并发 {args.workers}  |  共 {len(queries) * 3} 次调用\n", flush=True)

    verbose = not args.quiet
    t0 = time.perf_counter()
    p1 = run_pass("第1遍·基线（缓存空）", queries, verbose, args.workers)
    p2 = run_pass("第2遍·相同问题（精确命中）", queries, verbose, args.workers)
    p3q = [PARAPHRASES.get(q, q) for q in queries]
    p3 = run_pass("第3遍·改写问法（语义命中）", p3q, verbose, args.workers)
    elapsed = time.perf_counter() - t0

    s = cache.stats()
    print("\n| 轮次 | 请求数 | 命中 | 命中率 | 总成本(¥) | 平均延迟 | 相对基线成本 |")
    print("|---|---|---|---|---|---|---|")
    for p in (p1, p2, p3):
        ratio = p["cost"] / p1["cost"] if p1["cost"] else 0
        print(f"| {p['pass']} | {p['n']} | {p['hits']} | "
              f"{p['hit_ratio'] * 100:.1f}% | {p['cost']:.4f} | "
              f"{int(p['avg_lat'])}ms | {ratio * 100:.1f}% |")

    easy_n = sum(1 for c in cases if c["difficulty"] == "easy")
    print(f"\n说明：第2遍命中率的上限 = easy 用例占比（{easy_n}/{len(cases)}）——"
          f"hard 默认不缓存（CACHE_CACHE_HARD=1 可打开）。")
    print(f"缓存表现状：{s['entries']} 条，累计命中 {s['hits']} 次，"
          f"节省 {s['tokens_saved']} token / ¥{s['cost_saved_cny']:.4f}")
    print(f"本轮总耗时 {elapsed:.1f}s（并发 {args.workers}，"
          f"{len(queries) * 3} 次调用）")

    layer = store.stats().get("by_layer", {})
    if layer:
        print("\n分层追踪（decided_by）:")
        print("| 层 | 请求数 | 平均延迟 | 成本(¥) |")
        print("|---|---|---|---|")
        for name_, v in layer.items():
            print(f"| {name_} | {v['count']} | {v['avg_latency_ms']}ms "
                  f"| {v['cost_cny']:.4f} |")

    if args.mock:
        print("\n（MOCK 模式：链路与统计正确，绝对数字不可用于报告）")


if __name__ == "__main__":
    main()
