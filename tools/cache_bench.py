"""语义缓存基准：同一批请求跑多遍，输出命中率/延迟/成本的真实对比。

    python tools/cache_bench.py            # 真实模型（需要 .env 配置）
    python tools/cache_bench.py --mock     # 离线跑通流程（数字不可用于报告）

三轮实验设计：
  第 1 遍  原始问题（缓存空）    → 基线，全部 miss
  第 2 遍  相同问题再来一遍      → 精确命中（hash O(1) 通道）
  第 3 遍  换个说法再问一遍      → 语义命中（n-gram 相似度通道，暴露真实召回率）

产物：终端 markdown 表格，可直接贴进 README / 报告。
诚实原则：第 3 遍命不中就是命不中，n-gram 的召回边界如实呈现——
这份数据本身就是"为什么下一步该换真 embedding 模型"的论据。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

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


def run_pass(name: str, queries: list[str]) -> dict:
    hits = 0
    cost = 0.0
    lat = 0
    for q in queries:
        r = run_graph(q, session_id=f"bench-{uuid.uuid4().hex}", use_cache=True)
        if r.get("cache_hit"):
            hits += 1
        cost += r.get("cost_cny", 0)
        lat += r.get("latency_ms", 0)
    n = len(queries)
    return {"pass": name, "n": n, "hits": hits,
            "hit_ratio": hits / n if n else 0,
            "cost": cost, "avg_lat": lat / n if n else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="离线 mock 模式")
    args = parser.parse_args()

    store.init()
    cache.init()
    cases = load_cases()
    queries = [c["q"] for c in cases]
    print(f"测试集 {len(queries)} 条  |  mock={args.mock}  |  "
          f"阈值={cache.settings.cache_threshold}\n")

    p1 = run_pass("第1遍·基线（缓存空）", queries)
    p2 = run_pass("第2遍·相同问题（精确命中）", queries)
    p3q = [PARAPHRASES.get(q, q) for q in queries]
    p3 = run_pass("第3遍·改写问法（语义命中）", p3q)

    s = cache.stats()
    print("| 轮次 | 请求数 | 命中 | 命中率 | 总成本(¥) | 平均延迟 | 相对基线成本 |")
    print("|---|---|---|---|---|---|---|")
    for p in (p1, p2, p3):
        ratio = p["cost"] / p1["cost"] if p1["cost"] else 0
        print(f"| {p['pass']} | {p['n']} | {p['hits']} | "
              f"{p['hit_ratio'] * 100:.1f}% | {p['cost']:.4f} | "
              f"{int(p['avg_lat'])}ms | {ratio * 100:.1f}% |")

    print(f"\n缓存表现状：{s['entries']} 条，累计命中 {s['hits']} 次，"
          f"节省 {s['tokens_saved']} token / ¥{s['cost_saved_cny']:.4f}")

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
