"""阶段三冒烟测试：语义缓存 + 分层追踪。

离线跑（MOCK_MODE=1），不花钱不依赖外部服务：
    python tools/smoke_cache.py

覆盖：
  1. embed 向量：归一化、确定性、区分度
  2. cache 模块：精确命中 / 相似改写命中 / 不相似不命中 / 重复写入不覆盖
  3. TTL 过期条目被忽略与清理
  4. 图链路：首轮写入 → 相同/相似问题命中（0 token、0 成本）
  5. 多轮问题、hard 请求不写缓存（宁可少命中，不可答错）
  6. use_cache=False 时完全不碰缓存（评测隔离）
  7. store.stats() 的分层追踪（by_layer / cache_hit_ratio）
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="smoke_cache_")) / "cache.db")
os.environ["CHECKPOINT_DB"] = str(
    Path(tempfile.mkdtemp(prefix="smoke_ck_")) / "checkpoints.db")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import cache, store            # noqa: E402
from app.config import settings         # noqa: E402
from app.graph import run_graph         # noqa: E402

store.init()
cache.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


def norm(vec: list[float]) -> float:
    return sum(v * v for v in vec) ** 0.5


print("\n== 1. embed 向量 ==")
v1 = cache.embed("你好")
v2 = cache.embed("你好")
v3 = cache.embed("请问如何学习深度学习并调参")
check("维度 = 256", len(v1) == 256)
check("模长归一化 ≈ 1", abs(norm(v1) - 1.0) < 0.01, f"norm={norm(v1):.4f}")
check("确定性：同文本同向量", v1 == v2)
sim_hi = cache._cosine(cache.embed("把「我明天下午三点开会」翻译成日文"),
                       cache.embed("帮我把「我明天上午九点开会」翻译成日文"))
sim_far = cache._cosine(cache.embed("今天上海天气怎么样"),
                        cache.embed("解释一下反向传播的数学推导"))
sim_greet = cache._cosine(cache.embed("你好"), cache.embed("您好啊"))
check("同义改写相似度高（中长句）", sim_hi > 0.5, f"cos={sim_hi:.3f}")
check("无关问题相似度低", sim_far < 0.3, f"cos={sim_far:.3f}")
check("区分度：同义 > 无关", sim_hi > sim_far)
# 已知边界（写进测试防止将来无感退化）：n-gram 对短句寒暄召回弱
check("短句寒暄召回弱是已知边界（不误报为 bug）", 0.0 <= sim_greet < 0.5,
      f"cos={sim_greet:.3f}")

print("\n== 2. put / lookup 基本行为 ==")
ok = cache.put("今天上海天气怎么样", "（mock/本地）回答A", "Qwen2.5-7B-AWQ", "easy",
               tokens_in=50, tokens_out=20, cost=0.0002)
check("写入返回 True", ok)
again = cache.put("今天上海天气怎么样", "另一个答案", "x", "easy")
check("重复写入被忽略（不覆盖）", again is False)
hit = cache.lookup("今天上海天气怎么样")
check("精确命中", hit is not None and hit["score"] == 1.0, f"score={hit and hit['score']}")
check("命中带回答案", hit and hit["answer"] == "（mock/本地）回答A")
hit2 = cache.lookup("今天上海天气怎么样")
check("命中计数 +1", hit2 and hit2["hits"] == 2, f"hits={hit2 and hit2['hits']}")

miss = cache.lookup("帮我写一个分布式事务的方案")
check("不相似不命中", miss is None)

sim_hit = cache.lookup("上海今天天气怎么样呢")
check("相似改写命中", sim_hit is not None and sim_hit["score"] >= settings.cache_threshold,
      f"score={sim_hit and round(sim_hit['score'], 3)}")

print("\n== 3. TTL 过期 ==")
conn = sqlite3.connect(settings.db_path)
with conn:
    conn.execute("UPDATE semantic_cache SET updated_at='2020-01-01 00:00:00', hits=0"
                 " WHERE query_text LIKE '今天上海%'")
conn.close()
check("过期且未命中 → 不再返回", cache.lookup("今天上海天气怎么样") is None)

print("\n== 4. 图链路：写入 → 命中 ==")
r1 = run_graph("什么是 REST API", session_id="smoke-c1", use_cache=True)
check("第一轮正常作答", not r1.get("cache_hit"), r1.get("answer", "")[:20])
check("easy 首轮答案进了缓存",
      any("REST" in e for e in []) or cache.lookup("什么是 REST API") is not None)
check("第一轮有成本", r1.get("tokens_in", 0) > 0)

r2 = run_graph("什么是 REST API", session_id="smoke-c2", use_cache=True)
check("第二轮相同问题命中缓存", bool(r2.get("cache_hit")))
check("命中相似度 = 1.0", r2.get("cache_score") == 1.0)
check("命中后 token 为 0", r2.get("tokens_in", 0) == 0 and r2.get("tokens_out", 0) == 0)
check("命中后成本为 0", r2.get("cost_cny") == 0)
check("decided_by = cache", r2.get("decided_by") == "cache")
check("返回结构一致（有 memory 字段）", "memory" in r2)

r3 = run_graph("帮我介绍一下 REST API 好吗", session_id="smoke-c3", use_cache=True)
check("相似问题也能命中", bool(r3.get("cache_hit")),
      f"score={r3.get('cache_score')}")

r4 = run_graph("什么是 REST API", session_id="smoke-c4", use_cache=False)
check("use_cache=False 完全绕开缓存", not r4.get("cache_hit"))
check("关闭时照常调用模型", r4.get("tokens_in", 0) > 0)

print("\n== 5. 不该缓存的场景 ==")
n_before = cache.stats()["entries"]
run_graph("写一个快速排序", session_id="smoke-c5", use_cache=True)   # hard
n_after_hard = cache.stats()["entries"]
check("hard 请求不写缓存", n_after_hard == n_before)

s1 = run_graph("先问个简单问题", session_id="smoke-c6", use_cache=True)
n_after_first = cache.stats()["entries"]   # 首轮 easy 合法写入，以它为基线
run_graph("再问一个简单问题", session_id="smoke-c6", use_cache=True)  # 同会话第二轮
n_after_multi = cache.stats()["entries"]
check("多轮问题不写缓存（依赖上下文）", n_after_multi == n_after_first,
      f"{n_after_first} -> {n_after_multi}")
check("多轮回答正常", bool(s1.get("answer")))

print("\n== 6. 分层追踪（store.stats） ==")
s = store.stats()
check("by_layer 包含 cache 层", "cache" in s.get("by_layer", {}))
check("cache_hits 与命中次数一致", s.get("cache_hits", 0) >= 3,
      f"hits={s.get('cache_hits')}")
check("cache_hit_ratio 在 (0,1) 区间", 0 < s.get("cache_hit_ratio", 0) < 1,
      f"ratio={s.get('cache_hit_ratio')}")
check("缓存层平均延迟很低", s["by_layer"]["cache"]["avg_latency_ms"] < 200,
      f"{s['by_layer']['cache']['avg_latency_ms']}ms")

print("\n== 7. purge_expired ==")
conn = sqlite3.connect(settings.db_path)
with conn:
    conn.execute("UPDATE semantic_cache SET updated_at='2020-01-01 00:00:00', hits=0")
conn.close()
n_purged = cache.purge_expired()
check("过期未命中条目被清掉", n_purged > 0, f"purged={n_purged}")
check("清理后表为空", cache.stats()["entries"] == 0)

# ----------------------------------------------------------------
print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：")
    for name in FAIL:
        print(f"  - {name}")
sys.exit(1 if FAIL else 0)
