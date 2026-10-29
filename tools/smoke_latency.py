"""延迟记账冒烟测试（离线，不花钱）。

    python tools/smoke_latency.py

为什么单独测这一块
------------------
线上出过一条 51 亿毫秒（约 60 天）的延迟记录，一条就把"平均延迟"拉成了
5 亿毫秒——页面上所有性能数字瞬间变成废话。根因是端到端耗时用了
perf_counter（单调钟）：它能保证同一进程内单调，但**跨进程没有共同基准**。
而这张图恰好会在 interrupt（等用户确认）处把状态落盘、换个进程/换台机器继续跑。

这里断言三件事：
  1. 耗时改用墙钟 + 扣等待：正常、等待、等待过头、时间源失效，四种情况都要给出
     一个"人能用"的数（不会出现负数、也不会出现 60 天）
  2. 统计只拿"像真延迟"的行算平均，并把异常行的**个数**报出来（账不删，只不参与平均）
  3. 最近记录接口把脏值显示成 0 并打标，页面不会显示"5173251932 毫秒"
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="smoke_latency_"))
os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(TMP / "agent.db")
os.environ["CHECKPOINT_DB"] = str(TMP / "ckpt.db")
os.environ["ENABLE_CACHE"] = "0"
os.environ["ENABLE_RAG"] = "0"
os.environ["ENABLE_TOOLS"] = "0"
os.environ["MCP_SERVERS"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import graph, store   # noqa: E402

store.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


# ---------------------------------------------------------------- 1. 耗时计算
print("\n[1] 端到端耗时：墙钟 + 扣等待")

now = time.time()
normal = graph._elapsed_ms({"started_wall": now - 1.2})
check("正常请求：1.2 秒记成 1200 毫秒左右", 1150 <= normal <= 1350, f"实测 {normal}ms")

waited = graph._elapsed_ms({"started_wall": now - 5.0, "tool_waited_ms": 4200})
check("等人点确认的 4.2 秒从延迟里扣掉", 700 <= waited <= 900, f"实测 {waited}ms")

over = graph._elapsed_ms({"started_wall": now - 1.0, "tool_waited_ms": 30_000})
check("等待比总耗时还长 → 记 0，不记负数", over == 0, f"实测 {over}ms")

# 这一条就是线上那条脏数据的形状：起点来自另一个进程的单调钟（换算成墙钟≈60 天前）
ancient = graph._elapsed_ms({"started_wall": now - 60 * 24 * 3600})
check("时间源失效（起点在 60 天前）→ 记 0，不进统计", ancient == 0, f"实测 {ancient}ms")

legacy = graph._elapsed_ms({})
check("旧 checkpoint 没有 started_wall → 记 0，不猜", legacy == 0, f"实测 {legacy}ms")

# ---------------------------------------------------------------- 2. 统计口径
print("\n[2] 统计：异常行不参与平均，但个数要报出来")

BAD = 5_173_251_932          # 线上真实脏值（毫秒）
for lat, sid in ((100, "s-ok1"), (0, "s-cache"), (BAD, "s-bad")):
    store.record(
        query="q", route="easy", reason="r", decided_by="rule", model="m",
        latency_ms=lat, tokens_in=10, tokens_out=5, cost=0.001,
        baseline_cost=0.002, escalated=False, session_id=sid,
    )

st = store.stats()
check("平均延迟 = (100 + 0) / 2，脏值被排除", st["avg_latency_ms"] == 50,
      f"实测 {st['avg_latency_ms']}ms")
check("脏值个数单独报出来（1 条），不是悄悄丢掉", st["latency_outliers"] == 1,
      f"实测 {st['latency_outliers']}")
check("请求总数仍然如实（3 条都在账上）", st["runs"] == 3, f"实测 {st['runs']}")
check("按层统计也过滤了脏值", st["by_layer"]["rule"]["avg_latency_ms"] == 50,
      f"实测 {st['by_layer']['rule']['avg_latency_ms']}ms")

# ---------------------------------------------------------------- 3. 最近记录
print("\n[3] 最近记录：脏值显示成 0 并打标")

rows = store.recent(limit=5)
bad = next(r for r in rows if r["session_id"] == "s-bad")
ok = next(r for r in rows if r["session_id"] == "s-ok1")
check("脏值在接口层被归零（页面不会再显示 51 亿）", bad["latency_ms"] == 0,
      f"实测 {bad['latency_ms']}")
check("但标记了这一条是异常（可追查）", bad.get("latency_anomaly") is True)
check("正常记录原样返回，不被误伤", ok["latency_ms"] == 100 and "latency_anomaly" not in ok,
      f"实测 {ok['latency_ms']}")

# 库里原始值没被改写：记账是事实，只有"怎么用"变了
with store._conn() as conn:      # noqa: SLF001 — 冒烟脚本直接看原始行
    raw = conn.execute("SELECT latency_ms FROM runs WHERE session_id='s-bad'").fetchone()
check("库里的原始值保持不动（只改统计口径，不改事实）", raw["latency_ms"] == BAD,
      f"实测 {raw['latency_ms']}")

# ---------------------------------------------------------------- 汇总
print(f"\n{'=' * 46}\n通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    for name in FAIL:
        print("  失败：" + name)
    sys.exit(1)
print("全部通过")
