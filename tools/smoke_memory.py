"""阶段二冒烟测试：上下文预算 + 长期记忆 + 摘要压缩。

离线跑（MOCK_MODE=1），不花钱不依赖外部服务：
    python tools/smoke_memory.py

覆盖：
  1. context.build_messages 的预算分配与裁剪
  2. memory 抽取/落库/召回/遗忘
  3. 图链路：第一轮抽取事实 → 第二轮（新会话）召回注入
  4. 历史超限时触发摘要压缩
  5. user_id 为空时不注入（保证与基线链路一致）
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="smoke_mem_")) / "mem.db")
os.environ["CHECKPOINT_DB"] = str(
    Path(tempfile.mkdtemp(prefix="smoke_ck_")) / "checkpoints.db")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import context, memory, store   # noqa: E402

store.init()
memory.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


print("== 1. 上下文预算 ==")
msgs, stats = context.build_messages(
    system="你是助手", facts={}, summary="", history=[], query="你好")
check("无历史时只有 system+user", len(msgs) == 2, f"{len(msgs)} 条")
check("stats.total 有值", stats["total"] > 0)

long_history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": "这是一条比较长的历史消息" * 20}
    for i in range(30)
]
msgs, stats = context.build_messages(
    system="你是助手", facts={}, summary="",
    history=long_history, query="继续")
check("超长历史被预算裁剪", stats["history_dropped"] > 0,
      f"裁掉 {stats['history_dropped']} 条")
check("预算内 history token 不超限", stats["history"] <= context.DEFAULT_BUDGET["history"],
      f"{stats['history']} tok")
check("本轮问题永远保留", msgs[-1]["role"] == "user" and msgs[-1]["content"] == "继续")

facts = {"name": "小明", "working_on": "agent 项目"}
msgs, stats = context.build_messages(
    system="你是助手", facts=facts, summary="用户之前聊过Rust。",
    history=[], query="我在做什么")
check("记忆注入 system", stats["memory"] > 0 and "小明" in msgs[0]["content"])
check("摘要注入 system", stats["summary"] > 0 and "Rust" in msgs[0]["content"])
big_facts = {f"fact_{i}": "很长的描述" * 30 for i in range(10)}
_, stats = context.build_messages(
    system="你是助手", facts=big_facts, summary="", history=[], query="hi")
check("记忆块超预算时整块丢弃", stats["memory"] == 0 and stats["memory_dropped"] == 1)

print("== 2. 记忆抽取与召回 ==")
f1 = memory.extract("我叫小明，我在做 agent 项目，我最喜欢的语言是Rust")
check("抽取 name", f1.get("name") == "小明", str(f1))
check("抽取 working_on", f1.get("working_on") == "agent 项目")
check("抽取 favorite", f1.get("favorite_语言") == "Rust")
check("无关句子抽不出东西", memory.extract("今天天气怎么样") == {})
check("疑问句不误记", memory.extract("我在做什么项目？") == {})

uid = "smoke-user-" + uuid.uuid4().hex[:6]
saved = memory.remember(uid, "我叫小明")
check("remember 落库", "name" in saved)
memory.remember(uid, "我在做混合路由的 agent 项目")
recalled = memory.recall(uid)
check("recall 召回两条", recalled.get("name") == "小明"
      and "agent 项目" in recalled.get("working_on", ""), str(recalled))
memory.remember(uid, "我叫小红")   # 覆盖同名事实 = 记忆更新
check("同名事实被覆盖", memory.recall(uid).get("name") == "小红")
check("forget 清空", memory.forget(uid) >= 2 and memory.recall(uid) == {})

print("== 3. 图链路：跨会话长期记忆 ==")
from app.graph import run_graph   # noqa: E402

uid = "smoke-u-" + uuid.uuid4().hex[:6]
sid1 = "smoke-s1-" + uuid.uuid4().hex[:6]
r1 = run_graph("我叫小明，我在做混合路由 agent 项目", session_id=sid1, user_id=uid)
check("第一轮落库事实", "name" in r1.get("memory", {}).get("facts_saved", []),
      str(r1.get("memory", {}).get("facts_saved")))

sid2 = "smoke-s2-" + uuid.uuid4().hex[:6]   # 全新会话
r2 = run_graph("我在做什么项目？", session_id=sid2, user_id=uid)
mem2 = r2.get("memory", {})
check("新会话召回到事实", "name" in mem2.get("facts_injected", [])
      or "working_on" in mem2.get("facts_injected", []),
      str(mem2.get("facts_injected")))
check("召回计数可观测", mem2.get("ctx_tokens", {}).get("memory", 0) > 0,
      f"注入 {mem2.get('ctx_tokens', {}).get('memory')} tok")

r3 = run_graph("我在做什么项目？", session_id="smoke-anon-" + uuid.uuid4().hex[:6],
               use_cache=False)   # 这条验证的是记忆注入，与缓存无关，关掉避免干扰
check("无 user_id 时不注入（与基线一致）",
      r3["memory"]["facts_injected"] == [] and r3["memory"]["ctx_tokens"]["memory"] == 0)

print("== 4. 历史摘要压缩 ==")
sid3 = "smoke-s3-" + uuid.uuid4().hex[:6]
summary_seen = False
for i in range(12):   # 12 轮 = 24 条 > MAX_HISTORY_MESSAGES(20)，中途必触发
    r = run_graph(f"第{i}个问题：请简单回答数字 {i}", session_id=sid3, user_id="")
    if r["memory"]["summarized_now"]:
        summary_seen = True
check("历史超限时触发摘要", summary_seen)
state = __import__("app.graph", fromlist=["get_app"]).get_app().get_state(
    {"configurable": {"thread_id": sid3}})
h = state.values.get("history", [])
check("压缩后历史回落到保留线内",
      len(h) <= 12 + 2, f"{len(h)} 条")
check("summary 已生成且持久化", len(state.values.get("summary", "")) > 0)

print("== 5. 记账完整性 ==")
stats = store.stats()
check("所有调用都落库", stats["runs"] == 1 + 1 + 1 + 12, f"runs={stats['runs']}")
check("成本与基线非负", stats["cost_cny"] >= 0 and stats["baseline_cost_cny"] >= 0)

print(f"\n通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("失败项：", FAIL)
    sys.exit(1)
print("冒烟测试全部通过")
