"""LangGraph 版冒烟测试（离线 MOCK_MODE，不需要 GPU 和 API Key）。

验证四件事：
1. 三层路由在图编排下行为与 agent.py 一致（寒暄→easy 规则层、写代码→hard）；
2. 记账正常落库（bookkeep 节点）；
3. checkpointer 会话持久化：同一 session 第二轮能拿到第一轮的 history；
4. 不同 session 互相隔离。

运行：python tools/smoke_langgraph.py
"""

import os
import sys
import tempfile
from pathlib import Path

# 必须在导入 app.* 之前设好环境：离线假模型 + 临时数据库
os.environ["MOCK_MODE"] = "1"
_tmp = tempfile.mkdtemp(prefix="smoke_lg_")
os.environ["DB_PATH"] = str(Path(_tmp) / "smoke_runs.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import store  # noqa: E402
from app.graph import get_app, run_graph  # noqa: E402

PASS, FAIL = "✓", "✗"
failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {PASS if cond else FAIL} {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


print("== 1. 路由行为 ==")
store.init()  # 平时由 FastAPI startup 调用，冒烟脚本手动建表
r1 = run_graph("你好")
check("寒暄走规则层 easy", r1["route"] == "easy" and r1["decided_by"] == "rule", r1["reason"])
r2 = run_graph("帮我写一个 Python 函数实现快速排序")
check("写代码请求走 hard", r2["route"] == "hard", r2["reason"])
check("返回字段与 agent.run 一致",
      all(k in r2 for k in ("answer", "route", "latency_ms", "cost_cny", "escalated")))
check("engine 标记存在", "engine" not in r2)  # engine 由 server.py 加，graph 层不加

print("== 2. 记账 ==")
stats = store.stats()
check("前 2 次调用落库", stats["runs"] == 2, f"runs={stats['runs']}")
check("成本与基线均 > 0", stats["cost_cny"] > 0 and stats["baseline_cost_cny"] > 0)

print("== 3. 会话持久化（checkpointer）==")
# 随机 session_id：checkpoints.db 是跨运行持久化的，固定 id 会被上一次运行污染
sid = f"smoke-{os.urandom(4).hex()}"
run_graph("帮我写一个快速排序的代码", session_id=sid)  # 第一轮
run_graph("再讲讲刚才那个问题", session_id=sid)        # 第二轮
state = get_app().get_state({"configurable": {"thread_id": sid}})
history = state.values.get("history", [])
# 两轮 user+assistant 共 4 条
check("同 session 第二轮可见第一轮历史", len(history) == 4,
      f"history={len(history)} 条")
check("历史里含上一轮的用户消息",
      any(m["role"] == "user" and "快速排序" in m.get("content", "") for m in history))

print("== 4. 会话隔离 ==")
other = f"other-{os.urandom(4).hex()}"
run_graph("这是另一个会话的开场白", session_id=other)
state2 = get_app().get_state({"configurable": {"thread_id": other}})
h2 = state2.values.get("history", [])
check("不同 session 历史独立", len(h2) == 2 and not any(
    "快速排序" in m.get("content", "") for m in h2), f"history={len(h2)} 条")

print("== 5. 汇总 ==")
check("5 次冒烟调用全部落库", store.stats()["runs"] == 5,
      f"runs={store.stats()['runs']}")

print()
if failures:
    print(f"冒烟测试失败 {len(failures)} 项: {failures}")
    sys.exit(1)
print("冒烟测试全部通过（MOCK_MODE 离线，数据不可用于报告）")
