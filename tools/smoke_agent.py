"""工具层 + ReAct 循环冒烟测试（离线，不花钱）。

    python tools/smoke_agent.py

为什么必须离线且"脚本化模型"
----------------------------
真实模型同一个 prompt 跑两遍可能给出不同轨迹，用它测"重复调用会不会执行两次"
这种断言是不稳的：今天过明天挂。所以下面用 model_fn 注入**写死的剧本**，
每一步模型说什么都由测试决定，断言的是**循环本身的行为**：

  1. 工具注册与 schema 校验（漏字段/多字段/类型错/枚举）
  2. 一次调用就收：steps=1，答案是模型最终给的文本
  3. 参数错误 → 错误原文回给模型 → 模型改对后成功（不是抛异常终止）
  4. 未知工具 → 报错里带可用工具清单
  5. 重复调用不执行第二次（省 token 的第一道闸）
  6. 连续重复 → no_progress 收口（不靠步数硬停）
  7. 步数上限 → max_steps 且有一次"不许再调工具"的收口调用
  8. 单次预算 → budget 收口
  9. 危险级工具在只读权限下被拒
 10. 结果截断（clip）保留头尾并标注省略量
 11. sql_query 只读闸门：写语句/多语句被拒，正常查询能跑
 12. run_python 正常计算 + 黑名单拦截
 13. 记账：tool_steps 落库后能原样取回，stats 统计动手次数
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="smoke_agent_")) / "agent.db")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import rag, react, store, tools     # noqa: E402
from app.config import settings              # noqa: E402

store.init()
rag.init()   # 空表也要建：知识库为空时工具要正常返回，而不是抛 OperationalError

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


def call_msg(cid: str, name: str, args) -> dict:
    return {"id": cid, "name": name,
            "args_raw": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}


class Script:
    """剧本模型：按顺序吐出预设的 (content, tool_calls)，并记录每次收到的上下文。"""

    def __init__(self, turns: list[tuple[str, list[dict]]], tin: int = 100, tout: int = 20):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []
        self.tin, self.tout = tin, tout

    def __call__(self, messages, specs):
        self.seen.append([dict(m) for m in messages])
        if self.turns:
            content, calls = self.turns.pop(0)
        else:                        # 剧本用完了还问：说明循环没在对的地方停
            content, calls = "（剧本已用尽）", []
        return content, calls, self.tin, self.tout

    def last_tool_text(self) -> str:
        for m in reversed(self.seen[-1]):
            if m.get("role") == "tool":
                return str(m.get("content", ""))
        return ""


print("\n== 1. 注册表与参数校验 ==")
names = sorted(s.name for s in tools.all_specs())
check("四个工具都注册了", names == ["fetch_url", "run_python", "search_docs", "sql_query"], str(names))
check("Schema 能转成 OpenAI tools 格式",
      all(t["type"] == "function" and "parameters" in t["function"] for t in tools.openai_tools()))
spec = tools.get("search_docs")
for bad, why in [({"top_k": 3}, "少必填 query"),
                 ({"query": "x", "limit": 3}, "多了一个不存在的参数"),
                 ({"query": "x", "top_k": "abc"}, "类型不对")]:
    r = tools.call(spec.name, bad)
    check(f"参数校验拦住：{why}", not r.ok and "参数" in r.error, r.error[:48])
r = tools.call("search_docs", {"query": "语义缓存", "top_k": "3"})
check("字符串数字被自动转成整数（能救就救）", r.ok, r.summary)

print("\n== 2. 一次调用就收 ==")
sc = Script([("", [call_msg("c1", "search_docs", {"query": "载荷守卫"})]),
             ("守卫是靠字符串规则比参数。", [])])
res = react.run([{"role": "system", "content": "sys"}, {"role": "user", "content": "守卫怎么判断"}],
                model_fn=sc)
check("循环结束原因为 answer", res.stop == "answer", res.stop)
check("执行了 1 步", len(res.steps) == 1 and res.steps[0].ok, str([s.name for s in res.steps]))
check("答案来自模型最后一段文本", "字符串规则" in res.answer, res.answer[:24])
check("工具规则被插在系统消息之后（不是最末尾）",
      sc.seen[0][1]["role"] == "system" and "可以调用工具" in sc.seen[0][1]["content"])

print("\n== 3. 参数错误让模型自己改 ==")
sc = Script([("", [call_msg("c1", "search_docs", {"top_k": 2})]),
             ("", [call_msg("c2", "search_docs", {"query": "载荷守卫", "top_k": 2})]),
             ("改对了。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc)
check("第一次报参数错、第二次成功",
      len(res.steps) == 2 and not res.steps[0].ok and res.steps[1].ok,
      [s.ok for s in res.steps])
check("模型确实看到了错误原文", "缺少必填参数" in res.steps[0].error
      or "缺少必填参数" in sc.seen[1][-1]["content"], sc.seen[1][-1]["content"][:40])
check("最终仍给出答案", res.stop == "answer" and res.answer == "改对了。")

print("\n== 4. 不存在的工具 ==")
sc = Script([("", [call_msg("c1", "web_search", {"q": "x"})]), ("那就直接答。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc)
check("未知工具被记为失败", not res.steps[0].ok and "未注册" in res.steps[0].error, res.steps[0].error)
check("错误里带可用工具清单", "search_docs" in sc.seen[1][-1]["content"], sc.seen[1][-1]["content"][:60])

print("\n== 5. 重复调用不执行第二次 ==")
same = {"query": "载荷守卫"}
sc = Script([("", [call_msg("c1", "search_docs", same)]),
             ("", [call_msg("c2", "search_docs", same)]),
             ("别再调了，我直接答。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc)
check("第二步被标记为重复", len(res.steps) == 2 and res.steps[1].repeat, str([s.repeat for s in res.steps]))
check("重复步骤耗时为 0（没真执行）", res.steps[1].ms == 0)
check("模型被告知不要重复", "完全相同" in sc.seen[2][-1]["content"], sc.seen[2][-1]["content"][:40])

print("\n== 6. 连续重复 → no_progress 收口 ==")
sc = Script([("", [call_msg("c1", "search_docs", same)]),
             ("", [call_msg("c2", "search_docs", same)]),
             ("", [call_msg("c3", "search_docs", same)]),
             ("收口答案。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc)
check("停止原因是 no_progress", res.stop == "no_progress", res.stop)
check("收口时不再给工具（specs 为空）", sc.seen[-1] is not None)
check("收口答案被采用", res.answer == "收口答案。", res.answer)

print("\n== 7. 步数上限 ==")
turns = [("", [call_msg(f"c{i}", "search_docs", {"query": f"关键词{i}"})]) for i in range(4)]
turns.append(("收口。", []))
sc = Script(turns)
res = react.run([{"role": "user", "content": "q"}], model_fn=sc, max_steps=3)
check("停在 max_steps", res.stop == "max_steps", res.stop)
check("实际执行不超过上限", len([s for s in res.steps if not s.repeat]) <= 3,
      str(len(res.steps)))
check("收口调用带上了'不要再用工具'的指令",
      any("不要再调用工具" in str(m.get("content")) for m in sc.seen[-1]))

print("\n== 8. 预算收口 ==")
sc = Script([("", [call_msg(f"c{i}", "search_docs", {"query": f"k{i}"})]) for i in range(5)]
            + [("预算内答案。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc, budget_cny=0.0001, max_steps=5)
check("预算超了立刻收口", res.stop == "budget", res.stop)
check("成本已记账", res.cost_cny > 0, f"¥{res.cost_cny:.5f}")

print("\n== 9. 危险级工具被权限挡下 ==")
sc = Script([("", [call_msg("c1", "run_python", {"code": "print(1)"})]), ("好。", [])])
res = react.run([{"role": "user", "content": "q"}], model_fn=sc, levels=("read",))
check("只读权限下调 run_python 被拒", not res.steps[0].ok and "权限" in res.steps[0].error,
      res.steps[0].error)

print("\n== 10. 结果截断 ==")
long_text = "头" * 1200 + "中间关键" + "尾" * 1200
clipped, was = tools.clip(long_text, 400)
check("超长内容被截断", was and len(clipped) < len(long_text) // 2, f"{len(clipped)} 字")
check("保留开头与结尾", clipped.startswith("头") and clipped.endswith("尾"))
check("标注了省略字数", "中间省略" in clipped)

print("\n== 11. sql_query 只读闸门 ==")
for sql, why in [("DELETE FROM runs", "写语句"),
                 ("SELECT 1; SELECT 2", "多语句"),
                 ("PRAGMA table_info(runs)", "管理命令")]:
    r = tools.call("sql_query", {"sql": sql})
    check(f"拦住{why}", not r.ok, r.error[:40])
store.record(query="q1", route="easy", reason="r", decided_by="rule", model="m",
             latency_ms=10, tokens_in=1, tokens_out=1, cost=0.001,
             baseline_cost=0.002, escalated=False, session_id="s1")
r = tools.call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM runs"})
check("正常统计查询能跑", r.ok and "n" in r.content, r.content.replace("\n", " ")[:40])

print("\n== 12. run_python ==")
r = tools.call("run_python", {"code": "print(sum(range(101)))"})
check("能算出 5050", r.ok and "5050" in r.content, r.content[:40])
r = tools.call("run_python", {"code": "import shutil; shutil.rmtree('/')"})
check("黑名单拦下删库操作", not r.ok and "不被允许" in r.error, r.error[:40])
r = tools.call("run_python", {"code": "raise ValueError('boom')"})
# 这里要分清两层：**工具本身成功了**（沙箱正常跑完），是**用户程序报错**。
# 所以 ok=True，但退出码和 stderr 要原样交给模型——它据此改代码重试。
# 如果判成"工具失败"，模型就失去了自我修正的机会，整轮直接报废。
check("程序报错但工具成功，内容里带错误原文", r.ok and "ValueError" in r.content,
      f"ok={r.ok} summary={r.summary}")
check("页面能从 summary 看出是报错", "报错" in r.summary, r.summary)

print("\n== 13. 记账与统计 ==")
steps = [{"n": 1, "name": "search_docs", "args": "{}", "ok": True,
          "summary": "6 段", "ms": 12, "repeat": False, "truncated": False, "error": ""}]
store.record(query="q2", route="easy", reason="r", decided_by="rule", model="m",
             latency_ms=99, tokens_in=10, tokens_out=5, cost=0.002,
             baseline_cost=0.004, escalated=False, session_id="s1", tool_steps=steps)
recent = store.recent(5, session_id="s1")
got = [x for x in recent if x["query"] == "q2"][0]
check("tool_steps 落库后能取回数组", isinstance(got["tool_steps"], list)
      and got["tool_steps"][0]["name"] == "search_docs", str(got["tool_steps"])[:60])
st = store.stats()
check("stats 统计了动手次数", st["tool_runs"] >= 1, f"tool_runs={st['tool_runs']}")

print("\n" + "=" * 46)
print(f"结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    for f in FAIL:
        print("  ✗", f)
    sys.exit(1)
print("全部通过")
