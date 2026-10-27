"""人在环路（HITL）冒烟测试（离线，不花钱）。

    python tools/smoke_hitl.py

为什么还是"脚本化模型"
--------------------
真实模型同一句话跑两遍可能给出不同轨迹，用它断言"写操作会不会停下来等人"
是不稳的。所以这里注入写死的剧本，断言的是**图的行为**：

  1. 写操作会让整轮**停在"等你确认"**，并且这时工具一次都没执行
  2. 确认单内容正确（工具名、参数、级别、人话解释）
  3. 点同意 → 真的执行、答案正确，轨迹标 approved
  4. 点拒绝 → 一次都不执行，模型收到"用户拒绝"，轨迹标 rejected
  5. 只读工具不打扰人（一步到底，不弹确认）
  6. 一步里两个写操作 → **只问一次**，一次批完（问两次＝点两次，体验灾难）
  7. 你犹豫的时间不算系统慢（从延迟里扣掉，单独记在 waited_ms）
  8. 挂着没确认时又问新问题 → 旧操作自动取消、不再调模型（不为放弃的操作花钱）
  9. 没有待确认却点确认 → 明确报错，不是静默成功
 10. 拒绝后模型可以换方法继续（不是一拒就死）
 11. 到步数上限 / 超预算 → 收口，不是硬跑
 12. 关掉 ENABLE_TOOL_GRAPH 退回纯函数版：不挂起（这是对照组，数字才有得比）
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="smoke_hitl_"))
os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(TMP / "agent.db")
os.environ["CHECKPOINT_DB"] = str(TMP / "ckpt.db")
os.environ["ENABLE_CACHE"] = "0"          # 缓存会让第二次提问直接命中，测不了循环
os.environ["ENABLE_RAG"] = "0"
os.environ["ENABLE_TOOLS"] = "1"
os.environ["TOOL_LEVELS"] = "read,write"  # danger 不开：跑代码不该出现在这些用例里
os.environ["TOOL_CONFIRM_LEVELS"] = "write"
os.environ["MCP_SERVERS"] = ""            # 别为了测确认去起外部服务
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import graph, react, store, tools   # noqa: E402
from app.config import settings              # noqa: E402

store.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


# ---------------------------------------------------------------- 假工具

DONE: list[dict] = []      # 真的被执行过的写操作
READS: list[dict] = []     # 真的被执行过的读操作


def _note_handler(args: dict) -> dict:
    DONE.append(dict(args))
    return {"summary": f"已记下：{args.get('text', '')}",
            "content": f"已记下便签：{args.get('text', '')}"}


def _search_handler(args: dict) -> dict:
    READS.append(dict(args))
    return {"summary": "查到 1 条", "content": "#1 周五交开题报告"}


tools.register(tools.ToolSpec(
    name="fake_note", description="记一条便签（会写数据）",
    parameters={"type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]},
    handler=_note_handler, level="write"))
tools.register(tools.ToolSpec(
    name="fake_search", description="查便签（只读）",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]},
    handler=_search_handler, level="read"))


def call(cid: str, name: str, args) -> dict:
    return {"id": cid, "name": name,
            "args_raw": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}


class Script:
    """剧本模型：按顺序吐出预设的 (content, tool_calls)；用完之后一律给 tail。"""

    def __init__(self, *steps, tail: str = "（剧本结束，给最终答案）") -> None:
        self.steps = list(steps)
        self.tail = tail
        self.n = 0
        self.msgs: list = []          # 每次调用时看到的上下文（断言"话有没有传到"）

    def __call__(self, messages, specs):
        self.n += 1
        self.msgs.append(messages)
        if self.n <= len(self.steps):
            content, calls = self.steps[self.n - 1]
        else:
            content, calls = self.tail, []
        return content, calls, 100, 20


def game(script: Script) -> None:
    """把剧本装进图里：换模型 + 清编译缓存（图是全局缓存的，不清就还用旧的）。

    两处都要换：图版（toolgraph）用的是 `react.default_model`，纯函数版（react.run）
    用的是 `react._default_model`。只换一个，对照组的测试就会悄悄走真实（mock）模型，
    断言全过但什么都没测到——这种"假绿"比红灯更危险。
    """
    react.default_model = script
    react._default_model = script
    graph._apps.clear()


def ask(query: str, sid: str, **kw) -> dict:
    return graph.run_graph(query, session_id=sid, use_cache=False, use_rag=False, **kw)


def confirm(sid: str, approve: bool, waited_ms: int = 0, note: str = "") -> dict:
    return graph.resume_graph(sid, approve=approve, note=note, waited_ms=waited_ms)


# ================================================================ 1. 挂起
print("\n== 1. 写操作会停下来等人 ==")
DONE.clear()
s1 = Script((None, [call("c1", "fake_note", {"text": "周五交材料"})]),
            ("已记下，还需要别的吗？", []))
game(s1)
r = ask("帮我记一条便签：周五交材料", "hitl-1")
check("第一轮返回 awaiting_confirm", r.get("status") == "awaiting_confirm", str(r.get("status")))
check("挂起时工具一次都没执行", DONE == [], str(DONE))
check("没有答案字段（还没跑完）", not r.get("answer"))
pending = r.get("pending") or {}
check("确认单是 tool_confirm 类型", pending.get("kind") == "tool_confirm", str(pending.get("kind")))
check("确认单点名要执行什么", "fake_note" in (pending.get("question") or ""), pending.get("question"))
one = (pending.get("calls") or [{}])[0]
check("确认单给出参数", one.get("args", {}).get("text") == "周五交材料", str(one.get("args")))
check("确认单给出人话解释", one.get("why") == "会改动数据（写进去就回不去了）", one.get("why"))
# 问题原文要跟着确认单一起走：这一轮还没跑完，历史里没有它，刷新页面就只剩一张无头卡
check("确认单带回问题原文", pending.get("query") == "帮我记一条便签：周五交材料",
      str(pending.get("query")))

# ================================================================ 2. 同意
print("\n== 2. 点同意 → 真执行 ==")
r2 = confirm("hitl-1", approve=True)
check("恢复后拿到答案", r2.get("status") == "ok" and "已记下" in (r2.get("answer") or ""),
      str(r2.get("answer"))[:40])
check("工具真的执行了一次", len(DONE) == 1 and DONE[0]["text"] == "周五交材料", str(DONE))
steps = (r2.get("tools") or {}).get("steps") or []
check("轨迹里只有一步，且标为 approved", len(steps) == 1 and steps[0]["confirm"] == "approved",
      str([(x["name"], x["confirm"]) for x in steps]))
check("确认统计：问 1 次、批 1 次、拒 0 次",
      (r2.get("tools") or {}).get("confirm") == {"asked": 1, "approved": 1, "rejected": 0},
      str((r2.get("tools") or {}).get("confirm")))

# ================================================================ 3. 拒绝
print("\n== 3. 点拒绝 → 一次都不执行 ==")
DONE.clear()
s3 = Script((None, [call("c1", "fake_note", {"text": "把这条写进去"})]),
            ("好，那我不写了。", []))
game(s3)
ask("帮我记一条便签：把这条写进去", "hitl-3")
r3 = confirm("hitl-3", approve=False, note="不想让你写")
check("拒绝后仍然有答案", r3.get("status") == "ok" and bool(r3.get("answer")), str(r3.get("answer"))[:30])
check("工具一次都没执行", DONE == [], str(DONE))
st3 = (r3.get("tools") or {}).get("steps") or []
check("轨迹标 rejected", st3 and st3[0]["confirm"] == "rejected", str([(x["name"], x["confirm"]) for x in st3]))
# 拒绝的原文要真的进到模型的上下文里：只标一个 rejected 而不告诉模型，
# 它下一轮还会原样再试一次——这是"拒绝"和"什么都没发生"的区别。
check("拒绝原因进了模型上下文",
      "不想让你写" in json.dumps(s3.msgs[-1], ensure_ascii=False),
      (s3.msgs[-1][-1].get("content", "")[:60] if s3.msgs else ""))
check("拒绝统计：问 1 次、批 0 次、拒 1 次",
      (r3.get("tools") or {}).get("confirm") == {"asked": 1, "approved": 0, "rejected": 1},
      str((r3.get("tools") or {}).get("confirm")))

# ================================================================ 4. 只读不打扰
print("\n== 4. 只读工具不弹确认 ==")
READS.clear()
s4 = Script((None, [call("c1", "fake_search", {"query": "开题"})]), ("查到了 1 条。", []))
game(s4)
r4 = ask("便签里有哪些关于开题的", "hitl-4")
check("一步到底，不挂起", r4.get("status") == "ok", str(r4.get("status")))
check("读工具正常执行", len(READS) == 1, str(READS))
st4 = (r4.get("tools") or {}).get("steps") or []
check("轨迹标 auto（没打扰人）", st4 and st4[0]["confirm"] == "auto", str([x["confirm"] for x in st4]))
check("确认统计为 0", (r4.get("tools") or {}).get("confirm") ==
      {"asked": 0, "approved": 0, "rejected": 0})

# ================================================================ 5. 一次问清
print("\n== 5. 一步两个写操作 → 只问一次 ==")
DONE.clear()
s5 = Script((None, [call("c1", "fake_note", {"text": "第一条"}),
                    call("c2", "fake_note", {"text": "第二条"})]), ("两条都记下了。", []))
game(s5)
r5 = ask("记两条便签", "hitl-5")
pend5 = r5.get("pending") or {}
check("一张确认单里有两个调用", len(pend5.get("calls") or []) == 2,
      str([c.get("name") for c in (pend5.get("calls") or [])]))
check("挂起时一条都没执行", DONE == [], str(DONE))
r5b = confirm("hitl-5", approve=True)
check("批一次两个都执行", len(DONE) == 2, str(DONE))
check("轨迹两步都是 approved",
      [x["confirm"] for x in (r5b.get("tools") or {}).get("steps") or []] == ["approved", "approved"])

# ================================================================ 6. 等待时间
print("\n== 6. 你犹豫的时间不算系统慢 ==")
s6 = Script((None, [call("c1", "fake_note", {"text": "算一下延迟"})]), ("记好了。", []))
game(s6)
ask("帮我记一条便签：算一下延迟", "hitl-6")
r6 = confirm("hitl-6", approve=True, waited_ms=5000)
check("waited_ms 记录下来", (r6.get("tools") or {}).get("waited_ms") == 5000,
      str((r6.get("tools") or {}).get("waited_ms")))
check("延迟里没算那 5 秒", int(r6.get("latency_ms") or 0) < 5000,
      f"latency={r6.get('latency_ms')}ms")

# ================================================================ 7. 换问题=取消
print("\n== 7. 挂着没确认又问新问题 → 旧操作自动取消 ==")
DONE.clear()
s7 = Script((None, [call("c1", "fake_note", {"text": "被放弃的写入"})]),
            ("新问题的回答。", []))
game(s7)
ask("帮我记一条便签：被放弃的写入", "hitl-7")
calls_before = s7.n
r7 = ask("今天几号", "hitl-7")
check("新问题照常回答", r7.get("status") == "ok" and bool(r7.get("answer")), str(r7.get("answer"))[:30])
check("被放弃的写操作没执行", DONE == [], str(DONE))
check("取消没有额外调用模型", s7.n - calls_before <= 1, f"模型调用 {s7.n - calls_before} 次")

# ================================================================ 8. 空确认
print("\n== 8. 没有待确认却点确认 ==")
r8 = confirm("hitl-不存在的会话", approve=True)
check("明确报错而不是静默成功", bool(r8.get("error")), str(r8.get("error"))[:40])

# ================================================================ 9. 拒绝后换方法
print("\n== 9. 拒绝之后模型能换方法继续 ==")
DONE.clear()
s9 = Script((None, [call("c1", "fake_note", {"text": "写一条"})]),
            (None, [call("c2", "fake_search", {"query": "写一条"})]),
            ("写不了，但我查了下：没有相关记录。", []))
game(s9)
ask("帮我记一条便签：写一条", "hitl-9")
r9 = confirm("hitl-9", approve=False)
names = [x["name"] for x in (r9.get("tools") or {}).get("steps") or []]
check("拒绝后它改用了只读工具", names == ["fake_note", "fake_search"], str(names))
check("写操作依然没执行", DONE == [], str(DONE))
check("答案说明了自己写不了", "写不了" in (r9.get("answer") or ""), str(r9.get("answer"))[:30])

# ================================================================ 10. 上限与预算
print("\n== 10. 到上限/超预算会收口 ==")
settings.tool_max_steps = 2
DONE.clear()
READS.clear()
s10 = Script((None, [call("c1", "fake_search", {"query": "a"})]),
             (None, [call("c2", "fake_search", {"query": "b"})]),
             tail="（到上限，收口答案）")
game(s10)
r10 = ask("查一下 a 和 b 和 c", "hitl-10")
check("停在 max_steps", (r10.get("tools") or {}).get("stop") == "max_steps",
      str((r10.get("tools") or {}).get("stop")))
check("收口给了答案", "收口答案" in (r10.get("answer") or ""), str(r10.get("answer"))[:30])
check("读工具最多跑 2 步", len(READS) <= 2, f"{len(READS)} 次")

settings.tool_budget_cny = 0.0001     # 一次模型调用就超
game(Script((None, [call("c1", "fake_search", {"query": "x"})]),
            (None, [call("c2", "fake_search", {"query": "y"})]),
            (None, [call("c3", "fake_search", {"query": "z"})]),
            tail="（预算到了，收口答案）"))
r11 = ask("查一下 x", "hitl-11")
check("超预算会收口", (r11.get("tools") or {}).get("stop") == "budget",
      str((r11.get("tools") or {}).get("stop")))
settings.tool_max_steps = 5
settings.tool_budget_cny = 0.05

# ================================================================ 11. 基线对照
print("\n== 11. 关掉图版（ENABLE_TOOL_GRAPH=0）→ 不挂起 ==")
settings.tool_graph = False
DONE.clear()
game(Script((None, [call("c1", "fake_note", {"text": "基线写入"})]), ("基线里直接写了。", [])))
r12 = ask("帮我记一条便签：基线写入", "hitl-12")
check("纯函数版一步跑完，不挂起", r12.get("status") == "ok", str(r12.get("status")))
check("纯函数版直接执行了写操作", len(DONE) == 1, str(DONE))
check("它的轨迹没有 approved 标记（没人把关）",
      all(x.get("confirm") == "auto" for x in (r12.get("tools") or {}).get("steps") or []),
      str([x.get("confirm") for x in (r12.get("tools") or {}).get("steps") or []]))
settings.tool_graph = True

# ---------------------------------------------------------------- 收尾
print("\n" + "=" * 46)
print(f"结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    for n in FAIL:
        print("  ✗ " + n)
    sys.exit(1)
print("全部通过")
