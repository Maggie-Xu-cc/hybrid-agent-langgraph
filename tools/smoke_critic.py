"""多智能体审查（critic）冒烟测试（离线，不花钱）。

    python tools/smoke_critic.py

测什么
------
审查员是"第二个角色"，它的行为必须可断言，否则上线后只会变成一个神秘的加价项：

  1. 数字核对（零成本那层）：凭空数字被抓、千分位等价、年份/序数不误报
  2. 审查员自己的容错：返回垃圾、调用报错 → **放行**（不能因为它挂了把好答案毙掉）
  3. 图的行为：开关关着直通、只在工具轮次审、通过就放行、不过就打回返工
  4. 返工上限：不会无限打回（每轮都是真金白银）
  5. 规则层打回时**省掉那次 LLM 调用**（这是两层设计的意义）
  6. 审查成本计入总账（不藏起来，A/B 才有可比性）
  7. 没通过审查的答案**不写缓存**（否则错误会被复制到后面所有相似问题）
  8. 缓存命中的轮次不审（审一下就把缓存省的钱花回去了）
  9. 审查记录落库（不打进库，就没法回答"多花的钱买到了什么"）

为什么用脚本化模型：真模型同一句话跑两遍轨迹不同，断言"会不会打回"会飘。
这里断言的是**图的行为**，不是模型的能力。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="smoke_critic_"))
os.environ["MOCK_MODE"] = "1"
os.environ["DB_PATH"] = str(TMP / "agent.db")
os.environ["CHECKPOINT_DB"] = str(TMP / "ckpt.db")
os.environ["ENABLE_CACHE"] = "1"
os.environ["ENABLE_RAG"] = "0"
os.environ["ENABLE_TOOLS"] = "1"
os.environ["TOOL_LEVELS"] = "read"
os.environ["TOOL_CONFIRM_LEVELS"] = ""   # 这一篇不测确认，别让 interrupt 打断
os.environ["MCP_SERVERS"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import cache, graph, react, store, tools   # noqa: E402
from app import critic as critic_mod                # noqa: E402
from app.config import settings                     # noqa: E402

store.init()
cache.init()

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name + (f"（{detail}）" if detail else ""))


# ---------------------------------------------------------------- 假工具 + 剧本

tools.register(tools.ToolSpec(
    name="fake_search", description="查资料（只读）",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]},
    handler=lambda args: {"summary": "查到 12 条记录", "content": "查到 12 条记录"},
    level="read"))


def call(cid: str, name: str, args) -> dict:
    return {"id": cid, "name": name,
            "args_raw": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}


class Script:
    """剧本模型：按顺序吐 (content, tool_calls)，用完给 tail。"""

    def __init__(self, *steps, tail: str = "（剧本结束）") -> None:
        self.steps = list(steps)
        self.tail = tail
        self.n = 0

    def __call__(self, messages, specs):
        self.n += 1
        content, calls = (self.steps[self.n - 1] if self.n <= len(self.steps)
                          else (self.tail, []))
        return content, calls, 100, 20


class CloudScript:
    """审查员 / 返工共用的云端剧本：按调用顺序返回，并记账次数。"""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.n = 0

    def __call__(self, messages, max_tokens=None):
        self.n += 1
        text = self.texts[min(self.n - 1, len(self.texts) - 1)]
        return text, 80, 40


def verdict(passed: bool, score: int, issues=None, hint: str = "") -> str:
    return json.dumps({"pass": passed, "score": score, "issues": issues or [],
                       "hint": hint}, ensure_ascii=False)


def wire(worker: Script, critic_cloud: CloudScript, revise: CloudScript) -> None:
    """三处注入点，少一处就会有一路走真实客户端（离线必炸，或者"假绿"）。"""
    react.default_model = worker
    react._default_model = worker
    critic_mod.call_cloud = critic_cloud     # 审查员
    graph.call_cloud = revise                # 返工
    graph._apps.clear()


def ask(query: str, sid: str, use_cache: bool = False, **kw) -> dict:
    # 默认关缓存：大多数用例要测的是"模型/图的原始行为"，命中缓存会把它们跳过。
    # 只有第 8 组要测缓存与审查的交互，那里显式打开。
    return graph.run_graph(query, session_id=sid, use_cache=use_cache, use_rag=False, **kw)


TOOL_Q = "查一下资料里有多少条记录"

# ================================================================ 1. 数字核对
print("\n== 1. 数字核对（零成本那层）==")
g = critic_mod.grounding_issues
check("数字能在证据里找到 → 不报问题",
      g("一共 42 条", "工具返回：一共 42 条") == [], str(g("一共 42 条", "工具返回：一共 42 条")))
check("凭空数字 → 抓出来",
      len(g("一共 87 条", "工具返回：一共 12 条")) == 1,
      str(g("一共 87 条", "工具返回：一共 12 条")))
check("千分位与无逗号等价（1,048,576 / 1048576）",
      g("结果是 1,048,576", "算出来 1048576") == [])
check("问题里出现过的数字算合法",
      g("按你说的 87 条计算", "工具返回：无关内容", "那 87 条呢") == [])
check("个位数不误报（第 3 步 / 共 7 条）", g("第 3 步得到 7", "工具返回：无关") == [])
check("年份不误报", g("2026 年的数据", "工具返回：无关") == [])
check("没有证据时不判（交给 LLM 那层）", g("随便说 99 条", "") == [])
# 截断的证据不能定罪：工具结果超长会被截成头尾，中间的数字根本不在里面。
# 实测中因此误伤过一条正确的 SQL 聚合答案，所以这条必须钉死。
check("证据被截断时不判数字（宁漏不误）",
      g("一共 87 条", "工具返回：一共 12 条", "", truncated=True) == [])
r_cut = critic_mod.review("问题", "一共 87 条", "工具返回：12 条", truncated=True,
                          model_fn=CloudScript(verdict(True, 90)))
check("截断时走 LLM 那层而不是规则层", r_cut.by == "llm", r_cut.by)
# 派生数字必须放行：0.01 是 0.009561 四舍五入来的，判它"编的"就是误伤
check("四舍五入的派生数字 → 放行", g("平均 0.01 元", "avg = 0.009561") == [],
      str(g("平均 0.01 元", "avg = 0.009561")))
check("差得远的数字仍然报", len(g("平均 0.05 元", "avg = 0.009561")) == 1)
check("2% 以内的近似值 → 放行", g("约 101 条", "100 条") == [])
check("最多报 3 条（够提示，不刷屏）",
      len(g("1 个 22 个 33 个 44 个 55 个", "无关")) == 3,
      str(len(g("1 个 22 个 33 个 44 个 55 个", "无关"))))

# ================================================================ 2. 审查员自己的容错
print("\n== 2. 审查员自己出错时，必须放行 ==")
check("代码块包裹的 JSON 能解析",
      critic_mod._parse_json_obj('```json\n{"pass": true, "score": 90}\n```') ==
      {"pass": True, "score": 90})
check("JSON 前后有废话也能截出来",
      (critic_mod._parse_json_obj('我认为：{"pass": false, "score": 30} 以上') or {}).get("score") == 30)
check("完全不是 JSON → None", critic_mod._parse_json_obj("我觉得还行") is None)

r = critic_mod.review("问题", "答案", "证据 42", model_fn=CloudScript("不是 JSON"))
check("审查员返回垃圾 → 放行", r.passed and r.by == "error", f"{r.passed}/{r.by}")


def boom(messages, max_tokens=None):
    raise RuntimeError("审查员挂了")


r = critic_mod.review("问题", "答案", "证据 42", model_fn=boom)
check("审查员调用报错 → 放行（不拖垮主链路）", r.passed and r.by == "error", f"{r.passed}/{r.by}")

r = critic_mod.review("问题", "答案", "证据 42", model_fn=CloudScript(verdict(True, 90, ["有个小问题"])))
check("自己说 pass 但提了问题 → 打回（有意见就别放行）", not r.passed, str(r.brief()))

r = critic_mod.review("问题", "答案", "证据 42", min_score=70,
                      model_fn=CloudScript(verdict(True, 50)))
check("分数低于阈值 → 打回", not r.passed, f"score={r.score}")

msgs = critic_mod.build_revise_messages("问题", "答案", critic_mod.Review(False, 50, ["x"]), "证据")
check("返工提示禁止复述提示词骨架", "不要复述" in msgs[0]["content"] + msgs[1]["content"])

# ================================================================ 3. 图：开关与直通
print("\n== 3. 关掉开关：直通，不多花一分钱 ==")
settings.enable_critic = False
settings.critic_when = "tools"
w = Script((None, [call("c1", "fake_search", {"query": "记录"})]), ("查到 12 条记录", []))
cc, rv = CloudScript(verdict(True, 90)), CloudScript("不该被调用")
wire(w, cc, rv)
r = ask(TOOL_Q, "critic-off")
check("不审（skipped=True）", r["critic"]["skipped"] and r["critic"]["rounds"] == 0, str(r["critic"]))
check("审查员一次都没被调用", cc.n == 0, f"n={cc.n}")
cost_off = r["cost_cny"]

print("\n== 4. 打开开关，只审动过手的轮次 ==")
settings.enable_critic = True
w2 = Script((None, [call("c1", "fake_search", {"query": "记录"})]), ("查到 12 条记录", []))
cc2 = CloudScript(verdict(True, 90))
wire(w2, cc2, CloudScript("不该被调用"))
r = ask("你好呀", "critic-plain")
check("没动工具的轮次不审", r["critic"]["skipped"], str(r["critic"]))
check("普通问答的审查员调用为 0", cc2.n == 0, f"n={cc2.n}")

print("\n== 5. 动过手 + 审查通过 ==")
w3 = Script((None, [call("c1", "fake_search", {"query": "记录"})]), ("查到 12 条记录", []))
cc3 = CloudScript(verdict(True, 88))
wire(w3, cc3, CloudScript("不该被调用"))
r = ask(TOOL_Q, "critic-pass")
check("审了一轮（rounds=1）", r["critic"]["rounds"] == 1, str(r["critic"]))
check("判定通过", r["critic"]["pass"], str(r["critic"]))
check("没有返工（fixes=0）", r["critic"]["fixes"] == 0)
check("答案保持原样", "查到 12 条记录" in r["answer"], r["answer"][:40])
check("审完的花费高于不审（审查成本进总账）", r["cost_cny"] > cost_off,
      f"{r['cost_cny']} vs {cost_off}")
check("落库记下了审查轮次", r["critic"]["rounds"] == 1)
row = store.recent(1)[0]
check("记账表里 critic_rounds 落对了", row["critic_rounds"] == 1, str(row.get("critic_rounds")))

# ================================================================ 6. 规则层打回 → 返工
print("\n== 6. 凭空数字被打回 → 返工一次 → 再审通过 ==")
w4 = Script((None, [call("c1", "fake_search", {"query": "记录"})]),
            ("一共 87 条记录。", []))           # 87 不在证据里（证据是 12）
cc4 = CloudScript(verdict(True, 90))            # 返工后这一审通过
rv4 = CloudScript("一共 12 条记录。")
wire(w4, cc4, rv4)
r = ask(TOOL_Q, "critic-fix")
check("审了两轮（打回→返工→再审）", r["critic"]["rounds"] == 2, str(r["critic"]))
check("返工了一次", r["critic"]["fixes"] == 1)
check("最终通过", r["critic"]["pass"], str(r["critic"]))
check("答案是返工后的版本", "87" not in r["answer"] and "12" in r["answer"], r["answer"][:60])
check("规则层判的（by=rule 在返工前生效）", r["critic"]["by"] == "llm", str(r["critic"]["by"]))
check("返工调用了一次模型", rv4.n == 1, f"n={rv4.n}")
check("规则层打回时省掉了首轮 LLM 审查调用", cc4.n == 1, f"n={cc4.n}")
check("模型名标出了返工", "返工" in (r.get("model") or ""), str(r.get("model")))

# ================================================================ 7. 返工上限
print("\n== 7. 一直不过：返工到上限就收工 ==")
settings.critic_max_rounds = 1
w5 = Script((None, [call("c1", "fake_search", {"query": "记录"})]), ("查到 12 条记录", []))
cc5 = CloudScript(verdict(False, 30, ["答案没有说明数据来源"]))
rv5 = CloudScript("修了一下，还是没底气。")
wire(w5, cc5, rv5)
r = ask(TOOL_Q, "critic-loop")
check("返工次数等于上限（1 次）", r["critic"]["fixes"] == 1, str(r["critic"]))
check("审了两轮就停", r["critic"]["rounds"] == 2, str(r["critic"]))
check("没有无限打回（返工模型只调了 1 次）", rv5.n == 1, f"n={rv5.n}")
check("未通过也给出答案（不是整轮失败）", bool(r["answer"]), r["answer"][:30])
check("问题清单带出来了", r["critic"]["issues"], str(r["critic"]["issues"]))

# ================================================================ 8. 缓存与审查
print("\n== 8. 没过审查的答案不进缓存；缓存命中不审 ==")
settings.critic_when = "all"
settings.critic_max_rounds = 1
w6 = Script()                                  # 不调工具，走普通问答
cc6 = CloudScript(verdict(False, 20, ["凭空断言"]))
wire(w6, cc6, CloudScript("（返工）我不知道。"))
r = ask("项目最快支持多少并发？", "critic-nocache", use_cache=True)
check("非工具轮次也会审（when=all）", r["critic"]["rounds"] >= 1, str(r["critic"]))
check("没过审查的答案没进缓存", cache.lookup("项目最快支持多少并发？") is None)

cache.clear()
w7 = Script()
cc7 = CloudScript(verdict(True, 92))
wire(w7, cc7, CloudScript("不该被调用"))
r1 = ask("这个项目支持哪些省钱手段？", "critic-cache-1", use_cache=True)
check("第一轮审过并写了缓存（前提成立）",
      r1["critic"]["pass"] and cache.lookup("这个项目支持哪些省钱手段？") is not None,
      str(r1["critic"]))
before = cc7.n
r2 = ask("这个项目支持哪些省钱手段？", "critic-cache-2", use_cache=True)
check("第二轮命中缓存", r2.get("cache_hit"), str(r2.get("cache_hit")))
check("缓存命中的轮次不审", r2["critic"]["skipped"], str(r2["critic"]))
check("也没多调一次审查员", cc7.n == before, f"{before}->{cc7.n}")

print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：", "、".join(FAIL))
    raise SystemExit(1)
print("全部通过")
