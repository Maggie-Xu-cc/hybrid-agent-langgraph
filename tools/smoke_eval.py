"""轨迹评测打分器的离线冒烟。

为什么打分器值得单独测：它是整个轨迹评测里唯一"有观点"的代码——
什么算漏触发、什么算误触发、什么算打转，全在这里定义。
跑分器坏了，12 条真模型用例白跑（烧钱又得出错误结论）。

用法：python tools/smoke_eval.py   （零网络、零模型、秒级）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_traj import grade_case, summarize, traj_from_result  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{detail}]" if detail and not cond else ""))


def run_case(case: dict, traj: dict) -> dict:
    return grade_case(case, traj)


SQL = {"q": "记账表里有多少条？", "expect_tools": True, "names": ["sql_query"],
       "min_calls": 1, "max_calls": 3, "category": "t"}
CHAT = {"q": "你好呀", "expect_tools": False, "category": "t"}
ANY = {"q": "便签里有哪些？", "expect_tools": True, "names": ["notes_search", "notes_list"],
       "min_calls": 1, "max_calls": 4, "category": "t"}


def good_sql_traj() -> dict:
    return {"used": True, "calls": 1, "stop": "", "names": ["sql_query"], "repeats": 0,
            "cost": 0.01, "latency_ms": 1200, "answer": "一共 42 条", "confirm": {}}


print("== 1. 正确轨迹 ==========")
g = run_case(SQL, good_sql_traj())
check("该动手且动了手 → PASS", g["ok"] and g["verdict"] == "PASS", str(g))
check("PASS 时没有失分项", g["flags"] == [])

g = run_case(CHAT, {"used": False, "calls": 0, "stop": "", "names": [], "repeats": 0,
                    "cost": 0.002, "latency_ms": 500, "answer": "你好！", "confirm": {}})
check("不该动手也没动 → PASS", g["ok"])

# names 任选其一：期望两个、实际只调了 list，也算选对
g = run_case(ANY, {"used": True, "calls": 1, "stop": "", "names": ["notes_list"],
                   "repeats": 0, "cost": 0.01, "latency_ms": 900, "answer": "有 3 条",
                   "confirm": {}})
check("期望任选其一，实际命中其一 → PASS", g["ok"], str(g))

print("== 2. 触发决策的四象限 ==========")
g = run_case(SQL, {**good_sql_traj(), "used": False, "calls": 0, "names": [],
                   "answer": "一共大约 40 条左右"})   # 编的
check("该动手没动手 → missed_trigger", "missed_trigger" in g["flags"], str(g))
check("漏触发必 FAIL", not g["ok"])

g = run_case(CHAT, {**good_sql_traj(), "names": ["sql_query"], "answer": "查了库"})
check("不该动手动了手 → spurious", "spurious" in g["flags"], str(g))
check("误触发必 FAIL", not g["ok"])

print("== 3. 工具选择 ==========")
g = run_case(SQL, {**good_sql_traj(), "names": ["run_python"], "calls": 1})
check("动了手但选错工具 → wrong_tool", "wrong_tool" in g["flags"], str(g))
check("选错工具不算 spurious（触发本身没错）", "spurious" not in g["flags"])

print("== 4. 过程毛病（可叠加） ==========")
g = run_case(SQL, {**good_sql_traj(), "calls": 5, "repeats": 2,
                   "stop": "no_progress", "answer": "   "})
check("步数超限 → too_many_steps", "too_many_steps" in g["flags"])
check("有重复调用 → repeated", "repeated" in g["flags"])
check("打转收口 → wandered", "wandered" in g["flags"])
check("空答案 → empty_answer", "empty_answer" in g["flags"])
check("四个失分项同时报出（不是只报第一个）", len(g["flags"]) == 4, str(g["flags"]))
check("叠加后 ok=False", not g["ok"])

# stop 的白名单：正常收口不算打转
for ok_stop in ("", "answer", "cancelled"):
    g = run_case(SQL, {**good_sql_traj(), "stop": ok_stop})
    check(f"stop={ok_stop or '(空)'} 不算 wandered", "wandered" not in g["flags"])

print("== 5. 汇总与混淆矩阵 ==========")
cases = [SQL, SQL, CHAT, CHAT, ANY]
trajs = [
    good_sql_traj(),                                   # TP
    {**good_sql_traj(), "used": False, "calls": 0, "names": [], "answer": "大概 40 条"},  # FN
    {"used": False, "calls": 0, "stop": "", "names": [], "repeats": 0,
     "cost": 0.002, "latency_ms": 400, "answer": "你好", "confirm": {}},                  # TN
    {**good_sql_traj(), "names": ["sql_query"], "answer": "查了"},                        # FP
    {"used": True, "calls": 1, "stop": "", "names": ["notes_list"], "repeats": 0,
     "cost": 0.01, "latency_ms": 900, "answer": "有", "confirm": {}},                     # TP（names 命中）
]
rows = [{"case": c, "traj": t, "grade": run_case(c, t), "case_expect": c.get("expect_tools")}
        for c, t in zip(cases, trajs)]
s = summarize(rows)
check("TP=2 FN=1 FP=1 TN=1",
      (s["tp"], s["fn"], s["fp"], s["tn"]) == (2, 1, 1, 1), str(s))
check("触发正确率 = 3/5 = 0.6", s["trigger_acc"] == 0.6, str(s["trigger_acc"]))
check("工具选对率 = 2/2 = 1.0（FN 不进分母）", s["tool_choice_acc"] == 1.0,
      str(s["tool_choice_acc"]))
check("全对数 = 3", s["pass"] == 3, str(s["pass"]))
check("平均步数 = (1+0+0+1+1)/5 = 0.6", s["avg_calls"] == 0.6, str(s["avg_calls"]))
check("总成本四舍五入到 4 位", abs(s["total_cost"] - 0.042) < 1e-9, str(s["total_cost"]))

print("== 6. traj_from_result：从真返回结构里抽轨迹 ==========")
fake = {
    "answer": "一共 42 条",
    "cost_cny": 0.0139,
    "latency_ms": 2100,
    "tools": {
        "used": True, "calls": 1, "stop": "",
        "steps": [
            {"n": 1, "name": "sql_query", "ok": True, "repeat": False},
            {"n": 2, "name": "(收口)", "ok": True, "repeat": False},
        ],
        "confirm": {"asked": 1, "approved": 1},
    },
}
t = traj_from_result(fake)
check("used/calls/stop 抽对了", (t["used"], t["calls"], t["stop"]) == (True, 1, ""))
check("括号开头的内部步骤不算动手", t["names"] == ["sql_query"], str(t["names"]))
check("repeats 统计自步骤", t["repeats"] == 0)
check("确认统计原样带出", t["confirm"].get("approved") == 1)

print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：", "、".join(FAIL))
    raise SystemExit(1)
print("全部通过")
