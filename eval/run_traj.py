"""轨迹评测：量「过程对不对」，不只是「答案对不对」。

结果评测（eval/run_eval.py）只看路由和成本，回答蒙对了、过程是错的它发现不了：
该查库时没查（编了个数字）、不该动手时动了手（白花钱）、绕了四步才到答案、
同一工具反复调。轨迹评测把这四类都记下来，每类是一个独立的失分项。

用法（项目根目录；真模型，需要云端 Key）：
    TOOL_LEVELS=read,write,danger MCP_SERVERS="notes=python -m mcp_servers.notes" \
        python eval/run_traj.py
    python eval/run_traj.py --limit 5      # 快跑前 5 条
    python eval/run_traj.py --tag 实验名    # 结果文件加后缀，方便多轮对比

产物：
    eval/traj_result[_tag].csv   逐条明细（触发、选工具、步数、失分项）
    终端输出一份 markdown 表格 + 混淆矩阵，可直接贴进报告

为什么不用 runs.db 里已存的轨迹回放：记账表只有「实际发生」的轨迹，
没有「应该怎样」的期望标签。评测的核心是拿期望比对，所以必须现场重跑。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

# 收尾步骤的名字都带括号：它们不是"动手"，只是循环的收口动作
_INTERNAL = ("(", ")")


def traj_from_result(result: dict) -> dict:
    """从 run_graph 的返回里抽出「轨迹」这份纯数据。

    为什么要抽一层：打分器只认轨迹字典，不认整个 result。
    这样冒烟测试可以凭空造轨迹来测打分器，不用起模型、不用连云端。
    """
    tools = result.get("tools") or {}
    steps = tools.get("steps") or []
    return {
        "used": bool(tools.get("used")),
        "calls": int(tools.get("calls") or 0),
        "stop": tools.get("stop") or "",
        "names": [s.get("name", "") for s in steps
                  if not str(s.get("name", "")).startswith("(")],
        "repeats": sum(1 for s in steps if s.get("repeat")),
        "cost": float(result.get("cost_cny") or 0),
        "latency_ms": int(result.get("latency_ms") or 0),
        "answer": result.get("answer") or "",
        "confirm": tools.get("confirm") or {},
    }


def grade_case(case: dict, traj: dict) -> dict:
    """给一条轨迹打分。纯函数：case 是期望，traj 是实际，返回失分项与判定。

    每个失分项独立记录（flags 是列表，不是单一 fail 布尔）——
    一条轨迹可以同时「选错工具」且「多绕两步」，只报一个 fail 会丢信息。
    """
    expect_tools = bool(case.get("expect_tools"))
    flags: list[str] = []

    # ---- 触发决策对不对（该动没动 / 不该动动了）----
    if expect_tools and not traj["used"]:
        flags.append("missed_trigger")       # 漏触发：该动手没动手
    if not expect_tools and traj["used"]:
        flags.append("spurious")             # 误触发：不该动手动了手（白花钱）

    # ---- 工具选得对不对（只查"动了手"的用例）----
    want_names = case.get("names") or []
    if expect_tools and traj["used"] and want_names:
        # names 之间是"任选其一"的关系：比如统计题查库算或跑代码算都算对
        if not set(traj["names"]) & set(want_names):
            flags.append("wrong_tool")

    # ---- 步数是否节制 ----
    max_calls = case.get("max_calls")
    if max_calls and traj["calls"] > max_calls:
        flags.append("too_many_steps")

    # ---- 过程毛病 ----
    if traj["repeats"]:
        flags.append("repeated")             # 同工具同参数反复调
    if traj["stop"] in ("no_progress", "budget", "max_steps", "error"):
        flags.append("wandered")             # 打转/超限收口，不是干净利落的回答
    if not traj["answer"].strip():
        flags.append("empty_answer")

    return {
        "ok": not flags,
        "flags": flags,
        "verdict": "PASS" if not flags else "FAIL",
    }


def summarize(rows: list[dict]) -> dict:
    """把逐条结果聚合成能贴进报告的指标。

    触发决策按混淆矩阵算：TP=该动手动了，TN=不该动没动，
    FP=不该动动了（花钱买错误），FN=该动没动（最危险，会编答案）。
    FP 和 FN 的代价完全不对称，所以分开报，不合成一个"准确率"糊弄。
    """
    tp = sum(1 for r in rows if r["case_expect"] and r["traj"]["used"])
    fn = sum(1 for r in rows if r["case_expect"] and not r["traj"]["used"])
    fp = sum(1 for r in rows if not r["case_expect"] and r["traj"]["used"])
    tn = sum(1 for r in rows if not r["case_expect"] and not r["traj"]["used"])
    tool_cases = [r for r in rows if r["case_expect"] and r["traj"]["used"]]

    def _picked_right(r: dict) -> bool:
        # names 之间是"任选其一"：执行的里只要有一个在期望清单里就算对
        return bool(set(r["traj"]["names"]) & set(r["case"].get("names") or []))

    names_ok = sum(1 for r in tool_cases if _picked_right(r))
    return {
        "n": len(rows),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "trigger_acc": round((tp + tn) / len(rows), 3) if rows else 0,
        "tool_choice_acc": round(names_ok / len(tool_cases), 3) if tool_cases else None,
        "repeats": sum(r["traj"]["repeats"] for r in rows),
        "wandered": sum(1 for r in rows if "wandered" in r["grade"]["flags"]),
        "avg_calls": round(sum(r["traj"]["calls"] for r in rows) / len(rows), 2) if rows else 0,
        "total_cost": round(sum(r["traj"]["cost"] for r in rows), 4),
        "total_latency_s": round(sum(r["traj"]["latency_ms"] for r in rows) / 1000, 1),
        "pass": sum(1 for r in rows if r["grade"]["ok"]),
    }


def print_report(rows: list[dict], s: dict) -> None:
    print("\n## 轨迹评测明细\n")
    print("| # | 问题 | 类别 | 期望动手 | 实际 | 工具 | 步数 | 判定 | 失分项 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        c, t, g = r["case"], r["traj"], r["grade"]
        exp = "要" if c.get("expect_tools") else "不要"
        print(f"| {i} | {c['q'][:24]} | {c.get('category','')} | {exp} "
              f"| {'动手' if t['used'] else '没动'} "
              f"| {','.join(t['names']) or '—'} | {t['calls']} "
              f"| {'✅' if g['ok'] else '❌'} | {'；'.join(g['flags']) or '—'} |")

    print("\n## 汇总\n")
    print("| 指标 | 值 |")
    print("|---|---|")
    print(f"| 用例数 / 全对 | {s['n']} / {s['pass']} |")
    print(f"| 触发决策：该动手动了(TP) / 漏了(FN) / 误触发(FP) / 正确不动(TN) "
          f"| {s['tp']} / {s['fn']} / {s['fp']} / {s['tn']} |")
    print(f"| 触发决策正确率 | {s['trigger_acc'] * 100:.1f}% |")
    choice = "—" if s["tool_choice_acc"] is None else f"{s['tool_choice_acc'] * 100:.1f}%"
    print(f"| 工具选对率（动了手的用例里） | {choice} |")
    print(f"| 重复调用次数 / 打转收口次数 | {s['repeats']} / {s['wandered']} |")
    print(f"| 平均步数 | {s['avg_calls']} |")
    print(f"| 总成本 / 总延迟 | ¥{s['total_cost']:.4f} / {s['total_latency_s']}s |")

    if s["fp"] or s["fn"]:
        print("\n**误触发/漏触发清单**（这两类是最值得改的）：")
        for r in rows:
            if ("spurious" in r["grade"]["flags"]) or ("missed_trigger" in r["grade"]["flags"]):
                kind = "误触发 FP" if r["grade"]["flags"].count("spurious") else "漏触发 FN"
                print(f"- [{kind}] {r['case']['q']}（{r['case'].get('category','')}）"
                      f" → 实际：{'、'.join(r['traj']['names']) or '没动手'}")


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query", "category", "expect_tools", "used", "names", "calls",
                    "stop", "repeats", "cost_cny", "latency_ms", "verdict", "flags", "answer_head"])
        for r in rows:
            c, t, g = r["case"], r["traj"], r["grade"]
            w.writerow([c["q"][:80], c.get("category", ""), c.get("expect_tools"),
                        t["used"], "|".join(t["names"]), t["calls"], t["stop"],
                        t["repeats"], t["cost"], t["latency_ms"], g["verdict"],
                        "|".join(g["flags"]), t["answer"][:60]])


def preflight(cases: list[dict]) -> list[str]:
    """跑之前先报「哪些用例注定过不了」——环境缺什么，比跑完 12 条失败 4 条清楚得多。"""
    warns = []
    allow = settings.allowed_levels()
    if not settings.enable_tools:
        warns.append("ENABLE_TOOLS 未开：所有 expect_tools=true 的用例都会漏触发")
    if "danger" not in allow and any("run_python" in (c.get("names") or []) for c in cases):
        warns.append("danger 级未放行（TOOL_LEVELS 没含 danger）：run_python 用例会选错工具，"
                     "跑的时候请带 TOOL_LEVELS=read,write,danger")
    mcp_ok = False
    try:
        from app.mcp import bridge
        st = bridge.status()
        mcp_ok = any(s.get("connected") for s in (st.get("servers") or []))
    except Exception:
        pass
    if not mcp_ok and any("notes" in n for c in cases for n in (c.get("names") or [])):
        warns.append("MCP notes 服务未连接：便签用例会漏触发，"
                     "跑的时候请带 MCP_SERVERS=\"notes=python -m mcp_servers.notes\"")
    return warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全量）")
    ap.add_argument("--tag", default="", help="结果文件后缀，方便多轮对比")
    args = ap.parse_args()

    cases = [json.loads(line) for line in
             (ROOT / "eval" / "trajset.jsonl").open(encoding="utf-8") if line.strip()]
    if args.limit:
        cases = cases[:args.limit]


    # 评测跑在独立的记账库和便签库里：不污染日常记录，重跑也不用先清库。
    # 三个快照坑，都是 import 时读环境变量导致的，改 env 不够、要改到快照本身：
    # 1) settings.db_path 是导入时快照 → 直接赋值
    # 2) graph.CHECKPOINT_DB 同理 → 用 run_graph 的 checkpoint_db 参数绕开
    # 3) settings.mcp_servers 同理 → 直接赋值。不改的话 bootstrap 拿到空配置，
    #    便签工具根本不进工具表，模型只好去翻 search_docs——上一轮评测
    #    「便签查询调了 sql_query」的 wrong_tool 就是这么来的，不是模型的错。
    import os
    settings.db_path = str(ROOT / "eval" / "traj_runs.db")
    ckpt = str(ROOT / "eval" / "traj_ckpt.db")
    os.environ.setdefault("NOTES_DB", str(ROOT / "eval" / "traj_notes.db"))
    os.environ.setdefault("MCP_SERVERS", "notes=python -m mcp_servers.notes")
    settings.mcp_servers = os.environ["MCP_SERVERS"]

    # FastAPI 的 startup 事件在评测里不会触发，桥接要自己拉起，
    # 否则便签工具根本不在工具表里，模型想调也调不到
    from app.mcp import bridge
    bridge.bootstrap()

    from app.graph import resume_graph, run_graph
    from app import store

    # 独立的评测库是空的：建表这一步平时由服务启动代办，脚本里得自己来
    store.init()

    # preflight 放在 bootstrap 之后：桥接没拉起之前问"连上了吗"，
    # 答案永远是"没连上"，会报一条吓人的假警告
    for w in preflight(cases):
        print(f"[preflight 警告] {w}")

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        sid = f"traj-{i}-{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        result = run_graph(case["q"], session_id=sid, use_cache=False,
                           checkpoint_db=ckpt)

        # 写操作会停下来等人（TOOL_CONFIRM_LEVELS 默认含 write）。
        # 评测没有用户在场，替用户点头——这也顺带验证了确认链路不挡正常流程。
        waited = 0
        while result.get("status") == "awaiting_confirm":
            waited += 1
            if waited > 3:
                result["tools"] = result.get("tools") or {}
                result["tools"]["stop"] = "error"
                break
            result = resume_graph(sid, approve=True, checkpoint_db=ckpt)
        wall = int((time.time() - t0) * 1000)

        traj = traj_from_result(result)
        grade = grade_case(case, traj)
        rows.append({"case": case, "traj": traj, "grade": grade, "wall_ms": wall,
                     "case_expect": case.get("expect_tools")})
        marks = "；".join(grade["flags"]) or "ok"
        print(f"[{i}/{len(cases)}] {'✅' if grade['ok'] else '❌'} {case['q'][:30]} "
              f"→ {'、'.join(traj['names']) or '没动手'} | {marks}")

    s = summarize(rows)
    print_report(rows, s)

    suffix = f"_{args.tag}" if args.tag else ""
    out = ROOT / "eval" / f"traj_result{suffix}.csv"
    write_csv(rows, out)
    print(f"\n明细已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
