"""多智能体审查的 A/B 实测：同一套题，审查关 / 开各跑一遍。

为什么必须跑两遍而不是"看起来更靠谱就上"：审查员每次都要多花一次模型调用，
而且它可能根本抓不到东西（基线本来就对）。不量化就只能凭感觉，
而"凭感觉加一个角色"正是这类设计最常见的失败方式。

量的三件事：
  1. 事实命中率：有标准答案的题，答对了几条（facts 里任一词出现即算对）
  2. 数字无出处率：动过手的题里，答案出现"证据里没有的数字"的比例 —— 幻觉的主要形态
  3. 成本 / 延迟：审查的钱必须算进总账，否则 A/B 就没得比

用法（项目根目录；真模型）：
    TOOL_LEVELS=read,write,danger python eval/run_critic.py
    python eval/run_critic.py --limit 4      # 快跑前几条
    python eval/run_critic.py --tag 实验名

产物：
    eval/critic_ab.csv   逐条两组的答案、成本、审查结论
    终端输出对比表
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def load_cases(limit: int = 0) -> list[dict]:
    cases = [json.loads(line) for line in
             (ROOT / "eval" / "critic_set.jsonl").open(encoding="utf-8") if line.strip()]
    return cases[:limit] if limit else cases


def fact_hit(case: dict, answer: str) -> bool | None:
    """有标准答案的题：facts 里任一词出现就算对；没标准答案的返回 None（不进分母）。"""
    facts = [f.lower() for f in (case.get("facts") or [])]
    if not facts:
        return None
    text = (answer or "").lower()
    return any(f in text for f in facts)


def evidence_of(result: dict) -> tuple[str, bool]:
    """从返回里重建"这一轮手上的证据"，返回 (文本, 是否截断)。

    为什么不用 rag 片段：`result["rag"]["chunks"]` 只有出处和分数，没有正文，
    拿它当证据会把"有出处的数字"误判成"没出处的数字"。宁可少判，不要误判。
    也要用 `out`（工具真实输出）而不是 `summary`——后者是"1 行 × 2 列"这种摘要，
    拿它当证据等于没有证据（第一版指标因此虚高）。
    """
    parts, cut = [], False
    for s in (result.get("tools") or {}).get("steps") or []:
        if str(s.get("name", "")).startswith("("):
            continue
        is_cut = bool(s.get("truncated") or s.get("out_cut"))
        cut = cut or is_cut
        body = s.get("out") or s.get("summary", "")
        parts.append(f"[{s.get('name')} {s.get('args', '')}] {body}")
    return "\n".join(parts), cut


def run_arm(cases: list[dict], enabled: bool, tag: str) -> list[dict]:
    """跑一组（审查关 / 开）。settings 是运行时可改的，所以同一进程里能跑两遍。"""
    from app import graph
    from app import critic as critic_mod

    settings.enable_critic = enabled
    settings.critic_when = "tools"
    # 图里 review 节点读的是 settings（运行时读），本来不清也能生效；
    # 但既然两组之间要换行为，清一次编译缓存更不容易出"改了没生效"的糊涂账
    graph._apps.clear()

    rows = []
    for i, case in enumerate(cases, 1):
        sid = f"critic-{'on' if enabled else 'off'}-{i}-{uuid.uuid4().hex[:6]}"
        # checkpoint 也放评测目录：不复用 checkpoints.db，免得评测线程混进页面会话列表
        ckpt = str(ROOT / "eval" / "critic_ab_ckpt.db")
        result = graph.run_graph(case["q"], session_id=sid, use_cache=False,
                                 checkpoint_db=ckpt)
        # danger 级工具（run_python）会停下来等人点头。评测没人点，替它点——
        # 不点的话，计算类用例会静静停在"等你确认"，答案空白、成本 0（第一版就踩了这个）
        hops = 0
        while result.get("status") == "awaiting_confirm" and hops < 3:
            hops += 1
            result = graph.resume_graph(sid, approve=True, checkpoint_db=ckpt)
        if result.get("status") == "awaiting_confirm":
            print(f"  [跳过] {case['q'][:20]}：确认链路没走通")
        ev, cut = evidence_of(result)
        # 复用审查员那一层的数字核对：同一把尺子量两组，结论才可比
        unsupported = critic_mod.grounding_issues(result.get("answer") or "", ev,
                                                  case["q"], truncated=cut)
        rows.append({
            "case": case, "result": result, "unsupported": unsupported,
            "fact": fact_hit(case, result.get("answer") or ""),
        })
        hit = {True: "对", False: "错", None: "—"}[rows[-1]["fact"]]
        flag = "⚠数字无出处" if unsupported else ""
        print(f"[{tag} {i}/{len(cases)}] {hit} {case['q'][:26]} "
              f"¥{result.get('cost_cny', 0):.4f} {flag}")
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    fact_rows = [r for r in rows if r["fact"] is not None]
    ev_rows = [r for r in rows if (r["result"].get("tools") or {}).get("steps")]
    return {
        "arm": name,
        "n": n,
        "fact_acc": (sum(1 for r in fact_rows if r["fact"]) / len(fact_rows)) if fact_rows else None,
        "fact_n": len(fact_rows),
        "unsupported_rate": (sum(1 for r in ev_rows if r["unsupported"]) / len(ev_rows)) if ev_rows else None,
        "ev_n": len(ev_rows),
        "avg_cost": round(sum(r["result"].get("cost_cny", 0) for r in rows) / max(n, 1), 4),
        "avg_latency": round(sum(r["result"].get("latency_ms", 0) for r in rows) / max(n, 1)),
        "reviewed": sum(1 for r in rows if (r["result"].get("critic") or {}).get("rounds")),
        "caught": sum(1 for r in rows if (r["result"].get("critic") or {}).get("fixes")),
    }


def print_report(a: dict, b: dict, rows_a: list[dict], rows_b: list[dict]) -> None:
    def fmt(v, pct=True):
        if v is None:
            return "—"
        return f"{v * 100:.1f}%" if pct else str(v)

    def delta(x, y, unit="个点"):
        if x is None or y is None:
            return "—"
        return f"{(y - x) * 100:+.1f} {unit}"

    cost_pct = ((b["avg_cost"] / a["avg_cost"] - 1) * 100) if a["avg_cost"] else 0
    print("\n## 审查开/关对比\n")
    print("| 指标 | 关 | 开 | 差值 |")
    print("|---|---|---|---|")
    print(f"| 事实命中率（{a['fact_n']} 条有标准答案） | {fmt(a['fact_acc'])} | "
          f"{fmt(b['fact_acc'])} | {delta(a['fact_acc'], b['fact_acc'])} |")
    print(f"| 数字无出处率（{a['ev_n']} 条动过手） | {fmt(a['unsupported_rate'])} | "
          f"{fmt(b['unsupported_rate'])} | {delta(a['unsupported_rate'], b['unsupported_rate'])} |")
    print(f"| 平均成本 | ¥{a['avg_cost']:.4f} | ¥{b['avg_cost']:.4f} | "
          f"{cost_pct:+.0f}% |")
    print(f"| 平均延迟 | {a['avg_latency']}ms | {b['avg_latency']}ms | "
          f"{b['avg_latency'] - a['avg_latency']:+d}ms |")
    print(f"| 审查轮次 / 打回返工 | — | {b['reviewed']} / {b['caught']} | |")

    diff = [r for r in rows_b if r["unsupported"] or (r["case"].get("facts") and r["fact"] is False)]
    if diff:
        print("\n**开着审查时仍有问题/或被打回的条目**：")
        for r in diff:
            c = (r["result"].get("critic") or {})
            print(f"- {r['case']['q'][:30]} · 审查{c.get('score')}分 返工{c.get('fixes')}次"
                  f"{' · ' + '；'.join(r['unsupported']) if r['unsupported'] else ''}")


def write_csv(rows_a: list[dict], rows_b: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "query", "kind", "answer_head", "fact", "unsupported",
                    "cost_cny", "latency_ms", "critic_rounds", "critic_pass",
                    "critic_score", "critic_fixes", "critic_issues"])
        for arm, rows in (("off", rows_a), ("on", rows_b)):
            for r in rows:
                c = r["result"].get("critic") or {}
                w.writerow([arm, r["case"]["q"], r["case"].get("kind", ""),
                            (r["result"].get("answer") or "")[:80].replace("\n", " "),
                            r["fact"], "|".join(r["unsupported"]),
                            r["result"].get("cost_cny"), r["result"].get("latency_ms"),
                            c.get("rounds"), c.get("pass"), c.get("score"), c.get("fixes"),
                            "|".join(c.get("issues") or [])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cases = load_cases(args.limit)

    # 独立评测库 + 独立 checkpoint：不污染日常记账，重跑也不用先清库。
    # settings.db_path 是导入时快照，改 os.environ 没用，得直接赋值。
    from app import rag, store
    settings.db_path = str(ROOT / "eval" / "critic_ab.db")
    store.init()
    rag.init()
    if settings.enable_rag:
        # 换了库就等于换了知识库：索引存在库里，不重建的话文档类用例只能靠模型记忆
        # （表现为"答得挺像，但不是本项目的内容"——比答错更难发现）
        n = (rag.stats() or {}).get("chunks") or 0
        if not n:
            print("[准备] 评测库里没有索引，先把 docs/ 索引一遍……")
            built = rag.index_dirs(None, ROOT)
            print(f"[准备] 索引完成：{built['docs']} 篇 → {built['chunks']} 块")

    if "danger" not in settings.allowed_levels():
        print("[警告] danger 级未放行：run_python 不可用，计算类用例会退化。"
              "请带 TOOL_LEVELS=read,write,danger 运行")

    rows_off = run_arm(cases, enabled=False, tag="关")
    rows_on = run_arm(cases, enabled=True, tag="开")
    a, b = summarize("关", rows_off), summarize("开", rows_on)
    print_report(a, b, rows_off, rows_on)

    suffix = f"_{args.tag}" if args.tag else ""
    out = ROOT / "eval" / f"critic_ab{suffix}.csv"
    write_csv(rows_off, rows_on, out)
    print(f"\n明细已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
