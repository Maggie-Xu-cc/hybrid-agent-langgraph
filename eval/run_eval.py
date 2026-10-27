"""三种策略跑同一套测试集，输出对比表。

用法（在项目根目录）：
    python eval/run_eval.py                  # 默认评测 agent.py 基线链路
    python eval/run_eval.py --engine graph   # 评测 LangGraph 链路
    python eval/run_eval.py --engine both    # 两条链路都跑，并逐条比对一致性

产物：
    eval/result_{engine}.csv   逐条明细（路由判定、延迟、成本）
    终端输出一份 markdown 表格，可直接贴进报告
"""

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DB = ROOT / "eval" / "eval_runs.db"
os.environ["DB_PATH"] = str(EVAL_DB)

from app import store  # noqa: E402
from app.config import settings  # noqa: E402


def load_cases() -> list[dict]:
    path = ROOT / "eval" / "testset.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_runner(engine: str):
    """返回 (query, force_route) -> result 的调用函数。

    两条链路必须走同一个签名、同一份评测循环，否则"差异"可能来自调用方式
    而不是编排方式——对照实验的基本要求。
    """
    if engine == "graph":
        from app.graph import run_graph

        def _run(query: str, force: str | None) -> dict:
            # 评测是单轮任务：每条用例用独立 thread_id，
            # 避免上一条用例的对话历史污染下一条（否则成本/答案都不可比）。
            # use_cache=False：评测要测的是路由/模型链路本身，缓存命中会把
            # 同义用例的成本拉成 0，让对照实验失去意义。
            return run_graph(query, session_id=f"eval-{uuid.uuid4().hex}",
                             force_route=force, use_cache=False)

        return _run

    from app.agent import run

    def _run(query: str, force: str | None) -> dict:
        return run(query, force_route=force)

    return _run


def run_strategy(cases: list[dict], force: str | None, escalate: bool,
                 run_one) -> list[dict]:
    settings.enable_escalation = escalate
    out = []
    for case in cases:
        result = run_one(case["q"], force)
        out.append({"case": case, "result": result})
    return out


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    lat = sum(r["result"].get("latency_ms", 0) for r in rows) / max(n, 1)
    cost = sum(r["result"].get("cost_cny", 0) for r in rows)
    base = sum(r["result"].get("baseline_cost_cny", 0) for r in rows)
    ans_len = sum(len(r["result"].get("answer", "")) for r in rows) / max(n, 1)
    esc = sum(1 for r in rows if r["result"].get("escalated"))
    deg = sum(1 for r in rows if r["result"].get("degraded"))
    hit = sum(1 for r in rows if r["result"].get("route") == r["case"]["difficulty"])
    return {
        "strategy": name,
        "n": n,
        "avg_latency_ms": round(lat),
        "total_cost": round(cost, 4),
        "vs_cloud": round(cost / base, 3) if base else 0,
        "avg_answer_chars": round(ans_len),
        "routing_acc": round(hit / n, 3) if n else 0,
        "escalated": esc,
        "degraded": deg,
    }


def print_table(summary: list[dict]) -> None:
    header = "| 策略 | 请求数 | 平均延迟 | 总成本 | 相对全云端 | 平均答案长度 | 路由准确率 | 升级次数 | 降级次数 |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for s in summary:
        acc = "—" if s["strategy"] != "混合路由" else f"{s['routing_acc'] * 100:.1f}%"
        ratio = "—" if s["strategy"] == "全云端" else f"{s['vs_cloud'] * 100:.1f}%"
        print(
            f"| {s['strategy']} | {s['n']} | {s['avg_latency_ms']}ms | "
            f"¥{s['total_cost']:.4f} | {ratio} | "
            f"{s['avg_answer_chars']} 字 | {acc} | "
            f"{s['escalated']} | {s['degraded']} |"
        )


def write_csv(engine: str, hybrid_rows: list[dict]) -> Path:
    # plain 沿用原来的 result.csv（真要跑报告时就是它），graph 另存一份方便对照
    name = "result.csv" if engine == "plain" else f"result_{engine}.csv"
    out_csv = ROOT / "eval" / name
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query", "gold", "pred", "decided_by", "reason", "model",
             "latency_ms", "tokens_in", "tokens_out", "cost_cny", "baseline_cny",
             "escalated", "engine"]
        )
        for row in hybrid_rows:
            case, res = row["case"], row["result"]
            writer.writerow([
                case["q"][:80], case["difficulty"], res.get("route"), res.get("decided_by"),
                res.get("reason"), res.get("model"), res.get("latency_ms"),
                res.get("tokens_in"), res.get("tokens_out"),
                res.get("cost_cny"), res.get("baseline_cost_cny"), res.get("escalated"),
                engine,
            ])
    return out_csv


def run_engine(engine: str, cases: list[dict]) -> dict:
    """跑完一个引擎的三策略，返回 {"summary": [...], "hybrid": rows}。"""
    # 每个引擎开始前清空记账表：保证两条链路面对同样的干净环境，
    # 也让同一实验重复跑结果一致（可复现）。删文件不如清表——不依赖文件锁。
    store.init()
    store.clear()
    print(f"===== 引擎：{engine} =====")

    run_one = make_runner(engine)
    local_rows = run_strategy(cases, force="easy", escalate=False, run_one=run_one)
    cloud_rows = run_strategy(cases, force="hard", escalate=False, run_one=run_one)
    hybrid_rows = run_strategy(cases, force=None, escalate=True, run_one=run_one)

    summary = [
        summarize("全本地", local_rows),
        summarize("全云端", cloud_rows),
        summarize("混合路由", hybrid_rows),
    ]
    print_table(summary)

    out_csv = write_csv(engine, hybrid_rows)
    print(f"明细已写入：{out_csv}\n")

    # 误判清单，用来调规则阈值
    misses = [
        (r["case"]["q"][:40], r["case"]["difficulty"], r["result"].get("route"),
         r["result"].get("reason"))
        for r in hybrid_rows
        if r["result"].get("route") != r["case"]["difficulty"]
    ]
    if misses:
        print(f"路由误判 {len(misses)} 条：")
        for q, gold, pred, reason in misses:
            print(f"  [{gold} -> {pred}] {q} | {reason}")
        print()

    return {"summary": summary, "hybrid": hybrid_rows}


def compare(plain: dict, graph: dict) -> None:
    """逐条比对两条链路的混合路由结果，证明迁移没有引入行为回归。"""
    rows_a, rows_b = plain["hybrid"], graph["hybrid"]
    n = min(len(rows_a), len(rows_b))
    same_route = same_esc = same_ans = 0
    diffs = []
    for a, b in zip(rows_a[:n], rows_b[:n]):
        q = a["case"]["q"][:30]
        ra, rb = a["result"], b["result"]
        if ra.get("route") == rb.get("route"):
            same_route += 1
        else:
            diffs.append((q, f"路由 {ra.get('route')} != {rb.get('route')}"))
        if bool(ra.get("escalated")) == bool(rb.get("escalated")):
            same_esc += 1
        else:
            diffs.append((q, f"升级 {ra.get('escalated')} != {rb.get('escalated')}"))
        if ra.get("answer") == rb.get("answer"):
            same_ans += 1
        else:
            diffs.append((q, "答案文本不一致"))

    print("===== 重构一致性（混合路由逐条对比）=====")
    print(f"| 对比项 | 一致条数 | 占比 |")
    print(f"|---|---|---|")
    for name, ok in (("路由判定", same_route), ("是否升级", same_esc), ("答案文本", same_ans)):
        print(f"| {name} | {ok}/{n} | {ok / n * 100:.1f}% |")
    if diffs:
        print(f"\n差异 {len(diffs)} 处：")
        for q, why in diffs[:10]:
            print(f"  {q} | {why}")
    else:
        print("\n两条链路逐条完全一致：证明图重构是行为等价的改写。")

    s_plain = next(s for s in plain["summary"] if s["strategy"] == "混合路由")
    s_graph = next(s for s in graph["summary"] if s["strategy"] == "混合路由")
    print(f"\n成本差异：plain ¥{s_plain['total_cost']:.4f} vs graph "
          f"¥{s_graph['total_cost']:.4f}；延迟 {s_plain['avg_latency_ms']}ms vs "
          f"{s_graph['avg_latency_ms']}ms（图的编排开销即两者之差）")


def main() -> None:
    parser = argparse.ArgumentParser(description="三策略对比评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于快速验证")
    parser.add_argument("--engine", choices=["plain", "graph", "both"], default="plain",
                        help="评测哪条链路：plain=agent.py 基线，graph=LangGraph，both=都跑并对比")
    args = parser.parse_args()

    cases = load_cases()
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"测试集：{len(cases)} 条\n")

    if args.engine == "both":
        plain = run_engine("plain", cases)
        graph = run_engine("graph", cases)
        compare(plain, graph)
    else:
        run_engine(args.engine, cases)


if __name__ == "__main__":
    main()
