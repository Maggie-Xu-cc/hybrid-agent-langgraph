"""零依赖 + 不依赖 openai 包的三策略评测。AUP 环境专用，输出格式与 eval/run_eval.py 一致。

为什么另写一个：
1. AUP 镜像不一定装了 openai，也不一定能装；
2. GPU 会话可能只有 20 分钟，没时间排查 ModuleNotFoundError；
3. 这个脚本只用标准库，跑得通就是跑得通。

用法（项目根目录）：
    python eval/run_eval_stdlib.py                 # 全量 24 条
    python eval/run_eval_stdlib.py --limit 10      # 时间紧就少跑
    python eval/run_eval_stdlib.py --skip-cloud    # 云端不通时先只跑本地（数据不完整但能看路由准确率）

产物：
    eval/result_stdlib.csv   逐条明细
    终端 markdown 表         直接贴报告
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.router import decide          # 纯规则 + 可选模型分类，不依赖 openai
from app.config import settings

# ---- 下面两个函数替代 app.clients，去掉 openai 依赖 ----

def _post(base: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=h,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(base: str, key: str, model: str, messages: list, max_tokens: int,
         timeout: float = 180.0) -> tuple:
    """返回 (文本, 输入 token, 输出 token)。失败抛异常，由调用方处理。"""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    body = _post(base, payload, timeout, {"Authorization": f"Bearer {key}"})
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = body.get("usage") or {}
    return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


SYSTEM = (
    "你是一个严谨的助手。用中文回答，直接给结论，不要复述问题。"
    "如果不确定，明确说\"我不确定\"，不要编造。"
)
ESCALATE_MARKERS = ["我不确定", "不确定", "无法回答", "我不知道", "作为AI", "没有足够信息"]


def needs_escalation(text: str, min_chars: int = 12) -> bool:
    if len(text.strip()) < min_chars:
        return True
    return any(m in text for m in ESCALATE_MARKERS)


def run_one(query: str, force: str | None, allow_escalation: bool, no_cloud: bool) -> dict:
    started = time.perf_counter()
    route = decide(query)
    if force in ("easy", "hard"):
        route.difficulty = force
        route.reason = f"强制{force}"

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    local_in, local_out = route.tokens_in, route.tokens_out
    cloud_in = cloud_out = 0
    escalated = degraded = cloud_fail = False
    answer, model_used = "", ""

    def call_local(m, mt):
        return chat(settings.local_base_url, settings.local_api_key,
                    settings.local_model, m, mt, timeout=180)

    def call_cloud(m, mt):
        return chat(settings.cloud_base_url, settings.cloud_api_key,
                    settings.cloud_model, m, mt, timeout=120)

    if route.difficulty == "easy":
        try:
            answer, ti, to = call_local(messages, settings.local_max_tokens)
            local_in += ti
            local_out += to
            model_used = settings.local_model
        except Exception as exc:  # noqa: BLE001
            if no_cloud:
                return {"error": f"本地失败且禁止云端: {type(exc).__name__}"}
            degraded = True
            try:
                answer, ti, to = call_cloud(messages, settings.cloud_max_tokens)
                cloud_in += ti
                cloud_out += to
                model_used = settings.cloud_model
            except Exception as exc2:  # noqa: BLE001
                return {"error": f"本地与云端都失败: {exc2}"}
        if not degraded and allow_escalation and not no_cloud and needs_escalation(answer):
            escalated = True
            try:
                a2, ti2, to2 = call_cloud(messages, settings.cloud_max_tokens)
                cloud_in += ti2
                cloud_out += to2
                answer = a2
                model_used = f"{settings.local_model} -> {settings.cloud_model}"
            except Exception:  # noqa: BLE001
                cloud_fail = True
                escalated = False  # 升级未真正生效，不计入成功升级
    else:
        if no_cloud:
            try:
                answer, ti, to = call_local(messages, settings.local_max_tokens)
                local_in += ti
                local_out += to
                model_used = settings.local_model
            except Exception as exc:  # noqa: BLE001
                return {"error": f"本地失败: {exc}"}
        else:
            try:
                answer, ti, to = call_cloud(messages, settings.cloud_max_tokens)
                cloud_in += ti
                cloud_out += to
                model_used = settings.cloud_model
            except Exception:  # noqa: BLE001
                degraded = True
                cloud_fail = True
                answer, ti, to = call_local(messages, settings.local_max_tokens)
                local_in += ti
                local_out += to
                model_used = settings.local_model

    latency = int((time.perf_counter() - started) * 1000)
    t_in, t_out = local_in + cloud_in, local_out + cloud_out
    cost = ((local_in + local_out) / 1000 * settings.price_local_per_1k
            + cloud_in / 1000 * settings.price_cloud_in_per_1k
            + cloud_out / 1000 * settings.price_cloud_out_per_1k)
    baseline = (t_in / 1000 * settings.price_cloud_in_per_1k
                + t_out / 1000 * settings.price_cloud_out_per_1k)

    return {
        "answer": answer, "route": route.difficulty, "reason": route.reason,
        "decided_by": route.by, "model": model_used, "latency_ms": latency,
        "tokens_in": t_in, "tokens_out": t_out,
        "cost_cny": round(cost, 6), "baseline_cost_cny": round(baseline, 6),
        "escalated": escalated, "degraded": degraded, "cloud_fail": cloud_fail,
    }


def summarize(name: str, rows: list) -> dict:
    n = max(len(rows), 1)
    return {
        "strategy": name,
        "n": len(rows),
        "avg_latency_ms": round(sum(r["result"].get("latency_ms", 0) for r in rows) / n),
        "total_cost": round(sum(r["result"].get("cost_cny", 0) for r in rows), 4),
        "vs_cloud": round(sum(r["result"].get("cost_cny", 0) for r in rows)
                          / max(sum(r["result"].get("baseline_cost_cny", 0) for r in rows), 1e-9), 3),
        "avg_answer_chars": round(sum(len(r["result"].get("answer", "")) for r in rows) / n),
        "routing_acc": round(sum(1 for r in rows if r["result"].get("route") == r["case"]["difficulty"]) / n, 3),
        "escalated": sum(1 for r in rows if r["result"].get("escalated")),
        "degraded": sum(1 for r in rows if r["result"].get("degraded")),
        "cloud_fail": sum(1 for r in rows if r["result"].get("cloud_fail")),
        "errors": sum(1 for r in rows if r["result"].get("error")),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="三策略评测（零依赖版）")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-cloud", action="store_true", help="云端不可用时只测本地（会低估混合策略的收益）")
    args = p.parse_args()

    cases = [json.loads(ln) for ln in (ROOT / "eval" / "testset.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"测试集 {len(cases)} 条 | 本地 {settings.local_model} | 云端 {settings.cloud_model}")
    if args.skip_cloud:
        print("⚠ 已跳过云端：这张表不能直接当结论，只用于看路由准确率\n")

    def run_all(force, escalate):
        out = []
        for i, c in enumerate(cases, 1):
            r = run_one(c["q"], force, escalate, args.skip_cloud)
            out.append({"case": c, "result": r})
            flag = "✗" if r.get("error") else ("↑" if r.get("escalated") else " ")
            print(f"  [{i}/{len(cases)}] {flag} {r.get('route', '-'):<5} {c['q'][:28]}", flush=True)
        return out

    print("\n== 策略 1/3：全本地 ==")
    local_rows = run_all("easy", False)
    if args.skip_cloud:
        cloud_rows = local_rows
    else:
        print("\n== 策略 2/3：全云端 ==")
        cloud_rows = run_all("hard", False)
    print("\n== 策略 3/3：混合路由 ==")
    hybrid_rows = run_all(None, True)

    summary = [summarize("全本地", local_rows),
               summarize("全云端", cloud_rows),
               summarize("混合路由", hybrid_rows)]

    print("\n| 策略 | 请求数 | 平均延迟 | 总成本 | 占云端牌价 | 节省 | 平均答案长度 | 路由准确率 | 升级 | 降级 | 云端失败 | 报错 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        acc = "—" if s["strategy"] != "混合路由" else f"{s['routing_acc'] * 100:.1f}%"
        ratio = "—" if s["strategy"] == "全云端" else f"{s['vs_cloud'] * 100:.1f}%"
        saved = "—" if s["strategy"] == "全云端" else f"{(1 - s['vs_cloud']) * 100:.1f}%"
        print(f"| {s['strategy']} | {s['n']} | {s['avg_latency_ms']}ms | ¥{s['total_cost']:.4f} | "
              f"{ratio} | {saved} | {s['avg_answer_chars']} 字 | {acc} | {s['escalated']} | {s['degraded']} | "
              f"{s['cloud_fail']} | {s['errors']} |")

    bad = [s for s in summary if s["cloud_fail"] > 0]
    if bad:
        print("\n⚠ 云端调用失败（已降级到本地，该策略的云端数据不可用于报告）：")
        for s in bad:
            print(f"  - {s['strategy']}：{s['cloud_fail']} 次云端失败 / 共 {s['n']} 条")
        print("  查因：python tools/preflight_stdlib.py（看云端 API 是否 [OK]）、确认 CLOUD_API_KEY 与余额")
    if any(s["strategy"] == "全云端" and s["cloud_fail"] >= s["n"] for s in summary):
        print("  ⛔ '全云端' 全部失败：表中的\"全云端\"成本其实也是本地成本，不可作为云端基线。")

    out_csv = ROOT / "eval" / "result_stdlib.csv"    #输出原始数据，可复算
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query", "gold", "pred", "decided_by", "reason", "model", "latency_ms",
                    "tokens_in", "tokens_out", "cost_cny", "baseline_cny", "escalated", "error", "cloud_fail"])
        for row in hybrid_rows:
            c, r = row["case"], row["result"]
            w.writerow([c["q"][:80], c["difficulty"], r.get("route"), r.get("decided_by"),
                        r.get("reason"), r.get("model"), r.get("latency_ms"),
                        r.get("tokens_in"), r.get("tokens_out"), r.get("cost_cny"),
                        r.get("baseline_cost_cny"), r.get("escalated"), r.get("error", ""), r.get("cloud_fail", False)])
    print(f"\n明细：{out_csv}")

    misses = [(r["case"]["q"][:40], r["case"]["difficulty"], r["result"].get("route"))
              for r in hybrid_rows if r["result"].get("route") != r["case"]["difficulty"]]
    if misses:
        print(f"\n路由误判 {len(misses)} 条：")
        for q, g, pred in misses:
            print(f"  [{g} -> {pred}] {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
