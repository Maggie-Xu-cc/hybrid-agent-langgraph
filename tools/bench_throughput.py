"""并发压测：测单卡有效吞吐（tok/s）。这是算力测算脚本的唯一输入参数。

用法（云端机器、vLLM 已启动、项目根目录）：
    python tools/bench_throughput.py
    python tools/bench_throughput.py --concurrency 1,4,8 --num-prompts 16
    python tools/bench_throughput.py --pad-chars 6000 --label "8k上下文"   # 长上下文对照

产出：
    docs/evidence/bench.csv    每一档并发的明细（追加写，带时间戳和 label）
    终端打印 markdown 表，最后一行 SUGGESTED_GPU_TPS=<n> 供 run_all.py 抓取

口径说明（报告里必须写清，否则数据没人信）：
    吞吐 = 本轮全部请求的输出 token 之和 / 整体墙钟时间（wall time）。
    不是把单请求的 tok/s 相加——那样会高估。并发越高吞吐越高，但延迟同步上升。
"""

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"

BASE_PROMPT = "用不超过 80 字解释 KV Cache 是什么，以及它为什么会影响显存占用。"
FILLER = "这是一段用于拉长上下文的填充文本，不承载任何有效信息，仅用于模拟长对话场景。"


def build_prompt(pad_chars: int) -> str:
    if pad_chars <= 0:
        return BASE_PROMPT
    reps = pad_chars // len(FILLER) + 1
    return (FILLER * reps)[:pad_chars] + "\n\n" + BASE_PROMPT


def one_call(client: OpenAI, model: str, prompt: str, max_tokens: int) -> tuple:
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - started
    usage = resp.usage
    return elapsed, usage.completion_tokens, usage.prompt_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="单卡并发压测")
    parser.add_argument("--base-url", default=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL", "Qwen2.5-7B-Instruct-AWQ"))
    parser.add_argument("--api-key", default=os.getenv("LOCAL_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", default="1,4,8", help="逗号分隔，例如 1,4,8")
    parser.add_argument("--num-prompts", type=int, default=16, help="每档并发的总请求数，建议 >= 并发×2")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--pad-chars", type=int, default=0, help="在 prompt 前面填充多少字符，用于造长上下文")
    parser.add_argument("--label", default="", help="本轮实验标签，写进 CSV（例如 短上下文 / 8k上下文）")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    prompt = build_prompt(args.pad_chars)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    print(f"目标：{args.base_url}  模型：{args.model}")
    print(f"上下文填充：{args.pad_chars} 字符    每档请求数：{args.num_prompts}\n")

    try:
        available = [m.id for m in client.models.list().data]
        print(f"服务端可用模型：{available}")
        if args.model not in available:
            print(f"⚠ 配置的模型名 {args.model!r} 不在列表里，压测可能 404。\n")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 连不上 {args.base_url}（{type(exc).__name__}: {exc}）")
        print("  先确认 vLLM 已启动：curl -s localhost:8000/v1/models")
        return 1

    # 预热：首次调用含算子编译/显存分配，不预热会严重低估第一档
    print("预热中（1 次调用，不计入结果）...")
    try:
        one_call(client, args.model, prompt, 32)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 预热失败（{type(exc).__name__}: {str(exc)[:160]}）")
        return 1

    rows = []
    for level in levels:
        print(f"并发 {level} 跑 {args.num_prompts} 条 ...", flush=True)
        started = time.perf_counter()
        results, failures = [], 0
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(one_call, client, args.model, prompt, args.max_tokens)
                       for _ in range(args.num_prompts)]
            for fu in futures:
                try:
                    results.append(fu.result())
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
        wall = time.perf_counter() - started

        if not results:
            print(f"  ✗ 全部失败，最后一例：{last_err}")
            continue

        out_tok = sum(r[1] for r in results)
        in_tok = sum(r[2] for r in results)
        latencies = sorted(r[0] for r in results)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        tps = out_tok / wall if wall else 0
        rows.append({
            "label": args.label or "default",
            "pad_chars": args.pad_chars,
            "concurrency": level,
            "requests": len(results),
            "failures": failures,
            "wall_s": round(wall, 2),
            "out_tokens": out_tok,
            "in_tokens": in_tok,
            "tps": round(tps, 1),
            "avg_latency_ms": round(statistics.mean(latencies) * 1000),
            "p95_latency_ms": round(p95 * 1000),
            "avg_in_tokens": round(in_tok / len(results)),
        })
        print(f"  吞吐 {tps:.1f} tok/s   平均延迟 {rows[-1]['avg_latency_ms']}ms   "
              f"P95 {rows[-1]['p95_latency_ms']}ms   失败 {failures}")

    if not rows:
        print("\n没有任何成功数据，检查 vLLM 日志。")
        return 1

    print("\n| 并发 | 请求数 | 墙钟(s) | 输出tok | 吞吐(tok/s) | 平均延迟 | P95 | 平均输入tok |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['requests']} | {r['wall_s']} | {r['out_tokens']} | "
              f"{r['tps']} | {r['avg_latency_ms']}ms | {r['p95_latency_ms']}ms | {r['avg_in_tokens']} |")

    EVID.mkdir(parents=True, exist_ok=True)
    csv_path = EVID / "bench.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["ts"])
        if new_file:
            writer.writeheader()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            writer.writerow({**r, "ts": ts})
    print(f"\n明细已写入：{csv_path}")

    # 建议值：取中档并发（没有 4 就用最高档）的吞吐，用于容量测算偏保守
    pick = next((r for r in rows if r["concurrency"] == 4), rows[-1])
    print(f"SUGGESTED_GPU_TPS={int(pick['tps'])}  # 取自并发 {pick['concurrency']} 档")
    print("\n把上面这个数填进：python tools/capacity_plan.py --gpu-tps <该值>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
