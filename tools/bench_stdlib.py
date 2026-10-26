"""零依赖压测：只用 Python 标准库（urllib + threading + json），不需要 openai 包。

为什么单独写一个：AUP 镜像里不一定有 openai，而且 GPU 会话时长可能只有 20 分钟，
没时间排查依赖。/v1 服务的原始 HTTP 协议就是这段 JSON，直接发就行。

用法：
    python tools/bench_stdlib.py
    python tools/bench_stdlib.py --concurrency 1,4 --num-prompts 8 --max-tokens 128
    python tools/bench_stdlib.py --pad-chars 6000 --label 长上下文
"""

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"

FILLER = "这是一段用于拉长上下文的填充文本，不承载任何有效信息，仅用于模拟长对话场景。"
QUESTION = "用不超过 80 字解释 KV Cache 是什么，以及它为什么会影响显存占用。"


def build_prompt(pad_chars: int) -> str:
    if pad_chars <= 0:
        return QUESTION
    reps = pad_chars // len(FILLER) + 1
    return (FILLER * reps)[:pad_chars] + "\n\n" + QUESTION


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def one_call(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple:
    started = time.perf_counter()
    body = post_json(
        base.rstrip("/") + "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout,
    )
    elapsed = time.perf_counter() - started
    usage = body.get("usage") or {}
    return elapsed, int(usage.get("completion_tokens", 0)), int(usage.get("prompt_tokens", 0))


def main() -> int:
    p = argparse.ArgumentParser(description="零依赖并发压测")
    p.add_argument("--base-url", default=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.getenv("LOCAL_MODEL", "Qwen2.5-7B-AWQ"))
    p.add_argument("--concurrency", default="1,4")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--pad-chars", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    base = args.base_url
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    prompt = build_prompt(args.pad_chars)

    print(f"目标：{base}   模型：{args.model}")
    print(f"填充：{args.pad_chars} 字符   每档请求数：{args.num_prompts}\n")

    # 先确认服务活着
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=15) as r:
            ids = [m.get("id") for m in json.loads(r.read().decode()).get("data", [])]
        print(f"服务端模型：{ids}")
        if args.model not in ids:
            print(f"⚠ 配置的 {args.model!r} 不在列表里，可能 404。用上面对的名字重跑，或改 .env\n")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 连不上 {base}（{type(exc).__name__}: {exc}）")
        print("  确认 vLLM 已起：curl -s localhost:8000/v1/models")
        return 1

    print("预热 1 次（不计入）...")
    try:
        one_call(base, args.model, prompt, 32, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 预热失败：{type(exc).__name__}: {str(exc)[:200]}")
        return 1

    rows = []
    for level in levels:
        print(f"并发 {level}：{args.num_prompts} 条 ...", flush=True)
        results, failures, last_err = [], 0, ""
        lock = threading.Lock()

        def worker():
            nonlocal failures, last_err
            for _ in range(per_thread):
                try:
                    r = one_call(base, args.model, prompt, args.max_tokens, args.timeout)
                    with lock:
                        results.append(r)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        failures += 1
                        last_err = f"{type(exc).__name__}: {str(exc)[:120]}"

        per_thread = max(1, args.num_prompts // level)
        started = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(level)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - started

        if not results:
            print(f"  ✗ 全失败：{last_err}")
            continue

        out_tok = sum(r[1] for r in results)
        in_tok = sum(r[2] for r in results)
        lats = sorted(r[0] for r in results)
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
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
            "avg_latency_ms": round(statistics.mean(lats) * 1000),
            "p95_latency_ms": round(p95 * 1000),
            "avg_in_tokens": round(in_tok / len(results)),
        })
        print(f"  吞吐 {tps:.1f} tok/s   平均 {rows[-1]['avg_latency_ms']}ms   "
              f"P95 {rows[-1]['p95_latency_ms']}ms   失败 {failures}")

    if not rows:
        print("\n无有效数据。看 vLLM 日志。")
        return 1

    print("\n| 并发 | 请求数 | 墙钟(s) | 输出tok | 吞吐(tok/s) | 平均延迟 | P95 | 平均输入tok |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['requests']} | {r['wall_s']} | {r['out_tokens']} | "
              f"{r['tps']} | {r['avg_latency_ms']}ms | {r['p95_latency_ms']}ms | {r['avg_in_tokens']} |")

    EVID.mkdir(parents=True, exist_ok=True)
    path = EVID / "bench.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["ts"])
        if new:
            w.writeheader()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            w.writerow({**r, "ts": ts})
    print(f"\n已写入 {path}")

    pick = next((r for r in rows if r["concurrency"] == 4), rows[-1])
    print(f"SUGGESTED_GPU_TPS={int(pick['tps'])}  # 取自并发 {pick['concurrency']} 档")
    return 0


if __name__ == "__main__":
    sys.exit(main())
