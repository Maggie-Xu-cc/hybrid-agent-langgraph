"""AUP / AMD APU 保底方案：不起 vLLM，直接用 PyTorch 在 GPU 上跑一次真实推理。

为什么需要这个：
AUP Learning Cloud 默认单次会话约 20 分钟，装 vLLM + 拉 4.5 GB 权重 + 起服务 + 压测来不及。
但 proposal 要证明的是"真的用上了云端 AMD 算力"，而这件事用一次 GPU 前向推理就能证明：
    - 模型权重在 GPU 上（torch.cuda.memory_allocated 会给数字）
    - 推理过程消耗 GPU 算力（rocm-smi 能看到占用）
    - 有真实输出、真实延迟、真实 token 数

用法（AUP 终端，项目根目录）：
    python deploy/aup_gpu_demo.py
    python deploy/aup_gpu_demo.py --model Qwen/Qwen2.5-3B-Instruct        # 大一点
    python deploy/aup_gpu_demo.py --local-dir /home/jovyan/models/qwen1.5b # 离线权重
    python deploy/aup_gpu_demo.py --max-new-tokens 128 --cases 12

产出：
    docs/evidence/aup_gpu_run.md   报告可直接引用的实测记录（含环境、耗时、显存峰值）

注意：ROCm 版 PyTorch 仍然用 torch.cuda 接口访问 GPU，这是正常行为，不是笔误。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"
sys.path.insert(0, str(ROOT))

# 8 条覆盖 easy / hard 的典型请求，取自 eval/testset.jsonl 的风格
CASES = [
    ("你好，请介绍一下你自己", "easy"),
    ("把「今天天气不错」翻译成英文", "easy"),
    ("REST API 和 RPC 最核心的区别是什么？", "easy"),
    ("什么是 KV Cache？", "easy"),
    ("用 Python 写一个带超时控制的进程池，并解释任务队列为什么用 deque 而不是 list", "hard"),
    ("分析下面这段 SQL 在大表上很慢的原因：SELECT * FROM orders WHERE DATE(created_at) = '2024-01-01'", "hard"),
    ("对比 vLLM 与 Ollama 在吞吐和显存管理上的设计差异", "hard"),
    ("证明：在 n 个元素的数组上，任何基于比较的排序算法最坏情况至少需要 Ω(n log n) 次比较", "hard"),
]


def env_info() -> dict:
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["hip"] = torch.version.hip or "None（不是 ROCm 版！）"
        info["gpu_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 1)
            info["gpu_cu"] = getattr(props, "multi_processor_count", "?")
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers"] = "未安装"
    return info


def main() -> int:
    p = argparse.ArgumentParser(description="AUP GPU 直接推理（不依赖 vLLM）")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace 模型名")
    p.add_argument("--local-dir", default="", help="本地权重目录；给了就不联网下载")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--cases", type=int, default=len(CASES))
    p.add_argument("--cache-dir", default="/home/jovyan/hf_cache", help="HF 缓存放持久化目录")
    args = p.parse_args()

    info = env_info()
    print("== 环境 ==")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not info.get("gpu_available"):
        print("\n✗ GPU 不可用。这只说明当前镜像/会话没有分配到 GPU，不是代码问题。")
        print("  检查：1) 启动时选的是 GPU 镜像 2) rocm-smi 能看到卡 3) torch.version.hip 不是 None")
        return 1

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer  # noqa: F401
    except ImportError as exc:
        print(f"\n✗ 缺包：{exc}")
        print("  试：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers accelerate")
        return 1

    src = args.local_dir or args.model
    print(f"\n加载模型：{src}")
    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(src, cache_dir=args.cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            src, cache_dir=args.cache_dir, torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")  # 不用 device_map：省掉 accelerate 依赖
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 加载失败：{type(exc).__name__}: {str(exc)[:300]}")
        print("  常见原因：外网慢 -> export HF_ENDPOINT=https://hf-mirror.com；显存不足 -> 换更小模型")
        return 1
    load_s = time.perf_counter() - t0
    print(f"  加载完成，用时 {load_s:.1f}s")

    alloc_gb = torch.cuda.memory_allocated() / 1024 ** 3
    reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"  权重显存：allocated {alloc_gb:.2f} GB / reserved {reserved_gb:.2f} GB")

    rows = []
    for i, (q, gold) in enumerate(CASES[: args.cases], 1):
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        n_in = int(inputs["input_ids"].shape[-1])

        started = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started

        n_out = int(out.shape[-1] - n_in)
        text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        tok_s = n_out / elapsed if elapsed else 0
        rows.append({
            "q": q, "gold": gold, "answer": text, "latency_ms": round(elapsed * 1000),
            "in_tokens": n_in, "out_tokens": n_out, "tok_per_s": round(tok_s, 1),
        })
        print(f"  [{i}/{args.cases}] {gold:<4} {n_out:>3} tok  {elapsed * 1000:>6.0f}ms  "
              f"{tok_s:>5.1f} tok/s  {text[:40].replace(chr(10), ' ')}...", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    tps = statistics.mean(r["tok_per_s"] for r in rows)
    lat = statistics.mean(r["latency_ms"] for r in rows)
    print(f"\n峰值显存：{peak_gb:.2f} GB   平均吞吐：{tps:.1f} tok/s   平均延迟：{lat:.0f} ms")

    EVID.mkdir(parents=True, exist_ok=True)
    md = [
        "# AUP Learning Cloud · GPU 推理实测记录",
        "",
        f"- 采集时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{src}`（fp16，device_map=cuda）",
        f"- torch：{info.get('torch')}　HIP：{info.get('hip')}",
        f"- GPU：{info.get('gpu_name')}　显存 {info.get('gpu_mem_gb')} GB　CU {info.get('gpu_cu')}",
        f"- 权重加载耗时：{load_s:.1f}s　峰值显存：**{peak_gb:.2f} GB**",
        f"- 单请求最大生成 {args.max_new_tokens} token，greedy 解码",
        "",
        "## 逐条结果",
        "",
        "| # | 标注 | 输入tok | 输出tok | 延迟(ms) | tok/s | 回答（截断） |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        ans = r["answer"][:40].replace("|", "／").replace("\n", " ")
        md.append(f"| {i} | {r['gold']} | {r['in_tokens']} | {r['out_tokens']} | "
                  f"{r['latency_ms']} | {r['tok_per_s']} | {ans} |")
    md += [
        "",
        "## 汇总",
        "",
        f"- 平均吞吐：**{tps:.1f} tok/s**",
        f"- 平均延迟：**{lat:.0f} ms**",
        f"- 峰值显存：**{peak_gb:.2f} GB**",
        "",
        "## 结论口径（直接写进报告）",
        "",
        "> P0 预演已在 AUP Learning Cloud 的 AMD GPU（ROCm）上完成模型前向推理实测："
        f"模型 `{src}`，峰值显存 {peak_gb:.2f} GB，平均吞吐 {tps:.1f} tok/s，平均延迟 {lat:.0f} ms。",
        "> 生产形态的 vLLM 服务化部署与并发压测因平台单次会话时长限制未完成，列为 P1 任务。",
        "",
        "## 附：原始 JSON",
        "",
        "```json",
        json.dumps({"env": info, "load_s": round(load_s, 2), "peak_gb": round(peak_gb, 2),
                    "avg_tps": round(tps, 1), "avg_latency_ms": round(lat), "rows": rows},
                   ensure_ascii=False, indent=2),
        "```",
    ]
    out = EVID / "aup_gpu_run.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写入 {out}")
    print("\n下一步：把 docs/evidence/aup_gpu_run.md 下载下来发给 AI 助手，填进报告第六、八章。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
