"""云端窗口「一键采集」：系统信息 -> 压测 -> 评测 -> 算力重算 -> 归档证据。

前提：vLLM 已经在 LOCAL_BASE_URL 上跑起来，.env 已配好（MOCK_MODE=0，CLOUD_API_KEY 有值）。

用法（项目根目录）：
    python deploy/run_all.py                 # 全套：压测(1,4,8) + 24 条评测 + 算力表
    python deploy/run_all.py --limit 10      # 时间紧：评测只跑 10 条
    python deploy/run_all.py --quick         # 只压测 + 评测，不跑长上下文对照
    python deploy/run_all.py --skip-bench    # 已压过测，只补评测和算力表

产物全部落在 docs/evidence/：
    system.txt   实例与 GPU 信息（可替代部分截图）
    models.json  服务端模型列表
    bench.log / bench.csv   压测原始输出与明细
    eval.log     三策略对比表 + 误判清单
    capacity.md  用实测吞吐重算的算力预算
    RESULTS.md   汇总，直接抄进报告

跑完记得把 docs/evidence/ 整个目录下载回本地——实例被回收，云上的东西就没了。
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def sh(cmd: list[str], timeout: int = 60) -> str:
    """跑外部命令，失败也不抛异常，返回文本（用于信息采集）。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"[{type(exc).__name__}] {' '.join(cmd)} 执行失败: {exc}\n"


def run_step(cmd: list[str], log_path: Path, title: str) -> tuple[int, str]:
    """跑关键步骤：终端实时可见 + 落盘一份。"""
    print(f"\n{'=' * 62}\n== {title}\n{'=' * 62}")
    print(f"$ {' '.join(cmd)}\n", flush=True)
    started = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    print(out, flush=True)
    log_path.write_text(out, encoding="utf-8")
    print(f"-- {title} 结束，用时 {time.time() - started:.0f}s，退出码 {p.returncode}，日志 {log_path.name}")
    return p.returncode, out


def collect_system() -> Path:
    parts = [
        ("时间", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("主机", sh(["uname", "-a"]).strip()),
        ("Python", sh([sys.executable, "-V"]).strip()),
        ("rocm-smi", sh(["rocm-smi"])),
        ("显存信息", sh(["rocm-smi", "--showmeminfo", "vram"])),
        ("显卡型号", sh(["rocm-smi", "--showproductname"])),
        ("架构代号", sh(["bash", "-lc", "rocminfo | grep -m5 gfx || true"])),
        ("vLLM 版本", sh([sys.executable, "-c", "import vllm; print(vllm.__version__)"])),
        ("torch", sh([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.hip)"])),
    ]
    text = "\n\n".join(f"## {k}\n{v}" for k, v in parts)
    path = EVID / "system.txt"
    path.write_text(text, encoding="utf-8")
    return path


def check_service() -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    url = settings.local_base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = resp.read().decode()
        (EVID / "models.json").write_text(payload, encoding="utf-8")
        ids = [m.get("id") for m in json.loads(payload).get("data", [])]
        hit = settings.local_model in ids
        return True, (f"服务可达，模型 {ids}；配置里是 {settings.local_model}"
                      + ("（匹配）" if hit else "（不匹配，先改 .env 的 LOCAL_MODEL）"))
    except urllib.error.URLError as exc:
        return False, f"连不上 {url}：{exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def extract_table(text: str) -> str:
    """从输出里抠出 markdown 表格行。"""
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    return "\n".join(lines) if lines else "（未找到表格，见原始日志）"


def main() -> int:
    parser = argparse.ArgumentParser(description="云端窗口一键采集")
    parser.add_argument("--limit", type=int, default=0, help="评测只跑前 N 条（0 = 全量）")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-cache", action="store_true", help="跳过语义缓存基准")
    parser.add_argument("--quick", action="store_true", help="跳过 8k 长上下文对照压测")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--num-prompts", type=int, default=16)
    args = parser.parse_args()

    EVID.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    summary: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    print("本地配置：")
    print(f"  LOCAL_BASE_URL = {settings.local_base_url}")
    print(f"  LOCAL_MODEL    = {settings.local_model}")
    print(f"  CLOUD_MODEL    = {settings.cloud_model}")
    print(f"  MOCK_MODE      = {settings.mock_mode}")

    if settings.mock_mode:
        print("\n✗ MOCK_MODE=1，现在跑出来的全是假数据，不能进报告。先改 .env 再跑。")
        return 1

    print("\n采集系统信息 ...")
    sys_path = collect_system()
    print(f"  -> {sys_path}")

    ok, msg = check_service()
    print(f"  服务检查：{'✓' if ok else '✗'} {msg}")
    if not ok:
        print("\n先把本地服务起起来：")
        print("  AUP（无 vLLM）：python deploy/local_server.py")
        print("  有 vLLM 的卡：  bash deploy/start_vllm.sh")
        return 1

    # ---------- 压测 ----------
    tps = None
    if not args.skip_bench:
        code, out = run_step(
            [py, "tools/bench_throughput.py",
             "--concurrency", args.concurrency,
             "--num-prompts", str(args.num_prompts),
             "--label", "默认上下文"],
            EVID / "bench.log", "并发压测（默认上下文）",
        )
        summary["steps"]["bench"] = out
        m = re.search(r"SUGGESTED_GPU_TPS=(\d+)", out)
        if m:
            tps = int(m.group(1))
            print(f"\n>>> 实测建议单卡吞吐：{tps} tok/s")

        if not args.quick and code == 0:
            run_step(
                [py, "tools/bench_throughput.py",
                 "--concurrency", "4",
                 "--num-prompts", str(max(8, args.num_prompts // 2)),
                 "--pad-chars", "6000",
                 "--label", "长上下文约8k"],
                EVID / "bench_long.log", "并发压测（长上下文对照，写进敏感性分析）",
            )
    else:
        print("\n已跳过压测。")

    if tps is None:
        tps = 800
        print(f"⚠ 没拿到实测吞吐，算力表按 {tps} tok/s 估算——报告里必须换成实测值。")

    # ---------- 评测 ----------
    if not args.skip_eval:
        cmd = [py, "eval/run_eval.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run_step(cmd, EVID / "eval.log", f"三策略评测（{args.limit or '全量'} 条）")

    # ---------- 语义缓存基准 ----------
    if not args.skip_cache:
        run_step([py, "tools/cache_bench.py", "--quiet"],
                 EVID / "cache_bench.log", "语义缓存基准（基线/精确命中/语义命中）")

    # ---------- 算力重算 ----------
    run_step(
        [py, "tools/capacity_plan.py",
         "--gpu-tps", str(tps),
         "--cloud-in", str(settings.price_cloud_in_per_1k),
         "--cloud-out", str(settings.price_cloud_out_per_1k)],
        EVID / "capacity.md", f"算力需求测算（单卡 {tps} tok/s）",
    )

    # ---------- 汇总 ----------
    bench_txt = (EVID / "bench.log").read_text(encoding="utf-8") if (EVID / "bench.log").exists() else ""
    bench_long = (EVID / "bench_long.log").read_text(encoding="utf-8") if (EVID / "bench_long.log").exists() else ""
    eval_txt = (EVID / "eval.log").read_text(encoding="utf-8") if (EVID / "eval.log").exists() else ""
    cache_txt = (EVID / "cache_bench.log").read_text(encoding="utf-8") if (EVID / "cache_bench.log").exists() else ""
    cap_txt = (EVID / "capacity.md").read_text(encoding="utf-8") if (EVID / "capacity.md").exists() else ""

    results = f"""# 云端预演结果汇总（自动生成）

- 采集时间：{summary['time']}
- 实例：{socket.gethostname()}
- 模型：`{settings.local_model}`  服务：`{settings.local_base_url}`
- 云端兜底：`{settings.cloud_model}`
- **实测单卡吞吐：{tps} tok/s**（并发压测得到，非估算）

## 一、并发压测（默认上下文）

{extract_table(bench_txt)}

## 二、并发压测（长上下文约 8k）

{extract_table(bench_long) if bench_long else "（本次未跑）"}

> 用途：PROPOSAL 8.5 节承诺的「上下文长度对吞吐的敏感性分析」。
> 若长上下文吞吐明显下降，报告里要说明容量测算为何按低值取。

## 三、语义缓存基准

{extract_table(cache_txt) if cache_txt else "（本次未跑）"}

> 第 1 遍是基线（缓存空）；第 2 遍相同问题走精确命中；第 3 遍换说法测语义召回。
> 第 2 遍命中率的上限是 easy 用例占比——hard 默认不缓存（`CACHE_CACHE_HARD=1` 可开）。

## 四、三策略评测

{extract_table(eval_txt)}

## 五、算力需求测算

{cap_txt}

---

## 还差的 4 张截图（报告附件，缺一不可）

- [ ] `rocm-smi` 开机状态（显卡型号 + 显存总量）
- [ ] `rocm-smi` **vLLM 运行中**的显存占用 ← 最硬的「确实用了云端算力」证据
- [ ] vLLM 启动日志（能看到模型名、max-model-len、gpu-memory-utilization）
- [ ] 浏览器演示页（能看到「本地小模型」标签和成本数字）

## 备份（务必做）

```bash
tar czf evidence.tgz docs/evidence/
# 然后从本地执行：
# scp <用户>@<主机>:~/hybrid-agent-demo/evidence.tgz .
```

实例一旦回收，云上的数据就没了。**每一步跑完都备份一次**，不要攒到最后。
"""
    (EVID / "RESULTS.md").write_text(results, encoding="utf-8")

    print(f"\n{'=' * 62}")
    print(f"全部完成。汇总：{EVID / 'RESULTS.md'}")
    print(f"证据目录：{EVID}")
    print("\n现在做三件事：")
    print("  1) 截 4 张图（rocm-smi ×2、vLLM 日志、浏览器页面）存进 docs/evidence/")
    print("  2) tar czf evidence.tgz docs/evidence/ 并下载回本地")
    print("  3) 把 RESULTS.md 发给 AI 助手，让它把 PROPOSAL 的待填章节补实")
    return 0


if __name__ == "__main__":
    sys.exit(main())
