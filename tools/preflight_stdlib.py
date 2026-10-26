"""零依赖的预检：不用 openai 包，直接 HTTP 探一遍。AUP 环境专用。

用法（项目根目录）：
    python tools/preflight_stdlib.py

做四件事：
1. 告诉你当前目录、Python、持久化目录在哪
2. 查有没有 /dev/kfd（AMD GPU 设备节点）
3. 探本地推理服务（vLLM）通不通、模型名对不对
4. 探云端 API 通不通（用 1 个 token 试一下）

全 [OK] 就可以跑 eval/run_eval_stdlib.py 了。
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

OK, BAD, WARN = "[OK]  ", "[FAIL]", "[WARN]"


def _get(url: str, headers: dict | None = None, timeout: float = 15.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 60.0) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def check_platform() -> tuple[bool, str]:
    env = ROOT / ".env"
    notes = [f"工作目录 {Path.cwd()}"]
    notes.append(f".env {'存在' if env.exists() else '不存在（复制 .env.example）'}")
    if Path("/home/jovyan").is_dir():
        notes.append("持久化目录 /home/jovyan 可用")
    else:
        notes.append("没看到 /home/jovyan（非 AUP 环境？）")
    return env.exists(), "；".join(notes)


def check_gpu() -> tuple[bool, str]:
    has_kfd = Path("/dev/kfd").exists()
    smi = shutil.which("rocm-smi") or ("/opt/rocm/bin/rocm-smi" if Path("/opt/rocm/bin/rocm-smi").exists() else None)
    if not smi:
        return False, f"/dev/kfd {'有' if has_kfd else '无'}，但找不到 rocm-smi（试 export PATH=$PATH:/opt/rocm/bin）"
    try:
        out = subprocess.run([smi, "--showproductname"], capture_output=True, text=True, timeout=20).stdout
        card = ""
        for line in out.splitlines():
            if "Card series" in line or "Card SKU" in line or "Card Model" in line:
                card = line.split(":")[-1].strip()
                break
        return True, f"/dev/kfd 存在，显卡：{card or '(见 rocm-smi 输出)'}"
    except Exception as exc:  # noqa: BLE001
        return False, f"rocm-smi 执行失败：{exc}"


def check_local() -> tuple[bool, str]:
    url = settings.local_base_url.rstrip("/") + "/models"
    try:
        _, body = _get(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"连不上 {url}（{type(exc).__name__}: {str(exc)[:80]}）"
    try:
        ids = [m.get("id") for m in json.loads(body).get("data", [])]
    except Exception:  # noqa: BLE001
        return False, f"{url} 有响应但不是 OpenAI 格式：{body[:120]}"
    if not ids:
        return False, "服务可达但没有加载任何模型"
    if settings.local_model in ids:
        return True, f"可达，模型匹配：{settings.local_model}"
    return False, f"可达，但配置里是 {settings.local_model!r}，服务端实际是 {ids}"


def check_cloud() -> tuple[bool, str]:
    if not settings.cloud_api_key or settings.cloud_api_key.startswith("sk-xxxx"):
        return False, "CLOUD_API_KEY 没填或还是占位符"
    url = settings.cloud_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.cloud_api_key}"}
    try:
        _, body = _post(url, {
            "model": settings.cloud_model,
            "messages": [{"role": "user", "content": "只回复数字 1"}],
            "max_tokens": 8,
        }, headers, timeout=60)
        text = (json.loads(body).get("choices") or [{}])[0].get("message", {}).get("content", "")
        return True, f"可达，返回 {text[:20]!r}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:160]
        return False, f"HTTP {exc.code}：{detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:100]}"


def main() -> int:
    print("== AUP 环境预检（零依赖版）==\n")
    checks = [
        ("配置文件", check_platform()),
        ("AMD GPU", check_gpu()),
        ("本地推理服务", check_local()),
        ("云端 API", check_cloud()),
    ]
    failed = []
    for name, (ok, msg) in checks:
        tag = OK if ok else BAD
        print(f"{tag} {name:<8} {msg}")
        if not ok:
            failed.append(name)

    print(f"\n本地模型：{settings.local_model}  @ {settings.local_base_url}")
    print(f"云端模型：{settings.cloud_model}  @ {settings.cloud_base_url}")
    print(f"MOCK_MODE：{settings.mock_mode}（必须是 False）")

    if not failed:
        print("\n全部通过 → python eval/run_eval_stdlib.py --limit 8")
        return 0

    print("\n== 怎么修 ==")
    if "配置文件" in failed:
        print("- cp .env.example .env，然后改 LOCAL_MODEL / CLOUD_API_KEY")
    if "AMD GPU" in failed:
        print("- 确认选的是 GPU 镜像，不是 Code Server CPU / Basic Python")
        print("- rocm-smi 不在 PATH：export PATH=$PATH:/opt/rocm/bin")
    if "本地推理服务" in failed:
        print("- vLLM 没起：bash deploy/start_vllm.sh，等看到 Uvicorn running")
        print("- 名字不一致：让 .env 的 LOCAL_MODEL 和 /v1/models 返回的一字不差")
    if "云端 API" in failed:
        print("- 检查 key、余额、CLOUD_BASE_URL 是不是以 /v1 结尾")
        print("- 云端不通也能先跑：python eval/run_eval_stdlib.py --skip-cloud")
    return 1


if __name__ == "__main__":
    sys.exit(main())
