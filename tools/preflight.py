"""环境自检：跑评测之前先跑它，30 秒内告诉你哪里没配好。

用法（项目根目录）：
    python tools/preflight.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clients import chat, cloud_client, local_client  # noqa: E402
from app.config import settings  # noqa: E402

OK = "[OK]  "
BAD = "[FAIL]"


def check_env_file() -> tuple[bool, str]:
    env = ROOT / ".env"
    if not env.exists():
        return False, "没有 .env 文件（复制 .env.example 为 .env）"
    return True, f"已找到 {env.name}"


def check_mock() -> tuple[bool, str]:
    if settings.mock_mode:
        return False, "MOCK_MODE=1，回答是假的，不能用于报告"
    return True, "MOCK_MODE 已关闭"


def check_local() -> tuple[bool, str]:
    try:
        resp = local_client.models.list()
        ids = [m.id for m in resp.data][:3]
        if not ids:
            return False, "服务可达但没有加载模型"
        hit = settings.local_model in ids
        note = "模型名匹配" if hit else f"注意：配置里是 {settings.local_model}，实际有 {ids}"
        return True, f"可达，{note}"
    except Exception as exc:
        return False, f"连不上 {settings.local_base_url}（{type(exc).__name__}）"


def check_cloud() -> tuple[bool, str]:
    if not settings.cloud_api_key:
        return False, "CLOUD_API_KEY 为空"
    try:
        text, _, _ = chat(
            cloud_client,
            settings.cloud_model,
            [{"role": "user", "content": "只回复数字 1"}],
            max_tokens=8,
        )
        return True, f"可达，返回：{text[:20]!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}"


def main() -> None:
    print("== 环境自检 ==\n")
    checks = [
        ("配置文件", check_env_file()),
        ("Mock 模式", check_mock()),
        ("本地模型", check_local()),
        ("云端模型", check_cloud()),
    ]

    failed = []
    for name, (ok, msg) in checks:
        print(f"{OK if ok else BAD} {name:<8} {msg}")
        if not ok:
            failed.append(name)

    print(f"\n模型：本地 {settings.local_model} / 云端 {settings.cloud_model}")
    print(f"单价：本地 ¥{settings.price_local_per_1k}/千token，"
          f"云端 ¥{settings.price_cloud_in_per_1k}/¥{settings.price_cloud_out_per_1k}（入/出）")

    if not failed:
        print("\n全部通过，可以跑 python eval/run_eval.py")
        return

    print("\n== 怎么修 ==")
    if "配置文件" in failed:
        print("- 复制 .env.example 为 .env，填 CLOUD_API_KEY")
    if "Mock 模式" in failed:
        print("- .env 里删掉 MOCK_MODE=1 或改成 MOCK_MODE=0")
    if "本地模型" in failed:
        print("- 本地模型没起：ollama serve 后 ollama pull qwen2.5:7b")
        print("- 或确认 .env 的 LOCAL_BASE_URL 端口正确（Ollama 是 11434，vLLM 是 8000）")
    if "云端模型" in failed:
        print("- 检查 CLOUD_API_KEY、CLOUD_BASE_URL、CLOUD_MODEL 三行，以及账户余额")


if __name__ == "__main__":
    main()
