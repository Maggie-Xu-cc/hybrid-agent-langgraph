"""生成部署包：只挑运行时真正需要的文件，输出到 .deploy/。

为什么需要这一步：
    项目根目录里有 .venv313 / .venv 两个虚拟环境（合计 146MB）、本地数据库、
    证据目录。部署工具只自动排除 node_modules / .git / build，整个目录上传
    既慢又会把本地会话记录带上云。这里按**白名单**拷贝，部署包约 250KB。

用法：
    python deploy/prepare_bundle.py          # 生成/刷新 .deploy/
    然后对 .deploy 目录执行发布。

注意：每次改完代码要重新跑一次——部署包是快照，不会自动跟随源码。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".deploy"

# 白名单：少一个就跑不起来，多一个就不必要
FILES = ["requirements.txt", ".env", "demo_server.py"]
DIRS = ["app", "web"]

# 这些是本地开发产物，不能带上云
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", "*.db-wal", "*.db-shm", "*.log")


def main() -> int:
    if not (ROOT / ".env").exists():
        print("✗ 缺少 .env：云端服务需要 CLOUD_API_KEY 才能回答，先准备 .env")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for name in FILES:
        src = ROOT / name
        if not src.exists():
            print(f"✗ 缺少必需文件：{name}")
            return 1
        shutil.copy2(src, OUT / name)
        print(f"  + {name}")

    for name in DIRS:
        src = ROOT / name
        if not src.is_dir():
            print(f"✗ 缺少必需目录：{name}")
            return 1
        shutil.copytree(src, OUT / name, ignore=IGNORE)
        count = sum(1 for _ in (OUT / name).rglob("*") if _.is_file())
        print(f"  + {name}/ ({count} 个文件)")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n部署包：{OUT}  ({total / 1024:.0f} KB)")

    # 自查：包里绝对不能出现本地数据库
    leaked = [str(p.relative_to(OUT)) for p in OUT.rglob("*.db*")]
    if leaked:
        print(f"✗ 部署包里混进了数据库文件：{leaked}")
        return 1
    print("✓ 自查通过：无本地数据库文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
