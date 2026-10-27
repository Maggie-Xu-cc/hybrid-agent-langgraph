"""生成部署包：只挑运行时真正需要的文件，输出到 .deploy/。

为什么需要这一步：
    项目根目录里有 .venv313 / .venv 两个虚拟环境（合计 146MB）、本地数据库、
    证据目录。部署工具只自动排除 node_modules / .git / build，整个目录上传
    既慢又会把本地会话记录带上云。这里按**白名单**拷贝，部署包约 250KB。

用法：
    python deploy/prepare_bundle.py            # 生成/刷新 .deploy/（约 290KB，无模型）
    python deploy/prepare_bundle.py --with-bge # 连 91MB 向量模型一起打包（约 92MB）
    python deploy/prepare_bundle.py --with-bge --public   # 公网发布用（收紧工具权限）
    然后对 .deploy 目录执行发布。

--with-bge 会多做两件事，缺一不可：
    1. 拷 models/bge-small-zh/ 进包——没有权重文件，代码走到 embedder 就降级回 n-gram；
    2. 往包里的 requirements.txt 追加 onnxruntime / tokenizers——没有运行时，
       文件在也加载不了。
    两者都做了，服务器启动后 EMBED_BACKEND=auto 才会真的用上 BGE。
    实测代价：内存 +97MB、冷启动 +0.35 秒、每条向量 3~4ms（本机 CPU，2 线程）。

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
# docs 是 RAG 的知识库来源——不带上去，线上检索层就是空转（代码在、数据没在）
# mcp_servers 是外部工具服务（独立的 MCP server 进程），MCP_SERVERS 里配了就要带上
DIRS = ["app", "web", "docs", "mcp_servers"]

# 可选的向量模型（见 requirements-bge.txt 的说明）。
# 只带这 5 个文件：model.onnx 是 44KB 的图，真正的权重在 model.onnx_data 里（91MB），
# 漏了那个文件模型加载会直接报错。
BGE_SRC = ROOT / "models" / "bge-small-zh"
BGE_FILES = ["config.json", "model.onnx", "model.onnx_data",
             "tokenizer.json", "tokenizer_config.json"]
BGE_REQ = ["onnxruntime>=1.20", "tokenizers>=0.15"]

# 公网部署必须收紧的几项（--public 时覆盖进包里的 .env）。
# 为什么在打包脚本里做，而不是让大家记得改 .env：本地开发需要 run_python 才好调试，
# 一忙起来直接 `python deploy/prepare_bundle.py` 就发上去了——**默认路径必须是安全的**，
# 要放开得显式加参数。
PUBLIC_ENV = {
    # 只放只读 + 写。danger 级（run_python）等于把 shell 挂在公网页面上：
    # 提示注入一句 print(open('.env').read()) 就能把云端 API Key 读走。
    "TOOL_LEVELS": "read,write",
    "TOOL_ALLOW_DANGER": "0",
    # 写操作动手前必须有人点头。这是放开 write 的前提——没有这一条就只能开 read。
    "ENABLE_TOOL_GRAPH": "1",
    "TOOL_CONFIRM_LEVELS": "write",
    # 连一个好不相干的外部工具服务，演示"加工具只改配置"
    "MCP_SERVERS": "notes=python -m mcp_servers.notes",
    # 演示前先灌几条示例便签：一进来就能查到东西，不用等别人先写
    "NOTES_SEED": "1",
}

# 这些是本地开发产物，不能带上云
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.db", "*.db-wal", "*.db-shm", "*.log",
    "evidence",   # 实验日志，几个 MB，线上不需要
)


def _add_bge() -> int:
    """把模型文件与运行时依赖塞进部署包。返回 0 表示成功。"""
    if not BGE_SRC.is_dir():
        print(f"✗ 找不到模型目录：{BGE_SRC}\n  先跑 python tools/fetch_model.py 下载")
        return 1
    missing = [f for f in BGE_FILES if not (BGE_SRC / f).exists()]
    if missing:
        print(f"✗ 模型文件不全，缺：{missing}\n  重跑 python tools/fetch_model.py")
        return 1

    dst = OUT / "models" / "bge-small-zh"
    dst.mkdir(parents=True)
    for f in BGE_FILES:
        shutil.copy2(BGE_SRC / f, dst / f)
    size = sum((dst / f).stat().st_size for f in BGE_FILES) / 1048576
    print(f"  + models/bge-small-zh/（5 个文件，{size:.0f} MB）")

    # 依赖写进包里的 requirements.txt（不改仓库里的那份）：
    # 仓库那份是"本地能跑的最小集合"，模型是可选增强，混在一起会让
    # 每个克隆仓库的人都白装 20MB 的 onnxruntime。
    req = OUT / "requirements.txt"
    text = req.read_text(encoding="utf-8").rstrip("\n")
    req.write_text(text + "\n\n# 部署包启用 BGE 后端的运行时（prepare_bundle.py --with-bge 追加）\n"
                   + "\n".join(BGE_REQ) + "\n", encoding="utf-8")
    print(f"  + requirements.txt 追加：{', '.join(BGE_REQ)}")
    return 0


def _harden_env() -> None:
    """把公网安全项覆盖进包里的 .env（只改包里的那份，不动仓库里的 .env）。

    做法是"先删同名的键，再追加"——.env 是逐行读的，同名键后出现的会覆盖前面的，
    但留着旧的容易让人 grep 到错的那一行，干脆删干净。
    """
    env = OUT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines
            if not any(ln.strip().startswith(k + "=") for k in PUBLIC_ENV)]
    kept.append("")
    kept.append("# ===== 公网安全项（deploy/prepare_bundle.py --public 写入）=====")
    for k, v in PUBLIC_ENV.items():
        kept.append(f"{k}={v}")
    env.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print("  · 公网安全项：" + "，".join(f"{k}={v}" for k, v in PUBLIC_ENV.items()))


def main() -> int:
    with_bge = "--with-bge" in sys.argv[1:]
    public = "--public" in sys.argv[1:]

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

    if public:
        _harden_env()

    if with_bge and _add_bge() != 0:
        return 1

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    if with_bge:
        print(f"含模型：{total / 1048576:.0f} MB —— 上传会比平时慢，别以为卡住了")

    # 自查：包里绝对不能出现本地数据库
    leaked = [str(p.relative_to(OUT)) for p in OUT.rglob("*.db*")]
    if leaked:
        print(f"✗ 部署包里混进了数据库文件：{leaked}")
        return 1
    print("✓ 自查通过：无本地数据库文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
