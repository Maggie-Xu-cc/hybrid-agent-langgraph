"""下载本地向量模型（BGE-small-zh-v1.5 的 ONNX 版，约 91MB）。

    python tools/fetch_model.py            # 下载到 models/bge-small-zh
    python tools/fetch_model.py --check    # 只检查是否就绪，不下载

为什么要单独下载、不塞进 requirements
------------------------------------
pip 包只有几十 MB，模型是 91MB 的二进制权重，放 Git 仓库不合适；
而不装它系统也能跑（EMBED_BACKEND=auto 会自动退回 n-gram 后端）。
所以做成"想用再下"，而不是硬依赖。

为什么用 hf-mirror 而不是 huggingface.co
---------------------------------------
国内直连 huggingface.co 基本下不动。hf-mirror.com 是社区镜像，内容一致。

为什么是 onnx-community 这个仓库而不是 BAAI 官方仓库
-------------------------------------------------
BAAI/bge-small-zh-v1.5 只提供 PyTorch 权重，要跑起来得装 torch（2GB+）。
onnx-community 转好的 ONNX 版只要 onnxruntime（约 15MB），纯 CPU 就能推理，
单条十几毫秒——对"缓存判断"这个场景完全够用。

注意 model.onnx 只有 41KB：真正的权重在 model.onnx_data 里（ONNX 的外部数据格式），
两个文件必须放在同一目录，缺一个都加载不了。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "bge-small-zh"
BASE = "https://hf-mirror.com/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main"

# (远程路径, 本地文件名, 期望字节数下限)  下限用来识别"下载被截断"
FILES = [
    ("tokenizer.json", "tokenizer.json", 300_000),
    ("config.json", "config.json", 500),
    ("onnx/model.onnx", "model.onnx", 40_000),
    ("onnx/model.onnx_data", "model.onnx_data", 90_000_000),
]


def ready() -> bool:
    return all((DEST / name).exists() and (DEST / name).stat().st_size >= size
               for _, name, size in FILES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只检查，不下载")
    args = parser.parse_args()

    if ready():
        total = sum((DEST / n).stat().st_size for _, n, _ in FILES)
        print(f"模型已就绪：{DEST}（{total / 1e6:.0f}MB）")
        return

    if args.check:
        print(f"模型未就绪：{DEST}（跑 tools/fetch_model.py 下载）")
        sys.exit(1)

    DEST.mkdir(parents=True, exist_ok=True)
    for remote, name, min_size in FILES:
        dst = DEST / name
        if dst.exists() and dst.stat().st_size >= min_size:
            print(f"  跳过 {name}（已存在 {dst.stat().st_size / 1e6:.1f}MB）")
            continue
        print(f"  下载 {name} ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(f"{BASE}/{remote}", tmp)
        tmp.replace(dst)
        got = dst.stat().st_size
        if got < min_size:
            print(f"  下载不完整：{name} 只有 {got} 字节，期望 ≥{min_size}")
            sys.exit(1)
        print(f"  完成 {name}（{got / 1e6:.1f}MB）")

    print(f"\n模型就绪：{DEST}")
    print("验证：python tools/threshold_sweep.py --write")


if __name__ == "__main__":
    main()
