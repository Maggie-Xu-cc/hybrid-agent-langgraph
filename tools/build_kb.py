"""建/更新知识库索引：把 docs/ 下的 markdown 切块、向量化、入库。

用法（项目根目录）：
    python tools/build_kb.py                 # 索引 .env 里 KB_DIRS 指定的目录（默认 docs）
    python tools/build_kb.py --dirs docs,kb  # 指定目录
    python tools/build_kb.py --stats         # 只看现状，不重建

什么时候要重建：
- 第一次跑（或换了 KB_DIRS）
- 改了 docs 里的文档 —— 索引是快照，不是实时视图

为什么不做成"实时监视文件变化"：
文档改动不频繁，检索时多一次 stat 检查不值当；而且显式重建让人清楚
"刚才检索用的是哪一版知识库"，调试和复现都简单。生产环境才需要增量同步。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import rag, store  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建知识库索引")
    ap.add_argument("--dirs", default="", help="逗号分隔的目录，默认读 KB_DIRS")
    ap.add_argument("--stats", action="store_true", help="只显示现状")
    args = ap.parse_args()

    store.init()
    rag.init()

    if args.stats:
        s = rag.stats()
        print(f"知识库：{s['docs']} 篇 / {s['chunks']} 块 / {s['chars']} 字")
        print(f"最后更新：{s['updated_at'] or '（空）'}")
        return

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()] or \
           [d.strip() for d in settings.kb_dirs.split(",") if d.strip()]
    print(f"索引目录：{', '.join(dirs)}  （库：{settings.db_path}）")

    r = rag.index_dirs(dirs, ROOT)
    print(f"完成：{r['docs']} 篇 → {r['chunks']} 块")
    if r["skipped"]:
        print("跳过：")
        for s in r["skipped"]:
            print(f"  - {s}")

    st = rag.stats()
    print(f"\n当前知识库：{st['docs']} 篇 / {st['chunks']} 块 / {st['chars']} 字")
    if st["chunks"] == 0:
        print("\n⚠ 索引为空。检查 --dirs 路径是否正确、目录里有没有 .md 文件。")
        print("  索引为空时检索一律返回空，检索层会静默退化（不报错）——这是设计如此，")
        print("  但意味着你以为在用 RAG，实际没有。所以这里必须明确提示。")
        sys.exit(1)


if __name__ == "__main__":
    main()
