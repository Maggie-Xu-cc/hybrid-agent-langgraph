"""阈值扫描：同一套标注集，四种配置各跑一遍。

为什么必须重做这件事
--------------------
原来的验证是"24 个问题跑三遍，看命中率"。这个设计有个致命缺陷：
测试集里**只有"该命中"的样本，没有"不该命中"的样本**。
于是任何阈值看起来都很好——挑个最低的阈值，命中率最高，皆大欢喜。

学术界的做法（vCache, arXiv 2502.03771 / MeanCache 2024）：
  1. 测试集必须同时含"同义改写"（该命中）与"近似但不同"（**必须不命中**）；
  2. 指标不能只报命中率，要报 precision（服务出去的答案有多少是对的）；
  3. 阈值不能只有一个数字，要给出**整条曲线**，让人看见代价的走向。

指标对齐 vCache：
  Error rate = FP / n          错误命中占全部请求的比例（vCache 主指标）
  Hit rate   = (TP + FP) / n   缓存接管了多少请求（省钱的来源）
  Precision  = TP / (TP + FP)  接管的那些里，多少是真答对的
  Recall     = TP / (TP + FN)  该接管的，接住了多少
  F0.5       = precision 权重是 recall 的 4 倍（跟 MeanCache 一致）

为什么用 F0.5 而不是 F1：漏命中（FN）只是退回去调模型，多花一份钱，用户无感；
误命中（FP）是直接给用户一个错的答案。代价不对称，precision 必须加权。

AUC 是对"这两把尺子能不能把两类样本分开"的单一数字回答：
  AUC = P(随机取一个该命中的样本，分数高于随机取一个该拒绝的样本)
  0.5 = 完全分不开（等于瞎猜），< 0.5 = 方向反了，1.0 = 完美分开。
  它不依赖阈值，所以评价的是「尺子」本身，不是「刻度」。

四种配置对照：
  n-gram（旧）       字符重合度 + 固定阈值
  n-gram + 守卫      上面再加载荷校验
  BGE（新）          本地小模型 + 固定阈值
  BGE + 守卫         最终方案：向量粗筛 + 载荷校验两级过滤

用法：
  python tools/threshold_sweep.py                 # 四种配置全跑
  python tools/threshold_sweep.py --write         # 写 docs/evidence/threshold_sweep.md
  python tools/threshold_sweep.py --config bge_guard   # 只跑一种
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 不碰真实库：只算向量，但 import 链上会读 settings，指向临时目录最安全
_tmp = tempfile.mkdtemp(prefix="sweep_")
os.environ.setdefault("DB_PATH", str(Path(_tmp) / "sweep.db"))
os.environ.setdefault("CHECKPOINT_DB", str(Path(_tmp) / "ckpt.db"))

from app import cacheguard, embedder  # noqa: E402

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(16)] + [0.96, 0.97, 0.98, 0.99]
CURRENT = 0.45   # 旧线上阈值，表里标出来做对照

# (配置名, 后端, 是否开守卫)
CONFIGS = [
    ("n-gram", "ngram", False),
    ("n-gram + 守卫", "ngram", True),
    ("BGE", "bge", False),
    ("BGE + 守卫", "bge", True),
]


def load_pairs() -> list[dict]:
    path = ROOT / "eval" / "pairs.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def score_all(pairs: list[dict]) -> list[dict]:
    """每条查询算一次「与所有缓存条目的最大相似度」和最近邻。

    换阈值只是挪分数线，分数不用重算——所以每把尺子只算一次。
    """
    bases = sorted({p["base"] for p in pairs if p["label"] == 1})
    bvecs = dict(zip(bases, embedder.embed_batch(bases)))
    qvecs = embedder.embed_batch([p["q"] for p in pairs])

    scored: list[dict] = []
    for p, qv in zip(pairs, qvecs):
        best_base, best_score = "", -1.0
        for b in bases:
            s = cosine(qv, bvecs[b])
            if s > best_score:
                best_base, best_score = b, s
        scored.append({**p, "score": best_score, "nearest": best_base,
                       "correct_ok": p["label"] == 1 and best_base == p["base"]})
    return scored


def sweep(scored: list[dict], use_guard: bool) -> list[dict]:
    rows: list[dict] = []
    n = len(scored)
    for thr in THRESHOLDS:
        tp = fp = fn = tn = 0
        for s in scored:
            hit = s["score"] >= thr
            if hit and use_guard:
                # 过了分数线还要过载荷校验：向量只说"同一件事"，不说"参数也一样"
                ok, _ = cacheguard.check(s["q"], s["nearest"])
                hit = ok
            if hit and s["correct_ok"]:
                tp += 1
            elif hit:
                fp += 1                     # 过线但答的不是该答的 = 服务了错答案
            elif s["label"] == 1:
                fn += 1                     # 该命中没命中
            else:
                tn += 1                     # 正确拒绝，退回大模型
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f05 = (1.25 * precision * recall / (0.25 * precision + recall)
               if precision + recall else 0.0)
        rows.append({"thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": precision, "recall": recall, "f05": f05,
                     "hit_rate": (tp + fp) / n, "error_rate": fp / n,
                     "correct_rate": (tp + tn) / n})
    return rows


def auc(pos: list[float], neg: list[float]) -> float:
    """P(正样本分数 > 负样本分数)，平局算半分。不用 numpy，规模小算得动。"""
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def key_metrics(rows: list[dict]) -> dict:
    cur = min(rows, key=lambda r: abs(r["thr"] - CURRENT))
    best = max(rows, key=lambda r: r["f05"])
    safe = [r for r in rows if r["precision"] >= 0.95 and (r["tp"] + r["fp"]) > 0]
    return {"cur": cur, "best": best,
            "safe": max(safe, key=lambda r: r["recall"]) if safe else None}


def describe(scored: list[dict], kind: str) -> str:
    vals = sorted(s["score"] for s in scored if s["kind"] == kind)
    if not vals:
        return "—"
    q = statistics.quantiles(vals, n=4) if len(vals) > 3 else [vals[0], vals[0], vals[0]]
    return (f"{len(vals):>2} 条  最低 {vals[0]:.2f}  p25 {q[0]:.2f}  "
            f"中位 {q[1]:.2f}  p75 {q[2]:.2f}  最高 {vals[-1]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="写 docs/evidence/threshold_sweep.md")
    parser.add_argument("--config", default="", help="只跑某个配置（名字里的空格用下划线）")
    args = parser.parse_args()

    pairs = load_pairs()
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    n_base = len({p["base"] for p in pairs if p["label"] == 1})

    out: list[str] = []
    w = out.append
    w("# 语义缓存阈值扫描：四种配置对照")
    w("")
    w(f"评测集 `eval/pairs.jsonl`：{len(pairs)} 条查询，{n_base} 条缓存条目。")
    w("构成：" + "，".join(f"{k} {v} 条" for k, v in counts.items()) + "。")
    w("其中 `near_miss` 是**正确答案不在缓存里**的危险样本——任何命中都是错答。")
    w("")

    # 每把尺子只编码一次
    scored_by_backend: dict[str, list[dict]] = {}
    for _, backend, _ in CONFIGS:
        if backend in scored_by_backend:
            continue
        if embedder.force(backend) != backend:
            print(f"[warn] 后端 {backend} 不可用，跳过")
            continue
        scored_by_backend[backend] = score_all(pairs)

    w("## 一、两把尺子的分数分布")
    w("")
    w("| 样本类型 | n-gram | BGE |")
    w("|---|---|---|")
    for kind, name in (("paraphrase", "该命中 · 同义改写"), ("near_miss", "必须拒绝 · 危险近邻"),
                       ("unrelated", "必须拒绝 · 无关问题")):
        a = describe(scored_by_backend["ngram"], kind) if "ngram" in scored_by_backend else "—"
        b = describe(scored_by_backend["bge"], kind) if "bge" in scored_by_backend else "—"
        w(f"| {name} | {a} | {b} |")
    w("")
    for backend in ("ngram", "bge"):
        if backend not in scored_by_backend:
            continue
        sc = scored_by_backend[backend]
        pos = [s["score"] for s in sc if s["kind"] == "paraphrase"]
        near = [s["score"] for s in sc if s["kind"] == "near_miss"]
        far = [s["score"] for s in sc if s["kind"] == "unrelated"]
        w(f"- **{backend}**　AUC(改写 vs 近邻) = **{auc(pos, near):.3f}**，"
          f"AUC(改写 vs 无关) = {auc(pos, far):.3f}")
    w("")
    w("> AUC < 0.5 意味着分数方向反了：**随机挑一个该命中的，比挑一个该拒绝的还更容易得分低**。"
      "这说明问题不在刻度，在尺子。")
    w("")

    w("## 二、四种配置的关键指标")
    w("")
    w("| 配置 | 最优阈值 | 命中率 | Precision | Recall | F0.5 | 错误率 | P≥95% 时最高命中率 |")
    w("|---|---|---|---|---|---|---|---|")
    chosen: dict[str, tuple] = {}
    for name, backend, guard in CONFIGS:
        if backend not in scored_by_backend:
            continue
        if args.config and args.config not in (name, name.replace(" ", "_")):
            continue
        rows = sweep(scored_by_backend[backend], guard)
        m = key_metrics(rows)
        chosen[name] = (rows, m, backend, guard)
        b, s = m["best"], m["safe"]
        safe_note = (pct(s["hit_rate"]) + "（阈值 %.2f）" % s["thr"]) if s else "做不到"
        w(f"| {name} | {b['thr']:.2f} | {pct(b['hit_rate'])} | {pct(b['precision'])} | "
          f"{pct(b['recall'])} | {b['f05']:.3f} | {pct(b['error_rate'])} | {safe_note} |")
    w("")
    w("命中率 = 缓存接管了多少请求；错误率 = 其中答错的比例。")
    w("**这张表要横着看**：从第一行到第四行，每一步都在解决上一行暴露的问题。")
    w("")

    # 最终方案给完整曲线，其余只给关键点
    final_name = "BGE + 守卫" if "BGE + 守卫" in chosen else list(chosen)[-1]
    rows, m, backend, guard = chosen[final_name]
    w(f"## 三、最终方案（{final_name}）的完整曲线")
    w("")
    w("| 阈值 | 命中率 | 错误率 | Precision | Recall | F0.5 | 答对比例 |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        mark = "  ← 旧阈值" if abs(r["thr"] - CURRENT) < 1e-9 else ""
        w(f"| {r['thr']:.2f}{mark} | {pct(r['hit_rate'])} ({r['tp'] + r['fp']}) | "
          f"{pct(r['error_rate'])} ({r['fp']}) | {pct(r['precision'])} | "
          f"{pct(r['recall'])} | {r['f05']:.3f} | {pct(r['correct_rate'])} |")
    w("")
    b, s = m["best"], m["safe"]
    w(f"- F0.5 最优：阈值 **{b['thr']:.2f}**，命中率 {pct(b['hit_rate'])}，"
      f"precision {pct(b['precision'])}，错误率 {pct(b['error_rate'])}。")
    if s:
        w(f"- precision ≥ 95% 的最高召回：阈值 **{s['thr']:.2f}**，"
          f"命中率 {pct(s['hit_rate'])}，recall {pct(s['recall'])}。")
    else:
        w("- 没有阈值能做到 precision ≥ 95%。")
    w("")

    # 守卫拦下了什么（可解释性）
    if "bge" in scored_by_backend:
        blocked = []
        for s in scored_by_backend["bge"]:
            if s["label"] == 0 and s["score"] >= CURRENT:
                ok, why = cacheguard.check(s["q"], s["nearest"])
                if not ok:
                    blocked.append((s, why))
        if blocked:
            w("## 四、守卫拦下的典型误命中（本来会被服务出去）")
            w("")
            w("| 相似度 | 用户的真实问题 | 本该被复用成 | 拦住它的理由 |")
            w("|---|---|---|---|")
            for s, why in sorted(blocked, key=lambda x: -x[0]["score"])[:10]:
                w(f"| {s['score']:.3f} | {s['q']} | {s['nearest']} | {why} |")
            w("")

    text = "\n".join(out)
    print(text)

    if args.write:
        dst = ROOT / "docs" / "evidence" / "threshold_sweep.md"
        dst.write_text(text + "\n", encoding="utf-8")
        print(f"\n已写入 {dst}")


if __name__ == "__main__":
    main()
