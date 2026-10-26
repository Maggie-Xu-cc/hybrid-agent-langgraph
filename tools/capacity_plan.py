"""算力需求测算：把 demo 的实测吞吐外推成三阶段 GPU 预算。

这是 B 类"未来算力需求设计"的量化依据。核心公式只有一个：

    本地需要承载的 token/s = 峰值QPS × 简单请求占比 × 单请求token数
    所需卡数 = ceil(本地token/s ÷ (单卡有效吞吐 × 目标利用率))

用法：
    python tools/capacity_plan.py
    python tools/capacity_plan.py --gpu-tps 1200 --easy-ratio 0.55 --price 4
    python tools/capacity_plan.py --hours-per-day 24    # 按常驻算
"""

import argparse
import math

STAGES = [
    ("P0 概念验证（已完成）", 500),
    ("P1 小范围试点", 5_000),
    ("P2 规模推广", 50_000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="混合推理服务的 GPU 容量测算")
    p.add_argument("--avg-in", type=int, default=600, help="单请求输入 token 均值")
    p.add_argument("--avg-out", type=int, default=300, help="单请求输出 token 均值")
    p.add_argument("--gpu-tps", type=float, default=800, help="单卡有效吞吐（总 token/s，实测值）")
    p.add_argument("--util", type=float, default=0.7, help="目标利用率，留 30% 余量抗突发")
    p.add_argument("--easy-ratio", type=float, default=0.6, help="简单请求占比（路由走本地的比例）")
    p.add_argument("--peak-factor", type=float, default=5.0, help="峰值QPS ÷ 均值QPS")
    p.add_argument("--price", type=float, default=4.0, help="单卡每小时价格（元）")
    p.add_argument("--hours-per-day", type=float, default=8.0, help="每天实际开机小时数")
    p.add_argument("--cloud-in", type=float, default=0.005, help="云端输入价 元/千token")
    p.add_argument("--cloud-out", type=float, default=0.015, help="云端输出价 元/千token")
    p.add_argument("--ha-extra", type=int, default=1, help="生产环境额外冗余卡数")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    per_req = a.avg_in + a.avg_out
    cloud_per_req = a.avg_in / 1000 * a.cloud_in + a.avg_out / 1000 * a.cloud_out
    hours_month = a.hours_per_day * 30

    print("## 算力需求测算\n")
    print(
        f"假设：单请求 {a.avg_in}+{a.avg_out} token，单卡有效吞吐 {a.gpu_tps} tok/s，"
        f"目标利用率 {a.util:.0%}，简单请求占比 {a.easy_ratio:.0%}，"
        f"峰值系数 {a.peak_factor:g}，GPU ¥{a.price:g}/卡·小时，"
        f"每天开机 {a.hours_per_day:g} 小时，"
        f"云端 ¥{a.cloud_in:g}/¥{a.cloud_out:g} 每千 token（入/出）。\n"
    )
    print("| 阶段 | 日请求 | 峰值QPS | 本地tok/s | 卡数 | 月度GPU | 月度云端兜底 | 月度合计 | 全云端基线 | 节省 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, daily in STAGES:
        peak_qps = daily / 86400 * a.peak_factor
        local_tps = peak_qps * a.easy_ratio * per_req
        cards = max(1, math.ceil(local_tps / (a.gpu_tps * a.util)))
        if name.startswith("P2"):
            cards += a.ha_extra
        gpu_cost = cards * a.price * hours_month
        cloud_cost = daily * (1 - a.easy_ratio) * cloud_per_req * 30
        total = gpu_cost + cloud_cost
        baseline = daily * cloud_per_req * 30
        save = (1 - total / baseline) * 100 if baseline else 0
        print(
            f"| {name} | {daily:,} | {peak_qps:.2f} | {local_tps:,.0f} | {cards} | "
            f"¥{gpu_cost:,.0f} | ¥{cloud_cost:,.0f} | ¥{total:,.0f} | "
            f"¥{baseline:,.0f} | {save:+.1f}% |"
        )

    print("\n### 盈亏平衡点\n")
    easy_per_hour = a.price / (a.easy_ratio * cloud_per_req)
    print(
        f"单卡 ¥{a.price:g}/小时，每个请求走云端要 ¥{cloud_per_req:.4f}，"
        f"其中 {a.easy_ratio:.0%} 可被本地承接。\n"
    )
    print(
        f"- **按需每天 8 小时**：日均约 **{easy_per_hour * 8:,.0f}** 个请求即可回本；\n"
        f"- **24 小时常驻**：日均约 **{easy_per_hour * 24:,.0f}** 个请求才回本。\n"
    )
    print(
        "P0 阶段卡数打满下限（1 张），成本必然高于纯 API——这一阶段买的是"
        "「可行性验证 + 数据」，不是省钱。真正谈成本优化要等 P1/P2。\n"
    )

    print("### 其他资源\n")
    print("| 项目 | P0 | P1 | P2 | 说明 |")
    print("|---|---|---|---|---|")
    print("| 模型权重存储 | 20 GB | 100 GB | 500 GB | 每模型 6–18 GB，保留 2–3 个版本 |")
    print("| 日志与指标 | 1 GB/月 | 20 GB/月 | 200 GB/月 | 每条请求约 2 KB |")
    print("| 出网带宽 | 5 Mbps | 50 Mbps | 200 Mbps | 流式输出按 30 KB/s/并发估 |")
    print("| 微调算力 | 0 | 6 卡时/周 | 24 卡时/周 | 8B LoRA，按 5 万样本估 |")


if __name__ == "__main__":
    main()
