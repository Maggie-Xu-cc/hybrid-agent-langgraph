"""公开演示护栏：每日总额度 + 单 IP 每分钟频次。

为什么需要这一层：
    这个 demo 背后是真实的云端 API Key。公开链接一旦被转发、被爬虫扫到，
    一天就能烧掉几十块。限额不是安全机制，是**止损**——让最坏情况的支出
    可预期（额度 × 单价）。真正的防滥用要靠登录鉴权，这里只做"够用的护栏"。

为什么两个维度都要（它们防的不是同一件事）：
    每日总额度  防"有人一直点"      → 控住成本上限
    单 IP 频次  防"脚本瞬间狂刷"    → 否则一个人几分钟打满额度，正常访客反而用不了

默认值 0 = 不限制（本地开发和离线测试不受影响），只有部署到公网时才收紧。
"""

from __future__ import annotations

import threading
import time
from datetime import date

from .config import settings
from .dbutil import connect, with_retry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_quota (
    day  TEXT PRIMARY KEY,
    used INTEGER NOT NULL DEFAULT 0
);
"""

# ip -> 最近 60 秒内的时间戳列表。放内存不落库：频控本来就是"瞬时"语义，
# 进程重启后从零开始算没有危害，反而免了一次磁盘写。
_ip_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def init() -> None:
    conn = connect(settings.db_path)
    with conn:
        conn.executescript(_SCHEMA)
    conn.close()


@with_retry
def _try_consume(day: str, quota: int) -> tuple[bool, int]:
    """尝试占用一次额度，返回 (是否成功, 当前已用次数)。

    用带条件的 UPSERT 一条语句同时完成"判断+占用"：
        ON CONFLICT DO UPDATE SET used = used + 1 WHERE used < quota
    没超额才 +1，否则这一行不动、rowcount 为 0。

    为什么不能"先 SELECT 再 UPDATE"：并发下两个请求都读到 99、都写 100，
    额度就被突破了。为什么不能"先 +1 再判断"：被拒绝的请求也会把计数顶上去，
    统计里看到 used 比额度大很多，数字就不可信了。
    """
    conn = connect(settings.db_path)
    with conn:
        cur = conn.execute(
            "INSERT INTO demo_quota(day, used) VALUES(?, 1) "
            "ON CONFLICT(day) DO UPDATE SET used = used + 1 WHERE used < ?",
            (day, quota),
        )
        granted = cur.rowcount > 0
        row = conn.execute("SELECT used FROM demo_quota WHERE day = ?", (day,)).fetchone()
    conn.close()
    return granted, int(row["used"]) if row else 0


def _rate_ok(ip: str, per_min: int) -> bool:
    """滑动窗口限频：只保留最近 60 秒的请求时间戳。"""
    now = time.time()
    with _lock:
        hits = [t for t in _ip_hits.get(ip, []) if now - t < 60]
        if len(hits) >= per_min:
            _ip_hits[ip] = hits
            return False
        hits.append(now)
        _ip_hits[ip] = hits

        # 访客身份会不断变化，得清理否则字典无限膨胀
        if len(_ip_hits) > 2000:
            for key in [k for k, v in _ip_hits.items() if not v or now - v[-1] > 300]:
                _ip_hits.pop(key, None)
    return True


def check(ip: str) -> tuple[bool, str]:
    """(是否放行, 拒绝原因)。先查便宜的（内存），再查要写库的。"""
    per_min = settings.demo_rate_per_min
    if per_min > 0 and not _rate_ok(ip, per_min):
        return False, f"请求过于频繁（每分钟上限 {per_min} 次），稍等一会儿再问。"

    quota = settings.demo_quota_per_day
    if quota <= 0:
        return True, ""

    granted, used = _try_consume(date.today().isoformat(), quota)
    if not granted:
        return False, (f"今天的演示额度（{quota} 次）已经用完了。"
                       "这是个人项目，背后是真实 API 计费，请明天再来试。")
    return True, ""


def today_used() -> int:
    conn = connect(settings.db_path)
    row = conn.execute("SELECT used FROM demo_quota WHERE day = ?",
                       (date.today().isoformat(),)).fetchone()
    conn.close()
    return int(row["used"]) if row else 0
