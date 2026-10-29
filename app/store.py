"""运行记录与统计（SQLite，零依赖）。"""

import json
import sqlite3
import time

from .config import settings
from .dbutil import connect, with_retry

# 一条请求的真实耗时不可能超过一小时：工具循环有总超时（8s × 最多 5 步），
# 单次模型调用有 60s 超时，量级是"分钟"，不是"天"。
# 超过它的只可能是时间源本身失效——实测过一次：perf_counter 的基准在同一台
# 机器的不同进程之间并不通用，服务在 interrupt 处存盘、换进程恢复之后，
# 两个单调钟相减记出了一条 51 亿毫秒（约 60 天）的记录，把平均延迟整个带偏。
# 这个阈值只用来判断"这一行能不能进平均"，不改写原始账目——记账要如实。
MAX_PLAUSIBLE_LATENCY_MS = 60 * 60 * 1000

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 自增主键，一条请求一行
    ts TEXT NOT NULL,                        -- 时间戳
    query TEXT NOT NULL,                     -- 原始问题
    route TEXT NOT NULL,                     -- 判定结果 easy/hard
    reason TEXT,                             -- 判定理由
    decided_by TEXT,                         -- 谁判定的：rule/model/default/cache
    model TEXT,                              -- 模型
    latency_ms INTEGER,                      -- 端到端延迟
    tokens_in INTEGER,                       -- 输入 token
    tokens_out INTEGER,                      -- 输出 token
    cost_cny REAL,                           -- 实付成本
    baseline_cost_cny REAL,                  -- 全云端基线成本
    escalated INTEGER DEFAULT 0,             -- 是否触发升级重答
    session_id TEXT DEFAULT '',              -- 哪个会话产生的：页面按会话分开显示
    guard_blocked TEXT DEFAULT '',           -- 门卫拒绝原因（空 = 没被拦）
    cache_score REAL DEFAULT 0,              -- 本轮最高缓存相似度（含被拦下的候选）
    tool_steps TEXT DEFAULT '',              -- 工具调用轨迹（JSON 数组，空 = 没动手）
    -- 多智能体审查的账：审了几轮、最后过没过、几分、改了几次、提了什么问题。
    -- 打不进库就只剩"这一轮花了多少钱"，没法回答"多花的钱买到了什么"。
    critic_rounds INTEGER DEFAULT 0,         -- 审了几轮（0 = 没审）
    critic_pass INTEGER DEFAULT 1,           -- 最后一轮是否通过（0 = 提了问题还是放行了）
    critic_score INTEGER DEFAULT 0,          -- 审查员给的分数
    critic_fixes INTEGER DEFAULT 0,          -- 返工次数
    critic_issues TEXT DEFAULT ''            -- 提的问题（；分隔，截断 400 字）
);
"""

# 老库补列：SQLite 的 ALTER TABLE ADD COLUMN 很轻，不需要重建表
_MIGRATIONS = (
    ("session_id", "TEXT DEFAULT ''"),
    ("guard_blocked", "TEXT DEFAULT ''"),
    ("cache_score", "REAL DEFAULT 0"),
    ("tool_steps", "TEXT DEFAULT ''"),
    ("critic_rounds", "INTEGER DEFAULT 0"),
    ("critic_pass", "INTEGER DEFAULT 1"),
    ("critic_score", "INTEGER DEFAULT 0"),
    ("critic_fixes", "INTEGER DEFAULT 0"),
    ("critic_issues", "TEXT DEFAULT ''"),
)


def _conn() -> sqlite3.Connection:
    # timeout = busy_timeout：并发写时排队等锁而不是立刻抛 database is locked
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _conn() as conn:
        # WAL 是库级持久设置，设一次即可：读不阻塞写、写不阻塞读，
        # 并发跑基准/压测时不会互相卡死（默认的 rollback journal 会）
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        for name, decl in _MIGRATIONS:
            if name not in cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {decl}")


@with_retry
def record(
    query: str,
    route: str,
    reason: str,
    decided_by: str,
    model: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    baseline_cost: float,
    escalated: bool,
    session_id: str = "",
    guard_blocked: str = "",
    cache_score: float = 0.0,
    tool_steps: list | None = None,
    critic_rounds: int = 0,
    critic_pass: bool = True,
    critic_score: int = 0,
    critic_fixes: int = 0,
    critic_issues: str = "",
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO runs (ts, query, route, reason, decided_by, model, latency_ms,"
            " tokens_in, tokens_out, cost_cny, baseline_cost_cny, escalated,"
            " session_id, guard_blocked, cache_score, tool_steps,"
            " critic_rounds, critic_pass, critic_score, critic_fixes, critic_issues)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                query[:500],
                route,
                reason,
                decided_by,
                model,
                latency_ms,
                tokens_in,
                tokens_out,
                round(cost, 6),
                round(baseline_cost, 6),
                1 if escalated else 0,
                session_id,
                guard_blocked,
                round(cache_score or 0, 4),
                json.dumps(tool_steps, ensure_ascii=False) if tool_steps else "",
                critic_rounds,
                1 if critic_pass else 0,
                critic_score,
                critic_fixes,
                (critic_issues or "")[:400],
            ),
        )


def clear() -> int:
    """删除全部记账记录，返回删除的条数（不可恢复）。"""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM runs")
        return cur.rowcount or 0


def stats() -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(cost_cny),0) cost,"
            " COALESCE(SUM(baseline_cost_cny),0) base,"
            # 只拿"像真延迟"的行算平均。历史脏值（perf_counter 跨进程失效那条）
            # 不能删——账是真的，只是那一行的时间源错了——所以单独报个数让人看见，
            # 而不是悄悄把它从分母里拿掉。
            f" COALESCE(AVG(CASE WHEN latency_ms BETWEEN 0 AND {MAX_PLAUSIBLE_LATENCY_MS}"
            " THEN latency_ms END),0) lat,"
            f" COALESCE(SUM(CASE WHEN latency_ms > {MAX_PLAUSIBLE_LATENCY_MS}"
            " THEN 1 ELSE 0 END),0) lat_bad,"
            " COALESCE(SUM(escalated),0) esc,"
            # 门卫拦截次数：不是失败，是拦下了"字面很像但参数不同"的复用
            " COALESCE(SUM(CASE WHEN guard_blocked != '' THEN 1 ELSE 0 END),0) guarded"
            " FROM runs"
        ).fetchone()
        by_route = {
            r["route"]: r["n"]
            for r in conn.execute(
                "SELECT route, COUNT(*) n FROM runs GROUP BY route"
            ).fetchall()
        }
        # 分层追踪（tracing）：每个环节接了多少请求、平均延迟、花了多少钱。
        # decided_by 即请求走过的"最后一层"：cache=缓存命中, rule/model/default=路由结果
        # 工具调用统计：有多少轮动过手、平均几步。
        # 这两个数比"工具总数"有用得多——它回答"Agent 真的在用手，还是只写在 README 里"
        tool_row = conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN tool_steps != '' THEN 1 ELSE 0 END),0) uses,"
            " COALESCE(SUM(json_array_length(CASE WHEN json_valid(tool_steps)"
            " AND tool_steps != '' THEN tool_steps ELSE '[]' END)),0) steps"
            " FROM runs"
        ).fetchone()
        # 审查的账：审了多少轮、有几轮返工、几轮最终仍未通过。
        # 三个数要分开看：reviewed 是覆盖面，fixed 是"打回真起了作用"，
        # failed 是"打回也没改好"——后两者混在一起，就分不清审查员是有用还是空转。
        critic_row = conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN critic_rounds > 0 THEN 1 ELSE 0 END),0) reviewed,"
            " COALESCE(SUM(CASE WHEN critic_rounds > 0 AND critic_pass = 0 THEN 1 ELSE 0 END),0) failed,"
            " COALESCE(SUM(CASE WHEN critic_fixes > 0 THEN 1 ELSE 0 END),0) fixed"
            " FROM runs"
        ).fetchone()
        by_layer_raw = conn.execute(
            "SELECT decided_by layer, COUNT(*) n,"
            f" COALESCE(AVG(CASE WHEN latency_ms BETWEEN 0 AND {MAX_PLAUSIBLE_LATENCY_MS}"
            " THEN latency_ms END),0) lat,"
            " COALESCE(SUM(cost_cny),0) cost FROM runs GROUP BY decided_by"
        ).fetchall()

    total = row["n"] or 0
    cost = row["cost"] or 0.0
    base = row["base"] or 0.0
    saved = (1 - cost / base) if base > 0 else 0.0
    by_layer = {
        r["layer"] or "unknown": {
            "count": r["n"],
            "avg_latency_ms": int(r["lat"] or 0),
            "cost_cny": round(r["cost"] or 0, 6),
        }
        for r in by_layer_raw
    }
    return {
        "runs": total,
        "cost_cny": round(cost, 4),
        "baseline_cost_cny": round(base, 4),
        "saved_ratio": round(saved, 4),
        "avg_latency_ms": int(row["lat"] or 0),
        "latency_outliers": int(row["lat_bad"] or 0),
        "escalated": int(row["esc"] or 0),
        "easy": by_route.get("easy", 0),
        "hard": by_route.get("hard", 0),
        "by_layer": by_layer,
        "cache_hits": by_layer.get("cache", {}).get("count", 0),
        "cache_hit_ratio": round(by_layer.get("cache", {}).get("count", 0) / total, 4)
        if total else 0.0,
        "guard_blocks": int(row["guarded"] or 0),
        "tool_runs": int(tool_row["uses"] or 0),
        "tool_steps_total": int(tool_row["steps"] or 0),
        "critic_reviewed": int(critic_row["reviewed"] or 0),
        "critic_failed": int(critic_row["failed"] or 0),
        "critic_fixed": int(critic_row["fixed"] or 0),
    }

# 给演示页面底部“最近 10 条调用记录”
def recent(limit: int = 10, session_id: str | None = None) -> list[dict]:
    """最近请求。传 session_id 就只返回该会话的——页面按会话分开显示，
    全局混在一起既看不懂，也会让人以为别人的对话是自己的。"""
    sql = ("SELECT ts, query, route, decided_by, latency_ms, cost_cny,"
           " baseline_cost_cny, escalated, session_id, guard_blocked, cache_score,"
           " tool_steps, critic_rounds, critic_pass, critic_score, critic_fixes,"
           " critic_issues FROM runs")
    args: list = []
    if session_id is not None:
        sql += " WHERE session_id = ?"
        args.append(session_id)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        # 轨迹以 JSON 存 TEXT：页面和接口都用数组更好使，这里统一解一次。
        # 解析失败不抛错——旧库里可能是空串或半截 JSON，不该让整页打不开。
        try:
            d["tool_steps"] = json.loads(d.get("tool_steps") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["tool_steps"] = []
        # 单条延迟也可能是历史脏值：原样吐给页面会显示成"5173251932 毫秒"，
        # 用户只会以为页面坏了。这里置 0 并打标，页面照常显示，
        # 要追查时看 latency_anomaly 这一位，原始值仍在库里。
        if (d.get("latency_ms") or 0) > MAX_PLAUSIBLE_LATENCY_MS:
            d["latency_anomaly"] = True
            d["latency_ms"] = 0
        out.append(d)
    return out
