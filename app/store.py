"""运行记录与统计（SQLite，零依赖）。"""

import sqlite3
import time

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 自增主键，一条请求一行
    ts TEXT NOT NULL,                        -- 时间戳
    query TEXT NOT NULL,                     -- 原始问题
    route TEXT NOT NULL,                     -- 判定结果 easy/hard
    reason TEXT,                             -- 判定理由
    decided_by TEXT,                         -- 谁判定的：rule/model/default
    model TEXT,                              -- 模型
    latency_ms INTEGER,                      -- 端到端延迟
    tokens_in INTEGER,                       -- 输入 token
    tokens_out INTEGER,                      -- 输出 token
    cost_cny REAL,                           -- 实付成本
    baseline_cost_cny REAL,                  -- 全云端基线成本
    escalated INTEGER DEFAULT 0              -- 是否触发升级重答
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _conn() as conn:
        conn.executescript(SCHEMA)


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
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO runs (ts, query, route, reason, decided_by, model, latency_ms,"
            " tokens_in, tokens_out, cost_cny, baseline_cost_cny, escalated)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
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
            " COALESCE(AVG(latency_ms),0) lat,"
            " COALESCE(SUM(escalated),0) esc FROM runs"
        ).fetchone()
        by_route = {
            r["route"]: r["n"]
            for r in conn.execute(
                "SELECT route, COUNT(*) n FROM runs GROUP BY route"
            ).fetchall()
        }
        # 分层追踪（tracing）：每个环节接了多少请求、平均延迟、花了多少钱。
        # decided_by 即请求走过的"最后一层"：cache=缓存命中, rule/model/default=路由结果
        by_layer_raw = conn.execute(
            "SELECT decided_by layer, COUNT(*) n, COALESCE(AVG(latency_ms),0) lat,"
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
        "escalated": int(row["esc"] or 0),
        "easy": by_route.get("easy", 0),
        "hard": by_route.get("hard", 0),
        "by_layer": by_layer,
        "cache_hits": by_layer.get("cache", {}).get("count", 0),
        "cache_hit_ratio": round(by_layer.get("cache", {}).get("count", 0) / total, 4)
        if total else 0.0,
    }

# 给演示页面底部“最近 10 条调用记录”
def recent(limit: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, query, route, decided_by, latency_ms, cost_cny, escalated"
            " FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
