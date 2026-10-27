"""长期记忆：跨会话记住用户的关键事实。

设计取舍（面试可讲的三点）：
- 抽取用正则而不是模型：抽取发生在每一轮，调本地小模型也要花钱花延迟；
  规则对"我叫X / 我在做X"这类强模式足够准，命中不了的本来也不值得记。
  这是"判断成本要配得上判断价值"的路由思想在记忆上的复用。
- 存 SQLite 单表 (user_id, fact_key, fact_value)：覆盖旧值即是"记忆更新"，
  不需要向量库——事实数量级是个位数到几十条，全量召回就是最优检索。
- 召回是全量注入（带条数上限）：等事实多了再演进成向量召回，
  届时和 RAG 复用同一套 embedding + 检索机制。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime

from .config import settings
from .dbutil import with_retry

_FACT_RULES: list[tuple[re.Pattern, object]] = [
    # (正则, key) —— key 是字符串或接收 match 的函数
    (re.compile(r"我(?:叫|的名字是|名字叫)\s*([^\s，,。.!?！？]{1,12})"), "name"),
    (re.compile(r"我(?:最喜欢|最爱)(?:的)?([\u4e00-\u9fa5A-Za-z]{1,6})[是为]\s*([^，。；！？\n]{1,20})"),
     lambda m: f"favorite_{m.group(1)}"),
    (re.compile(r"我在(?:做|开发|写|研究|学|准备)\s*([^，。；！？\n]{1,24})"), "working_on"),
    (re.compile(r"我是([^\s，,。.]{1,12}?)(?:专业的?学生|专业)"), "major"),
    (re.compile(r"记住[:：]\s*(.{1,40})"), "memo"),  # memo 用内容 hash 做 key，可存多条
]


def _clean(v: str) -> str:
    """去掉值尾部的标点和空白，'agent 项目。' → 'agent 项目'。"""
    return v.rstrip("。，,;；！!？? .\u3000").strip()


# 值是疑问词 = 这是个问句不是陈述（"我在做什么项目？"），不能当事实记
_QUESTION_VALUE = re.compile(r"^(什么|啥|哪|怎样|怎么|如何|多少|几|谁|吗)")


def init() -> None:
    """建表（幂等）。与记账同库：事实量级很小，不值得为它单开一个文件。"""
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_facts ("
            " user_id TEXT NOT NULL,"
            " fact_key TEXT NOT NULL,"
            " fact_value TEXT NOT NULL,"
            " updated_at TEXT NOT NULL,"
            " PRIMARY KEY (user_id, fact_key))"
        )
    conn.close()


def extract(text: str) -> dict[str, str]:
    """从一句话里抽取事实。纯函数，不落库——方便单测。"""
    facts: dict[str, str] = {}
    for pattern, key in _FACT_RULES:
        m = pattern.search(text)
        if not m:
            continue
        k = key(m) if callable(key) else key
        v = _clean(m.group(m.lastindex if m.lastindex else 1))
        if not v or _QUESTION_VALUE.match(v):
            continue
        if k == "memo":   # 多条备忘共存，用内容 hash 区分
            k = f"memo_{hashlib.md5(v.encode()).hexdigest()[:6]}"
        facts[k] = v
    return facts


@with_retry
def remember(user_id: str, text: str) -> dict[str, str]:
    """抽取并落库。返回本轮实际写入/更新的事实。"""
    if not user_id or not text:
        return {}
    facts = extract(text)
    if not facts:
        return {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        for k, v in facts.items():
            conn.execute(
                "INSERT INTO user_facts (user_id, fact_key, fact_value, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id, fact_key) DO UPDATE SET"
                " fact_value=excluded.fact_value, updated_at=excluded.updated_at",
                (user_id, k, v, now),
            )
    conn.close()
    return facts


@with_retry
def recall(user_id: str, limit: int = 6) -> dict[str, str]:
    """召回某用户的事实，按最近更新排序，最多 limit 条。"""
    if not user_id:
        return {}
    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        rows = conn.execute(
            "SELECT fact_key, fact_value FROM user_facts WHERE user_id=?"
            " ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:   # 表还没建
        return {}
    return {k: v for k, v in rows}


@with_retry
def forget(user_id: str) -> int:
    """清空某用户的长期记忆，返回删除条数。"""
    if not user_id:
        return 0
    conn = sqlite3.connect(settings.db_path, timeout=10)
    with conn:
        n = conn.execute("DELETE FROM user_facts WHERE user_id=?", (user_id,)).rowcount
    conn.close()
    return n


def count() -> int:
    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        n = conn.execute("SELECT COUNT(*) FROM user_facts").fetchone()[0]
        conn.close()
        return n
    except sqlite3.OperationalError:
        return 0
