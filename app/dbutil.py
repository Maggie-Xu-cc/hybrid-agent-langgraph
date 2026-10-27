"""SQLite 小工具：统一连接参数 + 忙/只读重试。

为什么需要这一层：Windows 上杀毒软件、文件索引器会短暂独占数据库的
-wal / -shm 文件，SQLite 这时抛 `attempt to write a readonly database`；
换个时刻再试就正常。这类错误是**瞬时**的，重试几次即可。

重试一定要放在"这一条 SQL"的粒度上，不能放在业务外层：
业务外层重试会把模型调用整个再跑一遍，等于把 LLM 的钱再花一次。
"""

from __future__ import annotations

import functools
import sqlite3
import time

# 命中这些关键词才值得重试；语法错误、表不存在之类重试一万次也没用
_RETRY_HINTS = ("locked", "readonly", "busy")


def connect(path: str, timeout: float = 10) -> sqlite3.Connection:
    """统一连接：timeout 即 busy_timeout，写锁冲突时排队等而不是立刻抛。"""
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    return conn


def with_retry(fn=None, *, tries: int = 5, delay: float = 0.1):
    """给数据库操作加"忙就等一会儿再试"。

    两种用法：
        @with_retry
        def record(...): ...

        @with_retry(tries=6, delay=0.15)
        def put(...): ...

    最坏叠加等待 ≈ delay * (1+2+...+(tries-1))，默认 5 次约 1 秒，
    足够吃掉杀软扫描造成的秒级文件锁。
    """

    def wrap(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            for i in range(tries):
                try:
                    return f(*args, **kwargs)
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if i == tries - 1 or not any(h in msg for h in _RETRY_HINTS):
                        raise
                    time.sleep(delay * (i + 1))
            return None  # pragma: no cover

        return wrapper

    return wrap(fn) if fn is not None else wrap
