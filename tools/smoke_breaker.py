"""本地熔断器冒烟测试（离线，不发真实请求）。

验证三态转换：闭合 → 打开 → 半开 → 复位，以及"短路后不再等待"这个核心收益。

    python tools/smoke_breaker.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 指向一个必然连不上的端口：验证"失败会熔断"，不需要真的模型
os.environ["LOCAL_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["LOCAL_TIMEOUT"] = "2"
os.environ["LOCAL_BREAKER_THRESHOLD"] = "2"
os.environ["LOCAL_BREAKER_COOLDOWN"] = "2"

from app.clients import ProviderError, call_local, local_breaker  # noqa: E402

PASS = FAIL = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}" + (f"  ({extra})" if extra else ""))
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  ({extra})" if extra else ""))


def call() -> tuple[bool, float]:
    """调一次本地，返回 (是否成功, 耗时 ms)。"""
    t0 = time.perf_counter()
    try:
        call_local([{"role": "user", "content": "hi"}], max_tokens=8)
        return True, (time.perf_counter() - t0) * 1000
    except ProviderError:
        return False, (time.perf_counter() - t0) * 1000


print("== 1. 初始状态：闭合 ==")
check("初始 allow()", local_breaker.allow())
check("初始未打开", local_breaker.state()["open"] is False)

print("\n== 2. 连续失败触发熔断 ==")
ok1, ms1 = call()
check("第1次调用失败", ok1 is False, f"{ms1:.0f}ms")
check("失败1次后仍闭合（阈值2）", local_breaker.state()["open"] is False)
ok2, ms2 = call()
check("第2次调用失败", ok2 is False, f"{ms2:.0f}ms")
check("失败2次后打开", local_breaker.state()["open"] is True)

print("\n== 3. 打开期间短路：不发请求、不等待 ==")
ok3, ms3 = call()
check("熔断后调用仍失败", ok3 is False)
check("短路耗时 < 20ms（对比真实失败 %.0fms）" % ms1, ms3 < 20, f"{ms3:.2f}ms")
check("短路快于真实请求", ms3 < ms1)

print("\n== 4. 冷却后半开探测 ==")
time.sleep(2.2)
check("冷却结束 allow() 放行一次", local_breaker.allow())
check("半开时状态仍标记 open（探测中）", local_breaker.state()["open"] is False)  # 已过冷却
ok4, ms4 = call()
check("探测失败", ok4 is False)
check("探测失败后重新打开", local_breaker.state()["open"] is True)
ok5, ms5 = call()
check("再次短路", ok5 is False and ms5 < 20, f"{ms5:.2f}ms")

print("\n== 5. 成功后复位 ==")
local_breaker.ok()
check("手动复位后闭合", local_breaker.state()["open"] is False)
check("复位后允许调用", local_breaker.allow())

print("\n== 6. DISABLE_LOCAL 直接短路（纯云端跑基准用）==")
import app.clients as clients_mod  # noqa: E402
from app.config import settings  # noqa: E402

settings.disable_local = True
try:
    ok6, ms6 = call()
    check("DISABLE_LOCAL 时不发请求", ok6 is False and ms6 < 5, f"{ms6:.2f}ms")
finally:
    settings.disable_local = False

print(f"\n结果：{PASS} 通过，{FAIL} 失败")
sys.exit(1 if FAIL else 0)
