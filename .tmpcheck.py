import json, time, urllib.request
BASE = "https://hybrid-router-agent.app.workbuddy.host"
def ck():
    with urllib.request.urlopen(urllib.request.Request(BASE + "/?sharecode=CLU3I2I6TH"), timeout=20) as r:
        return "; ".join(c.split(";")[0] for c in (r.headers.get_all("Set-Cookie") or []))
def call(m, p, pay=None, c=""):
    d = json.dumps(pay, ensure_ascii=False).encode() if pay is not None else None
    req = urllib.request.Request(BASE + p, data=d, method=m,
        headers={"Content-Type": "application/json", "Cookie": c})
    try:
        with urllib.request.urlopen(req, timeout=180) as r: return json.load(r)
    except urllib.error.HTTPError as e: return {"err": e.code}
c, t = "", {}
for _ in range(25):
    try:
        c = ck(); t = call("GET", "/api/tools", c and c, )
        if t.get("levels"): break
    except Exception: pass
    time.sleep(3)
print("critic 配置:", t.get("critic"))
d = call("POST", "/api/chat", {"query": "统计一下记账表里被门卫拦过几次？", "session_id": "live-final-1"}, c)
if d.get("status") == "awaiting_confirm":
    d = call("POST", "/api/confirm", {"session_id": "live-final-1", "approve": True}, c)
cr = d.get("critic") or {}
print("工具:", [s.get("name") for s in ((d.get("tools") or {}).get("steps") or [])],
      "| 审查:", cr.get("rounds"), "轮", cr.get("score"), "分", "通过" if cr.get("pass") else "未通过",
      "| 成本 ¥", d.get("cost_cny"), "| 延迟", d.get("latency_ms"))
print("答案:", (d.get("answer") or "")[:110].replace("\n", " "))
s = call("GET", "/api/stats?recent_limit=2", cookie=c).get("summary") or {}
print("统计: 审过", s.get("critic_reviewed"), "打回返工", s.get("critic_fixed"), "仍未通过", s.get("critic_failed"))
