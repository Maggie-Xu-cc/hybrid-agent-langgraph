# 代码精读（面试准备版）

> 这份文档是给"面试前 30 分钟翻一遍"用的。结构：先看全景，再逐文件精读，最后背问答。
> 建议配合代码打开阅读：左边 IDE 打开 `app/graph.py`，右边看本文对应章节。
>
> 全文用大白话写成，每个技术点都配了 **面试话术**，可以直接说出口。

## 目录

| 节 | 内容 | 什么时候看 |
|---|---|---|
| 0 | 一句话讲清项目 + 三个必背数字 | 开场自述 |
| 1 | 一次请求的生命周期（8 步） | 讲流程 |
| 2 | 精读 `app/graph.py` | 讲代码 |
| 3 | 精读 `app/router.py` | 讲路由设计 |
| 4 | 精读 `app/clients.py` | 讲协议与容错 |
| 5 | 精读 `app/store.py` | 讲记账 |
| 6 | `agent.py` vs `graph.py` | **必问**：为什么用框架 |
| 7 | 精读 `eval/run_eval.py` | **必问**：怎么证明有效 |
| 8 | 精读 `tools/capacity_plan.py` | 讲成本测算 |
| 9 | 精读阶段二：上下文预算 + 长期记忆（v1.2） | **必问**：记忆怎么做的 |
| 10 | 高频面试问答（6 个） | 面试前突击 |
| 11 | 大白话术语表 | 卡壳时查 |
| 12 | 复习路径建议 | 按剩余时间选 |

---

## 0. 一句话讲清这个项目

> 本地 1.5B 小模型 + 云端大模型组成混合推理系统，用一个三层路由器把简单请求留在本地、复杂请求送去云端，并给每一次调用记账（token / 延迟 / 成本）。
> 核心问题：**在什么条件下，自建小模型 + 云端兜底比全部走 API 更划算？**

三个数字（背下来）：**实测成本降到全云端的 86.1%，路由准确率 95.8%，72 次调用零失败**。

---

## 1. 全景：一次请求的生命周期

用一个比喻记住整条链路：

> 这是一条**工厂流水线**。`AgentState` 是跟着工件走的**工单**，每个节点是一个**工位**（拿工单、干活、写几笔、往下传），条件边是传送带上的**分拣员**（看工单决定往左往右），checkpointer 是**游戏存档**（走完全程拍快照，下次同一个人来能接着上次的档）。

```
POST /api/chat                      server.py
   ↓
run_graph(query, session_id)        graph.py:313
   ↓ thread_id = session_id
[checkpointer 自动加载历史]          第 3 步，框架做的
   ↓
route_node                          graph.py:97     decide() 判难度
   ├── easy → local_answer           graph.py:128    本地小模型
   │            └→ self_check       graph.py:169    答不好？
   │                 └→ escalate    graph.py:179    云端重答
   └── hard → cloud_answer           graph.py:149    云端大模型
   ↓
bookkeep_node                       graph.py:195    算钱 + 落库 + 存历史
   ↓
[checkpointer 自动存快照]            第 8 步，框架做的
```

**紫色部分（加载/保存状态）是框架自动做的，你写的只是六个普通 Python 函数。** 这是理解 LangGraph 的关键：你不是在"学框架 API"，是在用框架重新组织你已经会写的东西。

---

## 2. 精读 `app/graph.py`（图编排层）

### 2.1 文件头注释（1-22 行）

那段 ASCII 图不是装饰，是**你的讲解稿**：被问"你系统的流程"，直接照着念。

必读的三行（14-17 行）：

```
- 路由逻辑 router.decide() 原样保留，从"函数调用"变成"条件边"；
- 记账逻辑 store.record() 原样保留，集中在 bookkeep 节点统一落库；
- agent.py 保留为基线实现，eval 可对比两条链路。
```

> **面试点**：这三句说明你是"有意识重构"而不是"推倒重写"。重构的第一原则是保住已验证正确的逻辑。

### 2.2 常量（38-54 行）

```python
SYSTEM = "你是一个严谨的助手..."          # 人设说明书
ESCALATE_MARKERS = ["我不确定", "不知道", ...]  # 认怂词表（0 成本质检）
CONTEXT_WINDOW_MESSAGES = 8               # 发给模型的（控制 token 账单）
MAX_HISTORY_MESSAGES = 20                 # 存进存档的（控制数据库大小）
```

**为什么两个窗口数字不同？** 很多人会混：8 是"发出去的成本"，20 是"存起来的体积"，两个不同的约束，不能用同一个值。

**为什么用关键词质检而不是模型质检？** 让模型判断"这个答案好不好"要再花一次钱和延迟，等于把省下的钱花回去。找"我不确定"这类词，0 成本 0 延迟。

### 2.3 `AgentState`（57-75 行）—— 全场最重要

```python
class AgentState(TypedDict, total=False):
    query: str
    history: list[dict]
    route: dict
    answer: str
    local_in: int   # 本地输入 token
    local_out: int
    cloud_in: int   # 云端输入 token
    cloud_out: int
    ...
```

`TypedDict` = 一张规定好每栏填什么的表格；`total=False` = 每栏可以不填。

**核心机制：节点只返回自己改的字段（partial update）。**

```python
def self_check_node(state):
    return {"needs_escalation": needs}   # 只交一张便利贴，框架负责贴到对应栏
```

> **面试话术**："节点之间不互相传一整个大对象，只通过 state 的约定字段通信。加一个节点不用改其他节点，耦合度低。这就是 LangGraph 相比'一个大函数里到处传参'的价值。"

**细节加分项**：token 拆成 `local_*` 和 `cloud_*` 四个变量，而不是一个 `tokens`。因为本地和云端单价不同，混在一起算不出真实成本。

### 2.4 两个工具函数（78-92 行）

```python
def _needs_escalation(text):
    if len(text.strip()) < 12: return True          # 太短 = 没答
    return any(m in text for m in ESCALATE_MARKERS) # 含认怂词
```

```python
def _build_messages(state):
    trimmed = history[-CONTEXT_WINDOW_MESSAGES:]    # 只留最近 8 条
    return [system, *trimmed, user]                 # 装信封
```

`[-8:]` 是 Python 切片"取最后 8 个"。**装信封前先扔掉老信**，控制 token。

### 2.5 `route_node`（97-125 行）—— 分诊台

```python
route = decide(query)                      # 调 router.py 的三层路由
if force in ("easy", "hard"):              # 演示/评测可强制指定
    route = Route(force, f"强制{force}", route.by)
return {
    "route": {...},                        # 判定结果
    "messages": _build_messages(...),       # 装好的信封
    "local_in": route.tokens_in,            # 路由的 token 记本地账
    "escalated": False, "degraded": False,  # 初始化标记
}
```

三个要点：
1. **它不返回答案**，只返回"判定 + 信封"。判定交给 `router.py`，节点只管调度——职责单一。
2. `{**state, "query": query}` = 复制工单再换掉 query（`strip()` 后的版本），**不原地修改 state**。
3. 路由用的 token 记在**本地账**上，因为路由是本地小模型干的活。

### 2.6 两个作答节点（128-146 / 149-166 行）—— 一对镜像

```python
try:
    answer, tin, tout = call_local(messages)
    update.update(..., degraded=False)
except ProviderError:                            # 本地挂了
    answer, tin, tout = call_cloud(messages)     # 云端顶上
    update.update(..., degraded=True,
                  route={**route, "reason": f"{route['reason']}(本地不可用)"})
```

大白话：**主备双活**。本地挂了云端顶，云端挂了本地顶。`degraded=True` 是"这次降级了"的标记，方便统计。

细节：`route={**route, "reason": ...}` 是**复制再改**，不是 `state["route"]["reason"] = ...`。函数式纪律，避免副作用。

### 2.7 `self_check_node`（169-176 行）—— 质检员

```python
needs = (settings.enable_escalation      # 开关开着
         and not state.get("degraded")   # 且这次没降级
         and _needs_escalation(answer))  # 且答案确实不好
```

**三道与门，全真才升级。** 为什么必须 `not degraded`？降级意味着答案已经来自云端，再"升级"一次就是重复付费。**很多实现会漏这一条，这是可以主动讲的细节。**

### 2.8 `escalate_node`（179-192 行）—— 叫外援

```python
update = {"escalated": True}
try:
    answer, tin, tout = call_cloud(messages)     # 云端重答
    update.update(answer=answer, ...)            # 覆盖本地答案
except ProviderError:
    pass                                          # ← 外援也叫不来，就用本地答案
```

> **面试点**：升级失败不能把请求搞崩。宁可给一个不太好的答案，也不能报错——可用性优先的取舍。

### 2.9 `bookkeep_node`（195-248 行）—— 收银台

四件事：

```python
# ① 算总账
tokens_in = local_in + cloud_in
latency_ms = int((time.perf_counter() - state["started"]) * 1000)
#                ↑ started 是 route 节点记的，端到端延迟要从头算

# ② 算钱（本地一个价，云端输入输出分别计价）
cost = (local_in+local_out)/1000 * price_local \
     + cloud_in/1000 * price_cloud_in \
     + cloud_out/1000 * price_cloud_out

# ③ 算基线：同样的 token 全走云端要花多少
baseline = tokens_in/1000 * price_cloud_in + tokens_out/1000 * price_cloud_out

# ④ 落库 + 追加历史
store.record(...)
history.append({"role":"user", ...}); history.append({"role":"assistant", ...})
return {"history": history[-MAX_HISTORY_MESSAGES:], "result": result}
```

三个精读点：
1. **延迟从 route 节点开始算**（`state["started"]`），否则路由时间不计入，数据是假的。
2. **`baseline` 是整个项目的北极星**：节省比例 = 1 − cost/baseline。没有 baseline 只能吹"我用了小模型"，有了它才能说"省了 13.9%"。
3. **history 只在最后追加，且截断到 20 条**。为什么不在 `local_answer` 里写？因为升级可能改答案，早写会污染上下文。

### 2.10 三个条件边（253-262 行）—— 交通指挥员

```python
def after_route(state):  return state["route"]["difficulty"]      # easy / hard
def after_local(state):  return "done" if degraded else "check"   # 降级就跳过自检
def after_check(state):  return "escalate" if needs_escalation else "done"
```

它们**不干活，只举手说往哪走**，返回的字符串去查下面的对照表。都是纯函数（只看不改），所以极好测试。

### 2.11 `build_graph`（267-289 行）—— 拼流水线

```python
g = StateGraph(AgentState)
g.add_node("route", route_node)              # 装工位：名字 + 函数
...
g.add_edge(START, "route")                   # 固定传送带
g.add_conditional_edges("route", after_route, {"easy": "local_answer", "hard": "cloud_answer"})
g.add_edge("bookkeep", END)
return g.compile(checkpointer=checkpointer) # ← 编译才变成可调用对象
```

**`compile()` 才是"开机"**：`add_node` 阶段只是画图纸，compile 之后才有 `invoke`。

### 2.12 存档系统（292-312 行）

```python
def _make_checkpointer():
    try:
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        return MemorySaver()          # 装不上就存内存

_app = None
def get_app():
    global _app
    if _app is None:                  # 单例：只建一次
        _app = build_graph(_make_checkpointer())
    return _app
```

- `check_same_thread=False`：SQLite 默认禁止跨线程共用连接，而 FastAPI 是多线程的，必须关掉。
- `_app` 懒加载单例：**每个请求重建图会重复编译**，这是"知道性能在哪"的体现。

### 2.13 `run_graph`（313-323 行）—— 大门

```python
config = {"configurable": {"thread_id": session_id}}   # ← 存档槽位号
out = app.invoke({"query": query, "force_route": force_route}, config)
return out["result"]
```

**`thread_id` 就是存档槽位号**：同一个号读同一份档。输入只有两栏，其余字段（如 `history`）是框架从存档读出来拼进工单的——**这是 checkpointer 最有价值的地方：你不用手动管理会话**。

### 2.14 三个自测题（先想再翻答案）

1. 如果把 `bookkeep` 里追加 history 的代码挪到 `local_answer_node`，会出什么 bug？
2. 为什么 `self_check_node` 要判断 `not degraded`？
3. 两个用户的 `session_id` 都是 `"default"`，会发生什么？

<details>
<summary>答案</summary>

1. 升级发生后，history 里留下的是被替换掉的错误答案，下一轮上下文被污染。
2. 降级时答案已来自云端，再升级等于重复付费。
3. 两人共享同一份 history，对话串台。线上必须给每个会话发唯一 id。

</details>

---

## 3. 精读 `app/router.py`（项目的灵魂）

### 3.1 设计动机（文件头注释）

> "为什么要有规则层：路由本身也是一次推理，规则能拦掉约一半请求的路由开销。"

**判断该用哪个模型，这件事本身也要花钱。** 所以判断成本随不确定性递增：确定的规则判（0 成本），模糊的才花 30 token 问小模型。

### 3.2 强特征正则（19-31 行）

```python
HARD_PATTERNS = [r"```", r"\bdef\s+\w+\s*\(", r"证明|推导|复杂度", ...]
HARD_RE = re.compile("|".join(HARD_PATTERNS), re.IGNORECASE)
```

- **`"|".join()` 合成一条大正则，只编译一次** —— 性能意识
- **`\b` 是单词边界**，`\bdef\s+\w+\s*\(` 抓 `def foo(` 但不误伤 `define`
- **`re.IGNORECASE`** 大小写都抓

**主动承认局限（加分）**：规则宁可错杀。"写个菜谱"会被 `写个` 判成 hard；"什么是装饰器"会走模型层。**误判代价不对称**，所以需要自检兜底。

### 3.3 `Route` 数据结构（55-62 行）

```python
@dataclass
class Route:
    difficulty: str   # easy | hard
    reason: str       # 判定理由（人要能读懂）
    by: str           # rule | model | default ← 谁判的
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
```

**判定结果必须带"证据"**。`by` 字段让评测能统计"规则层拦截率"。**可观测性不是事后加的，是设计数据结构时就留好的。**

### 3.4 规则层的判断顺序（65-83 行）

| 顺序 | 判断 | 为什么放这 |
|---|---|---|
| 1 | 空输入 | 直接短路 |
| 2 | 寒暄（`^...$` 全匹配） | `你好` 是 easy，`你好，帮我写排序` 不误伤 |
| 3 | 超长（>300 字） | 长文本必硬 |
| 4 | 强特征正则 | 成本低 |
| 5 | 短句（≤20 字） | 兜底闲聊 |
| — | **`return None`** | **"我不确定，交给下一层"** |

> `return None` 是最有设计感的一行。新手会在这里硬给答案，导致模型层永远用不上。**主动返回"不知道"是一种能力。**

### 3.5 `_parse_json` 容错解析（86-94 行）

```python
match = re.search(r"\{.*?\}", text, re.DOTALL)   # 从废话里抠出 {...}
data = json.loads(match.group(0))
```

**小模型不会乖乖只输出 JSON**，常回"好的，我的判断是：{...}"。所以正则可先把 JSON 抠出来再解析。

> **面试点**：这是 LLM 工程真实的坑。追问"怎么更稳"→ JSON mode / function calling 约束格式，但小模型支持不一定好，正则兜底更通用。

### 3.6 模型层的三个细节（97-117 行）

```python
call_local([{"role":"user", "content": ROUTER_PROMPT + query[:800]}], max_tokens=48)
```

- `query[:800]` 截断：分类只看开头，省 token
- `max_tokens=48`：只够一行 JSON，想啰嗦也没空间
- `except ProviderError: return None`：本地挂了不报错，直接放弃这层

### 3.7 `decide()` 三级瀑布（120-133 行）

```python
route = _rule_route(query)          # 规则层
if route is not None: return route
route = _model_route(query)         # 模型层
if route is not None: return route
if len(query) < 120: ...            # 兜底层
```

**永不返回 None、永不抛异常** —— fail-safe 设计。路由判错的代价远低于请求失败。

---

## 4. 精读 `app/clients.py`（协议设计）

### 4.1 最漂亮的一个决定（1-24 行）

```python
local_client = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key)
cloud_client = OpenAI(base_url=settings.cloud_base_url, api_key=... )
```

**本地服务和云端 API 说同一套"普通话"（OpenAI 兼容协议），所以"切换模型"退化成改一个地址。**

> **面试话术（建议背）**："我把本地推理服务也用 OpenAI 兼容协议暴露出来，路由在下游看只是换了个 base_url。好处是新增供应商成本几乎为零，坏处是失去各家专有特性——但这个项目里不值得为它付复杂度。"
>
> 能讲 trade-off 的两面，比只讲好处可信得多。

### 4.2 MOCK_MODE 是测试基础设施（31-54 行）

```python
if "请求路由器" in prompt: ...            # 模拟路由模型
if len(prompt) % 33 == 0:                 # 故意让 3% 请求"我不确定"
    return "我不确定这个问题的答案。"
```

- 没 GPU、没 Key、没网也能跑通全链路 → 离线联调
- **故意注入失败**，用来验证升级分支——你不可能等线上真出一次失败来测

> **面试点**："我用 mock 模式做离线联调，并故意注入失败来验证降级/升级分支。"

### 4.3 异常抽象（73-87 行）

```python
except Exception as exc:
    raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
```

把网络超时、401、模型名写错统一翻译成 `ProviderError`。上层只关心"这个 provider 能不能用"，不关心具体错误码。`from exc` 保留原始异常链，便于排查。

### 4.4 token 来源（83-87 行）

```python
usage = getattr(resp, "usage", None)
tin = getattr(usage, "prompt_tokens", 0) or 0
```

**不自己数 token，用服务端返回的真实用量**（那才是计费依据）。`getattr` 防某些供应商不返回 usage。

---

## 5. 精读 `app/store.py`（记账本）

### 5.1 表结构 → 指标（8-24 行）

| 字段 | 能回答什么问题 |
|---|---|
| `decided_by` | 规则层拦截率多少？ |
| `cost_cny` vs `baseline_cost_cny` | 省了多少钱？（核心卖点） |
| `escalated` | 升级率多少？自检是不是太敏感？ |
| `latency_ms` | 混合路由的延迟代价多大？ |

### 5.2 `stats()` 里的 SQL 细节（80-108 行）

```sql
SELECT COALESCE(SUM(cost_cny),0) cost ...
```

`COALESCE(x,0)`：表里没记录时 `SUM()` 返回 NULL，不处理的话 Python 里除法会炸。**新手最常踩的坑。**

```python
saved = (1 - cost / base) if base > 0 else 0.0
```

节省比例定义，`base > 0` 防除零。

### 5.3 为什么用 SQLite

> **面试话术**："零依赖、单文件，够 demo 量级。真上生产换 PostgreSQL，因为要多人并发写和跨实例共享——SQLite 的写锁是全库级的。"
>
> **主动说出边界，比假装完美可信。**

---

## 6. `agent.py` vs `graph.py`（面试必问）

| 维度 | `agent.py`（基线） | `graph.py`（重构版） |
|---|---|---|
| 流程表达 | 三层 if/else + try/except 嵌套 | 六节点 + 四条边，可画图 |
| 多轮对话 | ❌ 不支持 | ✅ checkpointer 持久化 |
| 状态管理 | 局部变量散落 | 统一在 `AgentState` |
| 可测试性 | 只能测整个 `run()` | 每个节点可单独测 |
| 加节点成本 | 改主函数，易漏分支 | `add_node` + 连边，不动老节点 |

原版难改的根源：

```python
if route.difficulty == "easy":
    try: call_local ... except: call_cloud ...
    if not degraded and _needs_escalation(answer):   # 已经两层嵌套
        try: call_cloud ... except: pass
else:
    try: call_cloud ... except: call_local ...
```

三种失败路径 × 两个分支 = **六条隐藏路径**挤在一个函数里。

> **标准答法**："第一版我故意不用框架，线性流程用 if/else 更好调试。加多轮记忆时发现状态散在局部变量里、加路径要改主函数多处，所以迁到 LangGraph。旧实现留在 agent.py 当基线，用 `ENABLE_LANGGRAPH` 开关切换，两条链路跑同一份评测集做对照。"

这段话包含三个加分项：**知道何时不用框架 / 知道重构触发点 / 有对照实验**。

---

## 7. 精读 `eval/run_eval.py`（怎么证明有效）

### 7.1 它在证明什么

> **同一批题、同一份代码，只改一个开关，看成本和延迟怎么变。**

```
                 ┌─ 全本地（force=easy，不升级）── 便宜但笨
24 条测试集 ──────┼─ 全云端（force=hard，不升级）── 聪明但贵
                 └─ 混合路由（自动判 + 升级）   ── 想两头都要
```

这是学术上的**消融实验（ablation）**。面试问"你说省了钱，凭什么"，答："三组对照，其他条件全一样，只有路由策略不同，所以差异可归因于路由。"

测试集：24 条，`easy 13 / hard 11`，字段 `{q, difficulty, category}`，按场景分类（寒暄/翻译/代码/数学…）标注。

### 7.2 开头几行藏着一个坑（18-26 行）

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # 让脚本能 import app.*

EVAL_DB = ROOT / "eval" / "eval_runs.db"
os.environ["DB_PATH"] = str(EVAL_DB)   # ← 必须在 import app 之前设置

from app import store                   # noqa: E402
from app.agent import run
```

`settings = Settings()` 是在**模块被 import 的那一刻**读环境变量的。顺序反了，读到的就是旧值。

> **面试点**：这是"配置在 import 时快照"的经典坑。更稳的做法是延迟读取（property）或 `pydantic-settings`。我保留简单写法，但知道问题在哪。

**评测用独立数据库**：不污染线上 `runs.db`，且每次跑前 `unlink()`（71-73 行）删掉重建 → **可复现**。

### 7.3 三种策略怎么"模拟"出来（35-41 行）

```python
def run_strategy(cases, force, escalate):
    settings.enable_escalation = escalate
    for case in cases:
        result = run(case["q"], force_route=force)
```

关键技巧是 `force_route`：**不写三套代码**，只给现有入口加一个"强制指定路由"的开关。

- 全本地 = `force="easy"`，全云端 = `force="hard"`，混合 = `force=None`（真的让路由器判）

> **为什么重要**：三组实验走**完全相同的代码路径**，唯一变量是路由决策。若为"全本地"另写一套调用逻辑，差异可能来自实现细节而非策略，实验就不可信。
>
> **面试话术**："我用同一个入口模拟三种策略，保证除路由外所有条件一致——这是对照实验的基本要求。"

### 7.4 七个指标，每个都有目的（44-63 行）

| 指标 | 代码 | 回答什么问题 |
|---|---|---|
| 平均延迟 | `lat` | 质量换速度的代价多大 |
| 总成本 | `cost` | 省了多少钱 |
| 相对全云端 | `vs_cloud = cost / base` | 86.1% 就是这个 |
| 答案长度 | `ans_len` | 质量代理指标 |
| 升级次数 | `esc` | 自检是不是太敏感 |
| 降级次数 | `deg` | 可用性（应恒为 0） |
| 路由准确率 | `hit / n` | 只对混合策略有效 |

两个值得单独讲的：

**`vs_cloud` 的分母是"逐请求按云端价折算的基线之和"**，不是"全云端策略的总成本"。因为混合策略的 token 消耗与全云端不同，逐请求折算才严格可比。**能讲清指标定义，说明你真想过，而不是随手除一下。**

**`ans_len` 是粗糙的质量代理**：太短通常意味着没答好，但衡量不了正确性。要主动承认。

### 7.5 产物不只是表格（104-132 行）

```python
csv 逐条明细（query / gold / pred / decided_by / reason / cost / escalated）
encoding="utf-8-sig"        # Windows Excel 打开中文 CSV 不乱码

misses = [(q, gold, pred, reason) for ... if pred != gold]
print(f"路由误判 {len(misses)} 条：")
```

> **评测的输出不是"我很棒"，而是"我哪里错了"。** 误判清单带 `reason` 字段，能直接看出是哪条正则误伤，然后回去调 `HARD_PATTERNS`。这叫 **error analysis**，是做实验和写论文的核心动作。

### 7.6 三个必须主动说的局限

| 局限 | 更好的做法 |
|---|---|
| 24 条样本太少，单次运行 | 扩到 100+，每例跑 3 次取均值报方差 |
| 质量只用"长度"代理 | 加 LLM-as-judge 打 1-5 分 + 人工抽检 20 条 |
| 只测单轮 | 加多轮测试集，验证记忆是否提升准确率 |

### 7.7 面试模板（背这段）

> "三策略对照实验：同一套 24 条标注测试集、同一份代码，只切换路由策略。指标分三类——**成本**（相对全云端 86.1%）、**质量**（路由准确率 95.8%，答案长度作代理）、**可用性**（72 次调用零降级零失败）。产物是逐条 CSV 加带理由的误判清单，能直接定位误判规则。局限我也清楚：样本量小、质量指标粗糙，下一步扩样本并引入 LLM-as-judge。"

杀伤力在于：**有数字、有归因、有产物、有局限、有下一步。**

---

## 8. 精读 `tools/capacity_plan.py`（规模化后还省吗）

### 8.1 它解决的问题

demo 阶段 1 张卡当然比 API 贵（大部分时间空转）。真问题是：**请求量涨到多少，自建才划算？**

输出 P0/P1/P2 三阶段的卡数、月成本、盈亏平衡点。

### 8.2 核心公式（文件头 3-6 行）

```
本地需要承载的 token/s = 峰值QPS × 简单请求占比 × 单请求token数
所需卡数 = ceil( 本地token/s ÷ (单卡有效吞吐 × 目标利用率) )
```

人话：**先算峰值时本地要扛多少 token，再除以一张卡能干多少活，向上取整。**

### 8.3 逐行精读（56-71 行）

```python
peak_qps  = daily / 86400 * a.peak_factor        # 日请求÷一天秒数×峰值系数(5)
local_tps = peak_qps * a.easy_ratio * per_req     # ×走本地比例×每请求token
cards     = max(1, math.ceil(local_tps / (a.gpu_tps * a.util)))  # 至少1张

if name.startswith("P2"):
    cards += a.ha_extra                            # 生产加冗余卡

gpu_cost   = cards * a.price * hours_month
cloud_cost = daily * (1 - easy_ratio) * cloud_per_req * 30   # 剩下40%走云端
baseline   = daily * cloud_per_req * 30
save       = (1 - total / baseline) * 100
```

三个参数的设计意图：

| 参数 | 值 | 为什么 |
|---|---|---|
| `peak_factor` | 5.0 | 按平均 QPS 买卡，峰值必崩；业界经验 3-5 倍 |
| `util` | 0.7 | 只用 70% 算力留 30% 抗突发；拉到 100% 就是排队和超时 |
| `ha_extra` | 1 | 生产必须冗余，只有 P2 加（P0 加是浪费） |

### 8.4 盈亏平衡点（73-82 行）—— 最有价值的一段

```python
easy_per_hour = a.price / (a.easy_ratio * cloud_per_req)
```

推导：一张卡每小时花 `price` 元；每个请求走云端要花 `cloud_per_req` 元，其中 `easy_ratio` 比例能被本地接住 → **每小时接住 `easy_per_hour` 个请求，省下的云端费用刚好等于卡租**。

脚本还对比"按需 8 小时"与"24 小时常驻"两种用法：卡按小时计费，开机越久，需要的请求量越大——**这就是为什么小团队按需启停更划算**。

### 8.5 最值钱的一句（83-86 行）

```python
print("P0 阶段卡数打满下限（1 张），成本必然高于纯 API —— 这一阶段买的是"
      "「可行性验证 + 数据」，不是省钱。真正谈成本优化要等 P1/P2。")
```

> **面试话术（强烈建议讲）**："我在成本测算里明确写了 P0 阶段自建一定比 API 贵，因为 1 张卡有固定成本、请求量摊不平。这一阶段买的是可行性和真实数据，不是省钱。"
>
> **为什么值钱**：大部分学生会硬吹"我省了钱"。主动说"这个阶段不省钱、什么时候才开始省"，说明你理解了成本结构，而不是在演算数字。面试官对**诚实的边界感**评价极高。

### 8.6 两个细节

**① 吞吐必须来自实测**：`--gpu-tps` 默认 800，但项目配了 `bench_throughput.py` 压测出真值再填。**"参数来自实测"是工程和作业的分界线。**

**② 不只算 GPU**（88-94 行）：模型权重存储（按模型数×版本数）、出网带宽（按流式输出×并发数）、微调算力（8B LoRA 按样本量）。**说明你想的是完整系统，不是一个显卡数字。**

### 8.7 两份文件合起来是一个完整论证

```
run_eval.py        →  现在有效吗？  24 条实测：省钱 13.9%，路由准确 95.8%
capacity_plan.py   →  规模化后还行吗？ P1/P2 的卡数与盈亏平衡点
```

**一个证明"已经发生"，一个证明"可以外推"。两条腿都站稳，方案才算成立。**

### 8.8 自测题

1. 为什么评测要在 `import app` 之前设置 `DB_PATH`？
2. 为什么三组策略要用同一个 `run()` 入口？
3. `vs_cloud` 为什么用逐请求基线累加，而不是"全云端策略总成本"？
4. `util` 为什么取 0.7 而不是 1.0？
5. "24 条样本够吗"怎么答？

<details>
<summary>参考答案</summary>

1. `settings` 在 import 时就把环境变量读成常量，之后改无效。
2. 保证除路由外条件全一致，差异才能归因于策略。
3. 混合策略 token 消耗与全云端不同，逐请求折算才严格可比。
4. 满负荷没有排队余量，突发会直接超时；30% 余量是必要成本。
5. 承认不够，说明会扩到 100+、重复 3 次取均值报方差、加 LLM-as-judge——把"样本少"变成"我知道怎么更严谨"。

</details>

---

## 9. 精读阶段二：上下文预算 + 长期记忆（v1.2）

阶段二在流水线两头各加了一个工位：

```
进门先查档案                    出门前干两件事
        ↓                              ↓
START → load_memory → route → … → bookkeep → save_memory → END
（查小本本）      （干活）        （算账+压缩历史）  （往小本本上记）
```

比喻贯穿全程：**session_id 是"今天这张桌子"，user_id 是"会员卡"**。换桌子不销会员卡，会员卡也查不到别的桌子的聊天——两个键管两件不同的事。

### 9.1 `memory.py` —— 那个小本本

**抽取规则（`memory.py:22-30`）——最该讲的设计决策**

```python
(re.compile(r"我(?:叫|的名字是|名字叫)\s*([^\s，,。.!?！？]{1,12})"), "name"),
```

> **面试必问：为什么用正则抽取，不用大模型？**
> "记忆抽取发生在**每一轮**对话上。用模型抽，每轮多一次推理，钱和延迟都翻倍；但'我叫X''我在做X'这种强模式，正则的准确率本来就够。这是路由思想的复用——**判断成本要配得上判断价值**，值得记的话本来就说得很直白。"

每条规则是 `(正则, key)` 对：正则抠值，key 归类。`favorite` 规则的 key 是 lambda（`favorite_语言`、`favorite_球队`……一句话能记多种偏好维度）。

注意 `working_on` 的值允许空格（`[^，。；！？\n]`）而 `name` 不允许（`[^\s...]`）——因为"agent 项目"中间有空格，名字没有。**字符类是按"这个字段的实际长相"设计的**。

**疑问句过滤（`memory.py:38-39,66`）——真实 bug 修出来的防线**

测试时问"我在做什么项目？"，`working_on` 被更新成了**"什么项目"**——用户的问题被当成事实记下来了。修复一行：抠出来的值以疑问词开头 → 是问句不是陈述 → 跳过。

> "教训是：**从文本里抽事实，必须区分'陈述'和'疑问'**。正则只管'像不像事实'，还得有人管'是不是在问'。"

**`extract` / `remember` 分离（`memory.py:57,74`）——可测试性**

`extract` 是纯函数（只算不存），冒烟测试不碰数据库就能断言抽取逻辑。**"算"和"存"分离，是可测试性的最低成本实现。**

**一条 SQL 完成"插入或更新"（`memory.py:85-91`）**

```sql
ON CONFLICT(user_id, fact_key) DO UPDATE SET fact_value=excluded.fact_value
```

用户今天"我叫小明"明天"我叫小红"，记忆自动覆盖到最新。**覆盖即更新，不需要先查再改。**

**全量召回，为什么不用向量检索（`memory.py:96`）**

> "一个用户的事实是个位数到几十条，全查出来也就几十行——**数据量小到全量就是最优检索**。向量库是给'几十万条文档'准备的。什么时候上向量？做 RAG 检索文档时，届时和 RAG 复用同一套 embedding。**先问数据量，再选检索方式**，不是为了简历好看就上向量库。"

`recall` 里 `except sqlite3.OperationalError: return {}`（108 行）——表还没建也不崩。**记忆系统坏了不能拖垮对话主流程。**

### 9.2 `context.py` —— 给 prompt 管预算的会计

**token 估算（`context.py:30` 附近）**

中文 0.75 token/字、英文 0.3 token/字符，误差 ±15%。**为什么不装 tokenizer？** 为"估算预算"这一个用途引入依赖不值得，预算本身留了余量。**知道自己估算有多粗、且这个精度够用**——比"我用 tiktoken 精确计算"更能体现工程判断。

**`build_messages` 四步（每步一个取舍）**

1. **记忆注入：整块丢弃原则**。记忆块装不下就整块不要——截断后变成"我叫小"，**半条记忆比没有记忆更危险**。
2. **摘要注入**：跟在记忆后面一起进 system，对话区保持干净。
3. **历史装配：从新往旧装**。`reversed(history[-40:])` 先粗剪防极端情况，然后从最新消息往回装，`history` 预算（800 tok）用完就停。`+4` 是每条消息的 role/格式开销。**阶段一按条数（固定 8 条），阶段二按钱分配**——短消息能装 20 条，长消息可能只装 3 条。
4. **本轮问题无条件保留**。用户的问题永远不能丢，没有预算能砍它。

`history_dropped`、`memory_dropped` 等统计随响应返回——**你砍了什么，系统自己要记得**，否则调优是瞎调。

### 9.3 `graph.py` 三处改动

**`load_memory_node`（`graph.py:119`）—— 三行代码里的关键约束**

user_id 为空 → facts 为空 dict → 后面什么都不注入。这是评测 6/6 一致性的保障：**新功能必须能被"关掉关得干干净净"**，否则对照实验全作废。

**`bookkeep_node` 里的摘要压缩（`graph.py:252`）—— 全文最值得精读**

```python
history.append(user+assistant 两条)      # ① 先追加本轮
if len(history) > 20:                     # ② 再判断超没超
    old, keep = history[:-12], history[-12:]
    piece, tin, tout = _summarize(old)    # ③ 压缩最老的部分
    local_in += tin; local_out += tout    # ④ 摘要的钱进总账！
    summary = 旧摘要[:120] + piece        # ⑤ 新旧合并，防摘要无限膨胀
    history = keep                        # ⑥ 回落到 12 条
store.record(...)                         # ⑦ 最后才落库
```

- **为什么摘要在 bookkeep 里做？** 摘要的 token 也要花钱，放在记账节点里做，摘要成本才能进 cost。省钱系统自己花出去的钱记不上账，说不过去。
- **第 ⑤ 步为什么要合并旧摘要？** 不合并的话每压缩一次 summary 就变长，10 轮后 summary 自己成了最长的 prompt——**压缩历史结果造出一个更大的历史**。
- **`20` 和 `12` 的差是节流阀**：压完留 12 条，意味着不会每轮都触发（12→14→16→18→20 才再触发），避免每轮都花摘要的钱。

**`_summarize`（`graph.py:230`）—— 降级思想无处不在**

```python
try:    call_local(SUMMARIZE_PROMPT + transcript[:3000], max_tokens=160)
except ProviderError: return "；".join(逐条截30字), 0, 0
```

摘要质量差一点没关系，**链路不能断**。与主流程"本地挂了降云端"是同一哲学，只是方向相反：宁可降质量也不降可用性。`transcript[:3000]`——摘要输入本身也要限长。

**`save_memory_node`（`graph.py:333`）—— 只记用户说的**

只抽 `query` 不抽 `answer`：模型回答里会出现"用户说他是学生"这类**转述**，直接抽会把转述错当陈述。**宁可少记，不可记错**——记忆会被反复注入 prompt，错一条污染后面所有轮。

### 9.4 三个自测题

1. 把摘要压缩从 bookkeep 挪到 save_memory（落库之后），会丢什么？
2. 如果 extract 不过滤疑问句，污染会怎么扩散？
3. user_id 不传时评测还能 6/6 一致，这个设计保障了什么？

<details>
<summary>参考答案</summary>

1. 摘要调用的 token 进不了 cost——账漏记，成本数据从此不真实，且与落库的 history 对不上。
2. "我在做什么项目？"把 working_on 覆盖成"什么项目"→ 每次召回都注入这条错误事实 → 模型照着胡说 → **记忆是正反馈循环，进一条脏数据会被放大**。
3. user_id 为空 → facts 空 → system 与基线完全相同 → 两条链路可比。**每个新功能都有干净的行为基线**，回归验证才有效——这是渐进式重构能持续的前提。

</details>

### 9.5 面试 30 秒串讲（背这段）

> "我的记忆分两层：短期是 checkpointer 按会话持久化的完整上下文，长期是跨会话的用户事实表。上下文组装按 token 预算分配——人设和本轮问题必留，记忆和摘要按优先级注入，历史从新到旧装到预算为止，装不下的老历史用本地小模型压成摘要，摘要的开销同样记账。事实抽取用正则而不是模型，因为抽取每轮都发生，判断成本要配得上判断价值；事实量级小，全量召回就是最优检索，数据量上来再演进到向量召回。整套东西的可观测性是关键：每轮注入了哪些记忆、裁掉几条历史、各部分 token 多少，全部随响应返回。"

这 30 秒里有：分层设计、预算控制、降级策略、成本意识、演进路线——面试官顺着任何一处往下挖，9.1-9.3 每节都有展开的弹药。

---

## 10. 高频面试问答（背到能脱稿讲 30 秒）

**Q1：为什么用 LangGraph，不用 LangChain 链或纯 if-else？**
> 见第 6 节标准答法。核心是"分支 + 回环 + 状态持久化"三件事需要框架，线性流程不需要。

**Q2：路由怎么判难度？判错了怎么办？**
> 三层：规则（0 成本，拦 70.8%）→ 本地小模型分类（~30 token）→ 长度兜底；外加质量自检兜网。
> **误判是不对称安全的**：误判 easy 顶多答不好被自检拉回来；误判 hard 多花钱但质量不损。实测准确率 95.8%。

**Q3：token 数从哪来？成本怎么算？**
> 云端用 API 返回的 `usage`；本地按"24GB 卡 ¥8/小时 ÷ 实测吞吐"折算单价——**本地不是免费的，这样对比才诚实**。
> 基线 = 同样 token 全走云端价，节省比例 = 1 − 实际/基线。

**Q4：并发下有什么问题？**
> SQLite 写锁粒度粗，`check_same_thread=False` 共享连接，demo 量级没问题。真上量要换 Postgres checkpointer + 连接池。**知道天花板在哪、怎么迁移，比装完美更可信。**

**Q5：记忆怎么做的？和 RAG 的区别？**
> 短期 = checkpointer 会话内状态 + 滑动窗口裁剪；长期（规划中）= 跨会话事实抽取进向量库。
> RAG 检索的是外部知识，记忆检索的是用户/会话相关状态——**检索机制可以复用，但数据源和生命周期不同**。

**Q6：怎么证明系统有效？**
> 自建标注测试集 + 三策略对照（全本地/全云端/混合），成本 86.1%、路由准确率 95.8%、72 次调用零失败。双链路开关还能做重构前后的回归对比。

---

## 11. 大白话术语表

| 术语 | 大白话 |
|---|---|
| State | 跟着工件走的工单 |
| Node | 工位，一个普通函数：拿工单 → 干活 → 交便利贴 |
| Conditional edge | 分拣员：看工单一栏，决定往左往右 |
| checkpointer | 游戏存档；`thread_id` 是存档槽位号 |
| partial update | 只交便利贴，不重写整张工单 |
| 降级 degraded | 主力挂了，备胎上（保可用性） |
| 升级 escalated | 主力答了但答得差，换更强的重答（保质量） |
| baseline 基线 | 假设全走云端的成本，用来算"省了多少" |
| fail-safe | 宁可给默认值，也不抛异常 |
| 容错解析 | 模型输出带废话，先用正则抠出 JSON 再解 |

---

## 12. 复习路径建议

1. **先看第 1 节全景图**，能自己画出八个步骤
2. **读 `graph.py` 的 `build_graph()`**（骨架）→ 六个节点（肉）→ `run_graph()`（入口）
3. **跑一遍** `python tools/smoke_langgraph.py`，对着输出复述每一步
4. **跑一遍** `python eval/run_eval.py --limit 5`，看表格和误判清单长什么样
5. **背第 9 节六个问答**，每个 30 秒
6. **练第 2.14 / 8.8 节自测题**，能讲出"为什么"而不只是答案

### 时间分配建议

| 剩余时间 | 只看这些 |
|---|---|
| 30 分钟 | 第 0 节数字 + 第 9 节问答 + 第 10 节术语表 |
| 2 小时 | 上面 + 第 1 节全景 + 第 2 节 graph.py 精读 |
| 半天 | 全篇通读 + 跑两个脚本 + 自测题 |
