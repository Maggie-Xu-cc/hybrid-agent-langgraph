# Hybrid Routing Agent

一个**成本感知的混合推理 Agent**：本地小模型处理简单请求，复杂请求自动升级到云端大模型，并把每次调用的 Token、延迟、成本都记下来。

> 主流程用 LangGraph `StateGraph` 编排，带会话记忆与可切换的双链路（图编排 / 显式流程）对照评测。
> 核心不是"又一个聊天机器人"，而是回答一个问题：**在什么条件下，自建小模型 + 云端兜底比全部走 API 更划算？**
>
> 三级降本思路：**语义缓存命中（0 成本）→ 规则层拦下大部分请求（0 成本）→ 本地小模型承接简单请求（低）→ 云端大模型只处理真正难的（高）**。

## 在线试用

**https://hybrid-router-agent.app.workbuddy.host/?sharecode=CLU3I2I6TH**

一句话说明它和本地的差别：**云端容器没有 GPU，所以线上是纯云端形态**——路由、缓存、记忆、记账的逻辑与本地完全一致，只是"简单请求"那一层由云端代答。页面上会明确标注当前是哪一层产出的答案，不会把云端的结果说成本地小模型的功劳。

额度 300 次/天、单 IP 10 次/分钟：这是个人项目，背后是真实 API 计费，限额是为了让最坏情况的支出可预期。

想看完整的混合链路（真·本地小模型承接），按下面「本地运行」起本地模型即可。

## 架构

```
                 ┌──────────────────────────────┐
   HTTP 请求 ──▶ │  FastAPI  (app/server.py)    │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  路由 router.py              │
                 │  规则层 → 模型层 → 兜底层    │
                 └───────┬──────────────┬───────┘
                    easy │              │ hard
                         ▼              ▼
              ┌────────────────┐  ┌────────────────┐
              │ 本地 Qwen2.5-1.5B│  │ 云端大模型 API │
              │ transformers/ROCm│  │ OpenAI 兼容    │
              └───────┬────────┘  └───────┬────────┘
                      │  自检不通过 → 升级 │
                      └──────────┬────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ 记账 store.py (SQLite) │
                    │ token / 延迟 / 成本     │
                    └────────────────────────┘
```

**设计要点**：本地推理服务与云端 API 都走 OpenAI 兼容协议，所以"路由"退化成一次 `base_url` 切换，不需要维护两套调用逻辑。平台镜像未提供 vLLM，本地服务由 transformers + 标准库 HTTP 自建（`deploy/local_server.py`，120 行），上层零改动。

## LangGraph 编排（v1.1）

主流程迁移到 LangGraph `StateGraph`（`app/graph.py`），行为与原显式流程（`app/agent.py`，保留为基线）完全一致：

```
START ─▶ load_memory ─▶ route ──easy──▶ local_answer ─▶ self_check ─需升级─▶ escalate ─┐
                              │                               │                        │
                              └──hard──▶ cloud_answer         └─通过─▶ bookkeep ─▶ save_memory ─▶ END
                                         (cloud_answer ─────────────▶ bookkeep)
```

- 路由逻辑 `router.decide()` 原样保留，从"函数调用"变成**条件边**；记账集中在 `bookkeep` 节点
- **会话记忆（短期）**：`SqliteSaver` checkpointer 按 `session_id` 持久化对话状态（`checkpoints.db`），天然支持多轮对话
- 双链路可切换：`ENABLE_LANGGRAPH=1/0`，两条链路跑同一份评测集可做对比实验
- 冒烟测试（离线）：`python tools/smoke_langgraph.py`

### 上下文工程（v1.2）

`app/context.py` 按 token 预算组装 prompt，而不是简单截断：

```
system 人设（必留）＋ 长期记忆事实（≤200 tok）＋ 早前摘要（≤300 tok）＋ 最近对话（≤800 tok）＋ 本轮问题（必留）
```

- token 估算用字数折算（中文 0.75/字、其它 0.3/字符），不为估算引入 tokenizer 依赖
- 记忆块装不下时**整块丢弃**，不塞半截；每轮的注入统计（各部分 token、裁剪条数）随响应返回（`memory.ctx_tokens`），上下文是可观测的

### 长期记忆（v1.2）

`app/memory.py` 跨会话记住用户的关键事实，`session_id`（短期）与 `user_id`（长期）两个键独立：**新会话不会忘掉用户，新身份读不到别人的会话**。

- 抽取用正则（`我叫X / 我在做X / 我最喜欢的X是Y / 记住：X`），带疑问句过滤（"我在做什么项目？"不会把"什么项目"记成事实）；每轮的判断成本是 0
- 存储 SQLite 单表 `(user_id, fact_key, fact_value)`，覆盖旧值即记忆更新；事实量级小，全量召回就是最优检索，暂不需要向量库
- **历史摘要压缩**：对话超过 20 条时，把最老的部分用本地小模型压成摘要（失败退化为逐条截取），保留最近 12 条——历史的上限从"硬丢弃"变成"软压缩"，摘要调用的 token 照样记账

前端身份存 `localStorage`（默认生成，可与 `session_id` 分别重置）；「忘掉我」按钮清空当前身份的长期记忆，历史会话不受影响。

- `GET /api/memory/{user_id}` → 查看已记住的事实
- `DELETE /api/memory/{user_id}` → 清空某身份的长期记忆
- 冒烟测试（离线，26 项断言）：`python tools/smoke_memory.py`

### 语义缓存（v1.3）

`app/cache.py` 插在路由**之前**：相似的问题直接复用上次的答案，成本降为 0。降本链路完整变成四级：

```
语义缓存命中(¥0) ─▶ 规则路由(¥0) ─▶ 本地小模型(低) ─▶ 云端大模型(高)
```

实现要点（每条都是面试可展开的取舍）：

- **查询向量化零成本**：字符 n-gram hashing 到 256 维归一化向量（`embed()` 纯 Python，毫秒级）。为缓存再调一次 embedding API 就本末倒置——`embed()` 是独立函数，将来换 BGE 等真 embedding 模型只改这一个函数
- **阈值实测校准**：同义改写对 n-gram 余弦通常 ≥0.5、无关问题 ≤0.3，默认阈值取 0.45（`CACHE_THRESHOLD`）。短句寒暄召回弱是已知边界（好在寒暄本来就走 0 成本规则路由）
- **只缓存首轮独立问题**：多轮问题（"继续""为什么"）依赖会话上下文，换会话直接复用会答错——宁可少命中，不可答错。hard 请求默认不缓存（质量敏感+重复率低），`CACHE_CACHE_HARD=1` 可打开
- **两段式查找**：归一化 hash 精确匹配 O(1) 优先，再相似度线性扫描（条目量级小，毫秒级）；等量级上万再上向量库
- **TTL 过期**（默认 7 天）+ 命中刷新：过期答案可能已过时（"今天天气"），一律不返回

评测时显式关缓存（`run_graph(..., use_cache=False)`），否则同义用例第二次出现命中缓存，污染对照实验。

- `GET /api/cache` → 缓存概况（条目/命中/省下的 token 与成本）
- `DELETE /api/cache` → 清空缓存
- 冒烟测试（离线，36 项断言）：`python tools/smoke_cache.py`
- 基准（两遍原始问题 + 一遍改写问法）：`python tools/cache_bench.py`（加 `--mock` 离线跑）

### 分层追踪（v1.3）

每次请求落在哪一层、各层延迟与成本，记账表按 `decided_by` 归因，`/api/stats` 直接给出：

```
| 层     | 请求数 | 平均延迟 | 成本(¥) |
| cache  | 22     | 0ms      | 0.0000  |
| model  | 17     | ...      | ...     |
| rule   | 33     | ...      | ...     |
```

前端每个浏览器会话自动生成 `session_id`（存 `localStorage`，刷新不丢），页面上的「新会话」按钮换 id 即隔离上下文；后端不传时默认 `default`。

### 历史会话可以切回去

页面下方的「历史会话」列表列出所有会话（标题取第一轮提问、轮数、最后活跃时间），点一条即切回该会话并回显消息记录，可以接着聊。

- `GET /api/sessions?limit=20` → 会话列表
- `GET /api/sessions/{session_id}` → 该会话的消息记录
- `DELETE /api/sessions/{session_id}` → 删除单个会话（会话行右侧的 `×`，点两次确认）
- `DELETE /api/sessions` → 清空全部会话（卡片右上角「清空全部会话」，点两次确认）

实现上没有另建索引表：直接读 checkpointer 自己的 `checkpoints` 表拿线程列表，再用 `get_state(thread_id)` 取历史。排序用表内 `rowid`，时间从 `checkpoint_id`（UUIDv6，前 60 bit 是时间戳）解出来。这样**会话的唯一真相就是 checkpointer**，不会出现"索引表和实际记忆不同步"的问题。评测/冒烟脚本的线程（`eval-*`/`smoke-*`/`other-*`）会被过滤掉，不进用户界面。

删除会话走 checkpointer 的 `delete_thread()`（会同时清 `checkpoints` 和 `writes` 两张表），版本不兼容时回退为直接删表，保证不会留下"半截会话"。删掉当前会话后前端会自动开一个新会话，不会停在空壳上。

### 重构前后一致性怎么验

评测脚本支持按链路跑，并在 `--engine both` 时逐条比对两条链路的结果：

```bash
python eval/run_eval.py                  # 基线 agent.py，产物 eval/result.csv
python eval/run_eval.py --engine graph   # LangGraph 链路，产物 eval/result_graph.csv
python eval/run_eval.py --engine both    # 跑两遍 + 逐条一致性对比
```

一致性对比输出示例（MOCK_MODE 下 24 条）：

```
| 对比项 | 一致条数 | 占比 |
| 路由判定 | 24/24 | 100.0% |
| 是否升级 | 24/24 | 100.0% |
| 答案文本 | 24/24 | 100.0% |
```

> 逐条一致 = 图重构是**行为等价的改写**，不是借重构偷偷调策略。这是重构最基本的验收标准。

评测时每条用例用独立 `thread_id`（`eval-<uuid>`），避免上一条用例的历史污染下一条，保证成本与答案可比。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入云端 API Key
python -m uvicorn app.server:app --host 0.0.0.0 --port 8080
```

`MOCK_MODE=1` 可以完全离线跑通（不需要 GPU、不需要 Key），用于先联调页面与记账逻辑。

**本地模型三种起法**（都走 OpenAI 兼容协议，改 `.env` 即可切换）：

| 场景 | LOCAL_BASE_URL | LOCAL_MODEL |
|---|---|---|
| 本机 Ollama（最省事） | `http://127.0.0.1:11434/v1` | `ollama list` 里的模型名，如 `qwen:latest` |
| 云 GPU vLLM | `http://127.0.0.1:8000/v1` | `Qwen2.5-7B-AWQ`（与 `--served-model-name` 一致） |
| 无本地模型 | 指向一个没人监听的端口 | 无所谓 |

最后一行是"纯云端模式"：本地连不上会自动降级到云端（`degraded=true`），链路照样跑通，
只是成本与延迟按云端算——适合只验证逻辑、或者本地机器扛不住的时候。

打开 http://localhost:8080 即可对话。页面会实时显示本次请求走了哪条路、花了多少钱、比全云端省了多少。

本地模型的部署（AMD ROCm + transformers 自建 OpenAI 兼容服务）见 [docs/deploy.md](docs/deploy.md)。

## 本机怎么跑最快

没有 GPU、只有一台普通笔记本时，跑一轮完整缓存基准有三个提速开关，**可以叠加**：

| 开关 | 作用 | 省多少 |
|---|---|---|
| `DISABLE_LOCAL=1` | 完全不调本地，所有请求直接走云端 | 省掉每条请求白等本地的 2~15 秒 |
| `--easy-only` | 只跑 easy 用例（hard 按设计不进缓存，跑了也不计入命中率） | 调用量 72 → 39 次 |
| `--workers 4~8` | 并发发起请求（瓶颈是网络等待，不是 CPU） | 串行 19.8s → 并发 7.2s |

最快的一条命令：

```bash
DISABLE_LOCAL=1 python tools/cache_bench.py --easy-only --workers 4
```

本机实测（DeepSeek 云端，16 核 CPU，无本地 GPU）：

| 跑法 | 调用次数 | 耗时 |
|---|---|---|
| 串行 + easy-only | 39 | 19.8 s |
| 并发 4 + easy-only | 39 | **7.2 s** |
| 并发 4 + 全量（含 hard） | 72 | 25.6 s |

### 两个必须知道的坑（都已修掉）

**1. 超时默认 600 秒。** openai SDK 不给 `timeout` 时默认十分钟，本地模型"端口通但不回"
（Ollama 半死、vLLM 还在加载权重）会让每条请求白等十分钟，看起来就像卡死。
现在 `LOCAL_TIMEOUT` 默认 15 秒，且失败 2 次后**熔断**——冷却 60 秒内直接跳过本地，
调用立刻返回（实测 2146 ms → 0.00 ms），由上层走降级。冷却结束放一次试探，成功即恢复。

**2. 并发写 SQLite 会偶发 `readonly database`。** 根因是 WAL 的自动 checkpoint 需要独占访问，
多连接写同一个库时它永远抢不到，WAL 只涨不缩（实测涨到 4MB 上限）后**所有写入失败
且不可自愈**。修法是反直觉的：**改回单连接 + 全局锁**，建库只做一次，DB 层再加一层重试
（只重试 SQL，不重跑模型）。压测结果：60 次并发请求 **0 失败**，`ck.db-wal` 稳定在 4MB
不再膨胀。落盘串行化的代价是毫秒级，而瓶颈在网络调用上，吞吐不受影响。

> ⚠️ `DISABLE_LOCAL=1` 会跳过路由的模型层（第二层小模型分类），
> 路由只能靠规则层和兜底层，**准确率会下降**——实测有 6 条标注 hard 的被规则层判成 easy。
> 所以：**要快速看效果就用它；要写进报告的数字，请在本地模型可用时跑。**

## 云端实验：三条命令

> 在 **AUP**（AMD 云 GPU）上跑，先看 [docs/aup-quickstart.md](docs/aup-quickstart.md)——照抄版，
> 写清了每步应该看到什么、20 分钟会话怎么拆开、权重怎么放持久分区、报错对照表。
> 那上面第一件事是 `bash deploy/aup_setup.sh`（自检 GPU / 依赖 / 权重 / .env）。

```bash
python deploy/local_server.py  # 起 OpenAI 兼容本地服务（transformers，平台无 vLLM）
cp .env.example .env           # 改 LOCAL_MODEL / CLOUD_API_KEY / MOCK_MODE=0
python deploy/run_all.py       # 压测 + 三策略评测 + 算力重算 + 归档证据
```

`run_all.py` 会把所有产物写进 `docs/evidence/`（压测结果、评测明细、结果汇总），可直接用于复盘与汇报。

## 算力需求测算

```bash
python tools/capacity_plan.py --gpu-tps 800 --easy-ratio 0.6 --price 4
```

输出三阶段（概念验证 / 试点 / 推广）的卡数、月度 GPU 成本、云端兜底成本与盈亏平衡点，回答"请求量涨到多少自建才划算"。

吞吐用实测值，别拍脑袋：

```bash
python tools/bench_throughput.py --concurrency 1,4,8 --num-prompts 16
# 最后一行 SUGGESTED_GPU_TPS=<n>，把它填进 capacity_plan.py 的 --gpu-tps
```

## 路由策略

三层，逐层降本：

| 层 | 触发条件 | 成本 | 覆盖场景 |
|---|---|---|---|
| 规则层 | 代码块 / 数学符号 / 超长文本 / 纯寒暄 | 0 | 实测 70.8% 请求 |
| 模型层 | 规则不确定时，本地模型输出 JSON 分类 | 约 30 token | 中间地带 |
| 兜底层 | 分类失败时按长度阈值 | 0 | 极少数 |

另有一层**质量自检**：本地答案过短或包含"不确定"类表述时，自动升级到云端重答，并单独计数。

## 实验结果

> 24 条自建测试集，云端全连通条件下实测（`python eval/run_eval_stdlib.py` 可复现），72 次调用零降级、零失败。

| 策略 | 平均延迟 | 总成本 | 相对全云端 | 路由准确率 | 升级次数 |
|---|---|---|---|---|---|
| 全本地 | 2858 ms | ¥0.0151 | 18.2% | — | 0 |
| 全云端 | 2067 ms | ¥0.0829 | 100% | — | 0 |
| 混合路由 | 2165 ms | ¥0.0714 | **86.1%** | **95.8%** | 3 |

### 语义缓存实测（v1.3）

> 2026-10-27，DeepSeek 云端真实调用，`DISABLE_LOCAL=1`（本机无 GPU），
> 13 条 easy 用例 × 3 遍，`python tools/cache_bench.py --easy-only --workers 4`

| 轮次 | 请求数 | 命中 | 命中率 | 总成本 | 平均延迟 | 相对基线 |
|---|---|---|---|---|---|---|
| 第 1 遍（缓存空，基线） | 13 | 0 | 0% | ¥0.0090 | 1051 ms | 100% |
| 第 2 遍（相同问题，精确命中） | 13 | 13 | **100%** | ¥0.0000 | **0 ms** | **0%** |
| 第 3 遍（改写问法，语义命中） | 13 | 9 | **69.2%** | ¥0.0055 | 391 ms | 61% |

三点如实说：

- **第 2 遍成本归零**：完全相同的问法走 md5 精确通道，命中即返回，0 token 0 元 0 ms。
- **第 3 遍 69.2% 是 n-gram 方案的真实天花板**：它只认字面 n-gram 重叠，换个说法问到
  "结构不同但意思一样"时就漏。这个数字正是下一步换本地 embedding 模型的论据——
  `cache.embed()` 是独立函数，换模型上层一行不改。
- 全量（24 条含 hard）跑出来第 2 遍命中 19/24 而不是 13/24，多出来的是标注 hard 却被
  规则层判成 easy 的用例——因为 `DISABLE_LOCAL=1` 关掉了路由的模型层。
  要出报告数字，本地模型必须在位。

### 线上实测（2026-10-27，公开链接上的两次真实调用）

| 请求 | 实际产出层 | 延迟 | 成本 |
|---|---|---|---|
| 「用一句话解释什么是 REST API」 | cloud（云端代答） | 1054 ms | ¥0.00073 |
| 「REST API 是什么？」 | **cache**（相似度 0.569） | **3 ms** | **¥0** |

同一句话换个说法再问一次：**延迟降 350 倍，成本归零**。这就是语义缓存存在的全部理由。

### 完整证据链

上面每个数字都有原始日志，不是手填的：

```
docs/evidence/
  RESULTS.md        汇总（自动生成，含缺失项说明）
  cache_bench.log   缓存基准原始输出
  capacity.md       算力测算（按实测/估算吞吐分别标注）
  system.txt        采集环境信息
  models.json       服务端模型列表
```

复现方式：`python deploy/run_all.py`（没有本地模型时用 `--cloud-only`，它会明确告诉你
哪几节缺、为什么缺）。

## 成本模型

```
本地成本 = (输入+输出 token) / 1000 × 单价_local
云端成本 = 输入 / 1000 × 单价_in + 输出 / 1000 × 单价_out
基线成本 = 同样 token 全部按云端价计算
节省比例 = 1 - 实际成本 / 基线成本
```

本地单价不是 0：按"24GB 卡 ¥8/小时 ÷ 实测吞吐"折算，这样对比才诚实。参数见 `.env`。

## 算力说明

本地推理服务部署于 **AMD 云端 GPU**（ROCm + transformers 自建服务；vLLM 因平台镜像未提供，列为换卡后的迁移目标），部署命令、显存占用、压测数据见 [docs/deploy.md](docs/deploy.md)。

## 目录结构

```
app/
  config.py    环境变量
  clients.py   OpenAI 兼容客户端（本地/云端）+ 离线 mock + 本地熔断器
  router.py    三层难度路由
  agent.py     基线主流程（显式三步，ENABLE_LANGGRAPH=0 时使用）
  graph.py     LangGraph 编排主流程（默认）：缓存查找 + 条件边 + 自检升级 + 记账 + 会话记忆
  cache.py     语义缓存：n-gram 向量相似度匹配（插在路由之前）
  context.py   上下文工程：token 预算分配与 prompt 组装（system/记忆/摘要/历史）
  memory.py    长期记忆：事实抽取（正则）、落库、召回、遗忘
  store.py     SQLite 记录与统计（含分层追踪 by_layer）
  dbutil.py    SQLite 连接参数 + 忙/只读重试（重试贴着 SQL，不重跑模型）
  guard.py     公开演示护栏：每日额度 + 单 IP 限频
  server.py    FastAPI 服务（按开关选择链路，返回 engine 与 layer 归因）
demo_server.py 云端部署入口（读 PORT、绑 0.0.0.0、自动判定要不要启用本地层）
run_server.py  本地 PyCharm 入口（127.0.0.1:8080，右键 Run 即用）
web/index.html 演示页面（自动生成并复用 session_id 与 user_id，「新会话」/「忘掉我」）
eval/
  testset.jsonl  标注测试集
  run_eval.py    三策略对比（--engine plain|graph|both 选链路，both 会逐条比对一致性）
tools/
  capacity_plan.py   算力需求测算（三阶段预算）
  bench_throughput.py 并发压测（测单卡 tok/s）
  cache_bench.py      语义缓存基准（命中率/成本对比，--mock 可离线，--workers 并发）
  smoke_langgraph.py  LangGraph 链路冒烟测试（离线）
  smoke_memory.py     记忆与上下文冒烟测试（离线，26 项断言）
  smoke_cache.py      语义缓存冒烟测试（离线，36 项断言）
  smoke_breaker.py    本地熔断器冒烟测试（离线，17 项断言）
deploy/
  local_server.py    本地模型服务（transformers + 标准库 HTTP，无 vLLM 的机器用）
  start_vllm.sh      环境判定 + 下权重 + 起 vLLM（有 GPU 且能装 vLLM 时用）
  run_all.py         一键采集：压测 + 评测 + 缓存基准 + 算力重算 + 归档（--cloud-only 无本地模型时）
  aup_setup.sh       AUP 环境自检 + 装依赖 + 预下权重（--install / --predownload）
  prepare_bundle.py  生成部署包（白名单拷贝，避免把 venv 和本地数据库传上云）
requirements-aup.txt   AUP 专用追加依赖（transformers；torch 用镜像自带的 ROCm 版）
docs/evidence/   实验结果与原始日志（RESULTS.md 为自动生成的汇总）
docs/aup-quickstart.md   AUP 照抄版上手指南（每步应该看到什么 / 会话怎么拆）
docs/deploy.md   AMD ROCm 部署手册
docs/data-dictionary.md  记账表字段说明与成本归因
docs/code-reading.md     代码精读（面试复习用）
```

## License

MIT
