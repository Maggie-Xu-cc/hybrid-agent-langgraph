# Hybrid Routing Agent

一个**成本感知的混合推理 Agent**：本地小模型处理简单请求，复杂请求自动升级到云端大模型，并把每次调用的 Token、延迟、成本都记下来。

> 主流程用 LangGraph `StateGraph` 编排，带会话记忆与可切换的双链路（图编排 / 显式流程）对照评测。
> 核心不是"又一个聊天机器人"，而是回答一个问题：**在什么条件下，自建小模型 + 云端兜底比全部走 API 更划算？**
>
> 三级降本思路：规则层拦下大部分请求（0 成本）→ 本地小模型承接简单请求（低）→ 云端大模型只处理真正难的（高）。

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

主流程已迁移到 LangGraph `StateGraph`（`app/graph.py`），行为与原显式流程（`app/agent.py`，保留为基线）完全一致：

```
route ──easy──▶ local_answer ──▶ self_check ──需升级──▶ escalate ──▶ bookkeep ──▶ END
  └──hard──▶ cloud_answer ─────────────────────────────────────────▶ bookkeep
```

- 路由逻辑 `router.decide()` 原样保留，从"函数调用"变成**条件边**；记账集中在 `bookkeep` 节点
- **会话记忆**：`SqliteSaver` checkpointer 按 `session_id` 持久化对话状态（`checkpoints.db`），天然支持多轮对话；历史带滑动窗口（发给模型最近 8 条）
- 双链路可切换：`ENABLE_LANGGRAPH=1/0`，两条链路跑同一份评测集可做对比实验
- 冒烟测试（离线）：`python tools/smoke_langgraph.py`

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

打开 http://localhost:8080 即可对话。页面会实时显示本次请求走了哪条路、花了多少钱、比全云端省了多少。

本地模型的部署（AMD ROCm + transformers 自建 OpenAI 兼容服务）见 [docs/deploy.md](docs/deploy.md)。

## 云端实验：三条命令

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
  clients.py   OpenAI 兼容客户端（本地/云端）+ 离线 mock
  router.py    三层难度路由
  agent.py     基线主流程（显式三步，ENABLE_LANGGRAPH=0 时使用）
  graph.py     LangGraph 编排主流程（默认）：条件边 + 自检升级 + 记账 + 会话记忆
  store.py     SQLite 记录与统计
  server.py    FastAPI 服务（按开关选择链路，返回 engine 字段）
web/index.html 演示页面（自动生成并复用 session_id，「新会话」按钮切换）
eval/
  testset.jsonl  标注测试集
  run_eval.py    三策略对比（--engine plain|graph|both 选链路，both 会逐条比对一致性）
tools/
  capacity_plan.py   算力需求测算（三阶段预算）
  bench_throughput.py 并发压测（测单卡 tok/s）
  smoke_langgraph.py  LangGraph 链路冒烟测试（离线）
deploy/
  start_vllm.sh  环境判定 + 下权重 + 起 vLLM（换独立显卡后的备用脚本）
  run_all.py     一键采集：压测 + 评测 + 算力重算 + 归档
docs/deploy.md   AMD ROCm 部署手册
docs/data-dictionary.md  记账表字段说明与成本归因
docs/code-reading.md     代码精读（面试复习用）
```

## License

MIT
