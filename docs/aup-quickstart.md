# AUP 上跑起来（照抄版）

> 这份是给 AUP（AMD 云 GPU，Jupyter 环境）用的。本机怎么跑见 README 的「本机怎么跑最快」。
>
> 一句话：**AUP 的价值是那个 GPU**。有 GPU 才能跑"全本地 / 全云端 / 混合"三策略对比和真实吞吐压测——
> 这两块在纯云端模式下是跑不出来的（缺数据，不是缺代码）。

---

## 先看清楚：AUP 有两个硬约束

| 约束 | 后果 | 对策 |
|---|---|---|
| **单次会话约 20 分钟** | 装依赖 + 下 3GB 权重会吃掉大半，跑实验时间不够 | 拆成两次会话：第一次专门准备，第二次专门跑数据 |
| **断线/会话回收后进程全没** | 起的服务消失，跑一半的数据丢失 | 服务放 `tmux` 里；产物存持久分区并当场下载回本地 |

持久分区是 `/home/jovyan/`（一般会保留），`/tmp` 和 `~` 之外的临时目录会被清掉。
所以：**权重缓存放 `/home/jovyan/hf_cache`，代码放 `/home/jovyan/`**。

---

## 第 0 步：启动实例，选对镜像

启动时**必须选 GPU 镜像**（名字里通常带 `rocm` 或 `gpu`）。选成纯 CPU 镜像，后面所有步骤都会
在"GPU 不可用"这一关失败——而且这个错很难看出来，因为代码本身完全正常。

进入 JupyterLab 后打开 **Terminal**。下面所有命令都在这个终端里敲。

---

## 第 1 步：确认真的分到卡了（30 秒，别跳过）

```bash
rocm-smi
```

**应该看到**：一张显卡的型号、显存、温度表格。

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

**应该看到**：两个版本号 + `True`，类似 `2.4.0+rocm6.0 6.0.32830 True`。

⚠️ 如果第二个是 `None`，说明是 **CPU 版 torch**，显卡用不了。这种情况别继续，
换镜像或按第 8 节的报错表处理。

---

## 第 2 步：把代码弄上去

**方式 A（推荐，仓库是公开的话）**

```bash
cd /home/jovyan
git clone https://github.com/Maggie-Xu-cc/hybrid-agent-langgraph.git
cd hybrid-agent-langgraph
```

**应该看到**：`Cloning into 'hybrid-agent-langgraph'...` 然后结束，`ls` 能看到 `app/ deploy/ eval/`。

**方式 B（仓库私有，或者 clone 一直卡）**

本地打包 → JupyterLab 左侧文件区点上传：

```bash
# 在本地项目根目录执行（Windows Git Bash）
cd /d/AgentLearningProject && tar czf hybrid-agent.tgz --exclude='.venv*' --exclude='.localdata' --exclude='.deploy' --exclude='.tmp*' hybrid-agent-langgraph
```

把生成的 `hybrid-agent.tgz` 拖进 JupyterLab 上传，然后在 AUP 终端：

```bash
cd /home/jovyan && tar xzf hybrid-agent.tgz && cd hybrid-agent-langgraph
```

---

## 第 3 步：一次自检，把所有问题看全

```bash
bash deploy/aup_setup.sh
```

这个脚本会检查 GPU / 依赖 / 权重缓存 / `.env` 四块，**并且不会动你的环境**。它同时会提醒
`.env` 里那个最容易踩的坑：本机是给 Ollama 配的 `qwen:latest`，AUP 上用的是
`Qwen/Qwen2.5-1.5B-Instruct`，两者不一致就会报 `404 model not found`。

缺依赖就装上：

```bash
bash deploy/aup_setup.sh --install
```

**应该看到**：`[OK] 依赖安装完成`。

> 为什么单独一个 `requirements-aup.txt`：本机不需要 torch，装进主依赖会让本机也去下 2GB。
> 而且 **AUP 的 torch 必须用镜像自带的 ROCm 版**，手动 `pip install torch` 很可能被换成 CPU 版。

---

## 第 4 步：下权重（单独一次会话做）

```bash
bash deploy/aup_setup.sh --predownload
```

**应该看到**：进度条，最后 `[OK] 权重就绪：/home/jovyan/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/...`

约 3GB，10~30 分钟。中途断了直接重跑，会续传。

**这一步做完就可以关会话了**——下次进来权重还在，不用重下。这就是为什么要单独一次会话。

---

## 第 5 步：配 `.env`

```bash
cp .env.example .env
nano .env
```

只改这三行（其余保持默认）：

```ini
LOCAL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
CLOUD_API_KEY=sk-你的真实key
```

`nano` 里改完：`Ctrl+O` 回车保存，`Ctrl+X` 退出。

再确认两行没跑偏：

```ini
MOCK_MODE=0          # 0 才是真调用；1 是假回答，数据不能进报告
DISABLE_LOCAL=0      # AUP 有 GPU，必须让本地上场，否则三策略对比失去意义
```

---

## 第 6 步：起本地模型（关键，用 tmux）

**不要直接 `python deploy/local_server.py`**——Jupyter 页面一断，进程就没了。

```bash
tmux new -s gpu
```

进入 tmux 后（底部有一行绿色状态栏），在里面执行：

```bash
cd /home/jovyan/hybrid-agent-langgraph
export HF_ENDPOINT=https://hf-mirror.com
python deploy/local_server.py
```

**应该看到**（第一次要等 1~3 分钟加载权重）：

```
[local_server] 加载 Qwen/Qwen2.5-1.5B-Instruct 到 cuda（首次运行会下载权重，请等它跑完）
[local_server] 就绪：http://0.0.0.0:8000/v1  模型名=Qwen/Qwen2.5-1.5B-Instruct
[local_server] 保持这个终端别关。评测/压测请在【另一个终端】跑。
```

看到「就绪」后按 `Ctrl+B`，松开再按 `D` ——**离开 tmux 但进程继续跑**。
回到 Jupyter 主界面（不是 tmux 里），新开一个 Terminal 做第 7 步。

> 想回去看服务日志：`tmux attach -t gpu`

---

## 第 7 步：验链路，再一键采集

新终端里：

```bash
cd /home/jovyan/hybrid-agent-langgraph

# ① 先确认两个模型都通
python tools/preflight.py
```

**应该看到**：四行 `[OK]`，最后一句「全部通过」。本地那行会显示「模型名匹配」。

```bash
# ② 一键采集：并发压测 → 三策略评测 → 语义缓存基准 → 算力重算
python deploy/run_all.py
```

**预计耗时**：15~25 分钟（压测最慢）。如果会话时间不够，拆开跑：

```bash
python deploy/run_all.py --skip-bench --skip-cache   # 只跑压测 + 评测（最耗时，但也是最有价值的）
python tools/cache_bench.py                          # 缓存基准单独跑，几分钟
```

**应该看到**：每步前有 `>>>` 标记，最后「全部完成」。

---

## 第 8 步：把数据带回本地（会话回收就没了）

```bash
cd /home/jovyan/hybrid-agent-langgraph
tar czf evidence.tgz docs/evidence/
ls -lh evidence.tgz
```

然后在 JupyterLab 左侧文件区找到 `evidence.tgz`，右键 **Download**。

本地解压后 `RESULTS.md` 里就是完整的三节数据（压测 / 三策略对比 / 语义缓存）。

---

## 一张表：报错对照

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `rocm-smi` 命令不存在 | 选的不是 GPU 镜像 | 换镜像重启实例 |
| `torch.version.hip` 是 `None` | 装成了 CPU 版 torch | 别自己 `pip install torch`，用镜像自带的 |
| `HIP out of memory` | 显存不够（1.5B 一般 4GB 够） | 换更小模型 `--model Qwen/Qwen2.5-0.5B-Instruct` |
| `404 model not found` | `.env` 的 `LOCAL_MODEL` 和服务端名字不一致 | 两者一字不差，`aup_setup.sh` 会提前提醒 |
| 加载卡在 `Downloading` 不动 | 直连 HuggingFace 不通 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 会话断开后 `preflight` 本地那行 FAIL | tmux 里的服务跟着没了 | 重新 `tmux attach -t gpu` 或重启服务 |
| `run_all.py` 说本地不可达直接退出 | 服务没起或端口不对 | 先 `curl localhost:8000/v1/models` 确认 |

---

## 附：如果这次拿不到 GPU

AUP 排队排不到时，用 `--cloud-only` 也能产出**一部分**真实数据（缓存基准），
但会明确标注缺哪几节、为什么缺：

```bash
python deploy/run_all.py --cloud-only
```

`RESULTS.md` 里会写清「本次为纯云端模式，缺并发压测与三策略对比，原因是它们测的就是本地小模型」。
**宁可缺数据并说明原因，也不要拿污染过的数据充数**——这一点在答辩和面试里都是加分项。
