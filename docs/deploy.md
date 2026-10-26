# AMD 云端 GPU 部署手册（ROCm + vLLM）

每一步都建议留截图，方便复盘与复现。

> 想省事：跳过手动步骤，直接看文末「一条命令跑通」。

## 0. 环境检查

```bash
rocm-smi                      # 应能看到显卡型号与显存
rocm-smi --showmeminfo vram   # 显存总量
rocm-smi --showproductname    # 卡型号
rocminfo | grep gfx           # 架构代号，例如 gfx90a
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

- [ ] 截图 `rocm-smi` 输出 → `docs/evidence/01-rocm-smi.png`

## 1. 模型权重

**不要**在算力窗口里现下。要么提前拉到持久化目录，要么接受 5–10 分钟的下载时间（走镜像还行）。

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', local_dir='/models/qwen2.5-7b-awq')
print('done')"
```

**为什么默认 7B 而不是 8B/14B**：Qwen3 系列要求 vLLM ≥ 0.8.5，云平台镜像通常更旧，报错是
`model type qwen3 not recognized`，排查要半小时。Qwen2.5-7B-AWQ 在所有版本上都稳，4bit 权重仅约 4.5 GB，
24GB 卡上余量充足，能把并发开到 8。

显存参考（ROCm + vLLM，8k 上下文）：

| 模型 | 权重占用 | 8k 上下文总占用（含 KV Cache） | 24GB 卡 |
|---|---|---|---|
| Qwen2.5-7B-AWQ | ~4.5 GB | ~10–13 GB | 宽松，并发可到 8 |
| Qwen2.5-14B-AWQ | ~9 GB | ~16–20 GB | 可用，并发受限 |
| Qwen2.5-32B-AWQ | ~18 GB | ~24 GB+ | 不建议 |

## 2. 启动 vLLM（官方 ROCm 镜像）

```bash
docker run -d --name vllm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -p 8000:8000 \
  -v /models:/models \
  rocm/vllm:latest \
  vllm serve /models/qwen2.5-7b-awq \
    --served-model-name Qwen2.5-7B-AWQ \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --host 0.0.0.0 --port 8000
```

> 没装 docker 时用 pip 路线（ROCm 版 torch 已就绪时通常直接可用）：
> ```bash
> pip install vllm
> python -m vllm.entrypoints.openai.api_server \
>   --model /models/qwen2.5-7b-awq --served-model-name Qwen2.5-7B-AWQ \
>   --max-model-len 8192 --gpu-memory-utilization 0.85 \
>   --max-num-seqs 16 --host 0.0.0.0 --port 8000
> ```

**关键参数**：

- `--max-model-len 8192`：Agent 场景上下文长，设太大 KV Cache 会吃光显存
- `--gpu-memory-utilization 0.85`：留 15% 给 kernel 与碎片，避免间歇性 OOM
- `--max-num-seqs 16`：单卡并发上限，压测时会摸到这个天花板
- AWQ 量化会被 vLLM 从 `config.json` 自动识别，不需要显式传 `--quantization`（显式传反而可能在旧版本上报错）

## 3. 验证

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-7B-AWQ","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

- [ ] 截图 vLLM 启动日志（模型名 + 参数） → `docs/evidence/02-vllm-log.png`
- [ ] 截图 `rocm-smi` 运行时显存占用 → **`docs/evidence/03-vram.png`，这是"确实用了云端算力"最硬的证据**

## 4. 压测（吞吐数据来源）

```bash
python tools/bench_throughput.py --concurrency 1,4,8 --num-prompts 16
python tools/bench_throughput.py --concurrency 4 --num-prompts 8 --pad-chars 6000 --label "长上下文约8k"
```

输出表格 + `docs/evidence/bench.csv`，最后一行 `SUGGESTED_GPU_TPS=<n>` 直接喂给容量测算：

```bash
python tools/capacity_plan.py --gpu-tps <n>
```

## 5. 常见错误

| 报错 | 原因 | 处理 |
|---|---|---|
| `HIP out of memory` | 上下文或并发过大 | 降 `--max-model-len` 到 4096；降 `--max-num-seqs` |
| `no kernel image is available` | 镜像不支持该 GPU 架构 | 换匹配 ROCm 版本的镜像 |
| `model type qwen3 not recognized` | vLLM 太旧 | 换 Qwen2.5 权重（本手册默认），或升级 vLLM |
| `AWQ is not supported` | 镜像未编译 AWQ kernel | 改用 fp16 权重（7B fp16 约 15GB，24GB 卡可跑） |
| 首次启动卡很久 | 正在编译算子 | 正常，等 3–10 分钟，别 Ctrl+C |
| 断连后进程没了 | 前台运行且没开 tmux | `tmux new -s gpu` 后再启；已死就重跑脚本 |
| 客户端 `404 model not found` | `.env` 的 `LOCAL_MODEL` 和服务端名字不一致 | 两者一字不差 |

## 6. 一条命令跑通（推荐）

```bash
# 1) 起服务：自动判环境（docker / pip vllm），缺权重自动下
bash deploy/start_vllm.sh

# 2) 另开一个 tmux 窗口，配 .env
cp .env.example .env && nano .env      # LOCAL_MODEL、CLOUD_API_KEY、MOCK_MODE=0

# 3) 自检 -> 一键采集
python tools/preflight.py
python deploy/run_all.py
```

`run_all.py` 产出全部落进 `docs/evidence/`，汇总在 `RESULTS.md`。

## 7. 没有 GPU 时的降级方案

算力窗口没批下来时，用 Ollama 先跑通全流程（代码零改动，只改 `.env` 两行）：

```bash
ollama serve
ollama pull qwen2.5:7b
# .env: LOCAL_BASE_URL=http://127.0.0.1:11434/v1, LOCAL_MODEL=qwen2.5:7b
```

之后切回 vLLM 只改一行配置——这正是"统一 OpenAI 兼容协议"带来的好处，可以作为一条工程决策记录下来。
但注意：这条路线与 vLLM 路径的吞吐、成本数据不可直接比较，压测要分别做。
