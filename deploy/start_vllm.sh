#!/usr/bin/env bash
# 一条命令搞定：判断环境 -> 缺权重就下 -> 起 vLLM。
#
# 用法（云端机器、在任意目录都行）：
#     bash deploy/start_vllm.sh
#     MODEL_ID=Qwen/Qwen3-8B-AWQ SERVED_NAME=Qwen3-8B-AWQ bash deploy/start_vllm.sh
#     MAX_LEN=4096 bash deploy/start_vllm.sh          # 显存紧张时降上下文
#
# 建议先开 tmux，否则关掉网页/终端，服务就没了：
#     tmux new -s gpu    （Ctrl+B 然后 D 退出但不结束）

set -uo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen2.5-7b-awq}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-7B-AWQ}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
PORT="${PORT:-8000}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

say "0. 环境检查"
echo "HF_ENDPOINT=$HF_ENDPOINT"
uname -a
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || warn "rocm-smi 报错，但先继续"
  rocm-smi --showmeminfo vram 2>/dev/null | head -20 || true
  rocm-smi --showproductname 2>/dev/null | head -20 || true
else
  warn "没有 rocm-smi。如果不是 AMD 环境，本脚本的 docker 路线走不通。"
  echo "  常见原因：需要 source /opt/rocm/.config 或把 /opt/rocm/bin 加进 PATH"
fi

say "1. 模型权重"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "已存在，跳过下载：$MODEL_DIR"
  du -sh "$MODEL_DIR" 2>/dev/null || true
else
  echo "下载 $MODEL_ID -> $MODEL_DIR （约 4-6 GB，走 hf-mirror）"
  python -m pip install -q -U huggingface_hub
  python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
model_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=model_id, local_dir=local_dir,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("下载完成:", local_dir)
PY
fi

VLLM_ARGS=(--model "$MODEL_DIR"
           --served-model-name "$SERVED_NAME"
           --max-model-len "$MAX_LEN"
           --gpu-memory-utilization "$GPU_UTIL"
           --max-num-seqs "$MAX_NUM_SEQS"
           --host 0.0.0.0 --port "$PORT")

say "2. 启动 vLLM"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && [ -e /dev/kfd ]; then
  echo "路线 A：docker + rocm/vllm 镜像（最省事，推荐）"
  docker rm -f vllm >/dev/null 2>&1 || true
  MOUNT_DIR="$(dirname "$MODEL_DIR")"
  INNER_MODEL="/models/$(basename "$MODEL_DIR")"
  exec docker run --rm --name vllm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size 16G \
    -p "${PORT}:8000" \
    -v "${MOUNT_DIR}:/models" \
    rocm/vllm:latest \
    vllm serve "$INNER_MODEL" \
      --served-model-name "$SERVED_NAME" \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --host 0.0.0.0 --port 8000
elif python -c "import vllm" >/dev/null 2>&1; then
  echo "路线 B：本机已装 vllm，直接启动"
  python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
  exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
else
  warn "既没有可用的 docker，也没装 vllm —— 别在这里死磕，走降级路线："
  cat <<'EOF'

  【降级 1：pip 装 vllm（有 root / conda 时通常可行）】
      pip install vllm
      bash deploy/start_vllm.sh          # 装完再跑一次本脚本

  【降级 2：Ollama（最快，10 分钟能出数据）】
      curl -fsSL https://ollama.com/install.sh | sh    # 需要 root
      ollama serve &
      ollama pull qwen2.5:7b
      # 然后改项目 .env 两行：
      #   LOCAL_BASE_URL=http://127.0.0.1:11434/v1
      #   LOCAL_MODEL=qwen2.5:7b
      # 代码一行不用改

EOF
  exit 1
fi
