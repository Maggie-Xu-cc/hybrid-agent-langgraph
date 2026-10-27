#!/usr/bin/env bash
# AUP 环境自检 + 准备：在 AUP 的 Terminal 里跑一次，把"该装什么、缺什么"一次说清。
#
# 用法（项目根目录）：
#   bash deploy/aup_setup.sh              # 只检查，不动环境
#   bash deploy/aup_setup.sh --install    # 缺什么装什么（pip 走清华镜像）
#   bash deploy/aup_setup.sh --install --predownload   # 顺便把权重下到持久分区
#
# 为什么要有这个脚本：
#   AUP 单次会话大约 20 分钟，装依赖 + 下 3GB 权重能吃掉大半。
#   把"检查"和"下载"拆成独立步骤，就可以分两次会话做，
#   第二次直接跑实验，不重下权重。
#
# 不要用 set -e：某一步失败也要把后面能查的都查完，一次把问题看全。

set -u
cd "$(dirname "$0")/.." || exit 1

DO_INSTALL=0
DO_PREDL=0
for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    --predownload) DO_PREDL=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（可用：--install / --predownload）"; exit 2 ;;
  esac
done

PY="${PYTHON:-python}"
MODEL="${LOCAL_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
CACHE_DIR="${HF_CACHE_DIR:-/home/jovyan/hf_cache}"
# 国内直连 HuggingFace 经常卡死，默认走镜像；要改就 export HF_ENDPOINT=...
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

fail=0
say()  { printf '%s\n' "$*"; }
ok()   { printf '[OK]   %s\n' "$*"; }
bad()  { printf '[FAIL] %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '[WARN] %s\n' "$*"; }

say "=============================================="
say " AUP 环境自检"
say "=============================================="
say "  Python : $($PY --version 2>&1)"
say "  工作目录: $(pwd)"
say "  HF 镜像 : $HF_ENDPOINT"
say ""

# ---------- 1. GPU：整个 AUP 路线的前提 ----------
say "-- 1. GPU 是否可用 ------------------------------------------"
GPU_OK=$($PY - <<'PY' 2>/dev/null
try:
    import torch
    print("1" if torch.cuda.is_available() else "0")
except Exception:
    print("0")
PY
)
$PY - <<'PY'
try:
    import torch
    hip = getattr(torch.version, "hip", None)
    print(f"  torch       : {torch.__version__}")
    print(f"  ROCm (hip)  : {hip or 'None  <-- 这不是 ROCm 版，装成了 CPU 版'}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU 型号    : {torch.cuda.get_device_name(0)}")
        print(f"  显存        : {p.total_memory / 1024**3:.1f} GB")
    else:
        print("  GPU         : 不可用")
except Exception as exc:
    print(f"  torch       : 导入失败 -> {type(exc).__name__}: {exc}")
PY

if [ "$GPU_OK" = "1" ]; then
  ok "GPU 可用"
else
  bad "GPU 不可用 —— 先确认启动实例时选的是 GPU 镜像（不是 CPU 镜像）"
  say "        另开一个终端跑 rocm-smi，能看到显卡才算真的分到了卡"
fi
say ""

# ---------- 2. 依赖 ----------
say "-- 2. Python 依赖 -------------------------------------------"
if $PY -c "import transformers" 2>/dev/null; then
  ok "transformers $($PY -c 'import transformers; print(transformers.__version__)')"
else
  bad "transformers 未安装（local_server.py 靠它加载模型）"
fi
if $PY -c "import fastapi, uvicorn, openai" 2>/dev/null; then
  ok "fastapi / uvicorn / openai 已装"
else
  warn "web 服务依赖不全（只跑评测/基准不影响，起页面才需要）"
fi

if [ "$DO_INSTALL" = "1" ]; then
  say ""
  say "  开始安装（requirements-aup.txt，走清华镜像）..."
  $PY -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
      -r requirements-aup.txt \
    && ok "依赖安装完成" \
    || bad "依赖安装失败，看上面的报错；常见是网络问题，重跑一次即可"
fi
say ""

# ---------- 3. 权重 ----------
say "-- 3. 模型权重 ---------------------------------------------"
say "  目标：$MODEL"
say "  缓存：$CACHE_DIR （放持久分区，会话回收后不用重下）"
mkdir -p "$CACHE_DIR" 2>/dev/null || warn "建不了 $CACHE_DIR，检查路径权限"

if [ "$DO_PREDL" = "1" ]; then
  say "  开始下载（约 3 GB，10~30 分钟，断了重跑会续传）..."
  $PY - <<PY
import os, sys
os.environ.setdefault("HF_ENDPOINT", "$HF_ENDPOINT")
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("$MODEL", cache_dir="$CACHE_DIR",
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"[OK]   权重就绪：{path}")
except Exception as exc:
    print(f"[FAIL] 下载失败：{type(exc).__name__}: {str(exc)[:200]}")
    print("       换个时间重跑，或先 export HF_ENDPOINT=https://hf-mirror.com")
    sys.exit(1)
PY
  [ $? -ne 0 ] && fail=$((fail+1))
else
  # 只检查缓存里有没有，不主动下
  if ls "$CACHE_DIR"/models--* >/dev/null 2>&1; then
    ok "缓存目录已有下载记录（大概率不用重下）"
  else
    warn "缓存目录是空的，第一次跑要先下 3 GB"
    say "        建议单独一次会话执行：bash deploy/aup_setup.sh --predownload"
  fi
fi
say ""

# ---------- 4. 配置 ----------
say "-- 4. 配置文件 ---------------------------------------------"
if [ -f .env ]; then
  ok ".env 已存在"
  grep -q "sk-xxxxxxxx" .env 2>/dev/null && bad ".env 里 CLOUD_API_KEY 还是占位符，没填真 key"
  grep -q "^MOCK_MODE=1" .env 2>/dev/null && warn "MOCK_MODE=1：回答是假的，不能进报告"

  LM=$(grep "^LOCAL_MODEL=" .env | cut -d= -f2- | tr -d '\r')
  LB=$(grep "^LOCAL_BASE_URL=" .env | cut -d= -f2- | tr -d '\r')
  say "        LOCAL_BASE_URL=$LB"
  say "        LOCAL_MODEL=$LM"

  # 这一条最容易踩：本机 .env 是给 Ollama 配的（qwen:latest + 11434），
  # 原样搬到 AUP 上，local_server 报的是 Qwen/Qwen2.5-1.5B-Instruct + 8000，
  # 表现为"服务明明起来了，却一直 404 model not found"。
  case "$LB" in
    *11434*) bad "LOCAL_BASE_URL 指向 Ollama（11434），AUP 上用的是 local_server.py（8000）" ;;
  esac
  if [ -n "$LM" ] && [ "$LM" != "$MODEL" ]; then
    warn "LOCAL_MODEL=$LM 与 local_server.py 默认的 $MODEL 不一致"
    say "        两者必须一字不差，否则报 404 model not found"
    say "        要么把 .env 改成 LOCAL_MODEL=$MODEL，要么起服务时加 --model $LM"
  fi
else
  bad "没有 .env（先 cp .env.example .env，再填 CLOUD_API_KEY）"
fi
say ""

# ---------- 结论 ----------
say "=============================================="
if [ "$fail" -eq 0 ]; then
  ok "自检通过"
else
  bad "有 $fail 项没通过，先按上面的提示修"
fi
say ""
say "下一步："
say "  ① 起本地模型（终端 A，保持不关）："
say "       python deploy/local_server.py"
say "     看到这行才算成功："
say "       [local_server] 就绪：http://0.0.0.0:8000/v1  模型名=$MODEL"
say ""
say "  ② 另开终端 B，确认链路："
say "       python tools/preflight.py"
say ""
say "  ③ 一键采集（压测 + 评测 + 缓存基准）："
say "       python deploy/run_all.py"
say "     产物在 docs/evidence/，汇总 docs/evidence/RESULTS.md"
say ""
say "  断线重连后服务会没，用 tmux 保命："
say "       tmux new -s gpu     # 里面起服务，Ctrl+B 然后 D 退出但进程还在"
say "       tmux attach -t gpu  # 回来接着看"
say "=============================================="
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
