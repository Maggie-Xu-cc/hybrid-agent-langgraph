"""轻量本地推理服务：transformers + 标准库 HTTP，OpenAI 兼容协议。

为什么存在：
    AUP 的 gfx1151 上没有（也很难装）vLLM，但整套 demo——评测、压测、Agent
    主流程——只要求一个 OpenAI 兼容的 localhost:8000/v1。这个脚本用约 100 行
    把 transformers 包成同样的接口，链路就能在真实 GPU 上全部跑通。

用法（项目根目录）：
    export HF_ENDPOINT=https://hf-mirror.com
    python deploy/local_server.py                    # 默认 Qwen2.5-1.5B-Instruct，端口 8000
    python deploy/local_server.py --port 8010        # 换端口

首次运行会下载权重（约 3 GB），之后 aup_gpu_demo.py 复用同一缓存，秒开。
注意：generate 有全局锁，串行生成——iGPU 上这是诚实的行为，压测得到的
吞吐是"排队后吞吐"，不是虚高的并发相加。
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCK = threading.Lock()
STATE = {"model": "", "tok": None, "model_obj": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默访问日志，避免刷屏干扰截图
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": STATE["model"], "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            messages = payload.get("messages", [])
            max_new = int(payload.get("max_tokens") or 256)
            if not messages:
                return self._json({"error": "messages is empty"}, 400)

            tok, model = STATE["tok"], STATE["model_obj"]
            t0 = time.perf_counter()
            with LOCK:
                inputs = tok.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        pad_token_id=tok.eos_token_id,
                    )
            text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            pt, ct = int(inputs.shape[1]), int(out.shape[1] - inputs.shape[1])
            elapsed = time.perf_counter() - t0

            self._json({
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "model": STATE["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                "server_latency_ms": int(elapsed * 1000),
            })
        except Exception as exc:  # 任何异常都以 500 返回，让调用方走降级路径
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    ap = argparse.ArgumentParser(description="transformers 版 OpenAI 兼容本地服务")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace 模型名")
    ap.add_argument("--cache-dir", default="/home/jovyan/hf_cache", help="权重缓存目录（放持久化分区）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[local_server] 加载 {args.model} 到 {device}（首次运行会下载权重，请等它跑完）", flush=True)
    STATE["model"] = args.model
    STATE["tok"] = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=args.cache_dir, dtype=dtype,
        )
    except TypeError:  # 旧版 transformers 用 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=args.cache_dir, torch_dtype=dtype,
        )
    STATE["model_obj"] = model.to(device).eval()

    print(f"[local_server] 就绪：http://{args.host}:{args.port}/v1  模型名={args.model}", flush=True)
    print("[local_server] 保持这个终端别关。评测/压测请在【另一个终端】跑。", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
