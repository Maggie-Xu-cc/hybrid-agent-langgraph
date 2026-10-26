"""PyCharm / 任意 IDE 的启动入口：右键 Run 即可。

等价于命令行：
    python -m uvicorn app.server:app --host 127.0.0.1 --port 8080
"""

import os

import uvicorn

if __name__ == "__main__":
    if not os.path.exists(".env"):
        os.environ.setdefault("MOCK_MODE", "1")
        print("[提示] 未找到 .env，已启用 MOCK 模式（回答是假的，但链路可跑）。")
        print("[提示] 想用真实模型：复制 .env.example 为 .env 并填写配置。")

    print("服务地址： http://127.0.0.1:8080")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8080, reload=False)
