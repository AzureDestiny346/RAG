"""FastAPI 服务层（W5 仅健康检查；W8 挂 /ask 调用 retrieval+generation）。

启动：.venv\\Scripts\\python.exe -m uvicorn server:app --reload
"""
from fastapi import FastAPI

app = FastAPI(title="rag-learn", description="最小可运行中文 RAG")


@app.get("/health")
def health() -> dict:
    """健康检查：服务化与演示的前置验证。"""
    return {"status": "ok"}
