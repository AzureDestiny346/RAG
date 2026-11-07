"""FastAPI 服务层（W8）：/health + /ask（复用 ask_pipeline，CLI/HTTP 同一入口）。

启动：.venv\\Scripts\\python.exe -m uvicorn server:app --reload
"""
from fastapi import FastAPI, HTTPException, Query

from ask import ask_pipeline

app = FastAPI(title="rag-learn", description="最小可运行中文 RAG")


@app.get("/health")
def health() -> dict:
    """健康检查：服务化与演示的前置验证。"""
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str = Query(min_length=1, max_length=200)) -> dict:
    """问答接口：检索 + 生成，检索溯源（渠道标记）随响应返回。

    q 超长直接 422（防超长输入打穿 LLM 上下文预算）；
    LLM 凭据未配置时 503 + 可操作提示，不裸 500。
    """
    try:
        return ask_pipeline(q)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
