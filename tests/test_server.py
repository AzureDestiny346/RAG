"""服务层接线单测（W8）：/health 与 /ask 的 HTTP 组装逻辑。

检索与 LLM 加载全部打桩——只测端点接线，不加载模型、不调真实 API。
运行：.venv\\Scripts\\python.exe -m pytest tests -q
"""
from fastapi.testclient import TestClient

import server


def test_health():
    assert TestClient(server.app).get("/health").json() == {"status": "ok"}


def test_ask_wiring(monkeypatch):
    """/ask 复用 ask_pipeline：打桩验证 HTTP 层（问题透传 + 结构原样返回）。"""
    captured = {}

    def fake_pipeline(q: str) -> dict:
        captured["q"] = q
        return {"question": q, "answer": "答案",
                "sources": [{"chunk_id": 7, "source": "s.md", "heading": "提示工程",
                             "in_bm25": True, "in_vector": False}]}

    monkeypatch.setattr(server, "ask_pipeline", fake_pipeline)
    resp = TestClient(server.app).get("/ask", params={"q": "提示工程资料在哪"})
    assert resp.status_code == 200
    assert captured["q"] == "提示工程资料在哪"
    body = resp.json()
    assert body["answer"] == "答案"
    assert body["sources"][0]["chunk_id"] == 7


def test_ask_requires_q():
    """/ask 缺参 -> 422（FastAPI 对必选 query 参数的标准行为）。"""
    assert TestClient(server.app).get("/ask").status_code == 422


def test_ask_rejects_overlong_query(monkeypatch):
    """/ask 的 q 超过 200 字符 -> 422（防超长输入打穿 LLM 上下文预算）。

    打桩 pipeline：若校验层失守，测试会快速失败而非触发真实 LLM 调用。
    """
    monkeypatch.setattr(server, "ask_pipeline", lambda q: {"ok": True})
    client = TestClient(server.app)
    assert client.get("/ask", params={"q": "字" * 201}).status_code == 422
    assert client.get("/ask", params={"q": "字" * 200}).status_code == 200


def test_ask_missing_key_returns_503(monkeypatch):
    """LLM 凭据未配置的 RuntimeError -> 503 + 人话 detail（不裸 500）。"""
    def boom(q: str) -> dict:
        raise RuntimeError("缺少 LLM Key：请在 .env 填入 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY。")

    monkeypatch.setattr(server, "ask_pipeline", boom)
    resp = TestClient(server.app).get("/ask", params={"q": "你好"})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["detail"]
