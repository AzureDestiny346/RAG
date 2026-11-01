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
