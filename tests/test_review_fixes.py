"""审查修复回归测试（code-review 双轴发现项）：配置优先级 / CRLF / 引擎缓存。"""
import importlib

import config


def test_deepseek_key_wins_over_stale_dashscope_url(monkeypatch):
    """配了 DeepSeek Key 时，残留的 DASHSCOPE_BASE_URL 不得劫持端点（Standards 轴）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://stale.example.com")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = importlib.reload(config)
    assert cfg.LLM_BASE_URL == "https://api.deepseek.com"
    assert cfg.LLM_MODEL == "deepseek-chat"
    importlib.reload(config)  # 恢复真实环境状态


def test_dashscope_key_uses_its_endpoint(monkeypatch):
    """DashScope Key + 显式 BASE_URL：端点与模型名跟随（既有行为锁定）。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://maas.example.com/v1")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = importlib.reload(config)
    assert cfg.LLM_BASE_URL == "https://maas.example.com/v1"
    assert cfg.LLM_MODEL == "qwen-plus"
    importlib.reload(config)


def test_chunking_handles_crlf():
    """Windows CRLF 语料：标题与正文不受 \\r 污染（特征化测试）。"""
    from chunking import chunk_markdown

    md = "# 标题甲\r\n正文一。\r\n## 标题乙\r\n正文二。\r\n"
    chunks = chunk_markdown(md, source="t.md")
    assert [c.heading for c in chunks] == ["标题甲", "标题甲 > 标题乙"]
    assert chunks[0].text.startswith("标题甲\n正文一。")


def test_engine_loaded_once_across_pipelines(monkeypatch):
    """/ask 高频调用不得反复加载模型与索引（双轴审查共同发现）。"""
    import ask

    loads = {"n": 0}

    def fake_load():
        loads["n"] += 1
        return ("bm25", "index", [{"chunk_id": 0}])

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    monkeypatch.setattr(ask, "load", fake_load)
    monkeypatch.setattr(ask, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(ask, "retrieve", lambda q, b, i, m, mo: [])
    monkeypatch.setattr(ask, "make_client", lambda: object())
    monkeypatch.setattr(ask, "answer", lambda c, q, ch: "ok")

    if hasattr(ask, "_engine"):
        ask._engine.cache_clear()
    ask.ask_pipeline("问题一")
    ask.ask_pipeline("问题二")
    assert loads["n"] == 1
    if hasattr(ask, "_engine"):
        ask._engine.cache_clear()
